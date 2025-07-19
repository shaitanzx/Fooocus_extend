import os
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageColor
import cv2
from tqdm import tqdm
import gradio as gr
##from modules import script_callbacks, shared, scripts
##from modules.paths_internal import models_path
##from modules.processing import StableDiffusionProcessing
##from modules.images import save_image
import torch
import tempfile
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread

class BlendMode:
    @staticmethod
    def _ensure_same_channels(target, blend):
        """Ensure both images have the same number of channels"""
        if target.shape[-1] == 4 and blend.shape[-1] == 3:
            alpha = np.ones((*blend.shape[:2], 1))
            blend = np.concatenate([blend, alpha], axis=-1)
        elif target.shape[-1] == 3 and blend.shape[-1] == 4:
            alpha = np.ones((*target.shape[:2], 1))
            target = np.concatenate([target, alpha], axis=-1)
        return target, blend

    @staticmethod
    def _apply_blend(target, blend, operation, opacity=1.0):
        """Apply blend operation with proper alpha handling"""
        target, blend = BlendMode._ensure_same_channels(target, blend)
        
        target_rgb = target[..., :3]
        blend_rgb = blend[..., :3]
        
        target_a = target[..., 3:] if target.shape[-1] == 4 else 1
        blend_a = blend[..., 3:] if blend.shape[-1] == 4 else 1
        
        result_rgb = operation(target_rgb, blend_rgb)
        result_a = target_a * blend_a
        
        result_rgb = result_rgb * opacity + target_rgb * (1 - opacity)
        result_a = result_a * opacity + target_a * (1 - opacity)
        
        return np.concatenate([result_rgb, result_a], axis=-1) if target.shape[-1] == 4 else result_rgb

    @staticmethod
    def normal(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: b, opacity)
    
    @staticmethod
    def multiply(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t * b, opacity)
    
    @staticmethod
    def screen(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: 1 - (1 - t) * (1 - b), opacity)
    
    @staticmethod
    def overlay(target, blend, opacity=1.0):
        def overlay_op(t, b):
            return np.where(t > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, overlay_op, opacity)
    
    @staticmethod
    def soft_light(target, blend, opacity=1.0):
        def soft_light_op(t, b):
            return np.where(b > 0.5,
                          t + (2 * b - 1) * (t - t * t),
                          t - (1 - 2 * b) * t * (1 - t))
        return BlendMode._apply_blend(target, blend, soft_light_op, opacity)
    
    @staticmethod
    def hard_light(target, blend, opacity=1.0):
        def hard_light_op(t, b):
            return np.where(b > 0.5,
                          1 - (1 - t) * (2 - 2 * b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, hard_light_op, opacity)
    
    @staticmethod
    def difference(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: np.abs(t - b), opacity)
    
    @staticmethod
    def exclusion(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t + b - 2 * t * b, opacity)
    
    @staticmethod
    def color_dodge(target, blend, opacity=1.0):
        def color_dodge_op(t, b):
            return np.where(b >= 1, 1, np.minimum(1, t / (1 - b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_dodge_op, opacity)
    
    @staticmethod
    def color_burn(target, blend, opacity=1.0):
        def color_burn_op(t, b):
            return np.where(b <= 0, 0, np.maximum(0, 1 - (1 - t) / (b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_burn_op, opacity)

class GeekyRemB:
    def __init__(self):
        self.session = None
        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")
        self.processing = False
        self.use_gpu = torch.cuda.is_available()
        self.frame_cache = {}
        self.max_cache_size = 100
        self.batch_size = 4  # Adjust based on your memory constraints
        self.max_workers = 4  # Adjust based on your CPU cores
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.blend_modes = {
            "normal": BlendMode.normal,
            "multiply": BlendMode.multiply,
            "screen": BlendMode.screen,
            "overlay": BlendMode.overlay,
            "soft_light": BlendMode.soft_light,
            "hard_light": BlendMode.hard_light,
            "difference": BlendMode.difference,
            "exclusion": BlendMode.exclusion,
            "color_dodge": BlendMode.color_dodge,
            "color_burn": BlendMode.color_burn
        }

    def process_frame_batch(self, frames, background_frames, *args):
        """Process multiple frames in parallel"""
        futures = []
        for frame, bg_frame in zip(frames, background_frames):
            future = self.executor.submit(self.process_frame, frame, bg_frame, *args)
            futures.append(future)
        return [future.result() for future in futures]

    def process_video(self, input_path, output_path, background_video_path, *args):
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            bg_cap = None
            if background_video_path:
                bg_cap = cv2.VideoCapture(background_video_path)
                bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_queue = queue.Queue(maxsize=self.batch_size * 2)
            result_queue = queue.Queue()
            
            # Use GPU-accelerated codec if available
            if self.use_gpu:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            def read_frames():
                frame_idx = 0
                while frame_idx < total_frames:
                    frames = []
                    bg_frames = []
                    for _ in range(self.batch_size):
                        if frame_idx >= total_frames:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        bg_frame = None
                        if bg_cap is not None:
                            bg_frame_idx = frame_idx % bg_total_frames
                            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
                            bg_ret, bg_frame = bg_cap.read()
                            if bg_ret:
                                bg_frame = cv2.resize(bg_frame, (width, height))
                        
                        frames.append(frame)
                        bg_frames.append(bg_frame)
                        frame_idx += 1
                    
                    if frames:
                        frame_queue.put((frames, bg_frames))
                frame_queue.put(None)

            def process_frames():
                while True:
                    batch = frame_queue.get()
                    if batch is None:
                        result_queue.put(None)
                        break
                    frames, bg_frames = batch
                    processed_frames = self.process_frame_batch(frames, bg_frames, *args)
                    result_queue.put(processed_frames)

            read_thread = Thread(target=read_frames)
            process_thread = Thread(target=process_frames)
            read_thread.start()
            process_thread.start()

            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while True:
                    processed_batch = result_queue.get()
                    if processed_batch is None:
                        break
                    for processed_frame in processed_batch:
                        out.write(processed_frame)
                        pbar.update(1)

            read_thread.join()
            process_thread.join()
            cap.release()
            if bg_cap:
                bg_cap.release()
            out.release()

            # Optimize final video encoding
            temp_output = output_path + "_temp.mp4"
            silent_output= output_path + "_silent.mp4"
            os.rename(output_path, temp_output)
            if self.use_gpu:
                os.system(f'ffmpeg -y -i "{temp_output}" -c:v h264_nvenc -preset p7 -tune hq -crf 23 "{silent_path}"')
            else:
                os.system(f'ffmpeg -y -i "{temp_output}" -c:v libx264 -preset faster -crf 23 "{silent_path}"')
            os.system(f'ffmpeg -y -i "{temp_output}" -i "{silent_output}" -c copy -map 1:v:0 -map 0:a? -map 0:s? "{output_path}"')    
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if os.path.exists(silent_output):
                os.remove(silent_output)

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise

        finally:
            if 'cap' in locals():
                cap.release()
            if 'bg_cap' in locals():
                bg_cap.release()
            if 'out' in locals():
                out.release()

    def apply_blend_mode(self, target, blend, mode="normal", opacity=1.0):
        if mode not in self.blend_modes:
            return blend
        
        target = target.astype(np.float32) / 255
        blend = blend.astype(np.float32) / 255
        
        result = self.blend_modes[mode](target, blend, opacity)
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if color == "green":
            lower = np.array([40 - color_tolerance, 40, 40])
            upper = np.array([80 + color_tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([90 - color_tolerance, 40, 40])
            upper = np.array([130 + color_tolerance, 255, 255])
        elif color == "red":
            lower = np.array([0, 40, 40])
            upper = np.array([20 + color_tolerance, 255, 255])
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
        if invert_mask:
            mask = 255 - mask

        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)

        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)

        return mask

    def parse_aspect_ratio(self, aspect_ratio_input):
        if not aspect_ratio_input:
            return None
        
        if ':' in aspect_ratio_input:
            try:
                w, h = map(float, aspect_ratio_input.split(':'))
                return w / h
            except ValueError:
                return None
        
        try:
            return float(aspect_ratio_input)
        except ValueError:
            pass

        standard_ratios = {
            '4:3': 4/3,
            '16:9': 16/9,
            '21:9': 21/9,
            '1:1': 1,
            'square': 1,
            'portrait': 3/4,
            'landscape': 4/3
        }
        
        return standard_ratios.get(aspect_ratio_input.lower())

    def calculate_new_dimensions(self, orig_width, orig_height, scale, aspect_ratio):
        new_width = int(orig_width * scale)
        
        if aspect_ratio is None:
            new_height = int(orig_height * scale)
        else:
            new_height = int(new_width / aspect_ratio)
        
        return new_width, new_height

    def remove_background(self, image, background_image, model, alpha_matting, alpha_matting_foreground_threshold, 
                      alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                      color_tolerance, background_mode, background_color, output_format="RGBA", 
                      invert_mask=False, feather_amount=0, edge_detection=False, 
                      edge_thickness=1, edge_color="#FFFFFF", shadow=False, shadow_blur=5, 
                      shadow_opacity=0.5, color_adjustment=False, brightness=1.0, contrast=1.0, 
                      saturation=1.0, x_position=0, y_position=0, rotation=0, opacity=1.0, 
                      flip_horizontal=False, flip_vertical=False, mask_blur=0, mask_expansion=0,
                      foreground_scale=1.0, foreground_aspect_ratio=None, remove_bg=True,
                      use_custom_dimensions=False, custom_width=None, custom_height=None,
                      output_dimension_source="Foreground", blend_mode="normal"):
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        if not isinstance(background_color, str) or not background_color.startswith('#'):
            background_color = "#000000"
        
        try:
            bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        except ValueError:
            bg_color = (0, 0, 0, 255)

        try:
            edge_color = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            edge_color = (255, 255, 255)

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        original_image = np.array(pil_image)

        if chroma_key != "none":
            chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
            input_mask = chroma_mask
        else:
            input_mask = None

        if remove_bg:
            removed_bg = remove(
                pil_image,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                post_process_mask=post_process_mask,
            )
            rembg_mask = np.array(removed_bg)[:, :, 3]
        else:
            removed_bg = pil_image.convert("RGBA")
            rembg_mask = np.full(pil_image.size[::-1], 255, dtype=np.uint8)

        if input_mask is not None:
            final_mask = cv2.bitwise_and(rembg_mask, input_mask)
        else:
            final_mask = rembg_mask

        final_mask = self.process_mask(final_mask, invert_mask, feather_amount, mask_blur, mask_expansion)

        orig_width, orig_height = pil_image.size
        bg_width, bg_height = background_image.size if background_image else (orig_width, orig_height)

        if use_custom_dimensions and custom_width and custom_height:
            output_width, output_height = int(custom_width), int(custom_height)
        elif output_dimension_source == "Background" and background_image:
            output_width, output_height = bg_width, bg_height
        else:
            output_width, output_height = orig_width, orig_height

        aspect_ratio = self.parse_aspect_ratio(foreground_aspect_ratio)
        new_width, new_height = self.calculate_new_dimensions(orig_width, orig_height, foreground_scale, aspect_ratio)

        fg_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        fg_mask = Image.fromarray(final_mask).resize((new_width, new_height), Image.LANCZOS)

        if background_mode == "transparent":
            result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        elif background_mode == "color":
            result = Image.new("RGBA", (output_width, output_height), bg_color)
        else:  # background_mode == "image"
            if background_image is not None:
                result = background_image.resize((output_width, output_height), Image.LANCZOS).convert("RGBA")
            else:
                result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))

        if flip_horizontal:
            fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
            fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)
            fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

        fg_image = fg_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
        fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

        paste_x = x_position + (output_width - fg_image.width) // 2
        paste_y = y_position + (output_height - fg_image.height) // 2

        # Apply blending mode
        if background_mode == "image" and background_image is not None:
            bg_array = np.array(result)
            fg_array = np.array(fg_image)
            
            # Ensure foreground array matches background dimensions before blending
            if bg_array.shape[:2] != fg_array.shape[:2]:
                # Resize foreground image to match background dimensions
                fg_image = fg_image.resize((output_width, output_height), Image.LANCZOS)
                fg_mask = fg_mask.resize((output_width, output_height), Image.LANCZOS)
                fg_array = np.array(fg_image)
            
            blended = self.apply_blend_mode(bg_array, fg_array, blend_mode, opacity)
            fg_with_opacity = Image.fromarray(blended)
            
            # Update paste coordinates since we resized
            paste_x = x_position
            paste_y = y_position
        else:
            fg_rgba = fg_image.convert("RGBA")
            fg_with_opacity = Image.new("RGBA", fg_rgba.size, (0, 0, 0, 0))
            for x in range(fg_rgba.width):
                for y in range(fg_rgba.height):
                    r, g, b, a = fg_rgba.getpixel((x, y))
                    fg_with_opacity.putpixel((x, y), (r, g, b, int(a * opacity)))

        # Ensure mask has same dimensions as image for pasting
        fg_mask_with_opacity = fg_mask.point(lambda p: int(p * opacity))
        if fg_mask_with_opacity.size != fg_with_opacity.size:
            fg_mask_with_opacity = fg_mask_with_opacity.resize(fg_with_opacity.size, Image.LANCZOS)

        result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask_with_opacity)

        if edge_detection:
            edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
            edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
            edge_overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
            edge_overlay.paste(Image.new("RGB", fg_image.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
            result = Image.alpha_composite(result, edge_overlay)

        if shadow:
            shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
            shadow_image = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
            shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (paste_x, paste_y), shadow_mask)
            result = Image.alpha_composite(result, shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur)))

        if color_adjustment:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)

        if output_format == "RGB":
            result = result.convert("RGB")

        return result, fg_mask

    def parse_color(self, color):
        """Safely parse color string to RGB tuple"""
        if isinstance(color, str) and color.startswith('#') and len(color) == 7:
            try:
                return tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        return (0, 0, 0)  # Default to black if parsing fails

    def process_frame(self, frame, background_frame=None, *args):
        """Process a single video frame with proper color handling"""
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_frame = frame
            
        args = list(args)
        
        if len(args) > 9:  # Handle background color
            bg_color = self.parse_color(args[9])
            args[9] = f"#{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}"
        
        if len(args) > 14:  # Handle edge color
            edge_color = self.parse_color(args[14])
            args[14] = f"#{edge_color[0]:02x}{edge_color[1]:02x}{edge_color[2]:02x}"
            
        if background_frame is not None:
            if isinstance(background_frame, np.ndarray):
                background_frame = Image.fromarray(cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB))
        
        args = tuple(args)
        processed_frame, _ = self.remove_background(pil_frame, background_frame, *args)
        return cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)

    def process_video(self, input_path, output_path, background_video_path, *args):
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            bg_cap = None
            if background_video_path:
                bg_cap = cv2.VideoCapture(background_video_path)
                bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    bg_frame = None
                    if bg_cap is not None:
                        bg_frame_idx = frame_idx % bg_total_frames
                        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
                        bg_ret, bg_frame = bg_cap.read()
                        if bg_ret:
                            bg_frame = cv2.resize(bg_frame, (width, height))
                    
                    processed_frame = self.process_frame(frame, bg_frame, *args)
                    out.write(processed_frame)
                    
                    frame_idx += 1
                    pbar.update(1)
            
            cap.release()
            if bg_cap:
                bg_cap.release()
            out.release()
            
            # Convert output video to MP4 container
            temp_output = output_path + "_temp.mp4"
            os.rename(output_path, temp_output)
            os.system(f'ffmpeg -i "{temp_output}" -c copy "{output_path}"')
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise

def on_ui_tabs():
    geekyremb_saving_path="outputs/geekyremb"
    geekyremb_max_video_length=3600
    geekyremb_max_image_size=8196
    with gr.Blocks(analytics_enabled=False) as geeky_remb_tab:
        gr.Markdown("# GeekyRemB: Background Removal and Image/Video Manipulation")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_type = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                foreground_input = gr.Image(label="Foreground Image", type="pil", visible=True)
                foreground_video = gr.Video(label="Foreground Video", visible=False)
                run_button = gr.Button(label="Run GeekyRemB")

                with gr.Group():
                    gr.Markdown("### Foreground Adjustments")
                    with gr.Group():
                        blend_mode = gr.Dropdown(
                            label="Blend Mode",
                            choices=["normal", "multiply", "screen", "overlay", "soft_light", 
                                    "hard_light", "difference", "exclusion", "color_dodge", "color_burn"],
                            value="normal"
                        )
                        opacity = gr.Slider(label="Opacity", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                    
                    foreground_scale = gr.Slider(label="Scale", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                    foreground_aspect_ratio = gr.Textbox(
                        label="Aspect Ratio",
                        placeholder="e.g., 16:9, 4:3, 1:1, portrait, landscape, or leave blank for original",
                        value=""
                    )
                    x_position = gr.Slider(label="X Position", minimum=-1000, maximum=1000, value=0, step=1)
                    y_position = gr.Slider(label="Y Position", minimum=-1000, maximum=1000, value=0, step=1)
                    rotation = gr.Slider(label="Rotation", minimum=-360, maximum=360, value=0, step=0.1)
                    
                    with gr.Row():
                        flip_horizontal = gr.Checkbox(label="Flip Horizontal", value=False)
                        flip_vertical = gr.Checkbox(label="Flip Vertical", value=False)

            with gr.Column(scale=1):
                result_type = gr.Radio(["Image", "Video"], label="Output Type", value="Image")
                result_image = gr.Image(label="Result Image", type="pil", visible=True)
                result_video = gr.Video(label="Result Video", visible=False)
                
                with gr.Group():
                    gr.Markdown("### Background Options")
                    remove_background = gr.Checkbox(label="Remove Background", value=True)
                    background_mode = gr.Radio(label="Background Mode", choices=["transparent", "color", "image", "video"], value="transparent")
                    background_color = gr.ColorPicker(label="Background Color", value="#000000", visible=False)
                    background_image = gr.Image(label="Background Image", type="pil", visible=False)
                    background_video = gr.Video(label="Background Video", visible=False)
                    duration_image = gr.Slider(label="Image2Video duration (sec)", minimum=0, maximum=600, value=5, step=1,visible=True)


        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Removal Settings")
                    model = gr.Dropdown(label="Model", choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"], value="u2net")
                    output_format = gr.Radio(label="Output Format", choices=["RGBA", "RGB"], value="RGBA")
                    alpha_matting = gr.Checkbox(label="Alpha Matting", value=False)
                    alpha_matting_foreground_threshold = gr.Slider(label="Alpha Matting Foreground Threshold", minimum=0, maximum=255, value=240, step=1)
                    alpha_matting_background_threshold = gr.Slider(label="Alpha Matting Background Threshold", minimum=0, maximum=255, value=10, step=1)
                    post_process_mask = gr.Checkbox(label="Post Process Mask", value=False)
                
                with gr.Column():
                    gr.Markdown("### Chroma Key Settings")
                    chroma_key = gr.Dropdown(label="Chroma Key", choices=["none", "green", "blue", "red"], value="none")
                    chroma_threshold = gr.Slider(label="Chroma Threshold", minimum=0, maximum=255, value=30, step=1)
                    color_tolerance = gr.Slider(label="Color Tolerance", minimum=0, maximum=255, value=20, step=1)
                
                with gr.Column():
                    gr.Markdown("### Effects")
                    invert_mask = gr.Checkbox(label="Invert Mask", value=False)
                    feather_amount = gr.Slider(label="Feather Amount", minimum=0, maximum=100, value=0, step=1)
                    edge_detection = gr.Checkbox(label="Edge Detection", value=False)
                    edge_thickness = gr.Slider(label="Edge Thickness", minimum=1, maximum=10, value=1, step=1)
                    edge_color = gr.ColorPicker(label="Edge Color", value="#FFFFFF")
                    shadow = gr.Checkbox(label="Shadow", value=False)
                    shadow_blur = gr.Slider(label="Shadow Blur", minimum=0, maximum=20, value=5, step=1)
                    shadow_opacity = gr.Slider(label="Shadow Opacity", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
                    color_adjustment = gr.Checkbox(label="Color Adjustment", value=False)
                    brightness = gr.Slider(label="Brightness", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                    contrast = gr.Slider(label="Contrast", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                    saturation = gr.Slider(label="Saturation", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                    mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=100, value=0, step=1)
                    mask_expansion = gr.Slider(label="Mask Expansion", minimum=-100, maximum=100, value=0, step=1)

            with gr.Row():
                gr.Markdown("### Output Settings")
                image_format = gr.Dropdown(label="Image Format", choices=["PNG", "JPEG", "WEBP"], value="PNG")
                video_format = gr.Dropdown(label="Video Format", choices=["MP4", "AVI", "MOV"], value="MP4")
                video_quality = gr.Slider(label="Video Quality", minimum=0, maximum=100, value=95, step=1)
                use_custom_dimensions = gr.Checkbox(label="Use Custom Dimensions", value=False)
                custom_width = gr.Number(label="Custom Width", value=512, visible=False)
                custom_height = gr.Number(label="Custom Height", value=512, visible=False)
                output_dimension_source = gr.Radio(
                    label="Output Dimension Source",
                    choices=["Foreground", "Background"],
                    value="Foreground",
                    visible=True
                )
        with gr.Row():
                gr.HTML('* \"Powered by GeekyGhost. <a href="https://github.com/GeekyGhost/Automatic1111-Geeky-Remb" target="_blank">\U0001F4D4 Document</a>')
        with gr.Row():
                gr.HTML('* Modification and adaptation for Fooocus is powered by Shahmatist^RMDA')

        def update_input_type(choice):
            return {
                foreground_input: gr.update(visible=choice == "Image"),
                foreground_video: gr.update(visible=choice == "Video")
            }

        def update_output_type(choice):
            return {
                result_image: gr.update(visible=choice == "Image"),
                result_video: gr.update(visible=choice == "Video")
            }

        def update_background_mode(mode):
            return {
                background_color: gr.update(visible=mode == "color"),
                background_image: gr.update(visible=mode == "image"),
                background_video: gr.update(visible=mode == "video")
            }

        def update_custom_dimensions(use_custom):
            return {
                custom_width: gr.update(visible=use_custom),
                custom_height: gr.update(visible=use_custom),
                output_dimension_source: gr.update(visible=not use_custom)
            }

        def process_image(image, background_image, *args):
            geeky_remb = GeekyRemB()
            result, _ = geeky_remb.remove_background(image, background_image, *args)
            return result

        def process_video(video_path, background_video_path, *args):
            geeky_remb = GeekyRemB()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                output_path = temp_file.name
            geeky_remb.process_video(video_path, output_path, background_video_path, *args)
            return output_path

        def run_geeky_remb(input_type, foreground_input, foreground_video, result_type, model, 
                          output_format, alpha_matting, alpha_matting_foreground_threshold,
                          alpha_matting_background_threshold, post_process_mask, chroma_key,
                          chroma_threshold, color_tolerance, background_mode, background_color,
                          background_image, background_video, invert_mask, feather_amount,
                          edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
                          shadow_opacity, color_adjustment, brightness, contrast, saturation,
                          x_position, y_position, rotation, opacity, flip_horizontal,
                          flip_vertical, mask_blur, mask_expansion, foreground_scale,
                          foreground_aspect_ratio, remove_background, image_format,
                          video_format, video_quality, use_custom_dimensions, custom_width,
                          custom_height, output_dimension_source, blend_mode,duration_image):
            
            if not isinstance(background_color, str) or not background_color.startswith('#'):
                background_color = "#000000"
            if not isinstance(edge_color, str) or not edge_color.startswith('#'):
                edge_color = "#FFFFFF"
            
            args = (model, alpha_matting, alpha_matting_foreground_threshold,
                   alpha_matting_background_threshold, post_process_mask, chroma_key,
                   chroma_threshold, color_tolerance, background_mode, background_color,
                   output_format, invert_mask, feather_amount, edge_detection,
                   edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity,
                   color_adjustment, brightness, contrast, saturation, x_position,
                   y_position, rotation, opacity, flip_horizontal, flip_vertical,
                   mask_blur, mask_expansion, foreground_scale, foreground_aspect_ratio,
                   remove_background, use_custom_dimensions, custom_width, custom_height,
                   output_dimension_source, blend_mode)

            if input_type == "Image" and result_type == "Image":
                result = process_image(foreground_input, background_image, *args)
                if image_format != "PNG":
                    result = result.convert("RGB")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format.lower()}") as temp_file:
                    result.save(temp_file.name, format=image_format, quality=95 if image_format == "JPEG" else None)
                    return temp_file.name, None
            elif input_type == "Video" and result_type == "Video":
                output_video = process_video(foreground_video, background_video if background_mode == "video" else None, *args)
                if video_format != "MP4":
                    temp_output = output_video + f"_temp.{video_format.lower()}"
                    os.system(f'ffmpeg -i "{output_video}" -c:v libx264 -crf {int(20 - (video_quality / 5))} "{temp_output}"')
                    os.remove(output_video)
                    output_video = temp_output
                return None, output_video
            elif input_type == "Image" and result_type == "Video":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    output_path = temp_file.name
                frame = cv2.cvtColor(np.array(foreground_input), cv2.COLOR_RGB2BGR)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
                for _ in range(24 * int(duration_image)):  # 5 seconds at 24 fps
                    out.write(frame)
                out.release()
                return None, process_video(output_path, background_video if background_mode == "video" else None, *args)
            elif input_type == "Video" and result_type == "Image":
                cap = cv2.VideoCapture(foreground_video)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = process_image(pil_frame, background_image, *args)
                    if image_format != "PNG":
                        result = result.convert("RGB")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format.lower()}") as temp_file:
                        result.save(temp_file.name, format=image_format, quality=95 if image_format == "JPEG" else None)
                        return temp_file.name, None
                return None, None

        input_type.change(update_input_type, inputs=[input_type], outputs=[foreground_input, foreground_video])
        result_type.change(update_output_type, inputs=[result_type], outputs=[result_image, result_video])
        background_mode.change(update_background_mode, inputs=[background_mode], 
                             outputs=[background_color, background_image, background_video])
        use_custom_dimensions.change(update_custom_dimensions, inputs=[use_custom_dimensions], 
                                   outputs=[custom_width, custom_height, output_dimension_source])

        run_button.click(
            fn=run_geeky_remb,
            inputs=[
                input_type, foreground_input, foreground_video, result_type,
                model, output_format, alpha_matting, alpha_matting_foreground_threshold,
                alpha_matting_background_threshold, post_process_mask, chroma_key,
                chroma_threshold, color_tolerance, background_mode, background_color,
                background_image, background_video, invert_mask, feather_amount,
                edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
                shadow_opacity, color_adjustment, brightness, contrast, saturation,
                x_position, y_position, rotation, opacity, flip_horizontal,
                flip_vertical, mask_blur, mask_expansion, foreground_scale,
                foreground_aspect_ratio, remove_background, image_format, video_format,
                video_quality, use_custom_dimensions, custom_width, custom_height,
                output_dimension_source, blend_mode,duration_image
            ],
            outputs=[result_image, result_video]
        )

    return [(geeky_remb_tab, "GeekyRemB", "geeky_remb_tab")]

#def on_ui_settings():
#    section = ("geeky-remb", "GeekyRemB")
#    shared.opts.add_option(
#        "geekyremb_saving_path",
#        shared.OptionInfo(
#            "outputs/geekyremb",
#            "GeekyRemB saving path",
#            gr.Textbox,
#            {"placeholder": "outputs/geekyremb"},
#            section=section
#        ),
#    )
#    shared.opts.add_option(
#        "geekyremb_max_video_length",
#        shared.OptionInfo(
#            300,
#            "Maximum video length in seconds",
#            gr.Number,
#            {"minimum": 1, "maximum": 3600},
#            section=section
#        ),
#    )
#    shared.opts.add_option(
#        "geekyremb_max_image_size",
#        shared.OptionInfo(
#            4096,
#            "Maximum image dimension",
#            gr.Number,
#            {"minimum": 512, "maximum": 8192},
#            section=section
#        ),
#    )

def update_input_type(choice):
    return {
        foreground_input: gr.update(visible=choice == "Image"),
        foreground_video: gr.update(visible=choice == "Video"),
    }

def update_output_type(choice):
    return {
        result_image: gr.update(visible=choice == "Image"),
        result_video: gr.update(visible=choice == "Video"),
    }

def update_background_mode(mode):
    return {
        background_color: gr.update(visible=mode == "color"),
        background_image: gr.update(visible=mode == "image"),
        background_video: gr.update(visible=mode == "video"),
    }

def update_custom_dimensions(use_custom):
    return {
        custom_width: gr.update(visible=use_custom),
        custom_height: gr.update(visible=use_custom),
        output_dimension_source: gr.update(visible=not use_custom)
    }

def process_image(image, background_image, *args):
    geeky_remb = GeekyRemB()
    result, _ = geeky_remb.remove_background(image, background_image, *args)
    return result

def process_video(video_path, background_video_path, *args):
    geeky_remb = GeekyRemB()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        output_path = temp_file.name
    geeky_remb.process_video(video_path, output_path, background_video_path, *args)
    return output_path

def run_geeky_remb(input_type, foreground_input, foreground_video, result_type, model, 
                   output_format, alpha_matting, alpha_matting_foreground_threshold,
                   alpha_matting_background_threshold, post_process_mask, chroma_key,
                   chroma_threshold, color_tolerance, background_mode, background_color,
                   background_image, background_video, invert_mask, feather_amount,
                   edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
                   shadow_opacity, color_adjustment, brightness, contrast, saturation,
                   x_position, y_position, rotation, opacity, flip_horizontal,
                   flip_vertical, mask_blur, mask_expansion, foreground_scale,
                   foreground_aspect_ratio, remove_background, image_format,
                   video_format, video_quality, use_custom_dimensions, custom_width,
                   custom_height, output_dimension_source, blend_mode):
    
    # Ensure color values are valid hex strings
    if not isinstance(background_color, str) or not background_color.startswith('#'):
        background_color = "#000000"
    if not isinstance(edge_color, str) or not edge_color.startswith('#'):
        edge_color = "#FFFFFF"
    
    args = (model, alpha_matting, alpha_matting_foreground_threshold,
           alpha_matting_background_threshold, post_process_mask, chroma_key,
           chroma_threshold, color_tolerance, background_mode, background_color,
           output_format, invert_mask, feather_amount, edge_detection,
           edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity,
           color_adjustment, brightness, contrast, saturation, x_position,
           y_position, rotation, opacity, flip_horizontal, flip_vertical,
           mask_blur, mask_expansion, foreground_scale, foreground_aspect_ratio,
           remove_background, use_custom_dimensions, custom_width, custom_height,
           output_dimension_source, blend_mode)

    if input_type == "Image" and result_type == "Image":
        result = process_image(foreground_input, background_image, *args)
        if image_format != "PNG":
            result = result.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format.lower()}") as temp_file:
            result.save(temp_file.name, format=image_format, quality=95 if image_format == "JPEG" else None)
            return temp_file.name, None
    elif input_type == "Video" and result_type == "Video":
        output_video = process_video(foreground_video, background_video if background_mode == "video" else None, *args)
        if video_format != "MP4":
            temp_output = output_video + f"_temp.{video_format.lower()}"
            os.system(f'ffmpeg -i "{output_video}" -c:v libx264 -crf {int(20 - (video_quality / 5))} "{temp_output}"')
            os.remove(output_video)
            output_video = temp_output
        return None, output_video
    elif input_type == "Image" and result_type == "Video":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            output_path = temp_file.name
        frame = cv2.cvtColor(np.array(foreground_input), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
        for _ in range(24 * 5):  # 5 seconds at 24 fps
            out.write(frame)
        out.release()
        return None, process_video(output_path, background_video if background_mode == "video" else None, *args)
    elif input_type == "Video" and result_type == "Image":
        cap = cv2.VideoCapture(foreground_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = process_image(pil_frame, background_image, *args)
            if image_format != "PNG":
                result = result.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format.lower()}") as temp_file:
                result.save(temp_file.name, format=image_format, quality=95 if image_format == "JPEG" else None)
                return temp_file.name, None
        return None, None

#script_callbacks.on_ui_tabs(on_ui_tabs)
#script_callbacks.on_ui_settings(on_ui_settings)
