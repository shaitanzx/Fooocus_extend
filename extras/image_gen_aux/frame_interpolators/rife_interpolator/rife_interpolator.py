# Originally based on code from - https://github.com/hzwer/ECCV2022-RIFE
# with code adaptations for this library

import os
import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from ..frame_interpolator import FrameInterpolator

# IMPORTANT: this file intentionally imports the single model (RIFE_HDv3).
# It does NOT try multiple model fallbacks
from .RIFE_HDv3 import Model


warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)


class RIFEFrameInterpolator(FrameInterpolator):
    """
    Real-Time Intermediate Flow Estimation (RIFE) for Video Frame Interpolation.

    This class handles loading the RIFE model and performing inference to generate
    intermediate frames between images or to upsample the frame rate of videos.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike], device=None, **kwargs):
        """
        Loads the RIFE model from a pretrained path or HuggingFace repository.

        Args:
            pretrained_model_or_path (str or os.PathLike): File path to the model checkpoint
                or the repo-id for HuggingFace Hub.
            device (torch.device, optional): The device to load the model onto.
                Defaults to None (auto-detects CUDA/CPU).
            **kwargs: Additional keyword arguments passed to the Model loader.

        Returns:
            RIFE: An instance of the RIFE interpolator with the loaded model.
        """
        model = Model.from_pretrained(pretrained_model_or_path, **kwargs)
        return cls(model, device=device)

    def __init__(self, model: Union[Model, str], device: Optional[torch.device] = None):
        """
        Initializes the RIFE interpolator.

        Args:
            model (Model or str): The loaded RIFE model instance.
            device (torch.device, optional): The execution device.
                If None, attempts to use CUDA if available, otherwise CPU.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = model.to(self.device)
        self.model.eval()

    # ----------------------------- helpers ---------------------------------
    def _read_image_tensor(self, path: str) -> Tuple[torch.Tensor, int, int, bool]:
        if path.lower().endswith(".exr"):
            img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            tensor = torch.tensor(img.transpose(2, 0, 1)).to(self.device).unsqueeze(0)
            return tensor, h, w, True
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            tensor = torch.tensor(img.transpose(2, 0, 1)).to(self.device).unsqueeze(0).float() / 255.0
            return tensor, h, w, False

    def _pad_pair_to_32(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        return img0, img1, padding

    def _make_inference(self, I0: torch.Tensor, I1: torch.Tensor, n: int) -> List[torch.Tensor]:
        middle = self.model.inference(I0, I1)
        if n == 1:
            return [middle]
        first = self._make_inference(I0, middle, n // 2)
        second = self._make_inference(middle, I1, n // 2)
        if n % 2:
            return [*first, middle, *second]
        return [*first, *second]

    def _transfer_audio(self, source_video: str, target_video: str):
        try:
            import shutil
            import tempfile

            import av
        except ImportError as e:
            raise ImportError(
                "Pyav is required for audio transfer to work in video interpolation logic. Run `pip install av` or `transfer_audio = False` to continue without errors."
            ) from e

        temp_dir = tempfile.mkdtemp()
        try:
            source_container = av.open(source_video)
            audio_streams = [s for s in source_container.streams if s.type == "audio"]
            if not audio_streams:
                source_container.close()
                return

            target_container = av.open(target_video)
            target_no_audio = os.path.splitext(target_video)[0] + "_noaudio" + os.path.splitext(target_video)[1]
            os.rename(target_video, target_no_audio)

            video_container = av.open(target_no_audio)
            output_container = av.open(target_video, "w")

            video_stream_mapping = {}
            for stream in video_container.streams.video:
                out_stream = output_container.add_stream(template=stream)
                video_stream_mapping[stream] = out_stream

            audio_stream_mapping = {}
            for stream in source_container.streams.audio:
                out_stream = output_container.add_stream(template=stream)
                audio_stream_mapping[stream] = out_stream

            for packet in video_container.demux():
                if packet.stream.type == "video":
                    packet.stream = video_stream_mapping[packet.stream]
                    output_container.mux(packet)

            source_container.close()
            source_container = av.open(source_video)

            for packet in source_container.demux():
                if packet.stream.type == "audio":
                    if packet.stream in audio_stream_mapping:
                        packet.stream = audio_stream_mapping[packet.stream]
                        output_container.mux(packet)

            video_container.close()
            source_container.close()
            output_container.close()

            if os.path.getsize(target_video) > 0:
                os.remove(target_no_audio)
            else:
                os.rename(target_no_audio, target_video)

        except Exception as e:
            raise Exception

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # --------------------------- public API --------------------------------
    def interpolate_images(
        self,
        img0_path: str,
        img1_path: str,
        exp: int = 4,
        ratio: float = 0.0,
        rthreshold: float = 0.02,
        rmaxcycles: int = 8,
        output_dir: Optional[str] = None,
    ) -> List[str] | List[torch.Tensor]:
        """
        Generates intermediate frames between two input images.

        Args:
            img0_path (str): Path to the first (start) image.
            img1_path (str): Path to the second (end) image.
            exp (int, optional): The exponent base 2 used to determine the number of frames generated.
                For example, `exp=2` generates 2^2 = 4 frames.
                This argument is ignored if `ratio` > 0. Defaults to 4.
            ratio (float, optional): A value between 0.0 and 1.0 determining the temporal position
                of the interpolated frame. 0.0 is img0, 1.0 is img1.
                If provided (not 0.0), a specific single frame is generated at this ratio
                using binary search recursion. Defaults to 0.0.
            rthreshold (float, optional): The acceptable error threshold for the `ratio` calculation.
                Used only when `ratio` is specified. Defaults to 0.02.
            rmaxcycles (int, optional): The maximum number of recursive inference cycles to perform
                when converging on the `ratio`. Used only when `ratio` is specified. Defaults to 8.
            output_dir (str, optional): The directory to save the interpolated frames.
                If None, the frames are returned as a list of torch.Tensors in memory.

        Returns:
            List[str] | List[torch.Tensor]:
                - If `output_dir` is provided: Returns a list of file paths to the saved images.
                - If `output_dir` is None: Returns a list of torch.Tensors (C, H, W) of the frames.

        Raises:
            ValueError: If input images have differing dimensions.
        """
        img0, h0, w0, exr0 = self._read_image_tensor(img0_path)
        img1, h1, w1, exr1 = self._read_image_tensor(img1_path)

        if h0 != h1 or w0 != w1:
            raise ValueError("Input images must have the same dimensions")

        img0, img1, _ = self._pad_pair_to_32(img0, img1)
        h, w = h0, w0

        # Mode 1: Arbitrary Ratio Interpolation (Binary Search)
        if ratio:
            img_list = [img0]
            img0_ratio, img1_ratio = 0.0, 1.0

            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp0, tmp1 = img0, img1
                for _ in range(rmaxcycles):
                    middle = self.model.inference(tmp0, tmp1)
                    mid_ratio = (img0_ratio + img1_ratio) / 2
                    if abs(mid_ratio - ratio) <= rthreshold / 2:
                        break
                    if ratio > mid_ratio:
                        tmp0 = middle
                        img0_ratio = mid_ratio
                    else:
                        tmp1 = middle
                        img1_ratio = mid_ratio

            img_list.extend([middle, img1])

        # Mode 2: Uniform Multi-Frame Interpolation (2^exp frames)
        else:
            img_list = [img0, img1]
            for _ in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.extend([img_list[j], mid])
                tmp.append(img1)
                img_list = tmp

        # Return Tensors if no output directory
        if output_dir is None:
            return [t[:, :, :h, :w].contiguous() for t in img_list]

        # Save images if output directory is provided
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for i, t in enumerate(img_list):
            ext = "exr" if exr0 and exr1 else "png"
            out_path = os.path.join(output_dir, f"img{i}.{ext}")
            img_np = t[0].cpu().numpy().transpose(1, 2, 0)[:h, :w]

            if ext == "exr":
                cv2.imwrite(out_path, img_np, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            else:
                cv2.imwrite(out_path, (img_np * 255).astype(np.uint8))

            saved.append(out_path)

        return saved

    def interpolate_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        exp: int = 1,
        fps: Optional[float] = None,
        scale: float = 1.0,
        montage: bool = False,
        ext: str = "mp4",
        transfer_audio: bool = True,
    ) -> str:
        """
        Upsamples a video file by interpolating intermediate frames.

        Args:
           video_path (str): Path to the source video file.
           output_path (str, optional): Directory or file path where the interpolated video will be saved.
                If None, saves to the same directory as input with a suffix.
           exp (int, optional): The exponent for the interpolation multiplier.
                The resulting frame rate will be `source_fps * (2^exp)`. Defaults to 1 (2x slow motion/smoothness).
           fps (float, optional): Specific target FPS for the output video.
                If provided, `exp` is used for frame generation count, but the video header is written
                with this specific FPS. **Note:** If `fps` is manually set, audio transfer is disabled.
           scale (float, optional): Flow scale multiplier (0.25, 0.5, 1.0, 2.0, 4.0).
                Controls the resolution processed by the model's optical flow. Defaults to 1.0.
           montage (bool, optional): If True, creates a side-by-side comparison of the input and output video.
                Useful for debugging or visualization. Defaults to False.
           ext (str, optional): The file extension for the output video container (e.g., "mp4"). Defaults to "mp4".
           transfer_audio (bool, optional): If True, attempts to transfer audio from source to output.
                Requires `fps` to be None. Defaults to True.

        Returns:
            str: The file path of the generated video.

        Raises:
            ValueError: If the video cannot be opened or `scale` is invalid.
        """
        import _thread
        from queue import Queue

        from tqdm import tqdm

        from .ssim_matlab import ssim_matlab

        multi = 2**exp
        assert scale in [0.25, 0.5, 1.0, 2.0, 4.0], "Scale must be one of [0.25, 0.5, 1.0, 2.0, 4.0]"

        videoCapture = cv2.VideoCapture(video_path)
        if not videoCapture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        source_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        success, lastframe = videoCapture.read()
        if not success:
            raise ValueError(f"Failed to read first frame of {video_path}")
        h, w, _ = lastframe.shape

        if fps is None:
            fps_not_assigned = True
            fps = source_fps * multi
        else:
            fps_not_assigned = False

        video_path_wo_ext, _ = os.path.splitext(video_path)
        if output_path is None:
            output_path = f"{video_path_wo_ext}_{multi}X_{int(np.round(fps))}fps.{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        def cv2_frame_generator(path):
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

        def clear_write_buffer(write_buffer):
            while True:
                item = write_buffer.get()
                if item is None:
                    break
                vid_out.write(item)

        def build_read_buffer(read_buffer, generator):
            for frame in generator:
                if montage:
                    frame = frame[:, left : left + w]
                read_buffer.put(frame)
            read_buffer.put(None)

        if montage:
            left = w // 4
            w = w // 2
            lastframe = lastframe[:, left : left + w]

        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        videogen = cv2_frame_generator(video_path)
        _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
        _thread.start_new_thread(clear_write_buffer, (write_buffer,))

        I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(self.device).unsqueeze(0).float() / 255.0
        I1 = F.pad(I1, padding)
        temp = None

        pbar = tqdm(total=tot_frame)
        while True:
            frame = temp if temp is not None else read_buffer.get()
            temp = None

            if frame is None:
                break

            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device).unsqueeze(0).float() / 255.0
            I1 = F.pad(I1, padding)

            I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            if ssim < 0.2:
                output = [I0] * (multi - 1)
            else:
                output = self._make_inference(I0, I1, multi - 1)

            vid_out.write(lastframe)
            for mid in output:
                mid = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                vid_out.write(mid[:h, :w])

            pbar.update(1)
            lastframe = frame

        vid_out.write(lastframe)
        pbar.close()
        vid_out.release()

        if transfer_audio and fps_not_assigned:
            try:
                self._transfer_audio(video_path, output_path)
            except Exception as e:
                raise Exception("Audio transfer failed") from e

        return output_path
