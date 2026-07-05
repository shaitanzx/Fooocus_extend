import cv2
import os
import gc
import torch
import threading
import numpy as np
from typing import Literal
from extras.basicsr.utils import img2tensor, tensor2img
from extras.basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (Literal): The GFPGAN architecture. Option: clean | bilinear | original | RestoreFormer | RestoreFormer++ | 
                        CodeFormer | GPEN | GPEN-1024 | GPEN-2048. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
        device (torch.device, optional): Device on which the models will run (CPU or GPU).
        det_model (Literal): The face detection model to use. Possible values are:
                             'retinaface_resnet50', 'YOLOv5l', 'YOLOv5n', 'dlib'. Default is 'retinaface_resnet50'.
        resolution (int): Specifies the input size based on the Face Restoration model.
                          Currently, only GPEN supports sizes of 1024 and 2048.
                          This value is set to the face_size parameter in FaceRestoreHelper. Default: 512.
    """

    def __init__(self, model_path, upscale=2, target_width=None, target_height=None, 
                 arch: Literal["clean", "bilinear", "original", "RestoreFormer", "RestoreFormer++", "CodeFormer", "GPEN", "GPEN-1024", "GPEN-2048", ] = "clean", 
                 channel_multiplier=2, bg_upsampler=None, device=None, model_rootpath="gfpgan/weights",
                 det_model: Literal["retinaface_resnet50", "YOLOv5l", "YOLOv5n", "dlib"] = "retinaface_resnet50", resolution = 512):
        self.upscale = upscale
        self.target_width = target_width
        self.target_height = target_height
        self.bg_upsampler = bg_upsampler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model_rootpath = model_rootpath
        self.resolution = resolution
        self.lock = threading.Lock()  # A thread lock for protecting shared resources

        # initialize the model
        if arch == 'clean':
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            from gfpgan.archs.gfpganv1_arch import GFPGANv1
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch and arch.startswith("RestoreFormer"):
            from gfpgan.archs.restoreformer_arch import VQVAEGANMultiHeadTransformer
            head_size = 4 if arch == "RestoreFormer++" else 8
            ex_multi_scale_num = 1 if arch == "RestoreFormer++" else 0
            self.gfpgan = VQVAEGANMultiHeadTransformer(head_size = head_size, ex_multi_scale_num = ex_multi_scale_num, resolution = self.resolution)
        elif arch == "CodeFormer":
            from gfpgan.archs.codeformer_arch import CodeFormer
            self.gfpgan= CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256'])
        elif arch and arch.startswith("GPEN"):
            from gfpgan.archs.gpen_arch import FullGenerator
            self.gfpgan = FullGenerator(size=self.resolution, style_dim=512, n_mlp=8, channel_multiplier=channel_multiplier, narrow=1)

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            target_width,
            target_height,
            face_size=self.resolution,
            crop_ratio=(1, 1),
            det_model=det_model,
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath=model_rootpath)

        # Load model weights
        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, model_rootpath), progress=True, file_name=None)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

        if "state_dict" in loadnet:
            loadnet = loadnet["state_dict"]
            new_weights = {}
            for k, v in loadnet.items():
                if "quantize.utility_counter" in k:
                    continue
                if k.startswith("vqvae."):
                    k = k.replace("vqvae.", "")
                new_weights[k] = v
            loadnet = new_weights

        if 'params_ema' in loadnet or 'params' in loadnet:
            keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
            self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        else:
            self.gfpgan.load_state_dict(loadnet, strict=True)

        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5, bg_upsample_img: np.ndarray=None, eye_dist_threshold:any = 10):
        """
        Enhance facial features in the input image using face alignment, restoration, and optional background upsampling.

        Args:
            img (np.ndarray): The input image to be processed.
            has_aligned (bool, optional): Indicates whether the input image is already aligned to 512x512. If True, no alignment will be performed. Defaults to False.
            only_center_face (bool, optional): If True, only the most prominent face in the center of the image will be processed. Defaults to False.
            paste_back (bool, optional): If True, the restored faces will be pasted back onto the original image or a background upsampled image. Defaults to True.
            weight (float, optional): The blending weight for the face restoration model. Defaults to 0.5.
            bg_upsample_img (np.ndarray, optional): An already upsampled background image to paste the restored faces onto. If None, background upsampling will be performed if supported. Defaults to None.
            eye_dist_threshold (any, optional): A threshold to filter out faces with too small an eye distance (e.g., side faces). Default is 10.

        Returns:
            tuple: 
                - cropped_faces (list): A list of cropped face images.
                - restored_faces (list): A list of restored face images.
                - restored_img (np.ndarray or None): The full restored image with enhanced faces pasted back, or None if `paste_back` is False.
        """
        with self.lock:
            self.face_helper.clean_all()

            try:
                if has_aligned:  # the inputs are already aligned
                    img = cv2.resize(img, (512, 512))
                    self.face_helper.cropped_faces = [img]
                else:
                    self.face_helper.read_image(img)
                    # get face landmarks for each face
                    self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=eye_dist_threshold)
                    # eye_dist_threshold=10: skip faces whose eye distance is smaller than 10 pixels
                    # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
                    # align and warp each face
                    self.face_helper.align_warp_face()
            
                # face restoration
                for cropped_face in self.face_helper.cropped_faces:
                    # prepare data
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            
                    try:
                        output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                        # convert to image
                        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                    except RuntimeError as error:
                        print(f'\tFailed inference for GFPGAN: {error}.')
                        restored_face = cropped_face
            
                    restored_face = restored_face.astype('uint8')
                    self.face_helper.add_restored_face(restored_face)
            
                if not has_aligned and paste_back:
                    # If a background upsampled image is provided, apply the face-enhanced image onto the provided image (bg upsample img must be an already upscaled image).
                    if bg_upsample_img is not None:
                        bg_img = bg_upsample_img
                    # upsample the background
                    elif self.bg_upsampler is not None:
                        # Now only support RealESRGAN for upsampling background
                        bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
                    else:
                        bg_img = None
            
                    self.face_helper.get_inverse_affine(None)
                    # paste each restored face to the input image
                    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
                    return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
                else:
                    return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
            finally:
                # Clean up resources to avoid memory leaks
                self._cleanup()

    def _cleanup(self):
        """
        Free GPU memory and clean up resources
        """
        torch.cuda.empty_cache()
        gc.collect()
        self.face_helper.clean_all()
