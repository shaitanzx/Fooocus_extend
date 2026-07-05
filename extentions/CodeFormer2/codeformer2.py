import gradio as gr


face_models = {
    "GFPGANv1.4.pth"      : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                            "https://github.com/TencentARC/GFPGAN/", 
"""GFPGAN: Towards Real-World Blind Face Restoration and Upscalling of the image with a Generative Facial Prior.
GFPGAN aims at developing a Practical Algorithm for Real-world Face Restoration.
It leverages rich and diverse priors encapsulated in a pretrained face GAN (e.g., StyleGAN2) for blind face restoration."""],

    "RestoreFormer++.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt",
                            "https://github.com/wzhouxiff/RestoreFormerPlusPlus", 
"""RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Pairs.
RestoreFormer++ is an extension of RestoreFormer. It proposes to restore a degraded face image with both fidelity and \
realness by using the powerful fully-spacial attention mechanisms to model the abundant contextual information in the face and \
its interplay with reconstruction-oriented high-quality priors."""],

    "CodeFormer.pth"      : ["https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                            "https://github.com/sczhou/CodeFormer", 
"""CodeFormer: Towards Robust Blind Face Restoration with Codebook Lookup Transformer (NeurIPS 2022).
CodeFormer is a Transformer-based model designed to tackle the challenging problem of blind face restoration, where inputs are often severely degraded.
By framing face restoration as a code prediction task, this approach ensures both improved mapping from degraded inputs to outputs and the generation of visually rich, high-quality faces.
"""],

    "GPEN-BFR-512.pth"    : ["https://huggingface.co/akhaliq/GPEN-BFR-512/resolve/main/GPEN-BFR-512.pth",
                            "https://github.com/yangxy/GPEN", 
"""GPEN: GAN Prior Embedded Network for Blind Face Restoration in the Wild.
GPEN addresses blind face restoration (BFR) by embedding a GAN into a U-shaped DNN, combining GAN’s ability to generate high-quality images with DNN’s feature extraction.
This design reconstructs global structure, fine details, and backgrounds from degraded inputs.
Simple yet effective, GPEN outperforms state-of-the-art methods, delivering realistic results even for severely degraded images."""],

    "GPEN-BFR-1024.pt"    : ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model.pt",
                            "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files", 
"""The same as GPEN but for 1024 resolution."""],

    "GPEN-BFR-2048.pt"    : ["https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/resolve/master/pytorch_model-2048.pt",
                            "https://www.modelscope.cn/models/iic/cv_gpen_image-portrait-enhancement-hires/files", 
"""The same as GPEN but for 2048 resolution."""],

    # legacy model
    "GFPGANv1.3.pth"    : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                          "https://github.com/TencentARC/GFPGAN/", "The same as GFPGAN but legacy model"],
    "GFPGANv1.2.pth"    : ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
                          "https://github.com/TencentARC/GFPGAN/", "The same as GFPGAN but legacy model"],
    "RestoreFormer.ckpt": ["https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt",
                          "https://github.com/wzhouxiff/RestoreFormerPlusPlus", "The same as RestoreFormer++ but legacy model"],
}
upscale_models = {
    # SRVGGNet(Compact)
    "realesr-general-x4v3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                                "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0", 
"""Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: add realesr-general-x4v3 and realesr-general-wdn-x4v3. They are very tiny models for general scenes, and they may more robust. But as they are tiny models, their performance may be limited."""],

    "realesr-animevideov3.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                                "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0", 
"""Anime, Cartoon, Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: update the RealESRGAN AnimeVideo-v3 model, which can achieve better results with a faster inference speed."""],
    
    "4xLSDIRCompact.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact/4xLSDIRCompact.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact", 
"""Realistic
Phhofm: Upscale small good quality photos to 4x their size. This is my first ever released self-trained sisr upscaling model."""],
     
    "4xLSDIRCompactC.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactC.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC", 
"""Compression Removal, JPEG, Realistic, Restoration
Phhofm: 4x photo upscaler that handler jpg compression. Trying to extend my previous model to be able to handle compression (JPG 100-30) by manually altering the training dataset, since 4xLSDIRCompact cant handle compression. Use this instead of 4xLSDIRCompact if your photo has compression (like an image from the web)."""],
         
    "4xLSDIRCompactR.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompactC/4xLSDIRCompactR.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompactC", 
"""Compression Removal, Realistic, Restoration
Phhofm: 4x photo uspcaler that handles jpg compression, noise and slight. Extending my last 4xLSDIRCompact model to Real-ESRGAN, meaning trained on synthetic data instead to handle more kinds of degradations, it should be able to handle compression, noise, and slight blur."""],

    "4xLSDIRCompactN.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Realistic
Phhofm: Upscale good quality input photos to x4 their size. The original 4xLSDIRCompact a bit more trained, cannot handle degradation.
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactC3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactC3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Compression Removal, 
JPEG, Realistic, Restoration
Phhofm: Upscale compressed photos to x4 their size. Able to handle JPG compression (30-100).
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactR3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Realistic, Restoration
Phhofm: Upscale (degraded) photos to x4 their size. Trained on synthetic data, meant to handle more degradations.
I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "4xLSDIRCompactCR3.pth": ["https://github.com/Phhofm/models/releases/download/4xLSDIRCompact3/4xLSDIRCompactCR3.pth",
                                "https://github.com/Phhofm/models/releases/tag/4xLSDIRCompact3", 
"""Phhofm: I am releasing the Series 3 from my 4xLSDIRCompact models. In general my suggestion is, if you have good quality input images use 4xLSDIRCompactN3, otherwise try 4xLSDIRCompactC3 which will be able to handle jpg compression and a bit of blur, or then 4xLSDIRCompactCR3, which is an interpolation between C3 and R3 to be able to handle a bit of noise additionally."""],

    "2xParimgCompact.pth": ["https://github.com/Phhofm/models/releases/download/2xParimgCompact/2xParimgCompact.pth",
                                "https://github.com/Phhofm/models/releases/tag/2xParimgCompact", 
"""Realistic
Phhofm: A 2x photo upscaling compact model based on Microsoft's ImagePairs. This was one of the earliest models I started training and finished it now for release. As can be seen in the examples, this model will affect colors."""],

    "1xExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xExposureCorrection_compact.pth",
                                         "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Restoration
Phhofm: This model is meant as an experiment to see if compact can be used to train on photos to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],
    
    "1xUnderExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xUnderExposureCorrection_compact.pth",
                                              "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Restoration
Phhofm: This model is meant as an experiment to see if compact can be used to train on underexposed images to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],
    
    "1xOverExposureCorrection_compact.pth": ["https://github.com/Phhofm/models/releases/download/1xExposureCorrection_compact/1xOverExposureCorrection_compact.pth",
                                             "https://github.com/Phhofm/models/releases/tag/1xExposureCorrection_compact", 
"""Restoration
Phhofm: This model is meant as an experiment to see if compact can be used to train on overexposed images to exposure correct those using the pixel, perceptual, color, color and ldl losses. There is no brightness loss. Still it seems to kinda work."""],

    "2x-sudo-UltraCompact.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-UltraCompact.pth",
                                "https://openmodeldb.info/models/2x-sudo-UltraCompact", 
"""Anime, Cartoon, Restoration
sudo: Realtime animation restauration and doing stuff like deblur and compression artefact removal.
My first attempt to make a REALTIME 2x upscaling model while also applying teacher student learning.
(Teacher: RealESRGANv2-animevideo-xsx2.pth)"""],

    "2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                                  "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-SuperUltraCompact", 
"""Anime, Compression Removal, Restoration
the-database: Real-time 2x Real-ESRGAN Compact/UltraCompact/SuperUltraCompact models designed for upscaling 1080p anime to 4K.
The aim of these models is to address scaling, blur, oversharpening, and compression artifacts while upscaling to deliver a result that appears as if the anime was originally mastered in 4K resolution."""],

    "2x_AnimeJaNai_HD_V3_UltraCompact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                             "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-UltraCompact", 
"""Anime, Compression Removal, Restoration
the-database: Real-time 2x Real-ESRGAN Compact/UltraCompact/SuperUltraCompact models designed for upscaling 1080p anime to 4K.
The aim of these models is to address scaling, blur, oversharpening, and compression artifacts while upscaling to deliver a result that appears as if the anime was originally mastered in 4K resolution."""],

    "2x_AnimeJaNai_HD_V3_Compact.pth": ["https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
                                                  "https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-Compact", 
"""Anime, Compression Removal, Restoration
the-database: Real-time 2x Real-ESRGAN Compact/UltraCompact/SuperUltraCompact models designed for upscaling 1080p anime to 4K.
The aim of these models is to address scaling, blur, oversharpening, and compression artifacts while upscaling to deliver a result that appears as if the anime was originally mastered in 4K resolution."""],

    # RRDBNet
    "RealESRGAN_x4plus_anime_6B.pth": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4", 
"""Anime, Cartoon, Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: We add RealESRGAN_x4plus_anime_6B.pth, which is optimized for anime images with much smaller model size. More details and comparisons with waifu2x are in anime_model.md"""],

    "RealESRGAN_x2plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1", 
"""Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: Add RealESRGAN_x2plus.pth model"""],

    "RealESRNet_x4plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1", 
"""Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: This release is mainly for storing pre-trained models and executable files."""],

    "RealESRGAN_x4plus.pth"         : ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                                      "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0", 
"""Compression Removal, General Upscaler, JPEG, Realistic, Research, Restoration
xinntao: This release is mainly for storing pre-trained models and executable files."""],

    # ESRGAN(oldRRDB)
    "4x-AnimeSharp.pth": ["https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth?download=true",
                         "https://openmodeldb.info/models/4x-AnimeSharp", 
"""Anime, Cartoon, Text
Kim2091: Interpolation between 4x-UltraSharp and 4x-TextSharp-v0.5. Works amazingly on anime. It also upscales text, but it's far better with anime content."""],

    "4x_IllustrationJaNai_V1_ESRGAN_135k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                               "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2", 
"""Anime, Cartoon, Compression Removal, Dehalftone, General Upscaler, JPEG, Manga, Restoration
the-database: Model for color images including manga covers and color illustrations, digital art, visual novel art, artbooks, and more. 
DAT2 version is the highest quality version but also the slowest. See the ESRGAN version for faster performance."""],

    "2x-sudo-RealESRGAN.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN.pth",
                               "https://openmodeldb.info/models/2x-sudo-RealESRGAN", 
"""Anime, Cartoon
sudo: Tried to make the best 2x model there is for drawings. I think i archived that. 
And yes, it is nearly 3.8 million iterations (probably a record nobody will beat here), took me nearly half a year to train. 
It can happen that in one edge is a noisy pattern in edges. You can use padding/crop for that. 
I aimed for perceptual quality without zooming in like 400%. Since RealESRGAN is 4x, I downscaled these images with bicubic.
Pretrained: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)"""],
    
    "2x-sudo-RealESRGAN-Dropout.pth": ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/2x-sudo-RealESRGAN-Dropout.pth",
                               "https://openmodeldb.info/models/2x-sudo-RealESRGAN-Dropout", 
"""Anime, Cartoon
sudo: Tried to make the best 2x model there is for drawings. I think i archived that. 
And yes, it is nearly 3.8 million iterations (probably a record nobody will beat here), took me nearly half a year to train. 
It can happen that in one edge is a noisy pattern in edges. You can use padding/crop for that. 
I aimed for perceptual quality without zooming in like 400%. Since RealESRGAN is 4x, I downscaled these images with bicubic.
Pretrained: Pretrained_Model_G: RealESRGAN_x4plus_anime_6B.pth / RealESRGAN_x4plus_anime_6B.pth (sudo_RealESRGAN2x_3.332.758_G.pth)"""],

    "4xNomos2_otf_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos2_otf_esrgan/4xNomos2_otf_esrgan.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xNomos2_otf_esrgan", 
"""Compression Removal, JPEG, Realistic, Restoration
Phhofm: Restoration, 4x ESRGAN model for photography, trained using the Real-ESRGAN otf degradation pipeline."""],

    "4xNomosWebPhoto_esrgan.pth": ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_esrgan/4xNomosWebPhoto_esrgan.pth",
                               "https://github.com/Phhofm/models/releases/tag/4xNomosWebPhoto_esrgan", 
"""Realistic, Restoration
Phhofm: Restoration, 4x ESRGAN model for photography, trained with realistic noise, lens blur, jpg and webp re-compression.
ESRGAN version of 4xNomosWebPhoto_RealPLKSR, trained on the same dataset and in the same way."""],


    "4x_foolhardy_Remacri.pth": ["https://civitai.com/api/download/models/164821?type=Model&format=PickleTensor",
                               "https://openmodeldb.info/models/4x-Remacri", 
"""Original
FoolhardyVEVO: A creation of BSRGAN with more details and less smoothing, made by interpolating IRL models such as Siax, 
Superscale, Superscale Artisoft, Pixel Perfect, etc. This was, things like skin and other details don't become mushy and blurry."""],

    "4x_foolhardy_Remacri_ExtraSmoother.pth": ["https://civitai.com/api/download/models/164822?type=Model&format=PickleTensor",
                               "https://openmodeldb.info/models/4x-Remacri", 
"""ExtraSmoother
FoolhardyVEVO: A creation of BSRGAN with more details and less smoothing, made by interpolating IRL models such as Siax, 
Superscale, Superscale Artisoft, Pixel Perfect, etc. This was, things like skin and other details don't become mushy and blurry."""],


    # DATNet
    "4xNomos8kDAT.pth"                     : ["https://github.com/Phhofm/models/releases/download/4xNomos8kDAT/4xNomos8kDAT.pth",
                                             "https://openmodeldb.info/models/4x-Nomos8kDAT", 
"""Anime, Compression Removal, General Upscaler, JPEG, Realistic, Restoration
Phhofm: A 4x photo upscaler with otf jpg compression, blur and resize, trained on musl's Nomos8k_sfw dataset for realisic sr, this time based on the DAT arch, as a finetune on the official 4x DAT model."""],

    "4x-DWTP-DS-dat2-v3.pth"               : ["https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/4x-DWTP-DS-dat2-v3.pth",
                                             "https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3", 
"""Dehalftone, Restoration
umzi.x.dead: DAT descreenton model, designed to reduce discrepancies on tiles due to too much loss of the first version, while getting rid of the removal of paper texture"""],

    "4xBHI_dat2_real.pth"                  : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real", 
"""Compression Removal, JPEG, Realistic
Phhofm: 4x dat2 upscaling model for web and realistic images. It handles realistic noise, some realistic blur, and webp and jpg (re)compression. Trained on my BHI dataset (390'035 training tiles) with degraded LR subset."""],

    "4xBHI_dat2_otf.pth"                   : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf/4xBHI_dat2_otf.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf", 
"""Compression Removal, JPEG
Phhofm: 4x dat2 upscaling model, trained with the real-esrgan otf pipeline on my bhi dataset. Handles noise and compression."""],

    "4xBHI_dat2_multiblur.pth"             : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblur.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Phhofm: the 4xBHI_dat2_multiblur checkpoint (trained to 250000 iters), which cannot handle compression but might give just slightly better output on non-degraded input."""],

    "4xBHI_dat2_multiblurjpg.pth"          : ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_multiblurjpg/4xBHI_dat2_multiblurjpg.pth",
                                             "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_multiblurjpg", 
"""Compression Removal, JPEG
Phhofm: 4x dat2 upscaling model, trained with down_up,linear, cubic_mitchell, lanczos, gauss and box scaling algos, some average, gaussian and anisotropic blurs and jpg compression. Trained on my BHI sisr dataset."""],

    "4x_IllustrationJaNai_V1_DAT2_190k.pth": ["https://drive.google.com/uc?export=download&confirm=1&id=1qpioSqBkB_IkSBhEAewSSNFt6qgkBimP",
                                             "https://openmodeldb.info/models/4x-IllustrationJaNai-V1-DAT2", 
"""Anime, Cartoon, Compression Removal, Dehalftone, General Upscaler, JPEG, Manga, Restoration
the-database: Model for color images including manga covers and color illustrations, digital art, visual novel art, artbooks, and more. 
DAT2 version is the highest quality version but also the slowest. See the ESRGAN version for faster performance."""],

    "4x-PBRify_UpscalerDAT2_V1.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_UpscalerDAT2_V1/4x-PBRify_UpscalerDAT2_V1.pth",
                                      "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_UpscalerDAT2_V1", 
"""Compression Removal, DDS, Game Textures, Restoration
Kim2091: Yet another model in the PBRify_Remix series. This is a new upscaler to replace the previous 4x-PBRify_UpscalerSIR-M_V2 model.
This model far exceeds the quality of the previous, with far more natural detail generation and better reconstruction of lines and edges."""],

    "4xBHI_dat2_otf_nn.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_dat2_otf_nn/4xBHI_dat2_otf_nn.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_otf_nn", 
"""Compression Removal, JPEG
Phhofm: 4x dat2 upscaling model, trained with the real-esrgan otf pipeline but without noise, on my bhi dataset. Handles resizes, and jpg compression."""],

    # HAT
    "4xNomos8kSCHAT-L.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-L.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-L", 
"""Anime, Compression Removal, General Upscaler, JPEG, Realistic, Restoration
Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. Since this is a big model, upscaling might take a while."""],

    "4xNomos8kSCHAT-S.pth"  : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCHAT/4xNomos8kSCHAT-S.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kSCHAT-S", 
"""Anime, Compression Removal, General Upscaler, JPEG, Realistic, Restoration
Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr. HAT-S version/model."""],

    "4xNomos8kHAT-L_otf.pth": ["https://github.com/Phhofm/models/releases/download/4xNomos8kHAT-L_otf/4xNomos8kHAT-L_otf.pth",
                              "https://openmodeldb.info/models/4x-Nomos8kHAT-L-otf", 
"""Faces, General Upscaler, Realistic, Restoration
Phhofm: 4x photo upscaler trained with otf, handles some jpg compression, some blur and some noise."""],

    "4xBHI_small_hat-l.pth": ["https://github.com/Phhofm/models/releases/download/4xBHI_small_hat-l/4xBHI_small_hat-l.pth",
                              "https://github.com/Phhofm/models/releases/tag/4xBHI_small_hat-l", 
"""Phhofm: 4x hat-l upscaling model for good quality input. This model does not handle any degradations.
This model is rather soft, I tried to balance sharpness and faithfulness/non-artifacts.
For a bit sharper output, but can generate a bit of artifacts, you can try the 4xBHI_small_hat-l_sharp version,
also included in this release, which might still feel soft if you are used to sharper outputs."""],

    # RealPLKSR_dysample
    "4xHFA2k_ludvae_realplksr_dysample.pth": ["https://github.com/Phhofm/models/releases/download/4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample", 
"""Anime, Compression Removal, Restoration
Phhofm: A Dysample RealPLKSR 4x upscaling model for anime single-image resolution."""],

    "4xArtFaces_realplksr_dysample.pth"    : ["https://github.com/Phhofm/models/releases/download/4xArtFaces_realplksr_dysample/4xArtFaces_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample", 
"""ArtFaces
Phhofm: A Dysample RealPLKSR 4x upscaling model for art / painted faces."""],

    "4x-PBRify_RPLKSRd_V3.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/4x-PBRify_RPLKSRd_V3/4x-PBRify_RPLKSRd_V3.pth",
                                             "https://github.com/Kim2091/Kim2091-Models/releases/tag/4x-PBRify_RPLKSRd_V3", 
"""Compression Removal, DDS, Debanding, Dedither, Dehalo, Game Textures, Restoration
Kim2091: This update brings a new upscaling model, 4x-PBRify_RPLKSRd_V3. This model is roughly 8x faster than the current DAT2 model, while being higher quality. 
It produces far more natural detail, resolves lines and edges more smoothly, and cleans up compression artifacts better.
As a result of those improvements, PBR is also much improved. It tends to be clearer with less defined artifacts."""],

    "4xNomos2_realplksr_dysample.pth"      : ["https://github.com/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
                                             "https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample", 
"""Compression Removal, JPEG, Realistic, Restoration
Phhofm: A Dysample RealPLKSR 4x upscaling model that was trained with / handles jpg compression down to 70 on the Nomosv2 dataset, preserves DoF.
This model affects / saturate colors, which can be counteracted a bit by using wavelet color fix, as used in these examples."""],

    # RealPLKSR
    "2x-AnimeSharpV2_RPLKSR_Sharp.pth": ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
                                        "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set", 
"""Anime, Compression Removal, Restoration
Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
RealPLKSR (Higher quality, slower) Sharp: For heavily degraded sources. Sharp models have issues depth of field but are best at removing artifacts
"""],

    "2x-AnimeSharpV2_RPLKSR_Soft.pth" : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_RPLKSR_Soft.pth",
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set", 
"""Anime, Compression Removal, Restoration
Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
RealPLKSR (Higher quality, slower) Soft: For cleaner sources. Soft models preserve depth of field but may not remove other artifacts as well"""],

    "4xPurePhoto-RealPLSKR.pth"       : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-RealPLSKR.pth",
                                        "https://openmodeldb.info/models/4x-PurePhoto-RealPLSKR", 
"""AI Generated, Compression Removal, JPEG, Realistic, Restoration
asterixcool: Skilled in working with cats, hair, parties, and creating clear images.
Also proficient in resizing photos and enlarging large, sharp images.
Can effectively improve images from small sizes as well (300px at smallest on one side, depending on the subject).
Experienced in experimenting with techniques like upscaling with this model twice and
then reducing it by 50% to enhance details, especially in features like hair or animals."""],

    "2x_Text2HD_v.1-RealPLKSR.pth"    : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2x_Text2HD_v.1-RealPLKSR.pth",
                                        "https://openmodeldb.info/models/2x-Text2HD-v-1", 
"""Compression Removal, Denoise, General Upscaler, JPEG, Restoration, Text
asterixcool: The upscale model is specifically designed to enhance lower-quality text images,
improving their clarity and readability by upscaling them by 2x.
It excels at processing moderately sized text, effectively transforming it into high-quality, legible scans.
However, the model may encounter challenges when dealing with very small text,
as its performance is optimized for text of a certain minimum size. For best results,
input images should contain text that is not excessively small."""],

    "2xVHS2HD-RealPLKSR.pth"          : ["https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/2xVHS2HD-RealPLKSR.pth",
                                        "https://openmodeldb.info/models/2x-VHS2HD", 
"""Compression Removal, Dehalo, Realistic, Restoration, Video Frame
asterixcool: An advanced VHS recording model designed to enhance video quality by reducing artifacts such as haloing, ghosting, and noise patterns.
Optimized primarily for PAL resolution (NTSC might work good as well)."""],

    "4xNomosWebPhoto_RealPLKSR.pth"   : ["https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.pth",
                                        "https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR", 
"""Realistic, Restoration
Phhofm: 4x RealPLKSR model for photography, trained with realistic noise, lens blur, jpg and webp re-compression."""],

    # DRCT
    "4xNomos2_hq_drct-l.pth"          : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.pth", 
                                        "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l",
"""General Upscaler, Realistic
Phhofm: An drct-l 4x upscaling model, similiar to the 4xNomos2_hq_atd, 4xNomos2_hq_dat2 and 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
"""],

    # ATD
    "4xNomos2_hq_atd.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_atd/4xNomos2_hq_atd.pth", 
                                         "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd",
"""General Upscaler, Realistic
Phhofm: An atd 4x upscaling model, similiar to the 4xNomos2_hq_dat2 or 4xNomos2_hq_mosr models, trained and for usage on non-degraded input to give good quality output.
"""],

    # MoSR
    "4xNomos2_hq_mosr.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos2_hq_mosr/4xNomos2_hq_mosr.pth", 
                                         "https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_mosr",
"""General Upscaler, Realistic
Phhofm: A 4x MoSR upscaling model, meant for non-degraded input, since this model was trained on non-degraded input to give good quality output.
"""],
    
    "2x-AnimeSharpV2_MoSR_Sharp.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Sharp.pth", 
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Compression Removal, Restoration
Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
MoSR (Lower quality, faster), Sharp: For heavily degraded sources. Sharp models have issues depth of field but are best at removing artifacts
"""],
    
    "2x-AnimeSharpV2_MoSR_Soft.pth"             : ["https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV2_Set/2x-AnimeSharpV2_MoSR_Soft.pth", 
                                         "https://github.com/Kim2091/Kim2091-Models/releases/tag/2x-AnimeSharpV2_Set",
"""Anime, Compression Removal, Restoration
Kim2091: This is my first anime model in years. Hopefully you guys can find a good use-case for it.
MoSR (Lower quality, faster), Soft: For cleaner sources. Soft models preserve depth of field but may not remove other artifacts as well
"""],

    # SRFormer
    "4xNomos8kSCSRFormer.pth"             : ["https://github.com/Phhofm/models/releases/download/4xNomos8kSCSRFormer/4xNomos8kSCSRFormer.pth", 
                                             "https://github.com/Phhofm/models/releases/tag/4xNomos8kSCSRFormer",
"""Anime, Compression Removal, General Upscaler, JPEG, Realistic, Restoration
Phhofm: 4x photo upscaler with otf jpg compression and blur, trained on musl's Nomos8k_sfw dataset for realisic sr.
"""],

    "4xFrankendataFullDegradation_SRFormer460K_g.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1PZrj-8ofxhORv_OgTVSoRt3dYi-BtiDj", 
                                                    "https://openmodeldb.info/models/4x-Frankendata-FullDegradation-SRFormer",
"""Compression Removal, Denoise, Realistic, Restoration
Crustaceous D: 4x realistic upscaler that may also work for general purpose usage. 
It was trained with OTF random degradation with a very low to very high range of degradations, including blur, noise, and compression. 
Trained with the same Frankendata dataset that I used for the pretrain model.
"""],

    "4xFrankendataPretrainer_SRFormer400K_g.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1SaKvpYYIm2Vj2m9GifUMlNCbmkE6JZmr", 
                                                    "https://openmodeldb.info/models/4x-FrankendataPretainer-SRFormer",
"""Realistic, Restoration
Crustaceous D: 4x realistic upscaler that may also work for general purpose usage. 
It was trained with OTF random degradation with a very low to very high range of degradations, including blur, noise, and compression. 
Trained with the same Frankendata dataset that I used for the pretrain model.
"""],

    "1xFrankenfixer_SRFormerLight_g.pth" : ["https://drive.google.com/uc?export=download&confirm=1&id=1UJ0iyFn4IGNhPIgNgrQrBxYsdDloFc9I", 
                                                  "https://openmodeldb.info/models/1x-Frankenfixer-SRFormerLight",
"""Realistic, Restoration
Crustaceous D: A 1x model designed to reduce artifacts and restore detail to images upscaled by 4xFrankendata_FullDegradation_SRFormer. It could possibly work with other upscaling models too.
"""],
}
def ui():
    with gr.Row():
        with gr.Column(variant="panel"):
            submit = gr.Button(value="Submit", variant="primary", size="lg")
            input_gallery = gr.Image(label="Input image",interactive=True,type="numpy")
            face_model                 = gr.Dropdown([None]+list(face_models.keys()), type="value", value='GFPGANv1.4.pth', label='Face Restoration version', info="Face Restoration and RealESR can be freely combined in different ways, or one can be set to \"None\" to use only the other model. Face Restoration is primarily used for face restoration in real-life images, while RealESR serves as a background restoration model.")
            
            upscale_model              = gr.Dropdown([None]+list(typed_upscale_models.keys()), type="value", value='SRVGG, realesr-general-x4v3.pth', label='UpScale version')
            upscale_scale              = gr.Number(label="Rescaling factor", value=4)
            face_detection             = gr.Dropdown(["retinaface_resnet50", "YOLOv5l", "YOLOv5n"], type="value", value="retinaface_resnet50", label="Face Detection type")
            face_detection_threshold   = gr.Number(label="Face eye dist threshold", value=10, info="A threshold to filter out faces with too small an eye distance (e.g., side faces).")
            face_detection_only_center = gr.Checkbox(value=False, label="Face detection only center", info="If set to True, only the face closest to the center of the image will be kept.")
            #with_model_name            = gr.Checkbox(label="Output image files name with model name", value=True)
            # Add a checkbox to always save the output as a PNG file for the best quality.
            #save_as_png                = gr.Checkbox(label="Always save output as PNG", value=True, info="If enabled, all output images will be saved in PNG format to ensure the best quality. If disabled, the format will be determined automatically (PNG for images with transparency, otherwise JPG).")

            # Event to update the selected image when an image is clicked in the gallery
            #selected_image = gr.Textbox(label="Selected Image", visible=False)
            #input_gallery.select(get_selection_from_gallery, inputs=None, outputs=selected_image)
            # Trigger update when gallery changes
            #input_gallery.change(limit_gallery, input_gallery, input_gallery)

            # with gr.Row():
            #     clear = gr.ClearButton(
            #         components=[
            #             input_gallery,
            #             face_model,
            #             upscale_model,
            #             upscale_scale,
            #             face_detection,
            #             face_detection_threshold,
            #             face_detection_only_center,
            #             with_model_name,
            #             save_as_png,
            #         ], variant="secondary", size="lg",)
        with gr.Column(variant="panel"):
            gallerys = gr.Image(label="Output image",visible=True,height=260,interactive=False)
            outputs = gr.File(label="Download the output ZIP file")
    # with gr.Row(variant="panel"):
    #     # Generate output array
    #     output_arr = []
    #     for file_name in example_list:
    #         output_arr.append([[file_name],])
    #     gr.Examples(output_arr, inputs=[input_gallery,], examples_per_page=20)
    with gr.Row(variant="panel"):
        # Convert to Markdown table
        header = "| Face Model Name | Info |\n|------------|------|"
        rows = [
            f"| [{key}]({value[1]}) | " + value[2].replace("\n", "<br>") + f" |"
            for key, value in face_models.items()
        ]
        markdown_table = header + "\n" + "\n".join(rows)
        gr.Markdown(value=markdown_table)

    for table in upscale_model_tables:
        with gr.Row(variant="panel"):
            gr.Markdown(value=table)

    # submit.click(
    #     upscale.inference, 
    #     inputs=[
    #         input_gallery,
    #         face_model,
    #         upscale_model,
    #         upscale_scale,
    #         face_detection,
    #         face_detection_threshold,
    #         face_detection_only_center,
    #         with_model_name,
    #         save_as_png,
    #     ],
    #     outputs=[gallerys, outputs],
    # )