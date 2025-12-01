I would like to introduce a new fork of the popular generative neural network **Fooocus - Fooocus extend**. 
I would like to point out that this fork can be run both locally on your computer and via Google Colab. 
Let's look at everything in order. 

**Full list of differences from the original Fooocus**

1. Modules running during generation
   - OneButtonPrompt - prompt generator with many settings
   - Prompt Translate
   - PhotoMaker - generating images with a reference face
   - InstantID - generating images with a reference face
   - Inswapper - face replacements in the generated image
   - CodeFormer - face enhancer
   - Vector - vector image generation
2. Additional modules
   - Image Batch - generation with a batch of reference images
   - Prompt Batch - generating a prompt batch
   - X/Y/Z Plot - a module that allows you to generate images by changing various parameters
   - Inswapper - face replacements in the generated image
   - CodeFormer - face enhancer
   - Remove Background
   - Vector - allows you to convert a raster image into a vector
3. Tools
   - Civitai Helper - A module for working with models and downloading them from Civitai
   - TextMask - A module for overlaying text on images and generating a mask for it for subsequent generation with text
   - SVGcode - allows you to convert a raster image into a vector
   - Roller - module for rolling images
   - OpenPoseEditor
   - Logo - overlaying a logo on an image
   - Photopea
4. Select the resolution and aspect ratio of the generated image
5. Wildcard
6. OpenPose ControlNet
7. Recolor ControlNet
8. Scribble ControlNet
9. Manga Recolor ControlNet
10. Save Image Grid for Each Batch
11. Filename Prefix
12. Paths and Presets - managing paths and presets
13. Load file of style
14. View LoRA trigger words and view the models page on civitai.com
15. Seamless tiling
16. Transparency - generating images on a transparent background
17. CFG control type (CFG Mimicking from TSNR, CFG rescale, Off)
18. Adetailer - extension for automatic masking and inpainting
19. Support external upscalers

**Launch**. If you will run it on a local machine, you can safely skip this item.
   
![image](https://github.com/user-attachments/assets/468487b8-8d4e-454c-ba92-1c9e5b60feb7)

Before launching you are offered to choose the following settings
Fooocus Profile - select which profile will be loaded at startup (default, anime, realistic).
Fooocus Theme - select a theme - light or dark.
Tunnel - select launch tunnel. When it happens that gradio stops working for some reason, you can choose cloudflared tunnel. However, the generation is a bit slower
Memory patch - adds a few keys to the launch bar that allow you to optimise your graphics card if you are using the free version of Google Colab. If you have paid access, this item can be disabled
GoogleDrive output - connects your GoogleDisk and save all of your generation directly to it.

![image](https://github.com/user-attachments/assets/323f6311-bbb4-477f-860d-3cfecae5d89b)

The **“Extensions”** panel is divided into three groups:
- in generation - modules working during generation
- modules - generation modules
- tools - auxiliary tools

**“in generation”** panel

**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions

<img width="1176" height="455" alt="image" src="https://github.com/user-attachments/assets/754584c0-a35b-4b67-a4e6-654d2fc3f47a" />

On the "Main" tab, you can select a preset for the hint generation theme, as well as specify additional hint prefixes and suffixes.
You can also create and save your own preset in this tab.

<img width="718" height="567" alt="image" src="https://github.com/user-attachments/assets/c8f1b30b-6472-4c94-b7b7-00ae6376f669" />

On the "Prompt assist" tab, you can generate 5 prompts and move each of them to the Fooocus prompt field.
When you select the "Prompt assist" mode, during image generation, not a new prompt will be generated, but variations of the prompt specified in the prompt field.
The strength of variation is selected by the slider located directly below it.

<img width="714" height="614" alt="image" src="https://github.com/user-attachments/assets/18102090-988f-4aca-bce6-fa9b3792e984" />

In this tab you can select the hint syntax for different generation models, the hint length, enable the hint generation enhancement tool, and specify the number of iterations of the prompt generation for one image.

<img width="1175" height="363" alt="image" src="https://github.com/user-attachments/assets/2bb26223-616e-4e21-a78f-8eb6b5083a58" />

This is where you can control the generation of negative prompts

**Prompt Translate**
![image](https://github.com/user-attachments/assets/6ddff3b1-5e98-43d4-b102-b61176c40c84)


Allows to translate both positive and negative prompts from any language into English, both before generation and directly during generation.

**PhotoMaker** (works only in the local version)

![image](https://github.com/user-attachments/assets/e15a668a-53aa-4607-a7ed-7d277a3726de)

The module allows you to generate an image with a reference face
Upload a photo of your face - your reference face should be uploaded here.
**Unlike normal generation, this 30-step generation on Nvidia 3060 12GB takes about 4 minutes.**


**InstantID** (works only in the local version) 
![image](https://github.com/user-attachments/assets/f12b7389-48b4-4659-a94d-5d01268ab179)

The module allows you to generate an image with a reference face
Upload a photo of your face - your reference face should be uploaded here.

Upload a reference pose image (Optional) - a reference pose image will be uploaded here. Optional.

Pregeneration image - if this mode is enabled, then a new image will be generated as a reference pose image based on your Prompt.

If you have not specified a pose reference image, an image will be generated from your prompt based only on the face image and the same resolution. 

If a pose reference image is present, the dimensions will be the same as the pose reference image.

IdentityNet strength (for fidelity) - responsible for the accuracy of face repetition.

Image adapter strength (for detail) - responsible for details

ContrloNet
- canny - used to define image contours
- depth - used to create a depth map of the image
Schedulers - select the Schedulers that will be used when generating the image with our face.
Enhance non-face region - enhance non-face parts of the image

**Unlike normal generation, this 30-step generation on Nvidia 3060 12GB takes about 7 minutes.**

**Inswapper**

<img width="746" height="381" alt="image" src="https://github.com/user-attachments/assets/138962ee-5dda-4e61-9b3d-24879922de99" />


This module is also intended for face replacement.

Source Image Index - index of the face in the reference image. Faces are numbered from left to right from top to bottom starting from zero. If you specify -1, the average mixed face of all available faces will be taken as the reference face

Target Image Index - index of the face in the output image. This is the index of the face to be replaced in the output image. If you specify -1, all faces will be replaced.

Source Face Image - image with face

Sace input image - saves the input image from the previous iteration (generation or previous Extention)

**CodeFormer**

<img width="743" height="252" alt="image" src="https://github.com/user-attachments/assets/ef798a4d-3329-4318-a4c0-7d704d82b8f7" />



Face enhancement module with upscale capability

Pre_Face_Align - aligns the face if it is tilted

Background Enchanced - improve background quality

Face Upsample - adjust reference face to the size of the input image face

Upscale - image enlargement

Codeformer_Fidelity - signability coefficient, inversely proportional to quality


Sace input image - saves the input image from the previous iteration (generation or previous Extention)
=======



**Vector**

<img width="747" height="354" alt="image" src="https://github.com/user-attachments/assets/6fe3d566-4630-49aa-b366-15af53c4be5d" />


This module allows you to get images in svg format in b/w mode.

Transparent PNG - Pre-Create Transparency

Noise Tolerance - setting for cutting off noise in the original image

Quantize - the level of image quantization

Save temp images - save the intermediate transparency file

Threshold - line curvature threshold



**“modules”** panel

**Image Batch** (batch image processing)

![image](https://github.com/user-attachments/assets/5279f1f1-ab59-4182-be78-b28e9b6c1043)

In a nutshell, this module allows you to perform group scaling of images, as well as create images based on a group of existing images using ImagePromt (ControlNet). To better understand this module, I suggest you conduct some experiments yourself. But I want to note that using it allows you to use your images as references and change their style depending on the prompt and model you choose. First, depending on the "Uplode ZIP-file" selector, you need to upload several reference images or a zip archive with them. The archive should not contain subfolders, file names should not contain non-Latin characters and spaces.

 Next, select the mode of changing the image resolution
- NOT scale - the source image resolution will not be taken into account during generation
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the source image will be selected.
- to OUTPUT - in this case, before generation the resolution of the source image will be changed to the generated one with preserving the proportions
  
Depending on what you want to do with the source images, select Action - Upscale or ImagePrompt. From the Method drop-down list, select the appropriate image processing method. If you are using ImagePrompt, you must also select the "Stop at" and "Weight" options.

Start batch - starts the process for execution. 

When the process is finished, click on Output->Zip button to create an archive with all previously created images from the output folder. The archive itself will appear in the "Download a Zip file" window. You can download it from there.

Clear Output - clear the output folder. It should be noted that not only the folder for the current date is cleared, but also the whole folder.


**Prompt Batch** (batch processing of prompts)

![image](https://github.com/user-attachments/assets/06ae0644-4e4b-4456-a92b-7699606d9ce0)

This module allows you to run generation of several hints sequentially one after another. To do this, you should fill in the table. Enter a positive hint in the hint column and a negative hint in the negative hint column respectively. Clicking the New Row button will add an empty row to the end of the table. Delete Last Row deletes the last row of the table. Start batch starts the execution of the list of prompts to generate.
You can also choose to add basic positive and negative hints.
None - no base hints will be added.
Prefix - base hints will be added before the table hints.
Suffix - basic hints will be added after hints from the table.

only positive prompts - if this item is active, all prompts in the file will be considered positive, otherwise, both positive and negative prompts will be loaded
Load prompts from file - allows to load the list of positive and negative prompts from a text file into the table. The file can have any extension

As an example, consider a file with the following contents

---- start of file -------

![image](https://github.com/user-attachments/assets/750dff48-0bcc-4a20-b754-2d40f443e938)

------ end of file --------

If the item ‘only positive prompts’ is active, then the table with prompts will have the following form


![image](https://github.com/user-attachments/assets/c5df613e-d5c4-4088-abad-bf727f84f041)


Otherwise

![image](https://github.com/user-attachments/assets/1f005878-cadc-443f-a76f-7f20dda088b2)

This means that in the first case all prompts in the file are treated as positive, and empty lines are ignored.
In the second case, the file first contains a line with a positive prompt, followed by a line with a negative prompt. If you don't need to specify a negative hint, leave this line blank, but the positive hint line must always be there.


**X/Y/Z Plot**

![019](https://github.com/user-attachments/assets/a0392f94-8ef0-4377-a2d6-9d260ba4a20c)


This extension allows you to make image grids to make it easier to see the difference between different generation settings and choose the best option. You can change the following parameters - Styles, Steps, Aspect Ratio, Seed, Sharpness, CFG (Guidance) Scale, Checkpoint, Refiner, Clip skip, Sampler, Scheduler, VAE, Refiner swap method, Softness of ControlNet, and also replace words in the prompt and change their order

**Inswapper**

<img width="1070" height="506" alt="image" src="https://github.com/user-attachments/assets/3e8e4005-e3be-481e-b04d-90fd4a5d3e28" />

The full analog of this module in the “in generation” panel, unlike which you need to load an additional input image

**CodeFormer**

<img width="1057" height="537" alt="image" src="https://github.com/user-attachments/assets/b0d1780f-0438-4e73-ba3f-281fb4b68a90" />


The full analog of this module in the “in generation” panel, unlike which you need to load an additional input image

**Remove Background**

![008](https://github.com/user-attachments/assets/f34a2aff-a2ad-4f53-8edb-0703a2c69386)


This extension is designed to add background removal, image/video processing, and blending to your projects. It provides precise background removal with support for multiple models, chroma keying, foreground adjustments, and advanced effects. Whether you’re working with images or videos, this extension provides everything you need to efficiently process visual content.

Key Features:

Multi-model background removal: Supports u2net, isnet-general-use, and other models.

Chroma keying support: Removes specific colors (green, blue, or red) from the background.

Blending modes: 10 powerful blending modes for image compositing.

Foreground adjustments: Scale, rotate, flip, and position elements precisely.

Video and image support: Easily process images and videos.

Multi-threaded processing: Efficiently process large files with streaming and GPU support. Customizable output formats: Export to PNG, JPEG, MP4, AVI, and more.

All processing results are automatically saved to the output folder without saving to History Log


**Vector**

<img width="674" height="586" alt="image" src="https://github.com/user-attachments/assets/8f042628-4ead-4db4-b68c-98b549674d0b" />


The full analog of this module in the “in generation” panel, unlike which you need to load an additional input image



**“tools”** panel

**Civitai Helper**

<img width="1180" height="2585" alt="image" src="https://github.com/user-attachments/assets/62c3a4ab-2423-47b4-9f2c-c65586ded0e9" />




This extension allows you to download models for generation from the civitai website.  To download a model you first need to specify your Civitai_API_key. In the Download Model section in the Civitai URL field you need to specify a link to the required model from the browser address bar and click Get Model Info by Civitai URL. After analysing the link you will be given information about the model. You will also be able to select the version of the model before downloading. This extension also allows you to find duplicates of downloaded models and check for updates. In addition, there is a group download option.

**TextMask**

![image](https://github.com/user-attachments/assets/1634a9b2-885d-49d8-a7cb-97417ed21568)

Fast text editor with mask creation for ControlNet and Inpaint! 

What this tool can do:
- Adds up to 5 text blocks to any image
- Supports fonts of any size up to 300px
- Works with Cyrillic, hieroglyphics and other non-Latin characters
- Generates black and white mask with one click
- Lets you forget Photoshop for 90% of simple text tasks
- Supports a sufficient number of fonts

How to use:
1. Upload an image
2. Add and edit the text as you like
3. Generate a mask in two clicks
4. Download and use in ControlNet (CPDS or PyraCanny) or as a mask for Inpaint to stylise text and blend into an image.

**SVGcode**

<img width="720" height="570" alt="image" src="https://github.com/user-attachments/assets/7203799b-ed3d-4b5d-89ef-58c9e5d170c0" />

Module for processing raster images for saving from to svg

**Roller**

<img width="749" height="482" alt="image" src="https://github.com/user-attachments/assets/d9c0d272-381e-4da8-9baa-21de1589d25c" />

This module is designed for rolling tiles to give it a more aesthetic appearance without losing the seamless effect.

Y_shift - vertical image shift.

Х_shift - horizontal image shift.

Save to OUTPUT FOLDER - saving the image to the output folder

**OpenPoseEditor**

<img width="1089" height="571" alt="image" src="https://github.com/user-attachments/assets/564e6ebc-3b49-4e50-bd61-819fab2ba090" />


This module allows you to create skeletons for subsequent image creation using OpenPose ControlNet. You can also create a skeleton from an existing image.

**Logo** - Logo insertion tool with automatic placement selection based on image content

<img width="1059" height="568" alt="image" src="https://github.com/user-attachments/assets/621355ca-4243-4709-80a1-726a9a0fcddd" />

Size ratio - the relative size of the logo to the image. Determines what percentage of the smaller side of the image the logo will occupy.

Margin Ratio - logo offset from image edges. Determines how far from the corner the logo will be placed.

Minimal complexity for background - the background complexity threshold at which a background is automatically added under the logo.

Сorner priority - select the priority of choosing the angle when overlaying the logo.


**Photopea** - a free online analogue of Photoshop

![007](https://github.com/user-attachments/assets/bd75f089-dee9-4c7a-a24b-2dfa5da7d2fd)

**In addition to these extensions, there are several other add-ons available**

**Select the resolution and aspect ratio of the generated image**

![image](https://github.com/user-attachments/assets/502cdf7e-ae4b-4bda-b2dc-b0076c962ae4)


This setting is located in the generation resolution selection tab. Here you can select the number of horizontal and vertical points, aspect ratio. To apply the settings, click the Set button and select this resolution from the list of proposed resolutions. Your resolution will be the second to last. 
You can also select a random aspect ratio for each generation from the available list of aspect ratios

**Wildcard**

![image](https://github.com/user-attachments/assets/45a4fc1f-72f6-479a-96ea-d61b6c62333e)

This module allows you not to memorise existing files with wildcard words, but to select them directly from a list of dictionaries. You can also select directly the item you need from the list.


**OpenPose ControlNet**

![016](https://github.com/user-attachments/assets/79e5b10a-0fcb-40d9-8d46-ec9869725919)

Allows you to create an image based on the pose skeleton.

**Recolor ControlNet**

![017](https://github.com/user-attachments/assets/2de6044f-2c23-4728-bf72-564d55d78300)

Allows you to colorize an image based on a black and white image.

**Scribble ControlNet**

![018](https://github.com/user-attachments/assets/ad18e4f4-d8f8-410a-b4c8-822de3dfadd4)

Allows you to color an image based on a sketch.

**Manga Recolor ControlNet**

<img width="1254" height="538" alt="image" src="https://github.com/user-attachments/assets/d9b15b34-801b-462a-a982-63de1e990033" />

This is a specially trained ControlNet model designed to automatically colorize grayscale images in anime style.

The model takes grayscale anime images as input and generates a colorized version.

An anime model is required for proper generation. Works with or without a clarifying prompt.

**Save Image Grid for Each Batch**

![020](https://github.com/user-attachments/assets/04aebc91-de87-428e-b664-83b6d597e23f)

**Filename Prefix**

![image](https://github.com/user-attachments/assets/c5e5dcb3-ab28-4063-9dec-c3b242442c32)

This setting may be useful when working on several projects to separate one from another.

**Paths and Presets**

![image](https://github.com/user-attachments/assets/db023eb1-3487-4d50-baed-9d0c0dc69cd5)

Here you can change the paths to your models if they are already in other folders on the disk. If your Сheckponts or LORAs are in different folders, then the paths to them can be specified separated by commas (,). After changing the path, it is best to restart FooocusExtend.

Also here you can create a new preset based on the existing settings, and delete any of the existing ones, except for default and initial.
The preset saves the following parameters: base model, refiner, refiner_switch, loras settings, cfg scale, sharpness, CFG Mimicking from TSNR, clip_skip, sampler, scheduler, Forced Overwrite of Sampling Step, Forced Overwrite of Refiner Switch Step, performance, image number, prompt negative, styles selections, aspect ratio, vae, inpaint_engine_version

**Load file of style**

![image](https://github.com/user-attachments/assets/8dfc51ce-f6f9-4ffa-b66d-2ae6d3412477)

Allows you to upload a file (in *.json format) with custom styles

**View LoRA trigger words and view the models page on civitai.com**

<img width="482" height="543" alt="image" src="https://github.com/user-attachments/assets/0d5d905c-738f-4e17-9fbb-44456def2fad" />

If trigger words or links to model pages are not displayed, you will need to scan the models in the Civitai Helper module in the "Scan Models" section.

**Seamless tiling**

<img width="368" height="138" alt="image" src="https://github.com/user-attachments/assets/71564718-7048-4ce9-96df-24a145ede961" />

Settings for creating seamless tiles. Located in Advanced - Developer Debug Mode - Control - Tiled. Sometimes a little edge refinement is required in any Photo Editor.

**Transparency**

<img width="452" height="152" alt="image" src="https://github.com/user-attachments/assets/a63e3921-f807-4872-9884-74698ce8a3e9" />

<img width="385" height="100" alt="image" src="https://github.com/user-attachments/assets/4b65d0f9-905f-4a8b-b6dd-8cd13870e3a5" />

Settings for creating images on a transparent background and a mask for it are located in the "Advanced" - "Developer Debug Mode" - "Controls" - "Transparency" section.

<img width="971" height="487" alt="image" src="https://github.com/user-attachments/assets/04db5df6-d9ed-4b75-9073-c5dbdd4549f7" />

None - the normal generation mode

Attention Injection - This mode uses LoRA rank 256, turning SDXL into a transparent image generator. It transforms the model's latent distribution into a "transparent latent space" that can be decoded by a dedicated VAE pipeline.

Conv Injection - This method uses an alternative model to transform SDXL into a transparent image generator. It uses biases on all convolutional layers (and, in fact, on all layers that are not q, k, v in any of the attention layers). These biases can be combined with any XL model to change the latent distribution to transparent images. Since learning the biases on all q, k, v layers was eliminated, the understanding of SDXL should be fully preserved. However, in practice, this first method has proven to yield better results. This method is used for some special cases that require special understanding. This method can have a strong impact on the style of the underlying model. This extension is based on layerdiffuse by lllyasviel (https://github.com/lllyasviel/sd-forge-layerdiffuse)

**CFG control type**

<img width="441" height="154" alt="image" src="https://github.com/user-attachments/assets/da27c255-7707-44d0-8228-a62aa7d9c820" />

CFG control type determines how guidance is applied during image generation:

Off - uses the standard CFG without adjustments;
CFG rescale - reduces artifacts at high Guidance Scale values by rescaling the guidance signal;
CFG Mimicking from TSNR -  adaptively adjusts guidance strength based on the noise level at each diffusion step, improving image quality and coherence.

**ADetailer**

<img width="830" height="965" alt="image" src="https://github.com/user-attachments/assets/30c8ea16-5c00-46f1-8d3e-740338ea1705" />
<img width="344" height="286" alt="image" src="https://github.com/user-attachments/assets/0fd1ecc4-b9e7-4670-bbca-f9c1b34dd4e3" />



| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          |                                                                                                                                      |
| ADetailer model classes           | Comma separated class names to detect. only available when using YOLO World models | If blank, use default values.<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |


| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

Support [SEP], [SKIP], [PROMPT] tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

The YOLO and Mediapipe models are used as detection models. Additional models can be downloaded from civitai.com using the Civitai Model Helper.

**Support external upscalers**

<img width="834" height="380" alt="image" src="https://github.com/user-attachments/assets/c25bd5f1-5103-44b8-a04d-b51ad0d2406c" />

You can load external upscalers using civatai helper. The upscaler is based on the data the model was trained on.


<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

![image](https://github.com/user-attachments/assets/5cf86b6d-e378-4d85-aed1-c48920b6c107)


***Change log***

v9.2.5
 1. Support external upscalers

v9.2.4
 1. Added CFG control type selection
 2. Add ADetailer
 3. Some bug fix and optimization

v9.2.3
 1. View the models page on civitai.com

v9.2.2
 1. Settings for Transparency generation

v9.2.1
 1. Add Logo in tools

v9.2.0
 1. Add Vector Style
 2. Add SVGcode
 3. Add Vector in generate mode
 4. Add Vector in module
 5. Add batch in Codeformer in module
 6. Add batch in Inswapper in module
 7. Optimized ImageBatch
 8. The x/y/z-plot module has been updated with the following parameters: Lora Name, Lora weight, Codeforeme, refiner settings, aspect ratio selection
 9. Final preview of ImageBatch
 10. Final preview of PromptBatch

v9.1.7
1. Add Roller in tools

v9.1.6
1. Settings for Seamless tiling

v9.1.5
1. A deeper integration of the OneButtonPrompt module has been carried out and it has been moved to the 'in generation' group

v9.1.4
1. Some bug fix

v9.1.3
1. Add Manga Recolor ControlNet

v9.1.2
1. Added selector for choosing to load individual files or zip archive in Image Batch module 

v9.1.1
1. Add view trigger words of LoRA
2. Bug fix in wildcards
3. Add Civitai_API_key saving
4. Add preview files for InstantId and Photomaker styles

v9.1.0
1. Add PhotoMaker module
2. Add PhotoMaker styles
3. Add Random Aspect Ratio

v9.0.0
1. Dividing extensions into groups
2. Add InstantID module
3. Add Inswapper module
4. Add CodeFormer module
5. Add InstanID styles

v8.1.0
1. Some bug fix
2. Add TextMask - Fast text editor with mask creation for ControlNet and Inpaint

v8.0.4
1. User style upload
2. Fixed maximum height of PromptBatch
3. Option to load only positive prompts

v8.0.3
1. Some bug fix
2. Add load prompt from files in PromptBatch
3. The ImageBatch extension interface has been simplified

v8.0.2
1. Some bug fix

v8.0.1
1. Add Filename Prefix
2. Add Paths and Presets Settings

V8
1. Save Image Grid for Each Batch
2. Add X/Y/Z Plot Extention
3. Prompt Batch is now in the extensions panel
4. Images Batch has become easier to manage while retaining its functionality
5. Images Batch is now in the extensions panel
6. Add support VAE and LyCoris in Civitai_Helper
7. The extension Remove Background has been changed
8. Add auto update on startup

V7
1. Add OpenPoseEditor
2. Fix bug in Image Batch Mode
3. Added cell selection in Image Batch Mode
4. Added selection of adding base prompts in Prompt Batch Mode
5. Add OpenPose ControlNet
6. Add Recolor ControlNet
7. Add Scribble ControlNet

V6

1. Add Prompt Batch Mode
2. Rename Batch Mode to Images Batch Mode
3. Fixed an incorrect start random number in Batch Mode
4. Add visual management of Wildcard and Words/phrases of wildcard
5. Added the ability to set any resolution for the generated image
6. Add OneButtonPrompt

V5
1. Model Downloader replaced with Civitai Helper

V4
1. Add VAE download
2. Add Batch mode

V3
1. Add Photopea
2. Add Remove Background
3. Add Extention Panel
4. All extensions are available in Extention Panel

V2
1. Added a Model Downloader to Fooocus webui instead of colab

V1
1. added the ability to download models from the civitai.com
2. saving the generated image to Google Drive
3. added prompt translator
4. added a patch for the ability to work in free colab mode 
