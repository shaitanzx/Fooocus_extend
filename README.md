I would like to introduce a new fork of the popular generative neural network **Fooocus - Fooocus extend**. 
I would like to point out that this fork can be run both locally on your computer and via Google Colab. 
Let's look at everything in order. 

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
![image](https://github.com/user-attachments/assets/3399327d-0fe4-46cc-8f34-430e94ab7910)

This module is also intended for face replacement.

Source Image Index - index of the face in the reference image. Faces are numbered from left to right from top to bottom starting from zero. If you specify -1, the average mixed face of all available faces will be taken as the reference face

Target Image Index - index of the face in the output image. This is the index of the face to be replaced in the output image. If you specify -1, all faces will be replaced.

Source Face Image - image with face

**CodeFormer**

![image](https://github.com/user-attachments/assets/512eb538-1923-439d-ad2a-826d7594e49e)

Face enhancement module with upscale capability

Pre_Face_Align - aligns the face if it is tilted

Background Enchanced - improve background quality

Face Upsample - adjust reference face to the size of the input image face

Upscale - image enlargement

Codeformer_Fidelity - signability coefficient, inversely proportional to quality

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

**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions
   
![009](https://github.com/user-attachments/assets/0f0005e7-1c0f-48b6-9c89-7ad1842d974e)


In the ‘Main’ tab, you can select the preset of the prompt generation theme, as well as specify unchanging prompt prefixes and suffixes.

![010](https://github.com/user-attachments/assets/e3c65841-362e-424a-9534-9a4be605483a)


In the ‘Workflow assist’ tab you can generate 5 prompts, transfer each of them to the Fooocus prompt field and also to the Workflow field. If you select Workflow mode, it will generate not a new prompt, but variations of the prompt specified in the Workflow text field. The level of variation is selected by the engine located just below it

![011](https://github.com/user-attachments/assets/83f70c92-a441-4f0f-b56e-b1c236f2e407)


In this tab you can select the prompt syntax for different generation models, the length of the prompt, and enable the prompt generation enhancer.

![012](https://github.com/user-attachments/assets/a215a97c-155d-4e3e-a729-74cdc8393148)


This is where you can control the generation of negative prompts

![013](https://github.com/user-attachments/assets/f05e26cf-1c2f-4341-b3cd-5fbb9c552217)


In this tab you can start the image generation queue by generated samples. Before starting you need to specify the aspect ratio of the generated image (Size to generate), the number of generated prompts (Generation of prompts), and the models to use (Model to use).

**X/Y/Z Plot**

![019](https://github.com/user-attachments/assets/a0392f94-8ef0-4377-a2d6-9d260ba4a20c)


This extension allows you to make image grids to make it easier to see the difference between different generation settings and choose the best option. You can change the following parameters - Styles, Steps, Aspect Ratio, Seed, Sharpness, CFG (Guidance) Scale, Checkpoint, Refiner, Clip skip, Sampler, Scheduler, VAE, Refiner swap method, Softness of ControlNet, and also replace words in the prompt and change their order

**Inswapper**
![image](https://github.com/user-attachments/assets/2df95460-eab8-420c-9388-246034ec27f4)

The full analog of this module in the “in generation” panel, unlike which you need to load an additional input image

**CodeFormer**

![image](https://github.com/user-attachments/assets/9c31d2a3-e8e1-409d-97e0-5a7bd425b434)

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


**“tools”** panel

**Civitai Helper**

![014](https://github.com/user-attachments/assets/27814499-2c24-4421-a4f9-0c3503b5f66b)



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

**OpenPoseEditor**

![015](https://github.com/user-attachments/assets/e0c7d4e6-e288-41da-ba57-622bfe601551)


This module allows you to create skeletons for subsequent image creation using OpenPose ControlNet. You can also create a skeleton from an existing image.


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

**View trigger words of LoRA**

![image](https://github.com/user-attachments/assets/0ea83134-b8ec-4960-a436-c8343d0762f7)

If trigger words are not shown, then you need to scan LoRa, in the Civitai Helper module in the Scan Models for Civitai section


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
