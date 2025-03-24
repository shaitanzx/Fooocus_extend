I would like to introduce a new fork of the popular generative neural network **Fooocus - Fooocus extend**. 
I would like to point out that this fork can be run both locally on your computer and via Google Colab. 
Let's look at everything in order. 

1.	**Launch**. If you will run it on a local machine, you can safely skip this item.
   
![image](https://github.com/user-attachments/assets/468487b8-8d4e-454c-ba92-1c9e5b60feb7)

Before launching you are offered to choose the following settings
Fooocus Profile - select which profile will be loaded at startup (default, anime, realistic).
Fooocus Theme - select a theme - light or dark.
Tunnel - select launch tunnel. When it happens that gradio stops working for some reason, you can choose cloudflared tunnel. However, the generation is a bit slower
Memory patch - adds a few keys to the launch bar that allow you to optimise your graphics card if you are using the free version of Google Colab. If you have paid access, this item can be disabled
GoogleDrive output - connects your GoogleDisk and save all of your generation directly to it.

2.	**Select the resolution and aspect ratio of the generated image**

![image](https://github.com/user-attachments/assets/ba5ce3d4-8f36-4f64-af82-760713c44c6a)

This setting is located in the generation resolution selection tab. Here you can select the number of horizontal and vertical points, aspect ratio. To apply the settings, click the Set button and select this resolution from the list of proposed resolutions. Your resolution will be the last one.

3.  **Wildcard**

![image](https://github.com/user-attachments/assets/45a4fc1f-72f6-479a-96ea-d61b6c62333e)

This module allows you not to memorise existing files with wildcard words, but to select them directly from a list of dictionaries. You can also select directly the item you need from the list.

4.	**Image Batch** (batch image processing)

![image](https://github.com/user-attachments/assets/c99f5a4a-b26f-42f6-9871-3a464f4bb73d)



 In a nutshell, this module allows you to perform group scaling of images, as well as create images based on a group of existing images using ImagePromt (ControlNet). To better understand this module, I suggest you do some experiments on your own. But I want to point out that using it allows you to use your images as references and change their style depending on the cue and the model you choose. First you need to create a zip archive with your images. The archive must not contain any subfolders, file names must not contain characters other than Latin. Upload the prepared archive to the "Upload zip file" window.

 Next, select the mode of changing the image resolution
- NOT scale - the source image resolution will not be taken into account during generation
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the source image will be selected.
- to OUTPUT - in this case, before generation the resolution of the source image will be changed to the generated one with preserving the proportions
  
Depending on what you want to do with the source images, select Action - Upscale or ImagePrompt. From the Method drop-down list, select the appropriate image processing method. If you are using ImagePrompt, you must also select the "Stop at" and "Weight" options.

Start batch - starts the process for execution. 

When the process is finished, click on Output->Zip button to create an archive with all previously created images from the output folder. The archive itself will appear in the "Download a Zip file" window. You can download it from there.

Clear Output - clear the output folder. It should be noted that not only the folder for the current date is cleared, but also the whole folder.


5.	**Prompt Batch** (batch processing of prompts)

![image](https://github.com/user-attachments/assets/fe1a2177-d579-4ffc-bc42-118b34c93d99)


This module allows you to run generation of several hints sequentially one after another. To do this, you should fill in the table. Enter a positive hint in the hint column and a negative hint in the negative hint column respectively. Clicking the New Row button will add an empty row to the end of the table. Delete Last Row deletes the last row of the table. Start batch starts the execution of the list of prompts to generate.
You can also choose to add basic positive and negative hints.
None - no base hints will be added.
Prefix - base hints will be added before the table hints.
Suffix - basic hints will be added after hints from the table.

Load prompts from file - allows to load the list of positive and negative prompts from a text file into the table. The file can have any extension

The file with prompts should have the following structure. First there is a line with a positive prompt, then a line with a negative one. If you don't need to specify the negative prompt, leave the line empty, but the line with the positive prompt must always be there.
As an example, let's look at the following file

---- start of file -------

![image](https://github.com/user-attachments/assets/750dff48-0bcc-4a20-b754-2d40f443e938)

------ end of file --------

After loading it, the table will look as follows

![image](https://github.com/user-attachments/assets/1f005878-cadc-443f-a76f-7f20dda088b2)


6.	**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions
   
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

7.	**Civitai Helper**

![014](https://github.com/user-attachments/assets/27814499-2c24-4421-a4f9-0c3503b5f66b)



This extension allows you to download models for generation from the civitai website.  To download a model you first need to specify your Civitai_API_key. In the Download Model section in the Civitai URL field you need to specify a link to the required model from the browser address bar and click Get Model Info by Civitai URL. After analysing the link you will be given information about the model. You will also be able to select the version of the model before downloading. This extension also allows you to find duplicates of downloaded models and check for updates. In addition, there is a group download option.

8.	**Prompt Translate**

![image](https://github.com/user-attachments/assets/6ddff3b1-5e98-43d4-b102-b61176c40c84)


Allows to translate both positive and negative prompts from any language into English, both before generation and directly during generation.

9.	**Photopea** - a free online analogue of Photoshop

![007](https://github.com/user-attachments/assets/bd75f089-dee9-4c7a-a24b-2dfa5da7d2fd)


10.	**Remove Background**

![008](https://github.com/user-attachments/assets/f34a2aff-a2ad-4f53-8edb-0703a2c69386)


This extension is designed to add background removal, image/video processing, and blending to your projects. It provides precise background removal with support for multiple models, chroma keying, foreground adjustments, and advanced effects. Whether you’re working with images or videos, this extension provides everything you need to efficiently process visual content.

Key Features:

Multi-model background removal: Supports u2net, isnet-general-use, and other models.

Chroma keying support: Removes specific colors (green, blue, or red) from the background.

Blending modes: 10 powerful blending modes for image compositing.

Foreground adjustments: Scale, rotate, flip, and position elements precisely.

Video and image support: Easily process images and videos.

Multi-threaded processing: Efficiently process large files with streaming and GPU support. Customizable output formats: Export to PNG, JPEG, MP4, AVI, and more.

11.	**OpenPoseEditor**

![015](https://github.com/user-attachments/assets/e0c7d4e6-e288-41da-ba57-622bfe601551)


This module allows you to create skeletons for subsequent image creation using OpenPose ControlNet. You can also create a skeleton from an existing image.

12.	**OpenPose ControlNet**

![016](https://github.com/user-attachments/assets/79e5b10a-0fcb-40d9-8d46-ec9869725919)

Allows you to create an image based on the pose skeleton.

13.	**Recolor ControlNet**

![017](https://github.com/user-attachments/assets/2de6044f-2c23-4728-bf72-564d55d78300)

Allows you to colorize an image based on a black and white image.

14.	**Scribble ControlNet**

![018](https://github.com/user-attachments/assets/ad18e4f4-d8f8-410a-b4c8-822de3dfadd4)

Allows you to color an image based on a sketch.

15. **X/Y/Z Plot**

![019](https://github.com/user-attachments/assets/a0392f94-8ef0-4377-a2d6-9d260ba4a20c)


This extension allows you to make image grids to make it easier to see the difference between different generation settings and choose the best option. You can change the following parameters - Styles, Steps, Aspect Ratio, Seed, Sharpness, CFG (Guidance) Scale, Checkpoint, Refiner, Clip skip, Sampler, Scheduler, VAE, Refiner swap method, Softness of ControlNet, and also replace words in the prompt and change their order

16. **Save Image Grid for Each Batch**

![020](https://github.com/user-attachments/assets/04aebc91-de87-428e-b664-83b6d597e23f)

17. **Filename Prefix**

![image](https://github.com/user-attachments/assets/c5e5dcb3-ab28-4063-9dec-c3b242442c32)

This setting may be useful when working on several projects to separate one from another.

18. **Paths and Presets**

![image](https://github.com/user-attachments/assets/db023eb1-3487-4d50-baed-9d0c0dc69cd5)

Here you can change the paths to your models if they are already in other folders on the disk. If your Сheckponts or LORAs are in different folders, then the paths to them can be specified separated by commas (,). After changing the path, it is best to restart FooocusExtend.

Also here you can create a new preset based on the existing settings, and delete any of the existing ones, except for default and initial.
The preset saves the following parameters: base model, refiner, refiner_switch, loras settings, cfg scale, sharpness, CFG Mimicking from TSNR, clip_skip, sampler, scheduler, Forced Overwrite of Sampling Step, Forced Overwrite of Refiner Switch Step, performance, image number, prompt negative, styles selections, aspect ratio, vae, inpaint_engine_version


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
