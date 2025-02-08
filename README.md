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

![004](https://github.com/user-attachments/assets/d11708d9-0386-4771-bf36-6b2dd90028af)




 In a nutshell, this module allows you to perform group upscaling of images, as well as create images based on a group of existing images using ImagePromt (ControlNet). To better understand this module, I advise you to conduct a few experiments yourself. But I want to note that its use allows you to use your images as references and change their style depending on the hint and the selected model. First, you need to create a zip archive with your images. The archive should not contain subfolders, the file names should not contain characters other than Latin. Upload the prepared archive to the "Upload zip file" window. Next, select the mode for changing the image resolution
- Not scale - the generation will not take into account the resolution of the original image
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the original image will be selected.
- to OUTPUT - in this case first the resolution of the original image will be changed to the generated one with preserving the proportions, and then the generation will start
  
Depending on what you want to do with your source images, select "Action" - Upscale or ImagePrompt. In the "Method" drop-down list, select the appropriate image processing method. In case of using ImagePrompt, you also need to select the "Stop at " and "Weight" parameters.

Clicking the Add to queue button will unpack the previously downloaded archive and put all jobs in the queue. The number of queued jobs will be indicated in brackets.

Start queue starts the queue for execution. 

Clear queue - clears the queue of existing jobs, but does not delete the last loaded archive.

When the queue is finished, click on Output->Zip to generate an archive with all previously generated images from the output folder. The archive itself will appear in the Download a Zip file window. From there you can actually download it.

Clear Output - clears the output folder. It should be noted that it clears not only the folder for the current date, but the whole folder.

If the execution proceeds without errors or interruptions, the queue will be cleared automatically, and the downloaded archive will remain in memory. Otherwise, the queue will not be cleared.

5.	**Prompt Batch** (batch processing of prompts)

![005](https://github.com/user-attachments/assets/15dffb7d-b5f9-4893-a930-6ac0e9d831cb)



This module allows you to start generating several prompts in the queue for execution. To do this, you need to fill in the table In the prompt column enter a positive prompt, and in the negative prompt column enter a negative prompt respectively. Clicking on New row will add an empty row to the end of the table. Delete last row deletes the last row of the table. Start batch starts execution of the prompt queue for generation.
You can also choose to add base positive and negative prompts.
None - basic prompts will not be added.
Prefix - base prompts will be added before prompts from the table.
Suffix - base samples will be added after samples from the table.


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

![016](https://github.com/user-attachments/assets/68af63fb-ab4f-48cb-aa2b-989dc57166a4)

Allows you to create an image based on the pose skeleton.

13.	**Recolor ControlNet**

![017](https://github.com/user-attachments/assets/b763ee5a-fc01-4f68-948a-beaf0dc39c3a)

Allows you to colorize an image based on a black and white image.

14.	**Scribble ControlNet**

![18](https://github.com/user-attachments/assets/39b4d80f-591d-4301-bb5a-c3cc47fb5325)

Allows you to color an image based on a sketch.

15. **X/Y/Z Plot**

![019](https://github.com/user-attachments/assets/a0392f94-8ef0-4377-a2d6-9d260ba4a20c)


This extension allows you to make image grids to make it easier to see the difference between different generation settings and choose the best option. You can change the following parameters - Styles, Steps, Aspect Ratio, Seed, Sharpness, CFG (Guidance) Scale, Checkpoint, Refiner, Clip skip, Sampler, Scheduler, VAE, Refiner swap method, Softness of ControlNet, and also replace words in the prompt and change their order

16. **Save Image Grid for Each Batch**

![20](https://github.com/user-attachments/assets/033a6a71-7e14-478a-a3d4-307f021fecec)

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
