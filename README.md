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

![image](https://github.com/user-attachments/assets/50e6b777-22ef-44e8-9b1e-35e2c91bb9d8)


 In a nutshell, this module allows you to upscale and create images based on existing images. To better understand this module, I advise you to do some experiments yourself. But I would like to note that its application allows you to use your images as references and change their style depending on the prompt and the selected model. First you need to create a zip-archive with your images. There should be no subfolders in the archive, file names should not use characters other than Latin. Upload the prepared archive to the Upload a zip file window. Next, you should choose the mode of changing the resolution of images
- Not scale - the generation will not take into account the resolution of the original image
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the original image will be selected.
- to OUTPUT - in this case first the resolution of the original image will be changed to the generated one with preserving the proportions, and then the generation will start
  
Since the whole generation process takes place in Input Image mode, you need to open this panel. Depending on what you want to do with your original images, select the Upscale or ImagePrompt tab. If you've decided to just upscale your images, then you just select the tab you want and proceed to launch. If you want to use your images as a reference, then open the ImagePrompt tab, adjust the weight and stop steps of the cell of the tab. In order not to make a mistake with the settings, I advise you to place one of your images in the cell beforehand and perform some generation in the normal mode, and after selecting the settings, proceed to group processing. I remind you that the Input Image panel should be active and open in the corresponding tab until the end of group generation.
If you are processing in ImagePromp mode, you can choose which one your images will be loaded into. This allows you to select other reference images and ControlNet modes for other cells.

Clicking the Add to queue button will unpack the previously downloaded archive and put all jobs in the queue. The number of queued jobs will be indicated in brackets.

Start queue starts the queue for execution. 

Stop queue - stops execution of the queue after the current iteration is generated. 

Clear queue - clears the queue of existing jobs, but does not delete the last loaded archive.

When the queue is finished, click on Output->Zip to generate an archive with all previously generated images from the output folder. The archive itself will appear in the Download a Zip file window. From there you can actually download it.

Clear Output - clears the output folder. It should be noted that it clears not only the folder for the current date, but the whole folder.

5.	**Prompt Batch** (batch processing of prompts)

![image](https://github.com/user-attachments/assets/ad6d71ed-8f3b-458f-af7e-b58ea1bd280d)

This module allows you to start generating several prompts in the queue for execution. To do this, you need to fill in the table In the prompt column enter a positive prompt, and in the negative prompt column enter a negative prompt respectively. Clicking on New row will add an empty row to the end of the table. Delete last row deletes the last row of the table. Start batch starts execution of the prompt queue for generation.
You can also choose to add base positive and negative prompts.
None - basic prompts will not be added.
Prefix - base prompts will be added before prompts from the table.
Suffix - base samples will be added after samples from the table.

Now let's look at the Extention panel. Here are some extensions for Stable Forge adapted for Fooocus

1.	**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions
   
![image](https://github.com/user-attachments/assets/64bc1e3d-efdc-4782-8070-fe5a181a2d45)

In the ‘Main’ tab, you can select the preset of the prompt generation theme, as well as specify unchanging prompt prefixes and suffixes.

![image](https://github.com/user-attachments/assets/6e7fee1b-a1d8-493f-a518-dc60c011c0f7)

In the ‘Workflow assist’ tab you can generate 5 prompts, transfer each of them to the Fooocus prompt field and also to the Workflow field. If you select Workflow mode, it will generate not a new prompt, but variations of the prompt specified in the Workflow text field. The level of variation is selected by the engine located just below it

![image](https://github.com/user-attachments/assets/66816520-eddd-40d7-984f-6e8eeb6ba0ef)

In this tab you can select the prompt syntax for different generation models, the length of the prompt, and enable the prompt generation enhancer.

![image](https://github.com/user-attachments/assets/f04d4527-d9fc-41e6-93f9-05ba9608b524)

This is where you can control the generation of negative prompts

![image](https://github.com/user-attachments/assets/94dd93f3-9d7e-4d70-9884-83a255d55414)

In this tab you can start the image generation queue by generated samples. Before starting you need to specify the aspect ratio of the generated image (Size to generate), the number of generated prompts (Generation of prompts), and the models to use (Model to use).

2.	**Civitai Helper**

![image](https://github.com/user-attachments/assets/b59821d4-8954-4012-ad12-f1bdc62d2dec)

This extension allows you to download models for generation from the civitai website.  To download a model you first need to specify your Civitai_API_key. In the Download Model section in the Civitai URL field you need to specify a link to the required model from the browser address bar and click Get Model Info by Civitai URL. After analysing the link you will be given information about the model. You will also be able to select the version of the model before downloading. This extension also allows you to find duplicates of downloaded models and check for updates. In addition, there is a group download option.

3.	**Prompt Translate**

![image](https://github.com/user-attachments/assets/de7d569a-3f1e-41ac-b47a-731ea28063be)

Allows to translate both positive and negative prompts from any language into English, both before generation and directly during generation.

4.	**Photopea** - a free online analogue of Photoshop

![image](https://github.com/user-attachments/assets/7c5b5cf2-292e-407a-9713-41b58034cbe0)

5.	**Remove Background**

![image](https://github.com/user-attachments/assets/398d6ba2-5413-490a-8fe0-11435a50e34c)


This extension allows you to remove the background on an uploaded image with a single button

6.	**OpenPoseEditor**

![image](https://github.com/user-attachments/assets/00acf39e-58f0-415c-bce3-f21a20a443a7)

This module allows you to create skeletons for subsequent image creation using OpenPose ControlNet. You can also create a skeleton from an existing image.

7.	**OpenPose ControlNet**

![image](https://github.com/user-attachments/assets/bd2459d0-7d88-4c09-9eb0-80c24ab5b61b)

Allows you to create an image based on the pose skeleton.

8.	**Recolor ControlNet**

![image](https://github.com/user-attachments/assets/5209e584-36ec-4577-a330-9b32070bc6de)

Allows you to colorize an image based on a black and white image.

9.	**Scribble ControlNet**

![image](https://github.com/user-attachments/assets/e356ad7d-89fd-4ff0-b5f4-e21494bac61a)

Allows you to color an image based on a sketch.

<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend without autoupdate. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

![image](https://github.com/user-attachments/assets/5cf86b6d-e378-4d85-aed1-c48920b6c107)




***Change log***

V7 (current version)
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
