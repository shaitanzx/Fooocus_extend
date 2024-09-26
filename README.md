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

![image](https://github.com/user-attachments/assets/d7f3e8ec-1d97-4f6d-b0a7-4d7122f325e7)

  In a nutshell, this module allows you to create images based on existing images. To better understand this module, I advise you to do some experiments yourself. But I would like to note that its application allows you to use your images as references and change their style depending on the prompt and the selected model. First you need to create a zip-archive with your images. There should be no subfolders in the archive, file names should not use characters other than Latin. Upload the prepared archive to the Upload a zip file window.  Next, you should choose the mode of changing the resolution of images
- Not scale - the generation will not take into account the resolution of the original image
- to ORIGINAL - this means that immediately before generation the resolution equal to the resolution of the original image will be selected.
- to OUTPUT - in this case first the resolution of the original image will be changed to the generated one with preserving the proportions, and then the generation will start
  
Since the whole generation process takes place in Input Image mode, you need to open this panel. Depending on what you want to do with your original images, select the Upscale or ImagePrompt tab. If you've decided to just upscale your images, then you just select the tab you want and proceed to launch. If you want to use your images as a reference, then open the ImagePrompt tab, adjust the weight and stop steps of the first cell of the tab. In order not to make a mistake with the settings, I advise you to place one of your images in the first cell beforehand and perform some generation in the normal mode, and after selecting the settings, proceed to group processing. I remind you that the Input Image panel should be active and open in the corresponding tab until the end of group generation.

Clicking the Add to queue button will unpack the previously downloaded archive and put all jobs in the queue. The number of queued jobs will be indicated in brackets.

Start queue starts the queue for execution. 

Stop queue - stops execution of the queue after the current iteration is generated. 

Clear queue - clears the queue of existing jobs, but does not delete the last loaded archive.

When the queue is finished, click on Output->Zip to generate an archive with all previously generated images from the output folder. The archive itself will appear in the Download a Zip file window. From there you can actually download it.

Clear Output - clears the output folder. It should be noted that it clears not only the folder for the current date, but the whole folder.

5.	**Prompt Batch** (batch processing of prompts)

![image](https://github.com/user-attachments/assets/1c73a095-d36b-4452-baf3-a86d7667118d)

This module allows you to start generating several prompts in the queue for execution. To do this, you need to fill in the table In the prompt column enter a positive prompt, and in the negative prompt column enter a negative prompt respectively. Clicking on New row will add an empty row to the end of the table. Delete last row deletes the last row of the table. Start batch starts execution of the prompt queue for generation.

Now let's look at the Extention panel. Here are some extensions for Stable Forge adapted for Fooocus

1.	**OneButtonPrompt** - allows you to generate prompts, generate variations of your prompt and runs to generate images from them. I will not dwell on the detailed description of this module - I will only point out the main functions
   
![image](https://github.com/user-attachments/assets/43827b70-a96a-4180-b177-f0b24d566a2c)

In the ‘Main’ tab, you can select the preset of the prompt generation theme, as well as specify unchanging prompt prefixes and suffixes.

![image](https://github.com/user-attachments/assets/6e7fee1b-a1d8-493f-a518-dc60c011c0f7)

In the ‘Workflow assist’ tab you can generate 5 prompts, transfer each of them to the Fooocus prompt field and also to the Workflow field. If you select Workflow mode, it will generate not a new prompt, but variations of the prompt specified in the Workflow text field. The level of variation is selected by the engine located just below it

![image](https://github.com/user-attachments/assets/aadadc7d-2641-4f1b-9ea4-4d99b14e11bc)
![image](https://github.com/user-attachments/assets/b2048816-522e-44cc-9b65-af7af5eb2f6c)
![image](https://github.com/user-attachments/assets/d24c6c43-cd1e-460d-b9e9-e49b926064f0)

In this tab you can select the prompt syntax for different generation models, the length of the prompt, and enable the prompt generation enhancer.

![image](https://github.com/user-attachments/assets/97ab92ea-c064-453a-bf51-a0de2f4d2fdb)

This is where you can control the generation of negative prompts

![image](https://github.com/user-attachments/assets/b8d3a7e4-4377-4d87-887f-6f1e472ab8ad)

In this tab you can start the image generation queue by generated samples. Before starting you need to specify the aspect ratio of the generated image (Size to generate), the number of generated prompts (Generation of prompts), and the models to use (Model to use).

2.	**Civitai Helper**

![image](https://github.com/user-attachments/assets/01250bff-947c-4aea-ab49-3c2a9e7715ac)
![image](https://github.com/user-attachments/assets/90a67800-9351-4dc7-a3e2-1c292d1f66b1)
![image](https://github.com/user-attachments/assets/1c032293-6d51-435f-bf38-b26be5b8de9d)
![image](https://github.com/user-attachments/assets/200bc5c8-1c06-4251-b63b-9a8d03e12a60)
![image](https://github.com/user-attachments/assets/5bb8f655-40df-46b7-8cbd-86dcaf6b8f14)
![image](https://github.com/user-attachments/assets/854de34d-5614-4a32-8f78-a0bf41038478)
![image](https://github.com/user-attachments/assets/8c36f474-098c-4ee8-85dc-f681652dc4ff)

This extension allows you to download models for generation from the civitai website.  To download a model you first need to specify your Civitai_API_key. In the Download Model section in the Civitai URL field you need to specify a link to the required model from the browser address bar and click Get Model Info by Civitai URL. After analysing the link you will be given information about the model. You will also be able to select the version of the model before downloading. This extension also allows you to find duplicates of downloaded models and check for updates. In addition, there is a group download option.

3.	**Prompt Translate**

![image](https://github.com/user-attachments/assets/833a8076-dc19-4876-b195-51a55c1b95e9)

Allows to translate both positive and negative prompts from any language into English, both before generation and directly during generation.

4.	**Photopea** - a free online analogue of Photoshop

![image](https://github.com/user-attachments/assets/161007bc-d3da-4548-a53c-864580ca24e9)

5.	**Remove Background**

![image](https://github.com/user-attachments/assets/3801c52d-97fc-4953-a250-421a8a3924ca)

This extension allows you to remove the background on an uploaded image with a single button

<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend without autoupdate. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

***Change log***

V6 (current version) (temporarily only Colab without autoupdate)

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
