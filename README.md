# Fooocus_extend
Extender for Fooocus

(Any borrowing of code without attribution and permission of the author is considered plagiarism and disrespect for the author - if any are found, they will be indicated here)

(Любые заимствования кода без указания авторства и разрешения автора считается плагиатом и неуважением к автору - если такие найдутся, то будут здесь указаны)
<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend without autoupdate. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend with autoupdate from original repository.</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

![image](https://github.com/user-attachments/assets/468487b8-8d4e-454c-ba92-1c9e5b60feb7)
![image](https://github.com/user-attachments/assets/88d51e58-49c3-4b46-9455-39de97c67f16)
![image](https://github.com/user-attachments/assets/05d3e317-3a05-434d-bb9d-9e017e6110bb)
![image](https://github.com/user-attachments/assets/e8ef96a2-5dc3-496f-8940-6fe531b80697)
![image](https://github.com/user-attachments/assets/28704aa3-7a95-4ad5-b6d6-ff044ff103dc)
![image](https://github.com/user-attachments/assets/6ff1277f-f0ae-4917-9773-505912351a3e)
![image](https://github.com/user-attachments/assets/257d2944-c5ef-40ca-b330-4f370c54bddc)
![image](https://github.com/user-attachments/assets/d4ab118e-06b8-4b3c-be70-83a7501b36ee)
![image](https://github.com/user-attachments/assets/e0138862-9ebb-4a84-8b46-96593e8e108c)
![image](https://github.com/user-attachments/assets/15750c0d-4597-468d-ba7b-3812d608d087)
![image](https://github.com/user-attachments/assets/f3e2e442-3c31-4038-ad75-103e5dc880b8)
![image](https://github.com/user-attachments/assets/749eabcb-2ef6-4809-8177-cdd6a690b36a)
![image](https://github.com/user-attachments/assets/afd366ed-827f-4004-a031-dde28dae99a5)


Сlick on the picture below to watch the video of previouse version


<a href="http://www.youtube.com/watch?feature=player_embedded&v=VuXzHu4PLsk
" target="_blank"><img src="http://img.youtube.com/vi/VuXzHu4PLsk/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

***Startup order***
1. Select a profile
2. Select a theme (light, dark)
3. Select a tunnel - when gradio does not work, you can use the alternative cloudflared tunnel
4. Memory_patch – reduces consumed video memory, which prevents errors when using various ImagePrompt modes
5. GoogleDrive_output – enable saving of all generation results to your Google Drive
6. Launch colab

***Civitai Helper***

This module allows you to download models from the website chivitai.com. To download, be sure to indicate the Civitai API key
To apply Embedding, in the prompt field use a record like (embedding:file_name:1.1)

***Prompt Translate***
1. Enable Translate - enable the extension
2. Translate - translation of the prompt into English without starting generation
3. Auto translate "Prompt and Negative prompt" before Generate - enable automatic translation of positive and negative prompts into English before generation
4. See translated prompts after click Generate - show translations of prompts after generation is completed

***Photopea*** - online image editor

***Remove Background*** - remove background from image

***Batch*** - batch image processing mode
Upload a ZIP file - uploading an archive with images. File names must not contain spaces or symbols in various language encodings other than Latin
Select method
- NOT Scale - during processing, the original image will not change its size and processing will proceed in accordance with the selected
- to ORIGINAL - when processing the image resolution, the original resolution will be preserved
- to OUTPUT - during processing, the image will first be changed to the output resolution, and then processed
Add to queue (0) - unpacks the downloaded archive and adds images to the queue. The number of images in the queue is indicated in parentheses. Before adding images to the queue, you need to make all the settings (selection, models, Lora, etc. The Input Image window must be open. Depending on what is open (Image Prompt or Upscale), processing will be carried out
Start queue - start the queue execution. During execution, it is possible to stop it after the current generation has completed
Clear queue - clearing the queue
Output-->ZIP - archiving processed images
Download a ZIP file - downloading the archive
Clear OUTPUT - deleting the archive and all processed images

***Порядок запуска***
1.	Выбрать профиль
2.	Выбрать тему (светлая, темная)
3.	Выбрать тунель - когда gradio не работает, можно воспользоваться альтернативным тунелем cloudflared
4.	Memory_patch – уменьшает потребляемую видеопамять, что не позволяет вывалиться в ошибку при использовании различных режимов ImagePrompt
5.	GoogleDrive_output – включение сохранения всех результатов генераций на свой гуглдиск
6.	Запустить колаб

***Civitai Helper***

Данный модуль позволяет загрузить модели с сайта civitai.com. Для скачивание обязательно укажите Civitai API key  
Для применения embedding, в поле промпта используйте запись типа (embedding:file_name:1.1)

***Prompt Translate***
1. Enable Translate - включение расширения
2. Translate - перевод промпта на англйский язык без запуска генерации
3. Auto translate "Prompt and Negative prompt" before Generate - включение автоматического перевода положительного и отрицательного промптов на английский язык перед генерацией
4. See translated prompts after click Generate - показывать переводы промптов после выполнения генерации

***Photopea*** - онлайн редактор изображений

***Remove Background*** - удаление фона с изображения

***Batch*** - режим групповой обработки изображений
Upload a ZIP file - зарузка арихива с изображеними. Файлы в именах не должны иметь пробелов и символов в различных языковых кодировках, кроме латинской
Select method
- NOT Scale - при обработке исходное изображене не будет изменять свои размеров и обработка будет идти в соответствии с выбранным 
- to ORIGINAL - при обработке разрешение изображений буде сохранено исходное разрешение
- to OUTPUT - при обработке изображение сначала будет измененено до выходного разрешения, а потом будет обраотано
Add to queue (0) - распаковка загруженного архива и добавление изображений в очеред. В скобках указано количество изображений в очереди. Перед добавлением изображений в очередь, необходимо произвести все настройки (выбор, модели, Lora и дак далее. Обязательно должно быть открыто окно Input Image. В зависимости, что открыто (Image Prompt или  Upscale) так и будет проводиться обработка
Start queue - запуск выполнения очереди. Во время выполнения имеется возможность ее остановить послевыполнения текущей генерации
Clear queue - очистка очереди
Output-->ZIP - архивирование обработанных изображений
Download a ZIP file - скачивание архива
Clear OUTPUT - удаление архива и всех обработанных изображений


***Change log***

V5(current version)
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
