# Fooocus_extend
Extender for Fooocus

(Any borrowing of code without attribution and permission of the author is considered plagiarism and disrespect for the author - if any are found, they will be indicated here)

(Любые заимствования кода без указания авторства и разрешения автора считается плагиатом и неуважением к автору - если такие найдутся, то будут здесь указаны)
<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_update.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Fooocus_extend without autoupdate. Base version 2.5.5</td>
  </tr>
  <tr>
    <td><a href="https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Original Fooocus Colab</td>
  </tr>
</table>

![image](https://github.com/user-attachments/assets/468487b8-8d4e-454c-ba92-1c9e5b60feb7)
![image](https://github.com/user-attachments/assets/874ae50a-3457-449e-b6bd-2df4a67cf8da)
![image](https://github.com/user-attachments/assets/381682e9-fa66-404b-9b5a-0eabdc0e7e8c)
![image](https://github.com/user-attachments/assets/485d4e5b-9aa7-48ff-b227-95697a41c500)
![image](https://github.com/user-attachments/assets/7c0e484b-1db6-42a1-b9ab-9dca6e2c454d)
![image](https://github.com/user-attachments/assets/e1e20029-43c7-42cb-b963-2e13c4710dfa)
![image](https://github.com/user-attachments/assets/7e4754b1-39f4-4426-91c6-79a5a61ae1bb)
![image](https://github.com/user-attachments/assets/9b155d93-6700-416a-b546-5b5ebc3cf4a6)
![image](https://github.com/user-attachments/assets/792ea7b7-491a-4f45-ae05-d150df809b8c)
![image](https://github.com/user-attachments/assets/0a84e6b3-b129-4af5-9e6b-a92f46c70f09)
![image](https://github.com/user-attachments/assets/8e9ad22c-9b43-4877-8024-e09d16d72a16) 
![image](https://github.com/user-attachments/assets/8f86b511-9e99-4c0c-b6e3-5e92af5c5e2a)
![image](https://github.com/user-attachments/assets/e8ef96a2-5dc3-496f-8940-6fe531b80697)
![image](https://github.com/user-attachments/assets/28704aa3-7a95-4ad5-b6d6-ff044ff103dc)
![image](https://github.com/user-attachments/assets/6ff1277f-f0ae-4917-9773-505912351a3e)
![image](https://github.com/user-attachments/assets/257d2944-c5ef-40ca-b330-4f370c54bddc)
![image](https://github.com/user-attachments/assets/d4ab118e-06b8-4b3c-be70-83a7501b36ee)
![image](https://github.com/user-attachments/assets/e0138862-9ebb-4a84-8b46-96593e8e108c)
![image](https://github.com/user-attachments/assets/f2f9de3c-f062-464d-bb19-fd16da4d8500)
![image](https://github.com/user-attachments/assets/7503c450-803a-413a-8b09-02c57f38ac76)
![image](https://github.com/user-attachments/assets/f88fdd72-ea5e-4b37-8764-a2db5e9a9e4a)
![image](https://github.com/user-attachments/assets/e5985c70-2414-4b5d-85af-0007ff0326a3)
![image](https://github.com/user-attachments/assets/07b4ceaf-ab8a-4b33-a14b-44828bd69e3f)
![image](https://github.com/user-attachments/assets/cb54945b-e5c4-4c5d-8a96-97bf3d0c9153)
![image](https://github.com/user-attachments/assets/cd59bc16-b66a-4a1a-8f46-919366be5158)




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

***Images Batch*** - batch image processing mode
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

***Images Prompts*** - group prompt processing mode
Enter positive and negative prompts into the table. Click Start batch to start executing the generation queue with the current settings for models, styles, and everything else.

***Wildcards*** - visual management of Wildcard and Words/phrases of wildcard for substitution into positive prompt

***Prompt Batch*** - allows you to enter multiple positive and negative prompts for sequential generation with the current settings

***OneButtonPrompt*** - prompt generator with the ability to customize styles and image themes

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

***Images Batch*** - режим групповой обработки изображений
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

***Images Prompts*** - режим групповой обработки промптов
Введите в таблицу положительные и отрицательные промпты. Нажмите Start batch для начала выполнения очереди генерации с текущими настройками моделей, стилей и всего остального

***Wildcards*** - визуальное управление Wildcard и подстановочными фразами для подстановки в положительный промпт

***Prompt Batch*** - позволяет ввести несколько положительных и отрицательных промптов для последовательной генерации с текущими настройками

***OneButtonPrompt*** - генератор промптов с возможность настрокаи стилей и тем изображений

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
