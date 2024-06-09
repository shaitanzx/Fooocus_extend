# Fooocus_extend
Extender for Fooocus

(Any borrowing of code without attribution and permission of the author is considered plagiarism and disrespect for the author - if any are found, they will be indicated here)

(Любые заимствования кода без указания авторства и разрешения автора считается плагиатом и неуважением к автору - если такие найдутся, то будут здесь указаны)


| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend.ipynb) | Fooocus_extend with autoupdate from Original Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_upd.ipynb) | Fooocus_extend without autoupdate from Original Colab. Base Version 2.4.1
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official






![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/83797030-bd0d-49a7-80a5-6398105b3c20)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/5b6fd1de-0d8c-4832-97b2-853260c4c313)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/09276692-f630-45c2-9605-9ff33d561e18)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/bfc44a92-dc55-4896-8e9d-ca62c36a664d)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/db746844-fae7-4e16-91f9-881df70a3c77)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/a117eea0-8278-4a03-b2d6-8f5d21ce6bb7)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/dede024d-ec0b-41a0-9b33-f85b1e986e5d)


Сlick on the picture below to watch the video of previouse version


<a href="http://www.youtube.com/watch?feature=player_embedded&v=VuXzHu4PLsk
" target="_blank"><img src="http://img.youtube.com/vi/VuXzHu4PLsk/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Startup order
1. Select a profile
2. Select a theme (light, dark)
3. Select tunnel (gradio, cloudflared)
4. Memory_patch – reduces consumed video memory, which prevents errors when using various ImagePrompt modes
5. GoogleDrive_output – enable saving of all generation results to your Google Drive
6. Launch colab

Model Dowloader
1. CivitAI_API_KEY - required for downloading models from civitai.com It is best to use your personal key, not a third-party one, since if necessary, you can always view ONLY YOUR download history on the site. To do this, you need to register on the website civitai.com and then in the settings you can get the key.
2. Checkpoint Link – adding links to models from civitai.com. If you need to load several models, links to them can be specified separated by commas (,) without spaces
3. Lora Link - adding links to Lora from the site civitai.com. If you need to download several Loras, links to them can be specified separated by commas (,) without spaces.
4. Embedding Link - adding links to Embedding from the site civitai.com. If you need to download several Loras, links to them can be specified separated by commas (,) without spaces.
5. Start Download - start downloading all files via links
6. If CivitAI_API_KEY is absent, then the download not started
7. After downloading all the files, in the Model tab in Advanced mode, you need to update the list of models (click Refresh All Files)
8. To apply Embedding, in the prompt field use a record like (embedding:file_name:1.1)

Prompt Translate
1. Enable Translate - enable the extension
2. Translate - translation of the prompt into English without starting generation
3. Auto translate "Prompt and Negative prompt" before Generate - enable automatic translation of positive and negative prompts into English before generation
4. See translated prompts after click Generate - show translations of prompts after generation is completed

Photopea - online image editor

Remove Background - remove background from image

Batch - mode of group processing of images using Image Prompt
1. Upload a ZIP file - upload ONE archive with reference images. File names must not contain spaces. The archive must not contain directories.
2. Scale Method - change permissions of input and output files
  - Not Scale - no additional changes will be made to the files
  - to ORIGINAL - the output file resolution will correspond to the resolution of the input file (reference).
  - to OUTPUT - before processing the generation, the resolution of the input file will be adjusted to the resolution of the output file with preserving the proportions.
3. Add to queue - unpacking of the downloaded archive into a folder for batch processing. The number of already existing unpacked files is specified in brackets
4. Start queue - start processing. After successful processing of all files, the folder is cleared. During the processing you can stop it. The stopping is performed only after the current task is completed
5. Clear queue - Clearing the folder with input files
6. Output-->ZIP - Archiving the folder with finished generations
7. Download a ZIP - window for downloading the archive with ready generations
8. Clear Output - clearing the folder with ready generations

To perform a batch operation, you need to 
- make all Fooocus settings in advance (select the models you need, make ImagePrompt settings, fix Seed if necessary, etc.).
- activate Input Image
- select the ‘Upscale or Variation’ or ‘Image Prompt’ tab, as the processing mode will be performed depending on the selected tab. In case the ‘Image Prompt’ tab will be opened, the loading of reference images will be performed only in the first cell.
- download and unpack the archive with reference images
- start processing
- after the end of processing, create an archive with ready generations and download it to your computer

Порядок запуска
1.	Выбрать профиль
2.	Выбрать тему (светлая, темная)
3.	Выбрать тунель (gradio, cloudflared)
4.	Memory_patch – уменьшает потребляемую видеопамять, что не позволяет вывалиться в ошибку при использовании различных режимов ImagePrompt
5.	GoogleDrive_output – включение сохранения всех результатов генераций на свой гуглдиск
6.	Запустить колаб

Model Dowloader
1.	CivitAI_API_KEY – необходим для загрузки моделей с civitai.com  Лучше всего использовать свой личный ключ, а не сторонний, так как в случае необходимости на сайте всегда можно посмотреть ТОЛЬКО СВОЮ историю загрузок. Для этого необходимо зарегистрироваться на сайте civitai.com и далее в настройках можно получить ключ.
2.	Checkpoint Link – добавление ссылок на модели с сайта civitai.com. При необходимости загрузки нескольких моделей, ссылки на них можно указывать через запятую (,) без пробелов
3.	Lora Link - добавление ссылок на Lora с сайта civitai.com. При необходимости загрузки нескольких Lora, ссылки на них можно указывать через запятую (,)  без пробелов.
4.	Embedding Link - добавление ссылок на Embedding с сайта civitai.com. При необходимости загрузки нескольких Lora, ссылки на них можно указывать через запятую (,)  без пробелов.
5.	Start Download - запуск скачивание всех файлов по ссылкам
6.	Если отстутсвует CivitAI_API_KEY, то загрузка производиться не будет
7.	После загрузки всех файлов, во вкладке Model в режиме Advanced, необходимо обновить список моделей (нажать Refresh All Files) 
8.	Для применения embedding, в поле промпта используйте запись типа (embedding:file_name:1.1)

Prompt Translate
1. Enable Translate - включение расширения
2. Translate - перевод промпта на англйский язык без запуска генерации
3. Auto translate "Prompt and Negative prompt" before Generate - включение автоматического перевода положительного и отрицательного промптов на английский язык перед генерацией
4. See translated prompts after click Generate - показывать переводы промптов после выполнения генерации

Photopea - онлайн редактор изображений

Remove Background - удаление фона с изображения

Batch - режим групповой обработки изображений с помощью Image Prompt
1. Upload a ZIP file - загрузка ОДНОГО архива с референсными изображениями. Имена файлов не должны содержать пробелы. Архив не должен содержать каталоги.
2. Scale Method - изменение разрешений входных и выходных файлов
  - Not Scale - дополнительных изменений файлов производиться не будет
  - to ORIGINAL - разрешение выходного файла будет соответствовать разрешению входному файлу (референсу)
  - to OUTPUT - перед обработкой генерацией разрешение входнго файла будет подгоняться под разрешение выходного файла с сохранением пропорций
3. Add to queue - распаковка загруженного архива в папку для групповой обработки. В скобках указывается количество уже имеющихся распакованных файлов
4. Start queue - запуск обработки. После успешной обработки всех файлов папка очищается. Во время обработки имеется возможность ее остановить. Остановка производится тольео после выполнения текущей задачи
5. Clear queue - Очистка папки с входными файлами
6. Output-->ZIP - Архивирование папки с готовыми генерациям
7. Download a ZIP - окно для скачивания архива с готовыми генерациями
8. Clear Output - очистка папки с готовыми генерациями

Для выполнения групповой операции вам необходимо 
- провести заранее все настройки Fooocus (выбрать нужне вам модели, произвести настройки ImagePrompt, по еобходимости зафикситовать Seed и т.д).
- активировать Input Image
- в нем выбрать вкладку "Upscale or Variation" или "Image Prompt", так как режим обработки будет производиться в зависимости от выбранной вкладки. В случае если будет открыта вкладка "Image Prompt", то загрузка референсных изображений будет производиться только в первую ячейку.
- загрузить и распаковать архив с референсными изображениями
- запустить обработку
- после окончания обработки создать архив с готовыми генерациями и скачать его себе на компьютер


Change log

V4 (current version)
1. Select tunnel (gradio, cloudflared)

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
