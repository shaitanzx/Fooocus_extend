# Fooocus_extend
Extender for Fooocus

(Any borrowing of code without attribution and permission of the author is considered plagiarism and disrespect for the author - if any are found, they will be indicated here)

(Любые заимствования кода без указания авторства и разрешения автора считается плагиатом и неуважением к автору - если такие найдутся, то будут здесь указаны)


| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend.ipynb) | Fooocus_extend with autoupdate from Original Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaitanzx/Fooocus_extend/blob/main/Fooocus_extend_wo_upd.ipynb) | Fooocus_extend without autoupdate from Original Colab. Base Version 2.3.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official






![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/83797030-bd0d-49a7-80a5-6398105b3c20)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/e59e3400-0d75-4428-bbee-46e7011dbaa1)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/5211a026-0eb9-4838-b650-8aeeed097e9b)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/bfc44a92-dc55-4896-8e9d-ca62c36a664d)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/db746844-fae7-4e16-91f9-881df70a3c77)
![image](https://github.com/shaitanzx/Fooocus_extend/assets/162459965/a117eea0-8278-4a03-b2d6-8f5d21ce6bb7)


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
