## What Civitai Helper is For
([Читать на русском языке](#для-чего-нужен-civitai-helper))

`Civitai Helper` is a set of tools for working with generation models. It helps download a model from Civitai, select its version, find information for an already installed model, check local models for duplicates, and find out if new versions are available. When downloading or scanning, Helper also writes description and metadata files next to the model.

## Where to Open It and What is Needed

1. In Fooocus Extend, expand **`Extention` → `tools` → `Civitai_helper`**. [4]
2. For regular public Civitai pages, you can start with the model URL. Some files require authorization to download. For such cases, there is a **`Civitai API key`** field and an **`Apply setting`** button at the top of the tab.
3. If downloading requires a key, create one in your Civitai account and paste it into this field.
4. Click **`Apply setting`** after pasting. The program will save the key in `civitai_api_key.txt` in the Fooocus Extend working folder and also use it in download requests.

## Setting up the API key

In the top **`Settings`** block, there is a single text field **`Civitai API key`** and an **`Apply setting`** button. Copy the key entirely, paste it into the field, and click the button. The status line below will show that the setting has been applied.

The code writes the key to the text file `civitai_api_key.txt` and reads it back on startup. This is not a separate mode setting: the key applies to Helper downloads that require authorization. Public information about models and some other operations may work without it.

## Scanning Installed Models

The **`Scan Models for Civitai`** block searches for models in local Fooocus Extend folders. For each supported file, the extension calculates a SHA256 hash and uses it to request information from Civitai. Scanning goes through nested folders as well. Supported file extensions: `.bin`, `.pt`, `.pth`, `.safetensors`, and `.ckpt`.

The block has the following settings:

- **`Model Types`** — types of local models to check. By default, all types known to Helper are selected: `ti`, `ckp`, `lora`, `lycoris`, `vae`, `detection`, and `upscaler`. 
- **`Replace Old Metadata Formats*`** — disabled by default. When enabled, Helper re-downloads information where existing metadata files are missing, outdated in format, or lack a format version. The option relates to metadata formats, not to updating model weights. The code tries not to overwrite metadata of a model with a mismatched ID and skips overwriting an empty description over a non-empty one. Important warning: if you manually edited metadata, make a backup first. 
- **`Scan`** — starts scanning the selected folders. Scanning takes time; the interface asks you to wait and watch the console for details. At the end, the number of successfully processed models is displayed.

Scanning does not download new model weights. It creates or updates adjacent information files. By default, Helper also generates a JSON with a description. If the hash is not found on Civitai, the source attempts to write minimal local information.

## Getting Information about a Local Model by URL

The **`Get Model Info from Civitai by URL`** block is designed for situations where a regular scan did not find information about a local model. Select a local model file, specify the Civitai page URL, and click **`Get Model Info from Civitai`**. This operation adds information to an already existing local model; it does not download its weights.

Settings and fields of the block:

- **`Model Type`** — type of the local model. Default is `ckp` (checkpoint). When changing the type, the model list updates. The interface displays types from the configured Helper folders. 
- **`Only Show Models have no Info`** — disabled by default. If enabled, the list is limited to local models that do not yet have a filled ID in the Civitai info file. This is a dropdown list filter; it does not delete files or trigger processing on its own.
- **`Model`** — the local model of the selected type. If the list is empty, check that a file of the appropriate type is in the folder for that type. Names are also searched in nested folders.
- **`Civitai URL`** — the URL of the model page.
- **`Get Model Info from Civitai`** — gets information for the specified page and selected local model, writes available metadata, and downloads a preview if necessary.

For a model without an existing metadata file, the **`Only Show Models have no Info`** toggle will also keep it in the list. If you select a model that already has a correct ID, the enabled filter will not show it.

## Download a Single Model: `Single` Tab

In the **`Download Model`** block, select the **`Single`** tab. This is the main mode when you need to download a single model with a chosen version.

1. In **`Civitai URL`**, paste the link to the model page and click **`Get Model Info by Civitai Url`**. For a Civitai version page, you can use a link with the version ID. After loading the information, the name, type, base model, versions, folder, and available files will appear.
2. Select **`Model Version`**. After receiving the data, the first version returned by Civitai is selected by default. For a link to a specific version, the corresponding version will be selected if the URL contains its ID.
3. Check **`Sub-folder`**. By default, the root folder `/` is selected. The list also shows existing subfolders inside the folder of this type.
4. Choose how to handle existing files, then check the list of file types that will be downloaded.
5. Optionally change the name in **`Rename Model`**. Then click **`3. Download Model`** and wait for a completion or error message.

Settings in this form:

- **`Model Name`**, **`Model Type`**, and **`Base Model`** — reference fields filled from Civitai after getting information. They cannot be edited in the form. They help verify that the correct model and its variant are selected.
- **`Model Version`** — the version to download. Switching the version updates the file list, original name, base model, and preview. Leave the default version if you need the very first version in the received list; otherwise, choose the needed one manually.
- **`Sub-folder`** — the save location inside the folder of the chosen type. `/` is the root. Other options are taken from existing subfolders.
- **`Duplicate File Behavior`** — behavior when the final file name matches. Default is `Skip`: do not download over an existing file. `Overwrite` replaces the file with a matching name. `Rename New` adds a suffix `_2`, `_3`, etc., to the new name.
- Lines with files **`Model`**, **`Training Data`**, **`Config`**, and **`VAE`** appear depending on the contents of the selected version. Checkboxes are enabled by default; the main `Model` line is mandatory and cannot be disabled. Uncheck additional files that are not needed. The VAE code directs it to the VAE folder, even if other version files are saved in the model type folder.
- **`Download All Files`** is disabled by default and appears when there is more than one file in the version. When enabled, all files of this version are downloaded, including types that the interface cannot classify. Therefore, enable it only when you really need all attachments, not just the model weights. Displaying individual file type checkboxes is hidden when this is enabled.
- **`Unhandled Files (Files that we don't know what to do with but will still downloaded with "Download All Files")`** — a reference line with files for which Helper has no separate type. It does not activate downloading on its own; such files are included via `Download All Files`.
- **`Preview Image`** — a gallery of example images from Civitai for the selected version. Available images may be filtered by the `Black Out NSFW` setting in the main Fooocus interface.
- **`Rename Model`** — the base file name without extension; after selecting a version, it is filled automatically from the file name on Civitai. Changing the name affects the downloaded files. The extension is shown by an internal `Model extension` field, which is hidden in the interface.
- **`3. Download Model`** — starts the download, writes progress and the final path in the status block and console.

Duplicate setting nuance: `Skip`, `Overwrite`, and `Rename New` manage path and file name matching. But before that, the code separately searches for metadata of the same Civitai version in the chosen folder. If this version is already listed in the folder, the operation stops with the message "This model version already exists"; changing the duplicate name behavior will not force a re-download of the same version.

## Download Multiple Models: `Batch Download` Tab

The **`Batch Download`** tab accepts a **text list of links**. Place each link on a separate line in the **`Models to download`** field, then click **`Download Models`**. Links to a model page and a version page are supported; you can specify a numeric ID instead of a link.

For a regular download, one link per line is sufficient. To add parameters to a specific link, put `::` after it, then list the parameters separated by `::`. Parameter names are case-insensitive.

- `AllFiles` — download all model files, including types that Helper usually does not process with separate checkboxes.
- `AllVersions` — download every published version of the model.
- `Subfolder=name` — save files to an existing subfolder inside the model type folder. This subfolder must already exist.

Example: `https://civitai.com/models/12345::AllFiles::AllVersions` requests all files of all versions of the model with ID `12345`. Parameters can be combined.

The batch mode has no duplicate behavior toggle: it uses `Skip`. One line can generate multiple downloads if `AllVersions` is specified. Links for which an ID or model information could not be obtained are skipped; errors of individual downloads are output in the final list.

## Find Information for Local Models

If a model is already downloaded manually, but Helper cannot link it to Civitai, first try **`Scan Models for Civitai`**. If the automatic search did not help, use the **`Get Model Info from Civitai by URL`** block: select the corresponding local model, paste the URL, and request information. This operation does not replace the model file and is not a weight download mode.

## Find Duplicates

In the **`Scan for duplicate models`** block, select the types and click **`Scan`**. By default, all types serviced by Helper are selected. Scanning compares the hashes of model files in local folders and shows matches; it does not delete them.

- **`Model Types`** — limits the check to selected types. Uncheck unnecessary ones to speed up the search.
- **`Use Hash from Metadata (May have false-positives but can be useful if you've pruned models)`** — disabled by default. In the disabled state, Helper calculates the hash from the file itself. If enabled, it first uses the SHA256 recorded in the metadata. This can be faster for metadata remaining after pruning, but may give a false positive; if the required hash is not in the metadata, Helper calculates it from the file.
- **`Scan`** — starts the analysis; it may take time.

## Check for New Versions

In the **`Check models' new version`** block, select the types and click **`Check New Version from Civitai`**. Helper goes through the selected local folders and compares the saved version with the actual version on Civitai. For reliable model matching, metadata files with its ID and version are needed. If a new version is found, a card with information and a **`Download Model's new version`** action will appear.

By default, `ti`, `hyper`, `ckp`, `lora`, `lycoris`, `vae`, `detection`, and `upscaler` are selected in **`Model Types`**.

The **`Download Model's new version`** button applies only to the found update card. It downloads the new version to the folder of the existing model and uses the saved binding to the model type; this is not a separate mass update mode for all models without prior checking.

## Where to Find Results

- **Model weights.** Saved to the folder corresponding to the model type: for example, checkpoints to the checkpoints directory, LoRA and LyCORIS to the LoRA directory, VAE to the VAE directory. Actual folders are taken from the Fooocus Extend configuration; the user-installed structure may differ from the standard one.
- **Subfolder.** For a single download, this is the root `/` or one of the offered existing subfolders. For `Batch Download`, the subfolder can be specified with the `Subfolder=...` parameter, but it must exist.
- **Metadata files.** After downloading and processing information, adjacent `.civitai.info` and JSON files usually appear. These files help the interface show information, trigger words, and match the version during future checks.
- **Preview.** Helper can save a local image next to the model with a `.preview.png` ending if such a preview does not yet exist and a suitable image is found.
- **Metadata overwrite.** In normal scanning, existing metadata recognized as current is not requested again. The `Replace Old Metadata Formats*` option enables selective updating of outdated formats. Do not enable it for manually modified metadata without a backup.

## What to Try First

For the first installation, it is safer to test Helper on a single model:

1. Open `Download Model` → `Single` and paste a public model link.
2. Select the desired version; leave `Sub-folder = /` if a separate existing subfolder is not needed.
3. Check the name, base model, and file list. Leave enabled only what is needed. For a regular download, do not enable `Download All Files`.
4. Leave `Duplicate File Behavior = Skip` so as not to overwrite an existing file with the same name.
5. Click `3. Download Model` and wait for the result. If Civitai responds that authorization is needed, enter the API key in `Settings` and try again.
6. Check the target model folder, metadata file, and preview. After that, use `Batch Download` for a list of URLs if you need to download multiple models.

## Common Problems

- **Cannot get information by link.** Ensure that a Civitai URL with a numeric model ID or the ID itself is pasted. For a version link, use a link containing the version ID. Check page and network accessibility.
- **Download returns an authorization error / 401.** An API key is required for some materials. Paste it into `Civitai API key` and click `Apply setting`. Do not send the key in correspondence or post it in public.
- **Cannot select a subfolder or `Model folder is not a dir` appears.** The specified subfolder must already exist inside the folder of the selected type. Select `/` or create the subfolder in the desired model directory in advance.
- **Helper reports that the version already exists.** This is a separate check by version ID in the metadata. Selecting `Overwrite` does not cancel it. If you really want to replace the local file, first check the existing metadata and files manually; do not delete them at random.
- **Scanning does not find the model.** Check the selected `Model Type`, file location, supported extension, and folder access. Scanning goes through nested folders but does not process arbitrary image files.
- **The version update button did not appear.** Check that the local model has Civitai metadata with a correct ID and that its types are selected. If there is no newer version on Civitai, Helper reports the absence of updates.

## Для чего нужен Civitai Helper
([Read in English](#what-civitai-helper-is-for))

`Civitai Helper` — это набор инструментов для работы с моделями для генераций. Он помогает скачать модель с Civitai, выбрать её версию, найти сведения для уже установленной модели, проверить локальные модели на дубликаты и узнать, появились ли у них новые версии. При скачивании или сканировании Helper также записывает рядом с моделью файлы с описанием и метаданными.

## Где открыть и что потребуется

1. В Fooocus Extend раскройте **`Extention` → `tools` → `Civitai_helper`**. [4]
2. Для обычных публичных страниц Civitai можно начать с URL модели. Некоторые файлы требуют авторизации для скачивания. Для таких случаев вверху вкладки есть поле **`Civitai API key`** и кнопка **`Apply setting`**.
3. Если скачивание требует ключ, создайте его в аккаунте Civitai и вставьте в это поле.
4. Нажмите **`Apply setting`** после вставки. Программа сохранит ключ в `civitai_api_key.txt` в рабочей папке Fooocus Extend, а также использует его в запросах на скачивание.

## Настройка API key

В верхнем блоке **`Settings`** есть единственное текстовое поле **`Civitai API key`** и кнопка **`Apply setting`**. Скопируйте ключ целиком, вставьте в поле и нажмите кнопку. Строка состояния ниже покажет, что настройка применена.

Код записывает ключ в текстовый файл `civitai_api_key.txt` и при запуске читает его обратно. Это не настройка отдельного режима: ключ применяется к скачиваниям Helper, которым требуется авторизация. Публичные сведения о моделях и некоторые другие операции могут работать без него.

## Сканирование установленных моделей

Блок **`Scan Models for Civitai`** ищет модели в локальных папках Fooocus Extend. Для каждого поддерживаемого файла расширение вычисляет SHA256-хеш и использует его, чтобы запросить сведения на Civitai. Сканирование проходит и по вложенным папкам. Поддерживаемые расширения файлов: `.bin`, `.pt`, `.pth`, `.safetensors` и `.ckpt`.

В блоке есть следующие настройки:

- **`Model Types`** — типы локальных моделей для проверки. По умолчанию выбраны все типы, которые Helper знает: `ti`, `ckp`, `lora`, `lycoris`, `vae`, `detection` и `upscaler`. 

- **`Replace Old Metadata Formats*`** — по умолчанию выключена. При включении Helper повторно скачивает информацию там, где существующие файлы метаданных отсутствуют, устарели по формату или не имеют версии формата. Опция относится к форматам метаданных, а не к обновлению весов модели. Код старается не затереть метаданные модели с несовпадающим ID и пропускает перезапись пустым описанием поверх непустого. Важное предупреждение: если вы вручную редактировали метаданные, сначала сделайте резервную копию. 
- **`Scan`** — запускает сканирование выбранных папок. Сканирование занимает время; интерфейс просит подождать и смотреть консоль для подробностей. В конце выводится число успешно обработанных моделей.

Сканирование не скачивает новые веса модели. Оно создаёт или обновляет соседние файлы сведений. По умолчанию Helper также формирует JSON с описанием. Если хеш не находится на Civitai, исходник пытается записать минимальные локальные сведения.

## Получение сведений о локальной модели по URL

Блок **`Get Model Info from Civitai by URL`** предназначен для ситуации, когда обычное сканирование не нашло сведения о локальной модели. Выберите локальный файл модели, укажите URL страницы Civitai и нажмите **`Get Model Info from Civitai`**. Эта операция добавляет сведения к уже существующей локальной модели; она не скачивает её веса.

Настройки и поля блока:

- **`Model Type`** — тип локальной модели. По умолчанию `ckp` (checkpoint). При смене типа обновляется список моделей. В интерфейсе доступны типы из настроенных папок Helper. 
- **`Only Show Models have no Info`** — по умолчанию выключен. Если включить, список ограничивается локальными моделями, у которых ещё нет заполненного ID в файле сведений Civitai. Это фильтр выпадающего списка, он не удаляет файлы и не запускает обработку сам по себе.
- **`Model`** — локальная модель выбранного типа. Если список пустой, проверьте, что файл подходящего типа находится в папке этого типа. Имена ищутся также во вложенных папках.
- **`Civitai URL`** — URL страницы модели.
- **`Get Model Info from Civitai`** — получает информацию для указанной страницы и выбранной локальной модели, записывает доступные метаданные и при необходимости загружает превью.

Для модели без существующего файла метаданных переключатель **`Only Show Models have no Info`** тоже оставляет её в списке. Если выбрать модель, у которой уже есть корректный ID, включённый фильтр её не показывает.

## Скачать одну модель: вкладка `Single`

В блоке **`Download Model`** выберите вкладку **`Single`**. Это основной режим, когда нужно скачать одну модель с выбранной версией.

1. В **`Civitai URL`** вставьте ссылку на страницу модели и нажмите **`Get Model Info by Civitai Url`**. Для страницы версии Civitai можно использовать ссылку с ID версии. После загрузки сведений появятся имя, тип, базовая модель, версии, папка и доступные файлы.
2. Выберите **`Model Version`**. После получения данных по умолчанию выбирается первая версия, которую вернул Civitai. Для ссылки на конкретную версию будет выбрана соответствующая версия, если URL содержит её ID.
3. Проверьте **`Sub-folder`**. По умолчанию выбрана корневая папка `/`. В списке также показаны существующие подпапки внутри папки данного типа.
4. Выберите, как поступать с существующим файлом, затем проверьте список типов файлов, которые будут загружаться.
5. При желании измените имя в **`Rename Model`**. Затем нажмите **`3. Download Model`** и дождитесь сообщения о завершении или ошибке.

Настройки в этой форме:

- **`Model Name`**, **`Model Type`** и **`Base Model`** — справочные поля, заполняемые с Civitai после получения информации. Их нельзя редактировать в форме. Они помогают проверить, что выбрана нужная модель и её вариант.
- **`Model Version`** — версия для скачивания. Переключение версии обновляет список файлов, исходное имя, базовую модель и превью. Оставьте версию по умолчанию, если нужна самая первая версия в полученном списке; иначе выберите нужную вручную.
- **`Sub-folder`** — место сохранения внутри папки выбранного типа. `/` — корень. Остальные варианты берутся из существующих подпапок.
- **`Duplicate File Behavior`** — поведение при совпадении имени конечного файла. По умолчанию `Skip`: не скачивать поверх существующего файла. `Overwrite` заменяет файл с совпадающим именем. `Rename New` добавляет к новому имени суффикс `_2`, `_3` и так далее.
- Строки с файлами **`Model`**, **`Training Data`**, **`Config`** и **`VAE`** появляются в зависимости от содержимого выбранной версии. Галочки по умолчанию включены; основная строка `Model` обязательна и её нельзя выключить. Снимайте отметку с дополнительных файлов, которые не нужны. VAE код направляет в папку VAE, даже если остальные файлы версии сохраняются в папке типа модели.
- **`Download All Files`** по умолчанию выключен и появляется, когда в версии больше одного файла. При включении скачиваются все файлы этой версии, включая типы, которые интерфейс не умеет классифицировать. Поэтому включайте его только когда действительно нужны все вложения, а не только веса модели. Отображение отдельных галочек типов файлов при этом скрывается.
- **`Unhandled Files (Files that we don't know what to do with but will still downloaded with "Download All Files")`** — справочная строка с файлами, для которых Helper не имеет отдельного типа. Она не активирует скачивание сама; такие файлы включаются через `Download All Files`.
- **`Preview Image`** — галерея изображений-примеров с Civitai для выбранной версии. Доступные изображения могут фильтроваться настройкой `Black Out NSFW` в основном интерфейсе Fooocus.
- **`Rename Model`** — базовое имя файла без расширения; после выбора версии оно заполняется автоматически из имени файла на Civitai. Изменение имени влияет на скачиваемые файлы. Расширение показывается внутренним полем `Model extension`, которое в интерфейсе скрыто.
- **`3. Download Model`** — стартует скачивание, пишет прогресс и итоговый путь в блоке статуса и в консоли.

Нюанс настройки дубликатов: `Skip`, `Overwrite` и `Rename New` управляют совпадением пути и имени файла. Но перед этим код отдельно ищет в выбранной папке метаданные той же версии Civitai. Если эта версия уже числится в папке, операция прекращается сообщением «This model version already exists»; смена поведения совпадающих имён не заставит повторно скачать ту же версию.

## Скачать несколько моделей: вкладка `Batch Download`

Вкладка **`Batch Download`** принимает **текстовый список ссылок**. Поместите каждую ссылку на отдельную строку в поле **`Models to download`**, затем нажмите **`Download Models`**. Поддерживаются ссылки на страницу модели и страницу версии; можно указывать числовой ID вместо ссылки.

Для обычной загрузки достаточно одной ссылки на строку. Чтобы добавить параметры к конкретной ссылке, поставьте после неё `::`, затем перечислите параметры через `::`. Имена параметров нечувствительны к регистру.

- `AllFiles` — скачать все файлы модели, включая типы, которые Helper обычно не обрабатывает отдельными галочками.
- `AllVersions` — скачать каждую опубликованную версию модели.
- `Subfolder=имя` — сохранить файлы в существующую подпапку внутри папки типа модели. Эта подпапка должна уже существовать.

Пример: `https://civitai.com/models/12345::AllFiles::AllVersions` запрашивает все файлы всех версий модели с ID `12345`. Параметры можно комбинировать.

У пакетного режима нет переключателя поведения дубликатов: он использует `Skip`. Одна строка может породить несколько скачиваний, если задан `AllVersions`. Ссылки, для которых не удалось получить ID или сведения о модели, пропускаются; ошибки отдельных загрузок выводятся в итоговом списке.

## Найти сведения для локальных моделей

Если модель уже скачана вручную, но Helper не может её связать с Civitai, сначала попробуйте **`Scan Models for Civitai`**. Если автоматический поиск не помог, используйте блок **`Get Model Info from Civitai by URL`**: выберите соответствующую локальную модель, вставьте URL и запросите сведения. Эта операция не заменяет файл модели и не является режимом загрузки весов.

## Найти дубликаты

В блоке **`Scan for duplicate models`** выберите типы и нажмите **`Scan`**. По умолчанию выбраны все типы, которые Helper обслуживает. Сканирование сравнивает хеши файлов моделей в локальных папках и показывает совпадения; оно не удаляет их.

- **`Model Types`** — ограничивает проверку выбранными типами. Снимите лишние отметки, чтобы ускорить поиск.
- **`Use Hash from Metadata (May have false-positives but can be useful if you've pruned models)`** — по умолчанию выключен. В выключенном состоянии Helper вычисляет хеш из самого файла. Если включить, он сначала использует SHA256, записанный в метаданных. Это может быть быстрее для оставшихся после чистки метаданных, но может дать ложное совпадение; если нужного хеша в метаданных нет, Helper вычисляет его по файлу.
- **`Scan`** — запускает анализ; он может занять время.

## Проверить новые версии

В блоке **`Check models' new version`** выберите типы и нажмите **`Check New Version from Civitai`**. Helper обходит выбранные локальные папки и сравнивает сохранённую версию с актуальной версией на Civitai. Для надёжного сопоставления модели нужны файлы метаданных с её ID и версией. Если новая версия найдена, появится карточка с информацией и действием **`Download Model's new version`**.

По умолчанию в **`Model Types`** выбраны `ti`, `hyper`, `ckp`, `lora`, `lycoris`, `vae`, `detection` и `upscaler`.

Кнопка **`Download Model's new version`** относится только к найденной карточке обновления. Она скачивает новую версию в папку существующей модели и использует сохранённую привязку к типу модели; это не отдельный режим массового обновления всех моделей без предварительной проверки.

## Где искать результаты

- **Вес модели.** Сохраняется в папку, соответствующую типу модели: например, checkpoints — в каталог checkpoints, LoRA и LyCORIS — в каталог LoRA, VAE — в каталог VAE. Фактические папки берутся из конфигурации Fooocus Extend; установленная пользователем структура может отличаться от стандартной.
- **Подпапка.** Для одиночной загрузки это корень `/` или одна из предложенных существующих подпапок. Для `Batch Download` подпапку можно указать параметром `Subfolder=...`, но она должна существовать.
- **Файлы метаданных.** После скачивания и обработки сведений обычно появляются соседние файлы `.civitai.info` и JSON. Эти файлы помогают интерфейсу показывать сведения, триггер-слова и сопоставлять версию при будущих проверках.
- **Превью.** Helper может сохранять локальное изображение рядом с моделью с окончанием `.preview.png`, если такого превью ещё нет и найдено подходящее изображение.
- **Перезапись метаданных.** В обычном сканировании уже существующие метаданные, распознанные как актуальные, повторно не запрашиваются. Опция `Replace Old Metadata Formats*` включает выборочное обновление устаревших форматов. Не включайте её для вручную изменённых метаданных без резервной копии.

## Что попробовать сначала

Для первой установки безопаснее проверить Helper на одной модели:

1. Откройте `Download Model` → `Single` и вставьте публичную ссылку модели.
2. Выберите нужную версию; оставьте `Sub-folder = /`, если не нужна отдельная существующая подпапка.
3. Проверьте имя, базовую модель и список файлов. Оставьте включённым только то, что нужно. Для обычного скачивания не включайте `Download All Files`.
4. Оставьте `Duplicate File Behavior = Skip`, чтобы не перезаписать существующий файл с тем же именем.
5. Нажмите `3. Download Model` и дождитесь результата. Если Civitai ответит, что нужна авторизация, внесите API key в `Settings` и повторите попытку.
6. Проверьте целевую папку моделей, файл метаданных и превью. После этого используйте `Batch Download` для списка URL, если требуется скачать несколько моделей.

## Типовые проблемы

- **Не удаётся получить сведения по ссылке.** Убедитесь, что вставлен URL Civitai с числовым ID модели или сам ID. Для ссылки на версию используйте ссылку, содержащую ID версии. Проверьте доступность страницы и сети.
- **Скачивание возвращает ошибку авторизации / 401.** Для некоторых материалов нужен API key. Вставьте его в `Civitai API key` и нажмите `Apply setting`. Не отправляйте ключ в переписку и не размещайте его в общем доступе.
- **Не удаётся выбрать подпапку или появилось `Model folder is not a dir`.** Указанная подпапка должна уже существовать внутри папки выбранного типа. Выберите `/` или заранее создайте подпапку в нужном каталоге моделей.
- **Helper сообщает, что версия уже существует.** Это отдельная проверка по ID версии в метаданных. Выбор `Overwrite` не отменяет её. Если действительно хотите заменить локальный файл, сначала проверьте существующие метаданные и файлы вручную; не удаляйте их наугад.
- **Сканирование не находит модель.** Проверьте выбранный `Model Type`, расположение файла, поддерживаемое расширение и наличие доступа к папке. Сканирование проходит по вложенным папкам, но не обрабатывает произвольные файлы изображений.
- **Не появилась кнопка обновления версии.** Проверьте, что локальная модель имеет метаданные Civitai с корректным ID и что выбраны её типы. Если на Civitai нет более новой версии, Helper сообщает об отсутствии обновлений.
