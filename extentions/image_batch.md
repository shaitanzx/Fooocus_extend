## What Image Batch is For
([Читать на русском языке](#для-чего-нужен-image-batch))

`Image Batch` sequentially processes a set of images. For each input image, it either uses it as a visual prompt to generate a new image (**Image Prompt**) or applies upscaling/variation to it (**Upscale**). All elements are processed sequentially, not simultaneously. The general text prompt, negative prompt, model, number of output images, format, and other main settings are taken from the standard Fooocus interface.

The module accepts either individual images or a single ZIP archive containing images.

## Before the First Launch

1. Open the **Image Batch** tab.
2. In the standard Fooocus interface, set the model and text **Prompt**. If necessary, fill in the **Negative Prompt**, choose a style, quality, and other generation parameters.
3. For the first test, use a single file. This will quickly show if the chosen method is suitable.
4. If other images are left in the standard **Input Image** block and it is enabled, during **Image Prompt** they might be used together with the current processing element in the module. Clear them if you want to test each file separately.
5. If the chosen mode requires additional models, Fooocus may download them on first use. This is especially likely for ImagePrompt and certain ControlNet modes.

In Image Batch, there is no separate text prompt for each file: the same **Prompt / Negative Prompt** pair is applied to all images in the set. If **Image Number** in the main interface is greater than 1, this number of outputs will be requested for each input image. With a random seed enabled, the handler assigns a new seed for the next input file; with a fixed seed, it uses the same value.

## Uploading Images

### Single File or Multiple Files

Leave the **Upload ZIP-file** checkbox unchecked. In the **Drag (Select) 1 or more reference images** field, drag and drop or select one or more image files.

If exactly one file is uploaded, the interface shows it and collapses the file list. This is just a preview: the handler still takes the original file from the upload list. With multiple files, the list remains visible, and the individual preview is hidden.

Files are processed by name in lexicographical order, not by the order of selection. For example, the names `1.png`, `10.png`, `2.png` might be processed in exactly that order.

### ZIP Archive

1. Enable **Upload ZIP-file**.
2. Select a single `.zip` archive in the **Upload a ZIP file** field.
3. Ensure that only images are inside and they are located directly in the root of the archive.

The extension unpacks the ZIP, but then searches for files only in the top level of the temporary folder. Images inside nested directories will not enter the queue. Do not add text files, thumbnails, or other non-graphic files to the root – this may cause an error. Also, you cannot use nested folders, spaces, or non-Latin characters in filenames. For reliability, adhere to these limitations. Upload only trusted ZIP archives.

Before each run, temporary directories are cleared, after which the selected images or ZIP are unpacked anew. This prevents mixing files from the previous set.

## Settings and When They Work

### **Upload ZIP-file**

Switches the input type. Default is unchecked: shows the individual image uploader. When checked, it hides the image list and shows a single ZIP upload.

### **Scale method**

Default is **NOT scale**. This parameter acts *only* when **Action = Image Prompt**; it is not used in **Upscale**.

- **NOT scale** — does not change the selected input canvas size to match the output. The image is still used as a prompt, and the preparation handler may resize it to the dimensions required by the chosen method.
- **to ORIGINAL** — before generation, sets the canvas width and height to the dimensions of the current source image.
- **to OUTPUT** — fits the source image to the output image dimensions while preserving aspect ratio; the frame is not cropped.

### **Action**

Default is **Image Prompt**. Switches between two processing paths and changes the visibility of related settings.

- **Image Prompt** shows the Image Prompt method, as well as **Stop At** and **Weight** parameters for each method. The source acts as a visual prompt for generation, while the text prompt helps change the style or content of the result.
- **Upscale** shows a separate upscaling/variation method. In this case, **Scale method** is not applied.

### **Method** (when **Action = Image Prompt**)

Default is **ImagePrompt**. List of methods: **ImagePrompt**, **PyraCanny**, **CPDS**, **FaceSwap**, **OpenPose**, **Recolor**, **Scribble**, **Manga**. The choice changes how the source features are extracted and used.

- **ImagePrompt** — general visual reference: the model tries to consider the overall content and impression of the source image.
- **FaceSwap** — uses a separate IP-Adapter branch for the face and passes the automatically found/cropped face area to it. This is not a literal pixel-to-pixel face insertion; the model still generates a new image.
- **PyraCanny** — focuses on the boundaries and contours of objects. Useful when preserving the layout and shape is more important than the colors of the source.
- **CPDS** — a structural prompt that helps maintain the shape and depth structure of the scene; this is not just converting the source to black and white.
- **OpenPose** — intended for pose control. As a pose reference, you must use a PNG file obtained using the OpenPoseEditor extension.
- **Recolor**, **Scribble**, and **Manga** — separate ControlNet variants for working with color, sketches, and manga structure.

### **Stop At**

Range **0–1**, step **0.001**. Initial value depends on the method. Visible and used only with **Image Prompt**. In the code, it is passed as the endpoint for applying ControlNet/IP-Adapter: a smaller value stops the prompt's influence earlier, a larger value keeps it longer during generation. This is not a quality slider; the correct value depends on how strongly the source should dictate the result.

### **Weight**

Range **0–2**, step **0.001**. Initial value depends on the method. Works only with **Image Prompt** and sets the weight of the visual condition: at a higher value, the source prompt usually influences the result more strongly. If the image starts to be copied excessively or interferes with the text prompt, try lowering the weight.

When changing the Image Prompt method, **Stop At** and **Weight** automatically revert to the default values for the new method. Default values in the code are: **ImagePrompt** — 0.5 / 0.6; **FaceSwap** — 0.9 / 0.75; **PyraCanny**, **CPDS**, and **OpenPose** — 0.5 / 1.0; **Recolor**, **Scribble**, and **Manga** — 1.0 / 1.0. Keep these values initially and change them only after testing the result.

### **Method** (when **Action = Upscale**)

This is a different **Method** field, not the Image Prompt list. Default is **Disabled**. List: **Disabled**, **Vary (Subtle)**, **Vary (Strong)**, **Upscale (1.5x)**, **Upscale (2x)**, **Final Upscale**. When switching to **Upscale**, do not forget to select a real mode: the default value **Disabled** does not trigger standard upscaling/variation of the source.

- **Vary (Subtle)** — regenerates the source with a moderate strength of change.
- **Vary (Strong)** — allows for more noticeable changes. In the code, these two variants have different levels of change; the main prompt influences the resulting picture.
- **Upscale (1.5x)** and **Upscale (2x)** — first upscale the image, then apply diffusion refinement. The scale refers to the upscaling operation; final dimensions may depend on model limits and settings.
- **Final Upscale** — final upscaling without the standard text prompt generation stage; suitable when you primarily need an upscaled version rather than changing the content.
- **Disabled** — upscaling/variation mode is turned off. Do not select this if the goal is to process the source image via Upscale.

### **Start batch**, **Output --> ZIP**, **Clear Output**

- **Start batch** prepares the input files and launches them sequentially. A notification displays the element number and file name. The **Stop** button halts the queue after the current generation call; already saved results remain.
- **Output --> ZIP** creates an archive with all files currently in the configured **Outputs path** folder, including results from previous runs and files from other Fooocus modes. The archive is not limited to just the current batch. It appears in **Download a ZIP file**; click the file to download it.
- **Clear Output** deletes the contents of the entire **Outputs path** directory, including nested folders and results from previous dates, without additional confirmation. This is not just clearing the current gallery. First, download or copy everything you need; press this button only if you truly want to clear the entire results directory.

## Workflow

### Generate New Images from Single Files or a ZIP

1. Set the general **Prompt** and, if necessary, **Negative Prompt** in the main interface.
2. In Image Batch, leave **Action = Image Prompt**.
3. For the first run, leave **Method = ImagePrompt** and the default **Stop At / Weight** values. If preserving contours is important, try **PyraCanny**; if preserving scene geometry, try **CPDS**.
4. Choose a **Scale method**. If unsure, start with **NOT scale**.
5. Upload one or more files via the image list, or enable **Upload ZIP-file** and specify the archive.
6. Click **Start batch**. Watch the notifications for the current file name and number.
7. Check the result in the main gallery. Files are saved to the standard Fooocus **Outputs path** directory.

Use the same prompt phrasing for different images if you need a unified style. To find the right influence strength, start with one image and change only one parameter at a time.

### Upscale Existing Images

1. Upload an image or a set.
2. Select **Action = Upscale**.
3. In the appearing **Method** field, select **Upscale (1.5x)** or **Upscale (2x)**. For final upscaling without standard text refinement, select **Final Upscale**.
4. Click **Start batch** and check the results. Test on a single file before running a large batch.

### Download Results

1. After the desired runs, click **Output --> ZIP**.
2. Check the **Download a ZIP file** field and download the created archive.
3. Remember that the archive will contain all files in the **Outputs path**, not just the latest Image Batch.
4. Download the ZIP before the next **Start batch** click: the next run clears the temporary folder where the generated ZIP is located. The generated results in the **Outputs path** are not deleted by this action.

## Where to Find Results

- Generated images are written to the standard Fooocus **Outputs path**, not next to the source files.
- Their format is determined by the main **Output Format** parameter: PNG, JPEG, and WEBP are available. The name is built using a template with an optional **Filename Prefix**, date, time, a random number, and the chosen format's extension; files are usually located in a date-named subfolder.
- The gallery shows the results of the current run; previous files remain on disk until deleted by the user or another operation.
- The ZIP from **Output --> ZIP** preserves the relative directory structure of the results folder and includes all files found there. Therefore, if old results have accumulated, the archive may be significantly larger than expected.

## Frequent Problems

- **Pressing Start batch ends in an error before generation.** Check that at least one file is selected in the active uploader. If **Upload ZIP-file** is enabled, select a ZIP first; if disabled, upload an image.
- **Image read error when running a ZIP.** Remove non-images from the archive and ensure the pictures are in the root, not in a nested folder. Simplify filenames: use Latin characters and avoid spaces.
- **In Upscale mode, the result did not upscale as expected.** Check that a specific mode is selected in the appearing **Method** field instead of **Disabled**.
- **The downloaded ZIP contains old results or files from other tabs.** This is expected: the entire current **Outputs path** is archived. Save the necessary files elsewhere, then configure a separate results directory if needed. Do not press **Clear Output** until you are sure old results are saved.
- **Unable to download the ZIP after the next run.** The created archive was in the temporary folder, which is cleared on the next **Start batch**. Download the ZIP immediately after it is created.

## Для чего нужен Image Batch
([Read in English](#what-image-batch-is-for))

`Image Batch` последовательно обрабатывает набор изображений. Для каждого входного изображения она либо использует его как визуальную подсказку для генерации нового изображения (**Image Prompt**), либо применяет к нему увеличение/вариацию (**Upscale**). Все элементы обрабатываются последовательно, а не одновременно. Общий текстовый промпт, негативный промпт, модель, число выходных изображений, формат и другие основные настройки берутся из обычного интерфейса Fooocus.

Модуль принимает либо отдельные изображения, либо один ZIP-архив с изображениями.

## Перед первым запуском

1. Откройте вкладку **Image Batch**.
2. В обычном интерфейсе Fooocus задайте модель и текстовый **Prompt**. При необходимости заполните **Negative Prompt**, выберите стиль, качество и остальные параметры генерации.
3. Для первого теста используйте один файл. Это быстрее покажет, подходит ли выбранный метод.
4. Если в обычном блоке **Input Image** оставлены другие изображения и он включён, при **Image Prompt** они могут использоваться вместе с текущим элементом обработки в модуле. Очистите их, если хотите проверять каждый файл отдельно.
5. Если выбранный режим требует дополнительные модели, Fooocus может скачать их при первом использовании. Это особенно вероятно для ImagePrompt и отдельных ControlNet-режимов.

В Image Batch нет отдельного текстового промпта для каждого файла: одна и та же пара **Prompt / Negative Prompt** применяется ко всем изображениям набора. Если в основном интерфейсе **Image Number** больше 1, это число выходов будет запрошено для каждого входного изображения. При включённом случайном seed обработчик назначает новый seed для следующего входного файла; при фиксированном seed использует одно и то же значение.

## Загрузка изображений

### Один файл или несколько файлов

Оставьте флажок **Upload ZIP-file** выключенным. В поле **Drag (Select) 1 or more reference images** перетащите или выберите один или несколько файлов изображений.

Если загружен ровно один файл, интерфейс показывает его и сворачивает список файлов. Это только предпросмотр: обработчик всё равно берёт исходный файл из списка загрузок. При нескольких файлах список остаётся видимым, отдельный предпросмотр скрывается.

Файлы обрабатываются по имени в лексикографическом порядке, а не по порядку выбора. Например, имена `1.png`, `10.png`, `2.png` могут идти именно в таком порядке.

### ZIP-архив

1. Включите **Upload ZIP-file**.
2. Выберите один архив `.zip` в поле **Upload a ZIP file**.
3. Убедитесь, что внутри лежат только изображения и они находятся прямо в корне архива.

Расширение распаковывает ZIP, но затем ищет файлы только в верхнем уровне временной папки. Изображения внутри вложенных каталогов в очередь не попадут. Не добавляйте в корень текстовые файлы, миниатюры или другие неграфические файлы - это может привести к ошибке. Также нельзя использовать вложенные папки, пробелы и нелатинские символы в именах файлов. Для надёжности соблюдайте эти ограничения. Загружайте только доверенные ZIP-архивы.

Перед каждым запуском очищаются временные каталоги, после чего выбранные изображения или ZIP распаковываются заново. Это предотвращает подмешивание файлов предыдущего набора.

## Настройки и когда они работают

### **Upload ZIP-file**

Переключает тип входа. По умолчанию выключен: показывается загрузчик отдельных изображений. При включении он скрывает список изображений и показывает одиночную загрузку ZIP.

### **Scale method**

По умолчанию выбрано **NOT scale**. Этот параметр действует только при **Action = Image Prompt**; в **Upscale** он не используется.

- **NOT scale** — не меняет выбранный размер входного холста по размеру выходного. Изображение всё равно используется как подсказка, и обработчик подготовки может привести его к размеру, требуемому выбранным методом.
- **to ORIGINAL** — перед генерацией выставляет ширину и высоту холста по размерам текущего исходного изображения.
- **to OUTPUT** — подгоняет исходное изображение к размерам выходного изображения, сохраняя пропорции; кадр не обрезается.

### **Action**

По умолчанию — **Image Prompt**. Переключает между двумя путями обработки и меняет видимость связанных настроек.

- **Image Prompt** показывает метод Image Prompt, а также параметры **Stop At** и **Weight** для каждого метода. Исходник выступает визуальной подсказкой для генерации, а текстовый промпт помогает изменить стиль или содержание результата.
- **Upscale** показывает отдельный метод увеличения/вариации. При этом **Scale method** не применяется.

### **Method** при **Action = Image Prompt**

По умолчанию — **ImagePrompt**. Список методов: **ImagePrompt**, **PyraCanny**, **CPDS**, **FaceSwap**, **OpenPose**, **Recolor**, **Scribble**, **Manga**. Выбор меняет способ извлечения и использования признаков исходника.

- **ImagePrompt** — общий визуальный ориентир: модель старается учитывать общее содержание и впечатление исходного изображения.
- **FaceSwap** — использует отдельную ветку IP-Adapter для лица и передаёт в неё автоматически найденную/обрезанную область лица. Это не буквальная вставка лица пиксель в пиксель; модель всё равно генерирует новое изображение.
- **PyraCanny** — ориентируется на границы и контуры объектов. Полезно, когда важнее сохранить расположение и форму, чем цвета исходника.
- **CPDS** — структурная подсказка, помогающая удержать форму и глубинную структуру сцены; это не просто перевод исходника в чёрно-белый цвет.
- **OpenPose** — предназначен для контроля позы. В качестве рференса озы необходимо использовать PNG-файл, полученный с помощью расширения OpenPoseEditor
- **Recolor**, **Scribble** и **Manga** — отдельные варианты ControlNet для работы с цветом, наброском и манга-структурой.

### **Stop At**

Диапазон **0–1**, шаг **0.001**. Начальное значение зависит от метода. Параметр виден и используется только при **Image Prompt**. В коде он передаётся как конечная точка применения ControlNet/IP-Adapter: меньшая величина прекращает влияние подсказки раньше, большая — сохраняет его дольше в ходе генерации. Это не ползунок качества; корректное значение зависит от того, насколько сильно исходник должен задавать результат.

### **Weight**

Диапазон **0–2**, шаг **0.001**. Начальное значение зависит от метода. Работает только при **Image Prompt** и задаёт вес визуального условия: при большем значении исходная подсказка обычно сильнее влияет на результат. Если изображение начинает чрезмерно копироваться или мешает промпту, попробуйте снизить вес.

При смене Image Prompt-метода **Stop At** и **Weight** автоматически возвращаются к значениям по умолчанию для нового метода. Исходные значения в коде такие: **ImagePrompt** — 0.5 / 0.6; **FaceSwap** — 0.9 / 0.75; **PyraCanny**, **CPDS** и **OpenPose** — 0.5 / 1.0; **Recolor**, **Scribble** и **Manga** — 1.0 / 1.0. Сначала оставьте эти значения и меняйте их только после пробного результата.

### **Method** при **Action = Upscale**

Это другое поле **Method**, не список Image Prompt. По умолчанию — **Disabled**. Список: **Disabled**, **Vary (Subtle)**, **Vary (Strong)**, **Upscale (1.5x)**, **Upscale (2x)**, **Final Upscale**. При переключении на **Upscale** не забудьте выбрать реальный режим: значение по умолчанию **Disabled** не запускает обычное увеличение/вариацию исходника.

- **Vary (Subtle)** — перегенерирует исходник с умеренной силой изменения.
- **Vary (Strong)** — допускает более заметные изменения. В коде для этих двух вариантов заданы разные уровни изменения; основной промпт влияет на получившуюся картинку.
- **Upscale (1.5x)** и **Upscale (2x)** — сначала увеличивают изображение, затем применяют диффузионное уточнение. Масштаб относится к операции увеличения; итоговые размеры могут зависеть от лимитов и настроек модели.
- **Final Upscale** — финальное увеличение без обычного этапа генерации по текстовому промпту; подходит, когда нужно в первую очередь получить увеличенную версию, а не изменить содержание.
- **Disabled** — режим увеличения/вариации выключен. Не выбирайте его, если цель — обработать исходное изображение через Upscale.

### **Start batch**, **Output --> ZIP**, **Clear Output**

**Start batch** подготавливает входные файлы и запускает их последовательно. В уведомлении отображается номер элемента и имя файла. Кнопка **Stop** останавливает очередь после текущего вызова генерации; уже сохранённые результаты остаются.

**Output --> ZIP** собирает архив со всеми файлами, которые сейчас находятся в настроенной папке **Outputs path**, включая результаты предыдущих запусков и файлы других режимов Fooocus. Архив не ограничен только текущим батчем. Он появляется в **Download a ZIP file**; нажмите на файл, чтобы скачать его.

**Clear Output** без дополнительного подтверждения удаляет содержимое всего каталога **Outputs path**, включая вложенные папки и результаты за предыдущие даты. Это не очистка только текущей галереи. Сначала скачайте или скопируйте всё нужное; нажимайте эту кнопку только если действительно хотите очистить весь каталог результатов.

## Порядок работы

### Сгенерировать новые изображения по одиночным файлам или ZIP

1. Задайте общий **Prompt** и, при необходимости, **Negative Prompt** в основном интерфейсе.
2. В Image Batch оставьте **Action = Image Prompt**.
3. Для первого прогона оставьте **Method = ImagePrompt** и значения **Stop At / Weight** по умолчанию. Если важно сохранить контуры — попробуйте **PyraCanny**; если геометрию сцены — **CPDS**.
4. Выберите **Scale method**. Если вы не уверены, начните с **NOT scale**.
5. Загрузите один или несколько файлов через список изображений либо включите **Upload ZIP-file** и укажите архив.
6. Нажмите **Start batch**. Смотрите на уведомления о текущем имени и номере файла.
7. Проверьте результат в основной галерее. Файлы сохраняются в обычный каталог Fooocus **Outputs path**.

Для разных изображений используйте одну формулировку промпта, если вам нужен единый стиль. Чтобы подобрать силу влияния, начните с одного изображения и меняйте только один параметр за раз.

### Увеличить уже готовые изображения

1. Загрузите изображение или набор.
2. Выберите **Action = Upscale**.
3. В появившемся поле **Method** выберите **Upscale (1.5x)** или **Upscale (2x)**. Для финального увеличения без обычного текстового уточнения выберите **Final Upscale**.
4. Нажмите **Start batch** и проверьте результаты. Перед большим набором сделайте пробу на одном файле.

### Скачать результаты

1. После нужных запусков нажмите **Output --> ZIP**.
2. Проверьте поле **Download a ZIP file** и скачайте созданный архив.
3. Помните, что в архив попадут все файлы в **Outputs path**, а не только последний Image Batch.
4. Скачайте архив до следующего нажатия **Start batch**: следующий запуск очищает временную папку, где лежит сформированный ZIP. Сами сгенерированные результаты в **Outputs path** этим действием не удаляются.

## Где искать результаты

- Сгенерированные изображения записываются в обычный **Outputs path** Fooocus, а не рядом с исходниками.
- Их формат определяется основным параметром **Output Format**: доступны PNG, JPEG и WEBP. Имя строится по шаблону с необязательным **Filename Prefix**, датой, временем, случайным числом и расширением выбранного формата; файлы обычно располагаются в подпапке с датой.
- Галерея показывает результаты текущего запуска; предыдущие файлы остаются на диске, пока их не удалит пользователь или другая операция.
- ZIP от **Output --> ZIP** сохраняет относительную структуру каталога результатов и включает все найденные там файлы. Поэтому при накопившихся старых результатах архив может быть значительно больше ожидаемого.

## Частые проблемы

- **Нажатие Start batch заканчивается ошибкой до генерации.** Проверьте, что выбран хотя бы один файл в активном загрузчике. Если **Upload ZIP-file** включён, сначала выберите ZIP; если выключен — загрузите изображение.
- **Ошибка чтения изображения при запуске ZIP.** Удалите из архива не-изображения и проверьте, что сами картинки лежат в корне, а не во вложенной папке. Упростите имена файлов: используйте латиницу и не ставьте пробелы.
- **В режиме Upscale результат не увеличился ожидаемым образом.** Проверьте, что в появившемся поле **Method** вместо **Disabled** выбран конкретный режим.
- **В скачанном ZIP оказались старые результаты или файлы других вкладок.** Это ожидаемо: архивируется весь текущий **Outputs path**. Сохраните нужные файлы в другом месте, затем при необходимости настройте отдельный каталог результатов. Не нажимайте **Clear Output**, пока не убедились, что старые результаты сохранены.
- **Не удаётся скачать ZIP после следующего запуска.** Созданный архив находился во временной папке, очищаемой при следующем **Start batch**. Скачивайте ZIP сразу после его создания.
