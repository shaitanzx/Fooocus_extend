## What Caption Does
([Читать на русском языке](#что-делает-caption))

`Caption` analyzes images and can create a tag list, write a text description, or do both. The interface offers three **Caption method** options:

- **WD** — generate tags with the selected tagger.
- **LLM** — generate a description with the selected language model or run one of the Florence-2 tasks.
- **WD+LLM** — generate tags first, then write a description. By default, the LLM can use the WD tags as additional context.

The extension has a **Single mode** tab for processing one image and a **Batch mode** tab for processing a set of images. Batch mode accepts individual images or a ZIP archive.

## Before Your First Run

First, choose a processing mode and the models you need, then click **Load Models**. If a selected model is not already on your computer, the extension downloads its files to the `models/caption` folder inside the Fooocus_extend launch directory. The first download may take some time; wait for the download-complete message.

After the models are loaded, the run buttons in the Single and Batch tabs become available. To change the method or a model, click **Unload Models** first, select the new settings, and click **Load Models** again. Unloading frees memory but does not delete the downloaded model files.

By default, the tagger runs on the CPU and the LLM runs on an available GPU. If there is not enough video memory to run a model on the GPU, enable CPU inference for the corresponding tagger or LLM. CPU processing is usually slower. When GPU use is selected, the code frees VRAM by unloading previously loaded Fooocus models; you may need to load them again before the next generation.

## Choosing Models

### Tagger: WD or PixAI

The **Tagger models** field combines standard WD models and two PixAI profiles. There is no separate backend switch: simply select the desired entry from the unified list. WD profiles differ by model and tag vocabulary. The PixAI profiles use the same model files but expose different sets of settings: `PixAI-Tagger-v1.0-Simple` or `PixAI-Tagger-v1.0-Advanced`.

PixAI is best suited to anime illustrations. The model produces tags in the **General**, **Character**, **Style**, **Copyright**, **Meta**, and **Rating** categories. These are model predictions, not guaranteed-correct labels; review the results, especially when exact characters, attributes, or ratings matter.

In the **Simple** profile, one threshold applies to the General and Style categories, while Character has its own threshold. You can also enable a trailing comma. In the **Advanced** profile, thresholds are set separately for General, Style, Copyright, Meta, and Rating, and a separate Character threshold remains available. Both profiles let you replace underscores with spaces and exclude specified tags. Although the model divides tags into categories, the extension outputs one combined tag list rather than a separate field for each category.

A **threshold** is the minimum confidence score required for a tag to be kept in the result. A higher threshold usually produces fewer tags; a lower threshold produces more, but may also include more incorrect ones. Start with the default values and adjust thresholds gradually.

### LLM: Joy, Qwen, or Florence

When a method that uses an LLM is selected in `Caption method`, **Joy**, **Qwen**, and **Florence** are available in **Choice LLM**. After choosing a backend, select a specific model from its corresponding list. The current configurations include two Joy Caption Alpha Two variants, Qwen2-VL 2B/7B, and four Florence-2 variants: base, large, base-ft, and large-ft. The `ft` suffix identifies a fine-tuned Florence variant in the model list.

For ordinary descriptions with Joy or Qwen, use the **system prompt** and **user prompt** fields. They define the model's role and what it should write. If you are unsure what to enter, leave the suggested values in place and test them on one image first.

Joy has a **Joy Formated Prompts** section. It lets you choose the type and length of the output and add extra instructions. Click **Generate prompts** to fill in the system/user prompts, then review and edit the text if needed. All available options are described below in the LLM Settings section.

For compatible Joy models, **Use LLM LoRA to avoid censored** may also be available. This switch selects an additional profile/weights when they are provided for the selected model. It does not change the prompt by itself and does not guarantee a particular response. If you are unsure whether to use it, leave it off.

### LLM dtype

`LLM dtype` sets the numerical precision of the model weights: **fp16**, **bf16**, or **fp32**. The default is `fp16`. For GPU inference, start with `fp16`; `bf16` is suitable if your graphics card supports it; `fp32` uses more memory and is mainly useful for compatibility or testing. This is a model-loading setting, not a way to make the description semantically more accurate. If **Use cpu for LLM inference** is enabled, the current code forces `fp32`, even if another value is selected in the list.

### LLM Quantization

`LLM Quantization` offers **none**, **4bit**, and **8bit**. The default is `none`, which loads the weights without additional quantization. `4bit` and `8bit` may reduce video-memory usage, but they change the precision of the weight representation; the effect on output and speed depends on the backend and hardware. For your first test, leave it at `none`; if VRAM is insufficient, try `8bit`, then `4bit`.

Not all models support this option. In the current code, quantization is automatically disabled for Florence-2. It is also disabled for Joy `Joy-Caption-Alpha-Two-Llava`. In these cases, the loader changes the selected value to `none`; check the log messages.

`LLM dtype` and `LLM Quantization` are applied when the weights are loaded. After changing either setting, click **Unload Models**, then click **Load Models** again; changing the value in an already-open interface is not enough.

## Tagger Settings

### WD Settings

This panel is displayed for a WD profile. By default, **Replace underscores with spaces** is enabled, and the thresholds are `Threshold = 0.35`, `General threshold = 0.35`, and `Character threshold = 0.85`. The other checkboxes are off and the text lists are empty. A threshold is the minimum confidence score for a tag to be kept. Increasing it usually reduces the number of tags; decreasing it adds more candidates. For WD models with General and Character categories, the main controls are `General threshold` and `Character threshold`. The general `Threshold` is a fallback for models that do not provide these categories separately.

The other WD Settings work as follows:

- **Replace underscores with spaces** replaces `_` with a space in tag names. Turn it off if you need the original tag format.
- **Adds rating tags to the first** puts the detected rating tag at the beginning of the list.
- **Adds rating tags to the last** puts it at the end. If both checkboxes are enabled, the code uses the “at the beginning” option.
- **Always put character tags before the general tags** moves character tags to the beginning of the combined list.
- **Expand tag tail parenthesis to another tag for character tags** splits the parenthesized tail of a character tag, such as `character (series)`, into the character name and a separate series tag. This applies only to models that provide WD character categories.
- **undesired tags to remove** excludes the listed tags. Enter them separated by commas, for example `blurry, text`.
- **Tags always put at the beginning** moves already-detected tags to the beginning of the list. This setting does not create a tag if the model did not produce it.
- **Tag replacement** replaces detected tag names. Format: `old,new;old2,new2`, for example `1girl,woman;solo,one person`. Use the exact source tag name after underscore replacement has been applied.

Rating- and character-related settings are useful only for models that provide the corresponding output categories. Other models may not have those categories, so the setting will not change their results.

### Common Tagger Settings

`Separator for tags` sets the delimiter between tags. The default is a comma followed by a space. `Extension for tag captions files` sets the suffix for WD tag files wherever this setting is used—for example, for the separate WD result in WD+LLM mode.

### PixAI Tagger Settings

List unwanted tags, separated by commas, in **Tags to exclude**. If **Replace underscores with spaces** is enabled, enter exclusions using the tag spelling after that replacement. If the task requires fewer irrelevant tags, raise the thresholds slightly; if the model misses important details, lower them slightly. The initial PixAI values in the configuration are intended as a starting point, not as universal settings for every image.

## LLM Settings

These options become available when a method that includes an LLM is selected. Florence uses the separate task fields described below; the standard system/user prompt fields are for Joy and Qwen.

**extension of LLM caption file** sets the extension for a separate LLM output file. The default is `.llmcaption`. For example, it is used in WD+LLM mode when saving the tags and description to separate files; Batch saving rules also depend on `Caption file extension`.

**llm will read wd caption for inference** asks the LLM to read an existing WD tag file during Batch processing. In WD+LLM `queue` mode, this allows the second pass to read the file created during the first pass. In LLM-only mode, the code looks for `.wdcaption` in the results folder, and the GUI clears this temporary folder before processing. Therefore, the checkbox will not automatically pick up a neighboring `.wdcaption` file from the folder containing the input images. If no file is found, the standard prompt without WD tags is used. Single mode also does not read a sidecar file: tags are available to the LLM there when the tagger runs as part of WD+LLM. If `{wd_tags}` remains in an LLM-only Single prompt, it will not be replaced—remove the placeholder or switch to WD+LLM.

**llm will not read wd caption for inference** primarily affects WD+LLM in `queue` mode: when checked, the second pass does not load the WD file. This switch does not rewrite a custom `user prompt` and does not remove `{wd_tags}` by itself. Check the prompt: if the placeholder remains, remove it when you do not need the tags; leave it in place if the tags are needed in Single mode or `sync` mode.

### Joy Formated Prompts

In **Caption Type**, you can select **Descriptive**, **Descriptive (Informal)**, **Training Prompt**, **MidJourney**, **Booru tag list**, **Booru-like tag list**, **Art Critic**, **Product Listing**, or **Social Media Post**. The default is **Descriptive**. The selection changes how the request is phrased, not which model is used.

**Caption Length** sets the desired length: `any`, `very short`, `short`, `medium-length`, `long`, `very long`, or a numeric word limit from 20 to 260. The default is `long`. A number is included in the prompt as a preference, not as a strict guarantee.

**Extra Options** adds additional instructions to the user prompt:

- **People and characters:** do not mention unchangeable traits; optionally refer to a character by the name entered in **Person/Character Name**.
- **Image details:** include lighting, camera angle, watermark, JPEG artifacts, composition, depth of field, and the likely natural or artificial light source.
- **Content and length:** keep the description PG, specify SFW/suggestive/NSFW, omit text or resolution, focus on the main elements, or avoid ambiguity.
- **Evaluative or speculative details:** assess subjective image quality or guess the camera and settings such as aperture, shutter speed, and ISO. The latter often leads to guesses—leave it off if you need verifiable facts.

Do not select conflicting instructions at the same time, such as “do not mention text” and “transcribe the text exactly.” Click **Generate prompts**, review the generated system/user prompts, and edit them if needed.

### System prompt, User prompt, and Advanced Options

The **system prompt** defines the model's general role and constraints. The **user prompt** specifies the task and output format. You can write your own text in both fields. In WD+LLM mode, keep `{wd_tags}` in the user prompt if you want to pass the LLM the tags; remove this placeholder if you do not need them.

The **Advanced Options** accordion contains three available settings. By default, temperature and max tokens are set to `0`, and image size is `1024`.

- **temperature for LLM model** controls variability; the slider defaults to `0`. In this implementation, `0` means “use the backend's default behavior,” not necessarily zero randomness: the current Joy code substitutes `0.6`, while Qwen uses its own generation default. For Joy/Qwen, start at `0.2–0.4` if you want more consistent text; increase the value for more variety. Florence runs without sampling, so this slider does not affect its result.
- **max token for LLM model** limits the length of newly generated text; the default is `0`, meaning a backend-specific value is used. For example, the Florence code sets 1024 tokens, while Joy uses 300. If the response is cut off, increase the limit; if it is too long, reduce the limit and specify the desired number of sentences in the prompt. A value that is too low can also truncate structured detection or segmentation output.
- **Resize image for inference** sets the image size for backends that use the shared resize path. In the current code, the slider applies to Qwen, for example. Joy forcibly resizes the image to 384 pixels, while Florence uses its own preprocessing, so this slider does not change their actual input size. For Qwen, a higher value may preserve small text and details but increases the workload; a lower value is usually faster at the cost of small details.

**Auto Unload Models after inference** is present in the code but hidden in the current interface, so it is not a standard user setting.

## How to Improve Prompts for LLMs

First decide what kind of result you need: a coherent description, a generation prompt, tags, an answer to one question, or an exact transcription of text. Then specify the format and length limit. For a reliable description, ask the model to rely on visible details, mention the position of objects, and omit anything it cannot determine confidently. Do not ask for both “as much detail as possible” and “only the essentials.” A direct question also works well with Qwen; official examples use short requests such as `Describe this image.` and questions about one visible attribute.

The examples below are for the **system prompt** and **user prompt** fields in Joy/Qwen. They are in English, as are the default prompts in the interface.

**1. Objective description without guessing**

System prompt:

```
You are a careful image captioner. Describe only details that are visibly supported by the image. Do not guess identities, relationships, locations, intentions, or exact camera settings. If a detail is unclear, omit it.
```

User prompt:

```
Write a factual caption in 1–2 sentences. Start with the main subject and action. Then mention distinctive visible clothing or objects, their spatial relationships, and the most relevant background detail. Include text only if it is legible. Do not begin with “This image shows”.
```

**2. Image-generation prompt**

```
Create one concise Stable Diffusion prompt as comma-separated phrases. Prioritize the main subject, visible appearance, clothing, pose or action, setting, composition, lighting, color palette, and visual medium. Include only details supported by the image. Do not add a negative prompt, weights, generic quality slogans, or an explanation.
```

In Joy, this corresponds to **Training Prompt**. If you need short tags with underscores, select **Booru tag list** instead of Training Prompt and explicitly specify the order and format of the tags. The upstream JoyCaption README provides separate instructions for descriptions, image-generation prompts, and Danbooru lists; the exact options in Fooocus_extend are determined by its own Joy prompt builder.

**3. Description using WD/PixAI suggestions in WD+LLM mode**

Leave `{wd_tags}` in the user prompt if the LLM should receive the tags. For example:

```
Candidate tags: {wd_tags}
Write one factual caption in 1–2 sentences. Treat the tags as suggestions, not as facts: keep only tags supported by the image, ignore incorrect ones, and do not repeat the raw tag list. Describe the main subject, action, and important spatial relations. Do not mention that tags were provided.
```

If you remove `{wd_tags}`, the tags will not be inserted into this request. Do not leave the placeholder in a Florence prompt: Florence receives the selected task and a separate text input, not the standard Joy/Qwen template.

**4. Answer to a specific visual question**

```
What color is the coat of the person on the left? Answer with one color, or say “unclear” if it cannot be determined. Use only visible evidence. Do not describe unrelated parts of the image.
```

This wording is useful when you need one attribute rather than a general caption. Instead of asking “Is the clothing red?”, ask for the color of a specific person's coat and identify the person's position in the frame. Qwen supports questions about visible attributes in images, but ambiguous or small details should still be checked.

**5. Exact text transcription**

```
Transcribe all legible text exactly. Preserve the reading order and line breaks. Do not summarize, translate, or correct spelling. Mark unreadable fragments as [unclear]. Do not invent missing characters.
```

For text recognition with Florence, it is usually simplest to select **OCR** or **OCR with Region** and leave `Florence user prompt` empty. Use the latter if you also need the text regions marked on the image.

After changing a prompt, first run Single inference on one image. If the result is too long, specify a limit or a number of sentences. If the model invents context, add an instruction not to draw conclusions that are unsupported by the image. If it misses a detail you need, specify the type of detail and its position in the frame instead of adding many generic words such as “as detailed as possible.”

## Florence-2: Tasks and Text Input

When you select **Florence** instead of the standard system/user prompt fields, the interface shows the **Florence2 system prompt** list and the **Florence user prompt** field. Select a task from the list; the text in the second field is added to its request. Leave that field empty for tasks that do not need additional text.

The following tasks are convenient for a first test:

- **Caption**, **Detailed Caption**, **More Detailed Caption** — generate a short, detailed, or more detailed description; additional text is usually not required.
- **Object Detection** — detect objects and display their names and regions on the image.
- **Dense Region Caption** — describe multiple regions of the image.
- **Region Proposal** — propose candidate regions without generating a standard text description.
- **OCR** — recognize text in the image.
- **OCR with Region** — recognize text and show the regions where it was found.
- **Open Vocabulary Detection** — find an object by its text label, for example `a green car`.
- **Caption to Phrase Grounding** — identify regions that correspond to phrases in the supplied description. For example: `A green car parked in front of a yellow building.`
- **Referring Expression Segmentation** — segment an object specified by a phrase, for example `a green car`.
- **Region to Segmentation**, **Region to Category**, **Region to Description** — advanced tasks for which Florence examples use a region notation with coordinate tokens such as `<loc_...>`. The Caption interface has no separate tool for selecting a region, so these are not the best choices for a first test.

For tasks involving regions, check **Florence Visualization**: supported tasks display boxes, polygons, or OCR regions there. In the current implementation, visual annotations are generated for Object Detection, Dense Region Caption, Region Proposal, Caption to Phrase Grounding, Referring Expression Segmentation, Region to Segmentation, Open Vocabulary Detection, and OCR with Region. Standard Caption and OCR return text without an overlay.

For detection or segmentation tasks, the text in `LLM Caption Output` may look like structured data rather than a polished, coherent description. That is expected: the list of regions and labels is needed for the task, while the overlay is easier to inspect in the visual preview.

The official Florence example for **Caption to Phrase Grounding** uses the request `A green car parked in front of a yellow building.`; for **Referring Expression Segmentation** and **Open Vocabulary Detection**, it uses `a green car`. For **Region to Segmentation**, the sample notebook supplies a region in the format `<loc_702><loc_575><loc_866><loc_772>`. Do not enter coordinates at random; this example is only meant to illustrate the format.

## Processing One Image — Single Mode

1. In `Caption method`, select `WD`, `LLM`, or `WD+LLM`.
2. Select a tagger and/or an LLM. If you are using Florence, also select a Florence model.
3. Configure CPU/GPU use and, if desired, tagger settings or prompts.
4. Click **Load Models** and wait for the loading-complete message.
5. Open the **Single mode** tab, upload an image to **Upload Image**, and click **Inference**.
6. Review the result in **WD Tags Output**, **LLM Caption Output**, and, if the Florence task supports visualization, **Florence Visualization**.

In Single mode, the text is displayed in the interface.

## Processing a Set of Images — Batch Mode

Batch mode accepts multiple uploaded images or one ZIP archive. To upload an archive, enable **Upload ZIP-file**. Supported image extensions are `.bmp`, `.jpg`, `.jpeg`, `.png`, and `.webp`. You cannot select a folder directly.

For a ZIP archive, place the images in the archive root. If you upload multiple files from different locations, use unique filenames.

Recommended workflow:

1. Select the method and models, then click **Load Models**.
2. Open **Batch mode** and add images or a ZIP archive.
3. Check `Caption file extension`. The default is `.txt`.
4. For WD+LLM, leave **Save WD and LLM captions in one file** enabled if you want both separate outputs and a combined file.
5. If using WD+LLM, start with `Run method = sync`. In this mode, the tagger and LLM process each image in sequence. `queue` processes the entire set with the tagger first, then passes the results to the LLM stage.
6. Click **Batch Process**. During processing, **Process preview** updates and notifications show the image number and filename: `Caption Batch: start element generation X/Y. Filename: filename.ext`.
7. When processing is complete, download the result using **Download a ZIP file**. A new run clears all temporary processing folders, so download the archive from the previous run first.

## What Files Are Created

In Batch mode, output files use the same base names as the images and are then packaged into a ZIP. For example, for an input named `portrait.png`, the following outputs are possible:

- **WD-only**: by default, `portrait.txt` is created with the tags because this branch uses the `Caption file extension` field. If you want the `.wdcaption` suffix, set it in `Caption file extension` before processing.
- **LLM-only**: by default, `portrait.txt` is created with the description.
- **WD+LLM**: when combined saving is enabled, separate `portrait.wdcaption` and `portrait.llmcaption` files are created, as well as a combined `portrait.txt`. In the combined file, the tags and description are separated by the value of `Seperator between WD tags and LLM captions`, which defaults to `|`.
- **Florence in LLM or WD+LLM**: if the task supports visualization, `portrait_visualization.png` is saved next to the text result. No PNG is created for Florence tasks without visual annotations.

Note that **Extension for tag captions files** in Common Tagger Settings and **Caption file extension** in Batch are not the same setting. In WD-only mode, the current generator uses the Batch field `Caption file extension`; in WD+LLM, the separate WD file gets its suffix from Common Tagger Settings.

## Passing WD Tags to the LLM

If you want the LLM to describe an image using the WD tags just generated for it, select **WD+LLM**. In this mode, the extension first generates the tags and can insert them into the LLM request through the `{wd_tags}` placeholder. If needed, edit the `user prompt` and keep this placeholder wherever you want the tags to appear.

## Troubleshooting

- **`Models not loaded!` appears** — click **Load Models** and wait for loading to finish.
- **The first run appears to do nothing for a long time** — the selected model may be downloading. Check your internet connection, available disk space, and the Fooocus console messages.
- **There are too many or too few tags** — adjust the threshold slightly. Raising it usually reduces the number of tags; lowering it usually increases the number.
- **PixAI performs worse on a photo than on an anime illustration** — the model card says PixAI Tagger is primarily designed for anime images. Review the output or try a WD model.
- **Florence does not show an annotated image** — check the selected task. Caption, Detailed Caption, More Detailed Caption, and OCR return text without this type of visualization. For an overlay, try Object Detection, Referring Expression Segmentation, or OCR with Region.
- **Batch did not process all images in a ZIP** — check that the images are in the archive root and have supported extensions.
- **The archive does not contain the expected suffix** — see the section above: WD-only uses `Caption file extension`, while WD+LLM creates separate WD/LLM files and a combined file.


## Что делает Caption
([Read in English](#what-caption-does))

`Caption` анализирует изображения и может либо создать список тегов, либо написать текстовое описание, либо выполнить оба действия. В интерфейсе есть три варианта `Caption method`:

- **WD** — создать теги с помощью выбранного tagger.
- **LLM** — получить описание с помощью выбранной языковой модели или выполнить одну из задач Florence-2.
- **WD+LLM** — сначала получить теги, затем сделать описание. По умолчанию LLM может использовать теги WD как дополнительную информацию.

В расширении есть вкладки **Single mode** для одного изображения и **Batch mode** для обработки набора изображений. Для Batch доступны отдельные изображения или ZIP-архив.

## Перед первым запуском

Сначала выберите режим обработки и нужные модели, затем нажмите **Load Models**. Если выбранной модели ещё нет на компьютере, расширение скачает её файлы в папку `models/caption` внутри каталога запуска Fooocus_extend. Первое скачивание может занять время; дождитесь сообщения о завершении загрузки.

После загрузки модели кнопки запуска во вкладках Single и Batch станут активными. Чтобы сменить метод или модель, сначала нажмите **Unload Models**, затем выберите новые настройки и снова нажмите **Load Models**. Выгрузка освобождает память, но не удаляет скачанные файлы модели.

По умолчанию tagger запускается на CPU, а LLM — на доступном GPU. Если для запуска модели на GPU не хватает видеопамяти, включите CPU для соответствующего tagger или LLM. Обработка на CPU обычно медленнее. При выборе GPU код освобождает VRAM, выгружая ранее загруженные модели Fooocus; это может потребовать их повторной загрузки перед следующей генерацией.

## Выбор моделей

### Tagger: WD или PixAI

Поле **Tagger models** объединяет обычные WD-модели и два профиля PixAI. Отдельного переключателя между ними нет: достаточно выбрать нужную запись в общем списке. WD-профили различаются моделью и теговым словарём. PixAI-профили используют одни и те же файлы модели, но показывают разные наборы настроек: `PixAI-Tagger-v1.0-Simple` или `PixAI-Tagger-v1.0-Advanced`.

PixAI лучше всего подходит для аниме-иллюстраций. Модель выдаёт категории тегов **General**, **Character**, **Style**, **Copyright**, **Meta** и **Rating**. Это подсказки модели, а не гарантированно верные метки: результат стоит просматривать, особенно если важны точные персонажи, атрибуты или рейтинг.

В профиле **Simple** общий порог применяется к категориям General и Style, а для Character есть отдельный порог (threshold). Здесь же можно включить запятую в конце списка тегов. В профиле **Advanced** порог задаётся отдельно для General, Style, Copyright, Meta и Rating, а отдельный порог Character остаётся доступен. В обоих профилях можно заменить подчёркивания пробелами и исключить перечисленные теги. Несмотря на деление по категориям, расширение выводит общий список тегов, а не отдельное поле для каждой категории.

**Порог** (threshold) — минимальная уверенность, при которой тег оставляют в результате. Более высокий порог обычно даёт меньше тегов; более низкий — больше, но среди них может быть больше ошибочных. Начните со значений по умолчанию и меняйте пороги постепенно.

### LLM: Joy, Qwen или Florence

Если в `Caption method` выбран режим с LLM, в поле **Choice LLM** доступны **Joy**, **Qwen** и **Florence**. После выбора backend выберите конкретную модель в соответствующем списке. В текущих конфигурациях доступны два варианта Joy Caption Alpha Two, Qwen2-VL 2B/7B и четыре варианта Florence-2: base, large, base-ft и large-ft. Суффикс `ft` обозначает fine-tuned-вариант из списка моделей Florence.

Для обычного описания Joy или Qwen используйте поля **system prompt** и **user prompt**. Они задают роль модели и то, что она должна написать. Если не уверены, оставьте предложенные значения и сначала проверьте их на одном изображении.

У Joy есть блок **Joy Formated Prompts**. В нём можно выбрать вид результата, его длину и дополнительные пожелания. Нажмите **Generate prompts**, чтобы заполнить system/user prompt; затем проверьте текст и при необходимости поправьте его вручную. Полный список вариантов описан ниже в разделе LLM Settings.

Для совместимой модели Joy также может показываться **Use LLM LoRA to avoid censored**. Этот переключатель выбирает дополнительный профиль/веса, если они предусмотрены для выбранной модели. Он не меняет prompt сам по себе и не гарантирует определённый ответ. Если не знаете, нужен ли этот вариант, оставьте его выключенным.

### LLM dtype

`LLM dtype` задаёт числовую точность весов: **fp16**, **bf16** или **fp32**; по умолчанию выбран `fp16`. Для GPU обычно начинают с `fp16`; `bf16` подходит, если видеокарта его поддерживает; `fp32` использует больше памяти и нужен главным образом для совместимости или проверки. Это настройка загрузки модели, а не способ сделать описание точнее по смыслу. Если включён **Use cpu for LLM inference**, текущий код принудительно использует `fp32`, даже если в списке выбрано другое значение.

### LLM Quantization

`LLM Quantization` предлагает **none**, **4bit** и **8bit**; по умолчанию выбрано `none`. Оно загружает веса без дополнительной квантизации. `4bit` и `8bit` могут уменьшить расход видеопамяти, но меняют точность представления весов; результат и скорость зависят от backend и оборудования. Для первого теста оставьте `none`; при нехватке VRAM попробуйте `8bit`, затем `4bit`.

Не все модели поддерживают этот выбор. В текущем коде для Florence-2 квантизация автоматически отключается. Для Joy `Joy-Caption-Alpha-Two-Llava` она тоже отключается. В этих случаях загрузчик заменит выбранное значение на `none`; проверьте сообщения в логе.

`LLM dtype` и `LLM Quantization` применяются во время загрузки весов. После изменения этих параметров нажмите **Unload Models**, затем снова **Load Models**; простого изменения значения в уже открытом интерфейсе недостаточно.

## Настройки tagger

### WD Settings

Эта панель отображается для WD-профиля. По умолчанию **Replace underscores with spaces** включён, а пороги равны `Threshold = 0.35`, `General threshold = 0.35` и `Character threshold = 0.85`; остальные флажки выключены, а текстовые списки пусты. Порог — это нижняя граница уверенности: тег сохраняется, когда его оценка не ниже выбранного значения. Увеличение порога обычно уменьшает число тегов; уменьшение добавляет больше кандидатов. Для WD-моделей с категориями General и Character главные регуляторы — `General threshold` и `Character threshold`. Общий `Threshold` служит запасным порогом для моделей, которые не отдают эти категории отдельно.

Остальные элементы WD Settings работают так:

- **Replace underscores with spaces** заменяет `_` на пробел в названиях тегов. Отключите, если нужен исходный формат тегов.
- **Adds rating tags to the first** добавляет найденный rating-тег в начало списка.
- **Adds rating tags to the last** помещает его в конец.
  Если включены оба флажка, код выбирает вариант «в начало».
- **Always put character tags before the general tags** переносит character-теги в начало общего списка.
- **Expand tag tail parenthesis to another tag for character tags** разделяет хвост character-тега вида `персонаж (серия)` на имя персонажа и отдельный тег серии. Это применимо только к моделям, которые отдают WD-категории персонажей.
- **undesired tags to remove** исключает перечисленные теги. Вводите их через запятую, например `blurry, text`.
- **Tags always put at the beginning** поднимает уже найденные теги в начало списка. Этот параметр не создаёт тег, если модель его не выдала.
- **Tag replacement** заменяет найденные имена тегов. Формат: `старый,новый;старый2,новый2`, например `1girl,woman;solo,one person`. Используйте точное исходное имя тега после применения замены подчёркиваний.

Rating- и character-настройки полезны только для моделей, которые имеют соответствующие выходные категории. У других моделей категория может отсутствовать, поэтому переключатель не изменит результат.

### Common Tagger Settings

`Separator for tags` задаёт разделитель между тегами. По умолчанию это запятая и пробел. 
`Extension for tag captions files` задаёт расширение файлов WD-тегов там, где этот путь используется — например, для отдельного WD-результата в режиме WD+LLM.

### PixAI Tagger Settings

У **Tags to exclude** перечислите ненужные теги через запятую. Если включён `Replace underscores with spaces`, сравнивайте исключения с написанием тегов после этой замены. Если задача требует меньше лишних тегов, немного увеличьте пороги; если модель пропускает нужные детали, немного уменьшите их. Начальные значения PixAI в конфигурации рассчитаны как рекомендуемая отправная точка, а не как универсальная настройка для любого изображения.

## LLM Settings

Эти элементы становятся доступны, когда выбран метод, включающий LLM. Florence использует отдельные task-поля, описанные ниже; обычные поля system/user prompt предназначены для Joy и Qwen.

**extension of LLM caption file** задаёт расширение отдельного LLM-файла. По умолчанию это `.llmcaption`. Он используется, например, при WD+LLM, когда включено сохранение тегов и описания в отдельных файлах; правила batch-сохранения также зависят от `Caption file extension`.

**llm will read wd caption for inference** просит LLM прочитать уже созданный файл тегов WD в Batch-процессе. В режиме WD+LLM с `queue` это позволяет второму проходу прочитать файл, созданный первым проходом. В режиме LLM-only код ищет `.wdcaption` в папке результатов, а GUI очищает эту временную папку перед запуском; поэтому галочка не подхватит автоматически соседний `.wdcaption` из папки с исходными изображениями. Если файл не найден, используется стандартный prompt без WD-тегов. Single mode также не читает sidecar: там теги доступны LLM, когда tagger запускается в WD+LLM. Если в LLM-only Single остался `{wd_tags}`, он не будет подставлен — удалите placeholder либо переключитесь на WD+LLM.

**llm will not read wd caption for inference** в первую очередь влияет на WD+LLM с `queue`: при включённой галочке второй проход не загружает WD-файл. Переключатель не переписывает произвольный `user prompt` и сам не удаляет `{wd_tags}`. Поэтому проверьте prompt: если placeholder остался, удалите его, когда теги не нужны; если теги нужны в Single mode или sync, оставьте его.

### Joy Formated Prompts

В поле **Caption Type** можно выбрать **Descriptive**, **Descriptive (Informal)**, **Training Prompt**, **MidJourney**, **Booru tag list**, **Booru-like tag list**, **Art Critic**, **Product Listing** или **Social Media Post**. По умолчанию выбрано **Descriptive**. Выбор меняет формулировку запроса, а не модель.

**Caption Length** задаёт желаемую длину: `any`, `very short`, `short`, `medium-length`, `long`, `very long` или числовое ограничение слов от 20 до 260; по умолчанию выбрано `long`. Число подставляется в prompt как пожелание, а не как строгая гарантия. 

**Extra Options** добавляет к user prompt дополнительные инструкции:

- **О людях и персонажах:** не указывать неизменяемые признаки; при необходимости назвать персонажа именем из **Person/Character Name**.
- **О деталях изображения:** включить освещение, угол камеры, водяной знак, JPEG-артефакты, композицию, глубину резкости и вероятный естественный/искусственный источник света.
- **О содержании и объёме:** оставить описание в рамках PG, обозначить SFW/suggestive/NSFW, не упоминать текст или разрешение, ограничиться главными элементами, избегать двусмысленности.
- **Оценочные и предположительные сведения:** оценить субъективное качество изображения либо угадать камеру и параметры вроде диафрагмы, выдержки и ISO. Последний пункт часто приводит к догадкам — не включайте его, если нужны проверяемые факты.

Не выбирайте одновременно противоречивые пожелания, например «не упоминать текст» и «точно переписать надпись». Нажмите **Generate prompts**, просмотрите созданные system/user prompts и при необходимости отредактируйте их. [2]

### System prompt, User prompt и Advanced Options

**system prompt** задаёт общую роль и ограничения модели. **user prompt** определяет конкретную задачу и формат ответа. В обоих полях можно написать свой текст. В режиме WD+LLM сохраните `{wd_tags}` в user prompt, если хотите передать LLM теги; удалите этот placeholder, если они не нужны.

В раскрывающемся блоке **Advanced Options** находятся три доступных параметра. По умолчанию temperature и max tokens установлены в `0`, а размер изображения — `1024`.

- **temperature for LLM model** регулирует вариативность; слайдер по умолчанию установлен в `0`. Значение `0` в этой реализации означает «использовать поведение backend по умолчанию», а не гарантированно нулевую случайность: текущий Joy-код подставляет `0.6`, а Qwen использует собственный generation default. Для Joy/Qwen начните с `0.2–0.4`, если нужен более стабильный текст; увеличивайте значение для большего разнообразия. Florence работает без sampling, поэтому этот slider на её результат не влияет.
- **max token for LLM model** ограничивает длину нового текста; по умолчанию — `0`, то есть backend-specific значение. Например, код Florence задаёт 1024 токена, а Joy — 300 токенов. Если ответ обрывается, увеличьте лимит; если он излишне длинный, уменьшите лимит и укажите желаемое число предложений в prompt. Слишком маленькое значение может обрезать также структурированный результат обнаружения или сегментации.
- **Resize image for inference** задаёт размер для backend, который использует общий resize-путь. В текущем коде slider применяется, например, к Qwen. Joy принудительно уменьшает изображение до 384 пикселей, а Florence использует собственный preprocessing, поэтому этот slider не меняет их фактический входной размер. Для Qwen более высокое значение может сохранить мелкий текст и детали, но увеличивает нагрузку; более низкое обычно ускоряет обработку ценой потери мелких деталей.

## Как улучшать промты для LLM

Сначала решите, какой результат нужен: связное описание, prompt для генерации, теги, ответ на один вопрос или точная расшифровка текста. Затем задайте формат и предел длины. Для надёжного описания попросите модель опираться на видимые детали, назвать положение объектов и пропускать то, чего нельзя уверенно установить. Не просите одновременно «очень подробно» и «только главное». Для Qwen подходит и прямой вопрос; официальные примеры используют короткие запросы вроде `Describe this image.` и вопросы об одном видимом признаке.

Ниже примеры для полей **system prompt** и **user prompt** Joy/Qwen. Формулировки оставлены на английском, как штатные prompts интерфейса.

**1. Объективное описание без догадок**

System prompt:

```
You are a careful image captioner. Describe only details that are visibly supported by the image. Do not guess identities, relationships, locations, intentions, or exact camera settings. If a detail is unclear, omit it.
```

User prompt:

```
Write a factual caption in 1–2 sentences. Start with the main subject and action. Then mention distinctive visible clothing or objects, their spatial relationships, and the most relevant background detail. Include text only if it is legible. Do not begin with “This image shows”.
```

**2. Prompt для генерации изображения**

```
Create one concise Stable Diffusion prompt as comma-separated phrases. Prioritize the main subject, visible appearance, clothing, pose or action, setting, composition, lighting, color palette, and visual medium. Include only details supported by the image. Do not add a negative prompt, weights, generic quality slogans, or an explanation.
```

В Joy это соответствует типу **Training Prompt**. Если нужны короткие теги с подчёркиваниями, выберите **Booru tag list** вместо Training Prompt и явно задайте порядок и формат тегов. Upstream README JoyCaption приводит отдельные инструкции для описаний, prompt-ов генерации и Danbooru-списков; точный набор вариантов в задаётся собственным Joy prompt builder.

**3. Описание с подсказками WD/PixAI в режиме WD+LLM**

Оставьте `{wd_tags}` в user prompt, если LLM должна получить теги. Например:

```
Candidate tags: {wd_tags}
Write one factual caption in 1–2 sentences. Treat the tags as suggestions, not as facts: keep only tags supported by the image, ignore incorrect ones, and do not repeat the raw tag list. Describe the main subject, action, and important spatial relations. Do not mention that tags were provided.
```

Если удалить `{wd_tags}`, теги не будут подставлены в этот запрос. Не оставляйте placeholder в Florence prompt: Florence получает выбранную задачу и отдельный текстовый ввод, а не обычный шаблон Joy/Qwen.

**4. Ответ на конкретный визуальный вопрос**

```
What color is the coat of the person on the left? Answer with one color, or say “unclear” if it cannot be determined. Use only visible evidence. Do not describe unrelated parts of the image.
```

Такая формулировка полезна, когда нужен один атрибут, а не общий caption. Вместо «красная ли одежда?» точнее спросить о цвете пальто конкретного человека и указать его положение в кадре. Qwen поддерживает запросы о видимых признаках на изображениях, но неоднозначные или мелкие детали всё равно нужно проверять.

**5. Точная расшифровка текста**

```text
Transcribe all legible text exactly. Preserve the reading order and line breaks. Do not summarize, translate, or correct spelling. Mark unreadable fragments as [unclear]. Do not invent missing characters.
```

Для распознавания текста во Florence обычно проще выбрать задачу **OCR** или **OCR with Region** и оставить `Florence user prompt` пустым. Используйте второй вариант, если нужны также области текста на изображении.

После изменения prompt сначала выполните Single inference на одном изображении. Если результат слишком длинный, задайте конкретный лимит или число предложений. Если модель додумывает контекст, добавьте запрет на выводы, которые не подтверждаются изображением. Если пропускает нужную деталь, укажите её тип и положение в кадре, а не добавляйте много общих слов вроде «максимально подробно».

## Florence-2: задачи и текстовый запрос

При выборе **Florence** вместо обычных полей system/user prompt появляются список **Florence2 system prompt** и поле **Florence user prompt**. В списке выбирается задача, а текст из второго поля добавляется к её запросу. Для задач без дополнительного текста оставьте поле пустым.

Наиболее удобные задачи для первого знакомства:

- **Caption**, **Detailed Caption**, **More Detailed Caption** — короткое, подробное или более подробное описание; дополнительный текст обычно не требуется.
- **Object Detection** — обнаружить объекты и вывести их названия с областями на изображении.
- **Dense Region Caption** — описать несколько областей изображения.
- **Region Proposal** — предложить области-кандидаты без обычного текстового описания.
- **OCR** — распознать текст на изображении.
- **OCR with Region** — распознать текст и показать области, где он найден.
- **Open Vocabulary Detection** — найти объект по текстовому названию, например `a green car`.
- **Caption to Phrase Grounding** — указать области, соответствующие фразам из введённого описания. Например: `A green car parked in front of a yellow building.`
- **Referring Expression Segmentation** — выделить маской объект, заданный фразой, например `a green car`.
- **Region to Segmentation**, **Region to Category**, **Region to Description** — продвинутые задачи, где в примерах Florence используется запись региона с токенами координат вида `<loc_...>`. В интерфейсе Caption нет отдельного инструмента для выбора области, поэтому это не лучший выбор для первого теста.

Если выбрана задача с областями, обратите внимание на **Florence Visualization**: для поддерживаемых задач там будут прямоугольники, полигоны или OCR-области. В текущей реализации визуальная разметка строится для Object Detection, Dense Region Caption, Region Proposal, Caption to Phrase Grounding, Referring Expression Segmentation, Region to Segmentation, Open Vocabulary Detection и OCR with Region. Обычный Caption и OCR возвращают текст без наложенной разметки.

Для задач обнаружения или сегментации текст в `LLM Caption Output` может выглядеть как структурированный результат, а не как красивое связное описание. Это нормально: перечень областей и меток нужен для соответствующей задачи, а наложение на картинку удобнее проверять в визуальном preview.

Официальный пример Florence для **Caption to Phrase Grounding** использует запрос `A green car parked in front of a yellow building.`; для **Referring Expression Segmentation** и **Open Vocabulary Detection** — `a green car`. Для задачи **Region to Segmentation** в sample notebook передаётся регион в формате `<loc_702><loc_575><loc_866><loc_772>`. Не вводите эти координаты наугад: такой пример нужен только как иллюстрация формата.

## Обработка одного изображения — Single mode

1. В `Caption method` выберите `WD`, `LLM` или `WD+LLM`.
2. Выберите tagger и/или LLM. Если используется Florence, дополнительно выберите модель Florence.
3. Настройте CPU/GPU и при желании параметры tagger или prompt.
4. Нажмите **Load Models** и дождитесь сообщения о завершении загрузки.
5. Во вкладке **Single mode**, загрузите изображение в **Upload Image** и нажмите **Inference**.
6. Посмотрите результат в **WD Tags Output**, **LLM Caption Output** и, если задача Florence поддерживает визуализацию, **Florence Visualization**.

В Single mode текст показывается в интерфейсе.

## Обработка набора изображений — Batch mode

Batch принимает несколько загруженных изображений либо один ZIP-архив. Если хотите отправить архив, включите **Upload ZIP-file**. Поддерживаются изображения `.bmp`, `.jpg`, `.jpeg`, `.png` и `.webp`. Папку напрямую выбрать нельзя.

Для ZIP-архива поместите изображения прямо в корень архива. Если загружаете несколько файлов из разных мест, используйте уникальные имена файлов.

Рекомендуемый порядок:

1. Выберите метод и модели, затем нажмите **Load Models**.
2. Откройте **Batch mode** и добавьте изображения или ZIP.
3. Проверьте `Caption file extension`. По умолчанию это `.txt`.
4. Для WD+LLM оставьте включённым **Save WD and LLM captions in one file**, если хотите получить и отдельные результаты, и общий файл.
5. Если выбран WD+LLM, начните с `Run method = sync`. В этом режиме tagger и LLM обрабатывают каждое изображение по очереди. `queue` сначала проходит весь набор tagger-ом, затем передаёт результаты этапу LLM.
6. Нажмите **Batch Process**. Во время обработки обновляется **Process preview**, а уведомления показывают номер изображения и его имя: `Caption Batch: start element generation X/Y. Filename: имя_файла`.
7. По окончании скачайте результат через **Download a ZIP file**. Новый запуск очищает все временные папки обработки, поэтому сначала скачайте архив с результатами предыдущего запуска.

## Какие файлы получатся

В Batch результаты создаются под теми же именами, что и изображения, затем упаковываются в ZIP. Например, для исходника `portrait.png` возможны такие варианты:

- **WD-only**: по умолчанию создаётся `portrait.txt` с тегами, потому что в этой ветке применяется поле `Caption file extension`. Если вам нужен суффикс `.wdcaption`, укажите его в `Caption file extension` перед запуском.
- **LLM-only**: по умолчанию создаётся `portrait.txt` с описанием.
- **WD+LLM**: при включённом объединении создаются отдельные `portrait.wdcaption` и `portrait.llmcaption`, а также общий `portrait.txt`. В общем файле теги и описание разделяются значением `Seperator between WD tags and LLM captions`, по умолчанию `|`.
- **Florence в LLM или WD+LLM**: если задача поддерживает визуализацию, рядом с текстовым результатом сохраняется `portrait_visualization.png`. Для задач Florence без визуальной разметки такого PNG не будет.

Обратите внимание: поле `Extension for tag captions files` в Common Tagger Settings и поле `Caption file extension` в Batch — не одно и то же. В WD-only текущий генератор использует именно batch-поле `Caption file extension`; в WD+LLM отдельный WD-файл получает суффикс из Common Tagger Settings.

## Как передать WD-теги в LLM

Если нужно, чтобы LLM описывала изображение с учётом только что полученных WD-тегов, выбирайте **WD+LLM**. В этом режиме расширение сначала получает теги и может вставить их в запрос LLM через placeholder `{wd_tags}`. При желании отредактируйте `user prompt`, сохранив этот placeholder там, где должны появиться теги.

## Если что-то пошло не так

- **Появилось `Models not loaded!`** — нажмите **Load Models** и дождитесь завершения.
- **Первый запуск долго ничего не показывает** — выбранная модель может скачиваться. Проверьте интернет-соединение, свободное место и сообщения в окне запуска Fooocus.
- **Тегов слишком много или слишком мало** — немного измените threshold. Увеличение порога обычно уменьшает количество тегов; уменьшение — увеличивает.
- **PixAI хуже работает на фотографии, чем на аниме-иллюстрации** — карточка модели указывает, что PixAI Tagger прежде всего рассчитан на аниме-изображения; проверьте выдачу вручную или попробуйте WD-модель.
- **Florence не показывает картинку с разметкой** — проверьте выбранную задачу. Caption, Detailed Caption, More Detailed Caption и OCR возвращают текст без такой визуализации. Для наложения используйте, например, Object Detection, Referring Expression Segmentation или OCR with Region.
- **Batch обработал не все картинки из ZIP** — проверьте, лежат ли изображения в корне архива и относятся ли их расширения к поддерживаемым.
- **В архиве нет ожидаемого суффикса** — сверьтесь с разделом выше: WD-only использует `Caption file extension`, а WD+LLM формирует отдельные файлы с WD/LLM-суффиксами и общий файл.

