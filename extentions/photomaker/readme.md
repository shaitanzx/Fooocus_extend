## What PhotoMaker Does
([Читать на русском языке](#что-делает-photomaker))

`PhotoMaker` creates a new image based on a text description and one or more photos of a person. It extracts facial features from the references and uses them during generation. This is not an insertion of the original photo or exact photomontage: the result is a new image, so the resemblance and style change along with the settings, prompt, and base model.

## Before You Start

Prepare at least one photo where the face is clearly visible. The panel accepts multiple images at once; adding several successful shots of the same person can give the model more information about their appearance. Do not upload an archive: PhotoMaker accepts images directly.

An enabled PhotoMaker expects the special word **`img`** in the positive prompt. Add it exactly once after the description of the person or their class, for example: `a photo of a woman img, wearing a red coat, standing in a garden`. For the best result, place the word next to the person's definition. The code checks that the `img` token is present exactly once: without it or with two occurrences, the generation will stop with an error. There is no need to add `img` to the negative prompt.

On the first activation, you may need to wait for the models to load.

## First Launch

1. Open **`Extensions` → `in generation` → `Photomaker`**.
2. Drag and drop one or more photos into **`Drag (Select) 1 or more photos of your face`**. After uploading, the files will appear in the `Your images` area.
3. Enable **`Enabled`**.
4. In the positive prompt, add `img` exactly once after the description of the person. For example: `portrait photo of a man img, wearing a dark blue jacket, soft window light`.
5. For the first attempt, leave **`Style strength (%)`** at the default value of `20`.
6. Click the standard **Generate** button.
7. Evaluate the facial resemblance, text correspondence, and style. For the next attempt, change one parameter at a time.

## Panel Settings

### `Enabled`

**Default:** disabled.

Enables PhotoMaker during the generation process. If disabled, the photos and other PhotoMaker settings do not affect the normal result. If enabled, the absence of a face reference will cause a `Cannot find any input face image!` error.

### `Drag (Select) 1 or more photos of your face`

Uploads one or more photos. This field accepts **multiple images** — they are passed to PhotoMaker as a set of references, not as a separate generation queue.

For each file, the module attempts to detect a face. If a face is found, the extracted feature vector is added to the common set. A photo without a detected face is skipped; an error occurs only if no face could be extracted from any of the images. Therefore, if you uploaded multiple files and one turned out to be unsuccessful, the module can continue with the rest.

The face search tries several detector sizes, reducing it until a face is found. This can help with small faces in the frame but does not guarantee recognition of a blurry, obstructed, or very small face.

The panel has a `Your images` gallery and a **`Remove and upload new ones`** button. The button clears the list to upload a different set. The interface does not provide separate control over the selected person or the selection of a specific face within a photograph.

### `Style strength (%)`

**Range:** `15–50`. **Step:** `1`. **Default:** `20`.

Despite the word "strength", this is not a direct regulator of the reference face weight. The value sets the switching point between regular text embeddings and PhotoMaker embeddings during the generation steps. The code calculates this point as a percentage of the number of steps, but limits it to a maximum of `30`. Before the boundary, the text description without a personalized embedding is used; after the boundary, the prompt with PhotoMaker is used.

Practical meaning: **a higher value prioritizes text and style for longer and may weaken the resemblance to the person**. A lower value activates personalization earlier. Therefore, start with `20`; if the style is not expressive enough, gradually increase the value. If the face stops looking similar, decrease it. The official PhotoMaker guide also states that a higher style strength usually reduces ID resemblance and enhances stylization.

## Single Image, Multiple Photos, and Archive

### One Photo

Upload a single file to the face photos field. Ensure that `Enabled` is turned on and the positive prompt contains `img` exactly once. This is the simplest way to test the functionality and adjust the `Style strength (%)`.

### Multiple Photos

Upload multiple photos at once through the same field. This is a single set of references for one generation, not a "one photo — one result" batch. The module extracts facial features from all suitable shots and combines them. Photos without a detected face are skipped; if none are left, the generation stops with the message `No face detected, please update the input face image(s)`.

The interface does not accept a folder as a separate object. Select photos as files from the folder or highlight multiple files in the upload dialog.

## In Which Modes PhotoMaker Works

In **normal generation**, it uses the selected photos, prompt, negative prompt, resolution, number of steps, CFG, and base model. The PhotoMaker code creates one image per call; the standard number of Fooocus images is processed by an external task loop, so with multiple outputs, PhotoMaker is called for each of them.

In **Prompt Batch**, tasks are launched sequentially with a modified prompt. PhotoMaker remains enabled for these generations and uses the same set of photos; do not forget `img` in each positive prompt.

## What is Saved

The PhotoMaker result is a standard generated Fooocus image. It appears in the results gallery and is saved to the standard output folder.

## Examples

### Photorealistic Portrait

Upload one clear photo and write: `portrait photo of a woman img, natural skin, looking at the camera, soft daylight`. Leave `Style strength (%) = 20` and perform one generation. If the recognizability is weak, try a lower value, such as `15`; if the portrait looks too much like the original shot and does not obey the stylistic prompt, try increasing it gradually.

### Multiple Angles of the Same Person

Upload several shots of the same person: for example, a straight-on portrait and a three-quarter view shot. Use a single prompt with a single `img`. If a face is completely invisible in one image, remove it from the set. If no face is found in any of the photos, PhotoMaker will report an error.

### Stylized Look

Keep `img` right after defining the person, for example: `portrait of an Asian woman img in a watercolor illustration style, soft colors`. Try increasing the `Style strength (%)` from `20` to `25` or `30`, keeping in mind the possible reduction in resemblance. For a more noticeable stylistic change, choose a suitable base model or LoRA.

## Troubleshooting

- **Message about the trigger word `img`.** The positive prompt is missing the token or it appears more than once. Add `img` exactly once next to the description of the person.
- **`Cannot find any input face image!`** PhotoMaker is enabled, but the photo field is empty. Upload at least one shot.
- **`No face detected, please update the input face image(s)`.** No face could be detected in the entire set. Choose a shot with a larger, clearer face and try uploading again.
- **The result has too weak a resemblance.** Check that `img` is placed after the word denoting the person, try a lower `Style strength (%)`, and use clearer photos. The official PhotoMaker V2 example recommends placing `img` after the class word, for example, `woman img`.
- **The image poorly follows the style or description.** Gradually increase the `Style strength (%)`, but keep the trade-off with resemblance in mind. A suitable base model and LoRA can change the style more effectively than trying to compensate for everything with a single parameter.
- **CUDA or out-of-memory error.** In PhotoMaker, the code strictly selects `cuda` and loads the base model, PhotoMaker V2, and T2I-Adapter. Check CUDA availability and the amount of free VRAM; if the problem persists, reduce the resolution and number of steps in the general settings, if possible in your configuration.

## Что делает PhotoMaker
([Read in English](#what-photomaker-does))

`PhotoMaker` создаёт новое изображение по текстовому описанию и одной или нескольким фотографиям человека. Он извлекает признаки лица из референсов и использует их при генерации. Это не вставка исходной фотографии и не точный фотомонтаж: результат — новое изображение, поэтому сходство и стиль меняются вместе с настройками, prompt и базовой моделью.

## Перед началом

Подготовьте хотя бы одну фотографию, на которой лицо хорошо различимо. Панель принимает несколько изображений одновременно; добавление нескольких удачных снимков одного человека может дать модели больше информации о внешности. Не загружайте архив: PhotoMaker принимает изображения напрямую.

Включённый PhotoMaker ожидает специальное слово **`img`** в положительном prompt. Добавьте его один раз после описания человека или его класса, например: `a photo of a woman img, wearing a red coat, standing in a garden`. Для лучшего результата расположите слово рядом с определением человека. Код проверяет, что токен `img` присутствует ровно один раз: без него или с двумя вхождениями генерация остановится с ошибкой. В отрицательный prompt добавлять `img` не нужно.

При первом включении может понадобиться дождаться загрузки моделей.

## Первый запуск

1. Откройте **`Extensions` → `in generation` → `Photomaker`**.
2. Перетащите одну или несколько фотографий в **`Drag (Select) 1 or more photos of your face`**. После загрузки файлы появятся в области `Your images`.
3. Включите **`Enabled`**.
4. В положительном prompt добавьте `img` ровно один раз после описания человека. Например: `portrait photo of a man img, wearing a dark blue jacket, soft window light`.
5. Для первой попытки оставьте **`Style strength (%)`** на значении по умолчанию `20`.
6. Нажмите обычную кнопку **Generate**.
7. Оцените сходство лица, соответствие тексту и стиль. Для следующей попытки изменяйте по одному параметру.

## Настройки панели

### `Enabled`

**По умолчанию:** выключено.

Включает PhotoMaker в процессе генерации. Если выключено, фотографии и остальные настройки PhotoMaker не влияют на обычный результат. Если включено, отсутствие референса лица вызовет ошибку `Cannot find any input face image!`.

### `Drag (Select) 1 or more photos of your face`

Загружает одну или несколько фотографий. Это поле допускает **несколько изображений** — они передаются PhotoMaker как набор референсов, а не как отдельная очередь генераций.

Для каждого файла модуль пытается обнаружить лицо. Если лицо найдено, извлечённый вектор признаков добавляется в общий набор. Фотография без найденного лица пропускается; ошибка возникает, только если не удалось извлечь лицо ни из одного изображения. Поэтому, если загрузили несколько файлов и один оказался неудачным, модуль может продолжить с остальными.

Поиск лица пробует несколько размеров детектора, уменьшая его до тех пор, пока лицо не будет найдено. Это может помочь с небольшими лицами в кадре, но не гарантирует распознавание нечёткого, закрытого или очень маленького лица.

На панели есть галерея `Your images` и кнопка **`Remove and upload new ones`**. Кнопка очищает список, чтобы загрузить другой набор. Отдельного управления выбранным человеком или выбора конкретного лица внутри фотографии интерфейс не предоставляет.

### `Style strength (%)`

**Диапазон:** `15–50`. **Шаг:** `1`. **По умолчанию:** `20`.

Несмотря на слово «strength», это не прямой регулятор веса референсного лица. Значение задаёт момент переключения между обычными текстовыми эмбеддингами и эмбеддингами PhotoMaker во время шагов генерации. Код вычисляет этот момент как процент от числа шагов, но ограничивает его максимумом `30`. До границы используется текстовое описание без персонализированного embedding, после границы — prompt с PhotoMaker.

Практический смысл: **большее значение дольше отдаёт приоритет тексту и стилю и может ослабить сходство с человеком**. Меньшее значение раньше подключает персонализацию. Поэтому начните с `20`; если стиль получается недостаточно выраженным, постепенно повышайте значение. Если лицо перестаёт быть похожим, уменьшите его. Официальное руководство PhotoMaker также указывает, что больший style strength обычно уменьшает ID-сходство и усиливает стилизацию.

## Одиночное изображение, несколько фотографий и архив

### Одна фотография

Загрузите один файл в поле фотографий лица. Убедитесь, что `Enabled` включён и в положительном prompt ровно один раз стоит `img`. Это самый простой способ проверить работу и подобрать `Style strength (%)`.

### Несколько фотографий

Загрузите сразу несколько фотографий через то же поле. Это единый набор референсов для одной генерации, а не пакет «одна фотография — один результат». Модуль извлекает признаки лиц из всех пригодных снимков и собирает их вместе. Фотографии без обнаруженного лица пропускаются; если подходящих не осталось, генерация останавливается с сообщением `No face detected, please update the input face image(s)`.

Папку как отдельный объект интерфейс не принимает. Выберите фотографии файлами из папки или выделите несколько файлов в диалоге загрузки.

## В каких режимах PhotoMaker работает

В **обычной генерации** он использует выбранные фотографии, prompt, negative prompt, размер, число шагов, CFG и базовую модель. Код PhotoMaker создаёт одну картинку на один вызов; обычное количество изображений Fooocus обрабатывается внешним циклом заданий, поэтому при нескольких выходах PhotoMaker вызывается для каждого из них.

В **Prompt Batch** задания запускаются по очереди с изменённым prompt. PhotoMaker остаётся включённым для этих генераций и использует тот же набор фотографий; не забудьте `img` в каждом положительном prompt.

## Что сохраняется

Результат PhotoMaker — обычное сгенерированное изображение Fooocus. Оно появляется в галерее результатов и сохраняется в стандартную папку вывода.

## Примеры

### Фотореалистичный портрет

Загрузите одну чёткую фотографию и напишите: `portrait photo of a woman img, natural skin, looking at the camera, soft daylight`. Оставьте `Style strength (%) = 20` и выполните одну генерацию. Если узнаваемость слабая, попробуйте меньшее значение, например `15`; если портрет слишком похож на исходную подачу и не слушается стилевой prompt, пробуйте повышать постепенно.

### Несколько ракурсов одного человека

Загрузите несколько снимков одного человека: например, прямой портрет и снимок в три четверти. Используйте один prompt с единственным `img`. Если в одном изображении лицо совсем не видно, удалите его из набора. Если не найдено лицо ни на одной фотографии, PhotoMaker сообщит об ошибке.

### Стилизованный образ

Сохраните `img` после определения человека, например `portrait of an Asian woman img in a watercolor illustration style, soft colors`. Попробуйте увеличить `Style strength (%)` с `20` до `25` или `30`, учитывая возможное снижение сходства. Для более заметного стилистического изменения подберите подходящую базовую модель или LoRA.

## Если что-то пошло не так

**Сообщение о trigger word `img`.** В положительном prompt нет токена или он встретился более одного раза. Добавьте `img` один раз рядом с описанием человека.

**`Cannot find any input face image!`** PhotoMaker включён, но поле с фотографиями пустое. Загрузите хотя бы один снимок.

**`No face detected, please update the input face image(s)`.** Не удалось обнаружить ни одного лица во всём наборе. Выберите снимок с более крупным, ясным лицом и повторите загрузку.

**В результате слишком слабое сходство.** Проверьте, что `img` стоит после слова, обозначающего человека, попробуйте меньшее `Style strength (%)` и используйте более ясные фотографии. Официальный пример PhotoMaker V2 рекомендует ставить `img` после class word, например `woman img`.

**Изображение плохо следует стилю или описанию.** Постепенно повысьте `Style strength (%)`, но учитывайте компромисс со сходством. Подходящая базовая модель и LoRA могут сильнее изменить стиль, чем попытка компенсировать всё одним параметром.

**Ошибка CUDA или нехватки памяти.** В PhotoMaker код жёстко выбирает `cuda` и загружает базовую модель, PhotoMaker V2 и T2I-Adapter. Проверьте доступность CUDA и объём свободной видеопамяти; если проблема сохраняется, уменьшите разрешение и число шагов в общих настройках, если это возможно в вашей конфигурации.
