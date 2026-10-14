## What InstantID is For
([Читать на русском языке](#для-чего-нужен-instantid))

`InstantID` helps create new images while preserving a person's recognizability from a single face photo. You describe the desired scene with text, upload a photo of the person, and Fooocus uses their facial features. Optionally, you can upload a separate photo with the desired pose: in this case, the identity is taken from the first image, while the facial and compositional guide comes from the second.

This is not a tool for direct insertion or pixel-perfect copying of the original face. It generates a new image, so the resemblance depends on the original photo, the model, the text description, and the control strength. InstantID selects the **largest detected face** in the photo. Working with multiple faces simultaneously is not supported.

## Where to Find It and What Files Are Needed

Open the **`Extensions` → `in generation` → `InstantID`** panel. InstantID is located in the group of additional settings applied during generation.

For the first test, you need the standard base model selected in Fooocus, a text prompt, and one face photo. The **`Upload a photo of your face`** field is mandatory when the `Enabled` checkbox is checked. Use a photo where the face is large enough, sharp, and not obscured by foreign objects. If the detector does not find a face, generation fails with the message `Reference Image: No face detected.`

The **`Upload a reference pose image (Optional)`** field accepts an additional single image. If you upload a shot, it must also have a detectable face: from it, the module extracts facial keypoints to set a guide. If a pose shot is not uploaded, the code uses the face photo for these keypoints as well.

On first use, the extension may download the necessary model and face detector files. If this is happening for the first time, let the download complete; in case of an error, check your internet connection and the Fooocus console messages.

## Quick First Launch

1. Open **`Extensions` → `in generation` → `InstantID`**.
2. Upload a face to **`Upload a photo of your face`**. For a start, it is better to choose a single shot with a clearly visible face.
3. Do not upload a pose reference yet. Without it, the module will use the face both as the basis for identity and as the guide for keypoints.
4. Enable **`Enabled`**.
5. Leave the other settings at their default values and set a prompt, for example: `portrait photo of a person standing in a garden, soft daylight, natural expression`.
6. Click the standard **Generate** button.
7. Evaluate not only the facial resemblance, but also the pose, saturation, and correspondence to the text. Change one setting at a time per run to see the effect.

The `Enabled` checkbox is initially turned off. If it is off, the presence of an uploaded photo and slider values will not trigger InstantID: generation will proceed normally.

## All Settings and Their Effects

### `Enabled`

**Default:** disabled. This setting enables InstantID at the generation stage. Without it, all other InstantID fields are ignored for the current task.

Enable it after uploading the facial reference. If you forget to check the box, Fooocus will still create an image, but it will not be controlled by this reference.

### `Upload a photo of your face`

Accepts **one** image file. This is the source of identity: the program runs face recognition and obtains a feature vector from the largest detected face, which it then passes to the adapter.

If there are multiple people in the shot, the code selects the face with the largest bounding box area, rather than asking the user to choose a person. Crop the photo in advance so that the desired face is the largest.

### `Upload a reference pose image (Optional)`

Accepts **one optional image** that sets the pose/head position guide. The program extracts facial keypoints from it. The identity is still taken from `Upload a photo of your face`, not from the pose photo.

Choose an image where the face is visible and large enough.

Despite the name "pose image", the implementation builds a control map from **facial keypoints**, not the entire body skeleton. Do not expect it to accurately set the position of hands and legs.

### `IdentityNet strength (for fidelity)`

**Range:** `0`–`1.5`. **Step:** `0.001`. **Default:** `0.80`.

Enhances the controlling influence of IdentityNet — the part of InstantID that helps maintain resemblance through the face and keypoints. If the result is not similar enough, increase the value in small steps. Too much influence can make the result less natural or restrict compositional freedom; adjust it together with Image adapter strength. To increase resemblance, gradually strengthen both parameters, and if the saturation is excessive, first reduce the Image adapter strength.

The parameter only acts when `Enabled` is on. In calculations, it is passed as the InstantID ControlNet strength.

### `Image adapter strength (for detail)`

**Range:** `0`–`1.5`. **Step:** `0.001`. **Default:** `0.80`.

Sets the weight of the image adapter, which passes the features of the reference face to the model. Increasing it can help with resemblance and details, but at too high a value, it may increase saturation or interfere with the accurate execution of the text description. In such cases, first reduce this parameter, and if the saturation remains excessive, also reduce the IdentityNet strength.

### `Start at` and `End at`

**Range of each:** `0`–`1`. **Step:** `0.001`. **Default:** `Start at = 0`, `End at = 1`.

These sliders set the time segment of the generation process during which InstantID control is active. By default, it acts from beginning to end. The code passes them as the start and end of the ControlNet control schedule; the same boundaries are used in the adapter settings.

For the first launch, leave the full range `0–1`. These fields do not set a resemblance percentage and do not change the original photo. The interface does not check if `Start at` is less than `End at`, so for a predictable result, keep the start lower than the end.

### `PyraCanny`

**Default:** enabled. This is a separate additional structural hint based on contours, built from the pose reference. If a pose reference is not uploaded, the face photo is substituted into the task state, so PyraCanny may use it. It does not replace the main InstantID face control.

Disable `PyraCanny` if you want to keep only the main identity/keypoints control and weaken the additional following of image contours. In normal generation mode, this checkbox determines whether the corresponding control is added to the reference image.

### `PyraCanny` → `Stop At`

**Range:** `0`–`1`. **Step:** `0.001`. **Default:** `0.5`.

Determines up to what part of the generation process the PyraCanny hint acts. The lower the value, the earlier it stops influencing. This parameter does not control the main IdentityNet — `End at` is for that. If the `PyraCanny` checkbox is disabled, `Stop At` is not used.

### `PyraCanny` → `Weight`

**Range:** `0`–`2`. **Step:** `0.001`. **Default:** `1`.

The strength of the contour hint. Increase it if you need to rely more heavily on the shape/contours of the reference image; decrease it if the composition is being copied too much or is interfering with the prompt. When `PyraCanny` is disabled, the slider is not applied.

### `CPDS`

**Default:** disabled. This is an alternative additional processing of the reference image via CPDS — a way to obtain a depth map. It can be enabled separately or together with PyraCanny if additional structural constraints are needed.

As with PyraCanny, the source will be the pose reference if uploaded, or the face photo if there is no separate pose reference. If `CPDS` is disabled, its two sliders are ignored.

### `CPDS` → `Stop At`

**Range:** `0`–`1`. **Step:** `0.001`. **Default:** `0.5`.

Sets the moment when CPDS influence stops during generation. It is independent of `CPDS` → `Weight` and the main `End at`. Not used when the `CPDS` checkbox is disabled.

### `CPDS` → `Weight`

**Range:** `0`–`2`. **Step:** `0.001`. **Default:** `1`.

Sets the strength of the additional CPDS control. Start with the default value. If the additional depth structure restricts the result too much, reduce the weight or disable `CPDS`.

## Result and Saving

InstantID does not create a separate face file, mask, JSON settings, or archive. It changes the control conditions of the standard Fooocus generation; the result is saved as a normally generated image in the standard output folder. If you need to repeat the same result, save the image and separately record the settings and prompt: do not rely on the temporary upload in the field as a reference archive.

## Practical Settings

### First, achieve resemblance
Leave `Start at = 0`, `End at = 1`, `IdentityNet strength = 0.80`, and `Image adapter strength = 0.80`. If the face is not recognizable enough, first increase one of the two weights in small steps. Choose a photo with a clear face, crop out extra people, and make one control run after each change.

### Reduce saturation or regain control over the prompt
First, reduce `Image adapter strength`. If the saturation is still too high, slightly reduce `IdentityNet strength`. If the text correspondence is weak, try reducing the adapter weight as well.

### Maintain a pose or contour guide
Upload a separate pose image with a visible face. Leave IdentityNet and the adapter at values that give the desired resemblance, then enable or disable `PyraCanny` and `CPDS` one by one. For the first test, leave the weights at `1` and `Stop At = 0.5`; then adjust only the type of structural hint that is actually needed.

## Troubleshooting

**The image generated, but it doesn't look like the uploaded face.** Ensure that `Enabled` is checked and the correct person's face is uploaded. If there are multiple people in the photo, the code automatically selected the largest face. Crop the shot so that the desired face is the largest. Then try gradually increasing the IdentityNet or adapter strength.

**The message `Reference Image: No face detected.` appears.** The detector did not find a face in the main photo. Take a clearer, more frontal shot where the face occupies a larger portion of the frame. This is an error specific to the facial reference.

**An error appears after uploading a separate pose.** Facial keypoints are also searched for in the pose picture. Check that the face is visible there. If you are unsure, remove the pose file and repeat the generation: the main photo will be used instead.

**The generation heavily copies the contours or composition of the reference.** Reduce `PyraCanny` → `Weight` or disable `PyraCanny`. If necessary, disable `CPDS` or reduce its weight. These parameters affect additional structural hints, not the direct recognition of identity.

## Для чего нужен InstantID
([Read in English](#what-instantid-is-for))

`InstantID` помогает создавать новые изображения с сохранением узнаваемости человека по одной фотографии лица. Вы описываете желаемый кадр текстом, загружаете фотографию человека, а Fooocus использует её признаки лица. При желании можно отдельно загрузить фотографию с нужной позой: тогда идентичность берётся из первого снимка, а ориентир лица и композиции — из второго.

Это не инструмент для прямой вставки или копирования исходного лица пиксель-в-пиксель. Он генерирует новое изображение, поэтому сходство зависит от исходного снимка, модели, текстового описания и силы управления. InstantID выбирает **самое крупное обнаруженное лицо** на снимке. Работа с несколькими лицами одновременно не поддерживается.

## Где найти и какие файлы нужны

Откройте панель **`Extensions` → `in generation` → `InstantID`**. InstantID находится в группе дополнительных настроек, применяемых во время генерации.

Для первого теста нужны обычная выбранная в Fooocus базовая модель, текстовый prompt и одна фотография лица. Поле **`Upload a photo of your face`** обязательно, когда установлен флажок `Enabled`. Используйте фото, где лицо достаточно крупное, резкое и не закрыто посторонними предметами. Если детектор не находит лица, генерация срывается с сообщением `Reference Image: No face detected.`

Поле **`Upload a reference pose image (Optional)`** принимает дополнительное одиночное изображение. Если загрузить снимок, на нём также должно обнаруживаться лицо: из него модуль извлекает ключевые точки лица, чтобы задать ориентир. Если снимок позы не загружен, код использует фотографию лица и для этих ключевых точек.

При первом использовании расширение может загрузить нужные файлы модели и детектора лица. Если это происходит впервые, дайте загрузке завершиться; при ошибке проверьте интернет-доступ и сообщения консоли Fooocus.

## Короткий первый запуск

1. Откройте **`Extensions` → `in generation` → `InstantID`**.
2. Загрузите лицо в **`Upload a photo of your face`**. Для начала лучше выбрать одиночный снимок с хорошо различимым лицом.
3. Не загружайте пока референс позы. Без него модуль возьмёт лицо и за основу идентичности, и за ориентир ключевых точек.
4. Включите **`Enabled`**.
5. Оставьте остальные настройки по умолчанию и задайте prompt, например: `portrait photo of a person standing in a garden, soft daylight, natural expression`.
6. Нажмите обычную кнопку **Generate**.
7. Оцените не только сходство лица, но и позу, насыщенность и соответствие тексту. Меняйте по одной настройке за запуск, чтобы видеть эффект.

Флажок `Enabled` изначально выключен. Если он выключен, наличие загруженной фотографии и значения слайдеров не запускают InstantID: генерация идёт обычным путём.

## Все настройки и их действие

### `Enabled`

**По умолчанию:** выключено. Эта настройка включает InstantID на этапе генерации. Без неё все остальные поля InstantID игнорируются для текущего задания.

Включите её после загрузки лицевого референса. Если забыть флажок, Fooocus всё равно создаст изображение, но оно не будет управляться этим референсом.

### `Upload a photo of your face`

Принимает **один** файл изображения. Это источник идентичности: программа запускает распознавание лица и получает по найденному самому крупному лицу вектор признаков, который затем передаёт адаптеру.

Если на снимке несколько людей, код выбирает лицо с наибольшей площадью рамки, а не просит пользователя выбрать человека. Обрежьте фотографию заранее так, чтобы нужное лицо было самым крупным.

### `Upload a reference pose image (Optional)`

Принимает **одно необязательное изображение**, которое задаёт ориентир позы/положения головы. Из него программа извлекает ключевые точки лица. Идентичность при этом по-прежнему берётся из `Upload a photo of your face`, а не из фотографии позы.

Выбирайте изображение, где лицо видно и оно достаточно крупное.

Несмотря на название «pose image», реализация строит управляющую карту из **лицевых ключевых точек**, а не скелет всего тела. Не рассчитывайте, что она точно задаст положение рук и ног.

### `IdentityNet strength (for fidelity)`

**Диапазон:** `0`–`1,5`. **Шаг:** `0,001`. **По умолчанию:** `0,80`.

Усиливает управляющее влияние IdentityNet — части InstantID, которая через лицо и ключевые точки помогает удерживать сходство. Если результат недостаточно похож, увеличьте значение небольшим шагом. Слишком сильное влияние может сделать результат менее естественным или ограничить свободу композиции; подбирайте его вместе с Image adapter strength. Для повышения сходства постепенно усиливать оба параметра, а при чрезмерной насыщенности сначала снижать Image adapter strength.

Параметр действует только при включённом `Enabled`. В вычислениях он передаётся как сила InstantID ControlNet.

### `Image adapter strength (for detail)`

**Диапазон:** `0`–`1,5`. **Шаг:** `0,001`. **По умолчанию:** `0,80`.

Задаёт вес адаптера изображения, который передаёт особенности референсного лица модели. Увеличение может помочь сходству и деталям, но при слишком сильном значении может повышать насыщенность или мешать точному исполнению текстового описания. В таких случаях сначала уменьшайте этот параметр, а если насыщенность остаётся чрезмерной — также IdentityNet strength.

### `Start at` и `End at`

**Диапазон каждого:** `0`–`1`. **Шаг:** `0,001`. **По умолчанию:** `Start at = 0`, `End at = 1`.

Эти ползунки задают временной участок генерационного процесса, на котором участвует управление InstantID. По умолчанию оно действует от начала до конца. Их код передаёт как начало и конец расписания управления ControlNet; те же границы используются в настройке адаптера.

Для первого запуска оставьте полный диапазон `0–1`. Эти поля не задают процент сходства и не меняют исходную фотографию. В интерфейсе нет проверки, что `Start at` меньше `End at`, поэтому для предсказуемого результата оставляйте начало ниже конца.

### `PyraCanny`

**По умолчанию:** включено. Это отдельная дополнительная структурная подсказка по контурам, построенная по референсу позы. Если референс позы не загружен, в состояние задания подставляется фотография лица, поэтому PyraCanny может использовать её. Она не заменяет основное управление InstantID по лицу.

Выключите `PyraCanny`, если хотите оставить только основное управление идентичностью/лицевыми точками и ослабить дополнительное следование контурам изображения. В режиме обычной генерации эта галочка определяет, добавляется ли соответствующее управление к изображению-референсу.

### `PyraCanny` → `Stop At`

**Диапазон:** `0`–`1`. **Шаг:** `0,001`. **По умолчанию:** `0,5`.

Определяет, до какой части процесса генерации действует именно подсказка PyraCanny. Чем меньше значение, тем раньше она прекращает влиять. Этот параметр не управляет основным IdentityNet — для него есть `End at`. Если флажок `PyraCanny` выключен, `Stop At` не используется.

### `PyraCanny` → `Weight`

**Диапазон:** `0`–`2`. **Шаг:** `0,001`. **По умолчанию:** `1`.

Сила контурной подсказки. Повышайте её, если нужно сильнее опираться на форму/контуры референсного изображения; уменьшайте, если композиция чрезмерно копируется или мешает prompt. При отключённом `PyraCanny` ползунок не применяется.

### `CPDS`

**По умолчанию:** выключено. Это альтернативная дополнительная обработка референсного изображения через CPDS - способ получить карту глубины. Её можно включать отдельно или вместе с PyraCanny, если нужны дополнительные структурные ограничения.

Как и для PyraCanny, источником будет референс позы, если он загружен, либо лицевое фото, если отдельного референса позы нет. Если `CPDS` выключен, его два ползунка игнорируются.

### `CPDS` → `Stop At`

**Диапазон:** `0`–`1`. **Шаг:** `0,001`. **По умолчанию:** `0,5`.

Задаёт момент прекращения влияния CPDS в ходе генерации. Он независим от `CPDS` → `Weight` и от основного `End at`. Не используется при выключенном флажке `CPDS`.

### `CPDS` → `Weight`

**Диапазон:** `0`–`2`. **Шаг:** `0,001`. **По умолчанию:** `1`.

Задаёт силу дополнительного CPDS-управления. Начните со значения по умолчанию. Если дополнительная глубинная структура слишком ограничивает результат, уменьшите вес или отключите `CPDS`.

## Результат и сохранение

InstantID не создаёт отдельный файл лица, маску, JSON-настройки или архив. Он меняет управляющие условия обычной генерации Fooocus; результат сохраняется как обычное сгенерированное изображение в стандартную папку вывода. Если нужно повторить тот же результат, сохраните изображение и отдельно запишите настройки и prompt: не полагайтесь на временную загрузку в поле как на архив референсов.

## Практические настройки

### Сначала добиться сходства

Оставьте `Start at = 0`, `End at = 1`, `IdentityNet strength = 0,80` и `Image adapter strength = 0,80`. Если лицо недостаточно узнаваемо, увеличивайте сначала один из двух весов небольшими шагами. Выберите фотографию с чётким лицом, уберите лишних людей кадрированием и сделайте один контрольный запуск после каждого изменения.

### Снизить насыщенность или вернуть контроль над prompt

Сначала уменьшите `Image adapter strength`. Если насыщенность всё ещё слишком высокая, немного уменьшите `IdentityNet strength`. При слабом соответствии тексту попробуйте также снизить вес адаптера.

### Удержать ориентир по позе или контурам

Загрузите отдельное изображение позы с видимым лицом. Оставьте IdentityNet и адаптер на значениях, которые дают нужное сходство, затем включайте или выключайте `PyraCanny` и `CPDS` по одному. Для первого теста оставьте веса `1` и `Stop At = 0,5`; затем корректируйте только тот вид структурной подсказки, который действительно нужен.

## Если что-то не работает

**Изображение сгенерировалось, но не похоже на загруженное лицо.** Убедитесь, что `Enabled` включён и загружено лицо нужного человека. Если на фото несколько людей, код автоматически выбрал самое крупное лицо. Обрежьте снимок так, чтобы нужное лицо было крупнейшим. Затем попробуйте постепенно усилить IdentityNet или адаптер.

**Появляется `Reference Image: No face detected.`** Детектор не нашёл лицо на главной фотографии. Возьмите более чёткий и фронтальный снимок, где лицо занимает большую часть кадра. Это ошибка именно лицевого референса.

**Ошибка появляется после загрузки отдельной позы.** В pose-картинке тоже ищутся лицевые ключевые точки. Проверьте, что лицо там видно. Если не уверены, уберите pose-файл и повторите генерацию: тогда вместо него будет использоваться основное фото.

**Генерация сильно копирует контуры или композицию референса.** Уменьшите `PyraCanny` → `Weight` или отключите `PyraCanny`. При необходимости отключите `CPDS` либо уменьшите его вес. Эти параметры влияют на дополнительные структурные подсказки, а не на распознавание личности напрямую.
