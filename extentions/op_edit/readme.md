## What the Extension is For
([Читать на русском языке](#для-чего-нужно-расширение))

**Open Pose Editor** is a human pose schematic editor. The canvas displays colored joint points and lines connecting them. You can create a schematic manually, detect a pose from a single image, load a saved pose from a JSON file, and download the result as a PNG or JSON.

## Preparation and First Launch

Open the **OpenPose Editor** tab. Upon launch, a standard figure already appears on the black canvas. The **Add** button adds another such figure. You can drag the joint points with the mouse; the connecting lines follow the points.

On the first use of **Detect from Image**, the extension automatically checks for the `body_pose_model.pth` weights and downloads them if necessary.

## Canvas Settings

The interface has two numeric settings:

- **width** — canvas width. By default, it is `512` pixels. The allowable range is from `64` to `2048`, with a step of `64`.
- **height** — canvas height. The value and range are the same: `512` by default, from `64` to `2048`, step `64`.

These sliders change the size of the canvas itself, meaning the coordinate system and the dimensions of the exported files. They do not automatically scale an already drawn pose: the points retain their coordinates, while the canvas boundaries change. Therefore, after a manual size change, the figure may appear shifted or occupy a smaller part of the canvas.

The dimensions also change automatically when you add a background image, detect a pose from an image, or load a JSON or preset with saved dimensions. In these cases, the image dimensions or data from the file are used as the canvas size.

## Creating and Editing a Pose

### Draw a Pose Manually

1. Leave the standard figure that already appears upon opening, or click **Add** to add a new one.
2. Drag the points to change the position of the joints. The lines between them move along with the points.
3. If necessary, set the **width** and **height**. Remember that this changes the canvas boundaries but does not scale existing points.
4. Download the schematic using **Save PNG** or save the coordinates via **Save JSON**.

Each call to **Add** creates a figure with the original coordinates of the standard template. Therefore, if the canvas is already significantly wider or taller than `512 × 512`, the new figure may appear small or off-center. It can be moved and adjusted manually.

### Detect a Pose from an Image

Click **Detect from Image** and select **one** image. The extension will run body pose detection, then display the found skeletons on the canvas. If multiple people are detected in the photograph, the code may add multiple figures. Simultaneously, the selected image is used as a semi-transparent background to make it easier to adjust the points. The canvas size is set to the original size of the uploaded image.

Detection builds a body skeleton from 18 points — for example, the head, shoulders, elbows, wrists, hips, knees, and ankles. Facial expressions and fingers are not detected.

If you need a pose based on a reference but do not want automatic detection, use **Add Background Image**. This button adds the photograph as a semi-transparent background and adjusts the canvas size to match it, but does not detect the pose itself. An already created schematic can be left and edited over the background.

You can also load a background image by dragging and dropping the file onto the canvas. Dragging a JSON file onto the canvas triggers loading the pose from the JSON.

### Reset the Canvas

The **Reset** button clears the poses and the background image, returning a black background. It does not reset the canvas size. After resetting, click **Add** to create the standard figure again.

## Loading and Using Presets

In the **Presets** block, you can save a frequently used pose so you do not have to create it from scratch.

- Select a preset from the list and click **Load Preset** to load it onto the canvas.
- Click **Save Preset** and enter a name in the prompt that appears. The canvas size and point coordinates are saved; the background photograph is not included in the preset.
- Select a preset and click **Delete Preset** to remove it.

## Loading a Pose from JSON

Click **Load from JSON** and select a single `.json` file. Loading replaces the current pose and sets the canvas size from the JSON.

A JSON file is convenient if you want to continue editing the coordinates later. It stores the canvas size and points, not the photograph used as a background.

## Saving the Result

**Save PNG** downloads an image named `pose.png`. The file contains the pose schematic on a black background, not a combination of the skeleton with the original snapshot.

**Save JSON** downloads a file named like `pose-<number>.json`. It saves the canvas width and height, as well as the point coordinates. The background photograph is not included in the file.

## Frequently Asked Questions and Troubleshooting

- **The Presets list is empty.** This is expected on the first launch: create your first preset via **Save Preset**.
- **The background disappeared after loading a JSON.** This is expected behavior: loading a JSON replaces the canvas content, and the photo background is not saved in the JSON. Add it again via **Add Background Image**.
- **The original photograph is missing from the PNG.** This is expected: during saving, the extension temporarily hides the photo background. If you only need the skeleton on a black background, no changes are required.
- **The first image check takes a long time to start or fails.** On the first pose detection, the model file may be downloading.
- **Points did not appear on the photo, or only part of the pose was detected.** Ensure that the person is clearly visible in the image and not obstructed.
- **Clicked Delete Preset on the wrong preset.** There is no confirmation prompt, and the deletion is immediately written to `presets.json`. If the desired preset is deleted, it must be recreated or restored from a file backup.

## Для чего нужно расширение
([Read in English](#what-the-extension-is-for))

**Open Pose Editor** — редактор схемы человеческой позы. На холсте отображаются цветные точки суставов и линии между ними. Можно создать схему вручную, определить позу на одном изображении, загрузить сохранённую позу из JSON и скачать результат как PNG или JSON.


## Подготовка и первый запуск

Откройте вкладку **OpenPose Editor**. При запуске на чёрном холсте уже появляется одна стандартная фигура. Кнопка **Add** добавляет ещё одну такую фигуру. Можно перетаскивать точки суставов мышью; соединительные линии следуют за точками.

При первом использовании **Detect from Image** расширение автоматически проверяет наличие весов `body_pose_model.pth` и при необходимости скачивается.

## Настройки холста

В интерфейсе есть две числовые настройки:

- **width** — ширина холста. При открытии равна `512` пикселям. Допустимый диапазон — от `64` до `2048`, шаг изменения — `64`.
- **height** — высота холста. Значение и диапазон такие же: `512` при открытии, от `64` до `2048`, шаг `64`.

Эти ползунки меняют размер самого холста, то есть систему координат и размеры экспортируемых файлов. Они не масштабируют автоматически уже нарисованную позу: точки сохраняют свои координаты, а границы холста меняются. Поэтому после ручного изменения размера фигура может оказаться смещена или занимать меньшую часть холста.

Размеры также меняются автоматически, когда вы добавляете фоновое изображение, распознаёте позу на изображении либо загружаете JSON или пресет с сохранёнными размерами. В этих случаях в качестве размера холста используются размеры изображения или данные из файла.

## Создание и редактирование позы

### Нарисовать позу вручную

1. Оставьте стандартную фигуру, уже показанную при открытии, или нажмите **Add**, чтобы добавить новую.
2. Перетаскивайте точки, чтобы менять положение суставов. Линии между ними перемещаются вслед за точками.
3. При необходимости задайте **width** и **height**. Помните, что это меняет границы холста, но не масштабирует существующие точки.
4. Скачайте схему с помощью **Save PNG** или сохраните координаты через **Save JSON**.

Каждый вызов **Add** создаёт фигуру с исходными координатами стандартного шаблона. Поэтому если холст уже значительно шире или выше `512 × 512`, новая фигура может оказаться небольшой или расположиться не по центру. Её можно передвинуть и настроить вручную.

### Определить позу на изображении

Нажмите **Detect from Image** и выберите **одно** изображение. Расширение запустит распознавание позы тела, затем покажет найденные скелеты на холсте. Если на фотографии распознаются несколько людей, код может добавить несколько фигур. Одновременно выбранное изображение используется как полупрозрачный фон, чтобы было удобно поправить точки. Размер холста устанавливается по исходному размеру загруженного изображения.

Распознавание строит скелет тела из 18 точек — например, головы, плеч, локтей, кистей, бёдер, коленей и лодыжек. Мимика и пальцы не распознаются.

Если нужна поза по образцу, но не автоматическое распознавание, используйте **Add Background Image**. Эта кнопка добавляет фотографию как полупрозрачный фон и подгоняет размер холста под её размер, но сама по себе позу не определяет. Уже созданную схему можно оставить и редактировать поверх фона.

Загрузить фоновое изображение также можно перетаскиванием файла на холст. Перетаскивание JSON на холст вызывает загрузку позы из JSON.

### Сбросить холст

Кнопка **Reset** очищает позы и фоновое изображение и возвращает чёрный фон. Размер холста она не сбрасывает. После сброса нажмите **Add**, чтобы снова создать стандартную фигуру.

## Загрузка и использование пресетов

В блоке **Presets** можно сохранить часто используемую позу, чтобы не создавать её заново.

- Выберите пресет в списке и нажмите **Load Preset**, чтобы загрузить его на холст.
- Нажмите **Save Preset** и в появившемся запросе введите название. Сохраняются размер холста и координаты точек; фоновая фотография в пресет не входит.
- Выберите пресет и нажмите **Delete Preset**, чтобы удалить его.

## Загрузка позы из JSON

Нажмите **Load from JSON** и выберите один файл `.json`. Загрузка заменяет текущую позу и задаёт размер холста из JSON.

Файл JSON удобен, если вы хотите позже продолжить редактирование координат. Он хранит размер холста и точки, а не фотографию, использованную как фон.

## Сохранение результата

**Save PNG** скачивает изображение с именем `pose.png`. Файл содержит схему позы на чёрном фоне, а не сочетание скелета с исходным снимком.

**Save JSON** скачивает файл с именем вида `pose-<число>.json`. В нём сохраняются ширина и высота холста, а также координаты точек. Фоновая фотография в файл не включается.

## Частые вопросы и проблемы

- **Список Presets пуст.** Это ожидаемо при первом запуске: создайте первый пресет через **Save Preset**.
- **После загрузки JSON пропал фон.** Это ожидаемое поведение: загрузка JSON заменяет содержимое холста, а фото-фон в JSON не сохраняется. Снова добавьте его через **Add Background Image**.
- **В PNG нет исходной фотографии.** Это ожидаемо: при сохранении расширение временно скрывает фото-фон. Если нужен только скелет на чёрном фоне, ничего менять не требуется.
- **Первая проверка изображения долго начинается или не проходит.** При первом обнаружении позы может скачиваться файл модели.
- **На фото не появились точки или обнаружена только часть позы.** Проверьте, что человек достаточно виден на изображении и не перекрыт.
- **Нажал Delete Preset не на тот пресет.** Подтверждения нет, а удаление сразу записывается в `presets.json`. Если нужный пресет удалён, его нужно создать заново или восстановить резервную копию файла.
