## What Vector Does
([Читать на русском языке](#что-делает-vector))

`Vector` converts a **raster image** — PNG, JPG, and other pictures — into an SVG file. It is not a color vector illustration generator or an outline editor. The module traces image areas and creates an SVG with **black fill and no stroke**. Colors, gradients, textures, and photographic details are not transferred as SVG colors.

High-contrast logos, silhouettes, stickers with a solid background, simple black-and-white drawings, and images pre-prepared in a flat graphic style work best. For photographs with shadows and many fine details, the result will usually be rough or overloaded with contours.

The extension has two independent ways of working:

- **`in generation` → `Vector`** — additional processing of an image that Fooocus has just generated.
- **`modules` → `Vector`** — conversion of existing files: a single image, multiple images, or a ZIP archive.

Do not confuse this module with the separate **`tools` → `SVGcode`** tab. This is a different tool with different logic and interface.

## What to Prepare Before Starting

For your first try, take an image with a large dark shape and a solid light background. If you plan to enable background removal, it is especially important that the **top-left pixel** of the image belongs to the background. The code takes its color as the background color. If a part of the object, a frame, or a logo is in this corner, preliminary background removal may delete the wrong areas.

## Two Modes and Settings Availability

In **`in generation`** mode, six items are available: `Enabled` and five general settings. While `Enabled` is off, Vector does not run after generation; the values of the other fields do not change anything for the generation result.

In **`modules` → `Vector`** mode, there is no `Enabled` toggle. Clicking `Start vectorization` always starts the conversion of the selected files. The same five general settings are available there.

`Transparent PNG`, `Noise Tolerance`, and `Quantize` form one group. The two sliders are used **only** when `Transparent PNG` is enabled. `Save temp images` also triggers only when `Transparent PNG` is enabled. If transparency is disabled, these three settings either do not participate in processing or do not create an additional file.

## All Settings in Simple Terms

### `Enabled` — only in `in generation`

**Default:** disabled.

Check this box if you need to additionally get an SVG after normal generation. When the box is unchecked, the image is created and saved normally, but the SVG is not formed.

This is not a toggle for the entire `modules` → `Vector` section: it does not exist in module mode.

### `Transparent PNG`

**Default:** disabled. Works in both modes.

Before tracing, the module reduces the image palette, selects the color of the top-left pixel as the assumed background, and makes pixels close to it transparent. Then the resulting image is passed to the tracer.

Enable this setting for a sticker or logo on a solid background when the background should not interfere with highlighting the silhouette. Do not enable it for a photograph, a complex gradient background, or an image where the object touches the top-left corner.

### `Noise Tolerance`

**Range:** `0` to `128`. **Default:** `16`. Works only when `Transparent PNG` is enabled.

This is the color similarity tolerance to the assumed background. A small value removes only an almost exactly matching background. A large value considers a wider range of similar shades as the background.

Leave `16` for the first attempt. If spots of the former solid background remain around the object, increase the value in small steps. If light parts of the drawing start to disappear along with the background, decrease it.

This slider does not set the detail of vector lines and does not affect Vector when `Transparent PNG` is disabled.

### `Quantize`

**Range:** `2` to `255`. **Default:** `16`. Works only when `Transparent PNG` is enabled.

Before comparing with the background, the image is reduced to the specified number of colors. A smaller number makes the palette rougher and sometimes helps with a monochrome sticker. A larger number leaves more shades of the original, but with a non-uniform background, it can make its removal less predictable.

For a solid background, start with `16`. Try `2–8` if it is a very simple flat icon. If important shades are lost at the object's boundary, return to `16` or increase the value gradually.

### `Save temp images`

**Default:** disabled. Works only together with `Transparent PNG`.

The checkbox saves an intermediate version after background removal. It is needed to check exactly what will be traced.

In **`modules` → `Vector`** mode, a file named `name_trans.extension` will appear next to the SVG. For example, for `cat.png` it will be `cat_trans.png`. Because of the intermediate file, a single source drawing will already yield two results, so the interface will offer a ZIP instead of a single SVG.

In **`in generation`** mode, the intermediate image is added to the normal generation results for subsequent saving. The SVG is still created separately and is not displayed in the gallery.

If the source file is JPG/JPEG, do not enable this checkbox unnecessarily: the code tries to save the RGBA image with the original extension, and JPEG does not support an alpha channel. In case of an error, disable `Save temp images` or use a PNG.

### `Threshold`

**Range:** `0` to `1` with a step of `0.05`. **Default:** `0.5`. The element is shown in both modes.

This parameter sets the tolerance threshold for curve simplification — a higher value allows more simplification and fewer segments, but at the cost of accuracy.

## Invisible but Important Conversion Rules

The code has a fixed black-and-white conversion threshold of `0.5`; there is no separate visible slider for it. Also, small closed areas up to two units of area are discarded. Therefore, small dots, dust, and thin isolated details may disappear.

Vector always writes SVG paths as black fill. The background in the SVG is not drawn as a separate white rectangle, so in a viewer it usually looks transparent. This is normal behavior, not a sign of a corrupted file.

## Mode 1: Get SVG from a New Generation

1. Set up normal Fooocus generation as usual. For a logo, sticker, or flat drawing, you can select a Vector variant in the general style selector. The Vector interface itself only recommends such styles; this is not a mandatory condition.
2. Open `in generation`, then the `Vector` tab.
3. Enable `Enabled`.
4. For the first attempt, leave `Transparent PNG` and `Save temp images` disabled. The values of `Noise Tolerance`, `Quantize`, and `Threshold` in such a run will not provide useful result tuning.
5. Click the standard Fooocus generation button.

The SVG does not replace the generated PNG/JPEG and is not added to the gallery. The code saves it separately in the output folder. The name looks unusual: the internal name generator by default creates a name with `.png`, and Vector adds `.svg`. Therefore, expect a file like `2026-10-12_16-56-00_1234.png.svg`.

For the current internal task, Vector takes the last image from the results list. Do not use this mode as a guaranteed exporter of every variant from a complex internal batch: check that the SVG appeared for the desired image.

## Mode 2: Process a Single File in `modules` → `Vector`

1. Open the `modules` section and the `Vector` tab.
2. Ensure `Upload ZIP-file` is disabled.
3. Drag and drop a single file into the `Drag (Select) 1 or more reference images` field.
4. After uploading, the interface hides the original field and shows `Reference image`. This is just a convenient preview of the selected file; the same uploaded file is processed.
5. For a simple first run, leave `Transparent PNG` and `Save temp images` disabled.
6. Click `Start vectorization`.
7. If exactly one result file is created, a direct SVG download will appear in the `Download a ZIP file` area, despite the field name. Download the file before the next run.

The `Process preview` window shows the source image that the module has currently taken into work. This is not an interactive SVG preview nor the final vector image.

## Multiple Images

1. Leave `Upload ZIP-file` disabled.
2. Upload two or more images into the same field.
3. Configure transparency only if all images have a similar, simple background. The settings are applied equally to all files in the batch.
4. Click `Start vectorization`.
5. The module sorts the filenames alphabetically and processes them in that order. After completion, it will collect the results into a ZIP.

There is no separate folder mode: the interface supports multiple selected files or a single ZIP, but not selecting a local folder. If you need to process the contents of a folder, select its files in the selection dialog or create a flat ZIP archive according to the rules below.

## ZIP Archive: How to Prepare and Process

ZIP mode exists and is intended for batch processing.

1. Create a **flat ZIP archive**: images must lie directly in the root of the archive.
2. Do not add non-images, system files, or service folders to the archive. Do not put images in nested folders.
3. Open `modules` → `Vector`.
4. Enable `Upload ZIP-file`. The images field will hide, and `Upload a ZIP file` will appear instead.
5. Upload a single `.zip` file.
6. Select the desired settings and click `Start vectorization`.
7. Download the provided ZIP with the results.

Why a flat archive is needed: the code unpacks the ZIP entirely, but then Vector iterates only over regular files in the top level of its temporary folder. Nested directories are not traversed, and their images will be skipped. Files in the root are not filtered: each of them is passed as an image. A single text file or other service file in the root can stop processing with an error.

Use unique base names. For example, `logo.png` and `logo.jpg` both try to create `logo.svg`; one result can overwrite the other. The same applies to two uploads with the same name: only the last copied file will remain in the temporary folder.

## Where Results Are Located and What You Will Download

### In `modules` → `Vector`

For an input `flower.png` without transparency, `flower.svg` is created. If both `Transparent PNG` and `Save temp images` are enabled, `flower_trans.png` is additionally created.

When one file is being processed, the interface outputs this file directly. When there are two or more files, it creates a ZIP. Therefore, a single source with intermediate image saving enabled will usually return a ZIP archive of two files.

### In `in generation` → `Vector`

The SVG is written to the normal Fooocus results folder, not to the temporary folder of the batch module. The original raster generation is saved in the usual way. With `Save temp images` enabled, an intermediate raster version after background removal is also added.

## Practical Scenarios

### Black Logo on a White Background

Open `modules` → `Vector`, upload the logo PNG, leave `Transparent PNG` disabled, and click `Start vectorization`. Start with this option: SVG does not draw a white background anyway, and extra preliminary processing can damage the light details of the logo.

### Sticker on a Solid Background

Upload the PNG to `modules` → `Vector`. Enable `Transparent PNG`, leave `Noise Tolerance = 16` and `Quantize = 16`. The first time, you can enable `Save temp images` to look at `*_trans` and understand if the background was removed correctly. If background spots remain, slightly increase `Noise Tolerance`; if the light edges of the sticker disappear, decrease it.

### SVG for Your Own Fooocus Generation

Create an image with a simple composition and high contrast. In `in generation` → `Vector`, enable only `Enabled`. First, find the SVG in the `outputs` folder and evaluate the shape. Only after that try `Transparent PNG` for images with a specially prepared solid background.

## Troubleshooting

- **After generation, PNG is visible, but SVG is not in the gallery.** This is expected. The SVG is not added to the list of result images. Look for a file ending in `.png.svg` in the Fooocus output folder for the current date.
- **After launching in the module, a file is downloaded, not a ZIP.** This is a normal scenario: if the final temporary directory contains exactly one file, the interface outputs it directly.
- **Images from subfolders did not make it into the ZIP.** In this version, nested folders are not traversed. Recreate the archive so that all images lie in its root.
- **Batch processing stops on the archive.** The likely reason is that there is a non-image in the ZIP root: the code tries to open every top-level file as an image. Delete text files, `.DS_Store`, other service files, and repeat the run.
- **Part of the object is deleted along with the background.** Preliminary processing considers the top-left pixel as the background. Place a uniform background in this corner, crop the image, or disable `Transparent PNG`. Then, if necessary, decrease `Noise Tolerance`.
- **An error occurs only with `Transparent PNG` + `Save temp images` for JPG.** The intermediate image has RGBA transparency, and the code saves it with the original file's extension. JPEG does not store transparency. Disable `Save temp images` or process a PNG.
- **Part of the results disappeared or was overwritten.** Check the names. Files with the same name upon upload can be overwritten, and different extensions with the same base name create the same SVG. Rename the input files uniquely.
- **The vector turned out too simple, small dots were lost.** This can be a normal result of the built-in suppression of small isolated areas. Prepare a cleaner and larger source image; there is no separate setting for this in the interface.

## Что делает Vector
([Read in English](#what-vector-does))

`Vector` превращает **растровое изображение** — PNG, JPG и другие картинки — в файл SVG. Это не генератор цветной векторной иллюстрации и не редактор контуров. Модуль обводит области изображения и создаёт SVG с **чёрной заливкой без обводки**. Цвета, градиенты, текстуры и фотографическая детализация не переносятся как цвета SVG.

Лучше всего подходят контрастные логотипы, силуэты, наклейки с ровным фоном, простые чёрно-белые рисунки и изображения, заранее подготовленные в плоском графическом стиле. Для фотографии с тенями и множеством мелких деталей результат обычно будет грубым или перегруженным контурами.

В расширении есть два независимых способа работы:

- **`in generation` → `Vector`** — дополнительная обработка изображения, которое Fooocus только что сгенерировал.
- **`modules` → `Vector`** — преобразование уже имеющихся файлов: одного изображения, нескольких изображений или ZIP-архива.

Не путайте этот модуль с отдельной вкладкой **`tools` → `SVGcode`**. Это другой инструмент, с другой логикой и интерфейсом.

## Что подготовить перед началом

Для первого опыта возьмите изображение с крупной тёмной фигурой и одноцветным светлым фоном. Если собираетесь включать удаление фона, особенно важно, чтобы **верхний левый пиксель** изображения принадлежал фону. Именно его цвет код принимает за цвет фона. Если в этом углу находится часть объекта, рамка или логотип, предварительное удаление фона может удалить не те области.

## Два режима и доступность настроек

В режиме **`in generation`** доступны шесть элементов: `Enabled` и пять общих настроек. Пока `Enabled` выключен, Vector не запускается после генерации; значения остальных полей ничего не меняют для результата генерации.

В режиме **`modules` → `Vector`** переключателя `Enabled` нет. Нажатие `Start vectorization` всегда запускает преобразование выбранных файлов. Там доступны те же пять общих настроек.

`Transparent PNG`, `Noise Tolerance` и `Quantize` образуют одну группу. Два ползунка используются **только**, когда включён `Transparent PNG`. `Save temp images` также срабатывает только при включённом `Transparent PNG`. Если прозрачность отключена, эти три настройки либо не участвуют в обработке, либо не создают дополнительный файл.

## Все настройки простыми словами

### `Enabled` — только в `in generation`

**Значение по умолчанию:** выключено.

Включите этот флажок, если после обычной генерации нужно дополнительно получить SVG. При выключенном флажке изображение создаётся и сохраняется обычным способом, но SVG не формируется.

Это не переключатель всего раздела `modules` → `Vector`: в модульном режиме его нет.

### `Transparent PNG`

**Значение по умолчанию:** выключено. Работает в обоих режимах.

Перед трассировкой модуль уменьшает палитру изображения, выбирает цвет верхнего левого пикселя как предполагаемый фон и делает близкие к нему пиксели прозрачными. Затем полученная картинка передаётся в трассировщик.

Включайте эту настройку для наклейки или логотипа на ровном фоне, когда фон не должен мешать выделению силуэта. Не включайте её для фотографии, сложного градиентного фона или изображения, у которого объект касается верхнего левого угла.

### `Noise Tolerance`

**Диапазон:** от `0` до `128`. **По умолчанию:** `16`. Работает только при включённом `Transparent PNG`.

Это допуск похожести цвета на предполагаемый фон. Малое значение удаляет только почти точно совпадающий фон. Большое значение считает фоном более широкий диапазон похожих оттенков.

Оставьте `16` для первой попытки. Если около объекта остаются пятна прежнего ровного фона, увеличивайте значение небольшими шагами. Если вместе с фоном начинают исчезать светлые части рисунка, уменьшайте его.

Этот ползунок не задаёт детализацию векторных линий и не влияет на Vector, когда `Transparent PNG` выключен.

### `Quantize`

**Диапазон:** от `2` до `255`. **По умолчанию:** `16`. Работает только при включённом `Transparent PNG`.

До сравнения с фоном картинка сводится к указанному числу цветов. Меньшее число делает палитру более грубой и иногда помогает с однотонной наклейкой. Большее число оставляет больше оттенков исходника, но при неоднородном фоне может сделать его удаление менее предсказуемым.

Для ровного фона начните с `16`. Попробуйте `2–8`, если это очень простая плоская иконка. Если на границе объекта теряются важные оттенки, вернитесь к `16` или увеличьте значение постепенно.

### `Save temp images`

**Значение по умолчанию:** выключено. Работает только вместе с `Transparent PNG`.

Флажок сохраняет промежуточную версию после удаления фона. Он нужен, чтобы проверить, что именно будет трассироваться.

В режиме **`modules` → `Vector`** рядом с SVG появится файл вида `имя_trans.расширение`. Например, для `cat.png` это `cat_trans.png`. Из-за промежуточного файла один исходный рисунок уже даст два результата, поэтому интерфейс предложит ZIP, а не одиночный SVG.

В режиме **`in generation`** промежуточная картинка добавляется в обычные результаты генерации для последующего сохранения. SVG всё равно создаётся отдельно и в галерее не отображается.

Если исходный файл — JPG/JPEG, не включайте этот флажок без необходимости: код пытается сохранить RGBA-картинку с исходным расширением, а JPEG не поддерживает альфа-канал. При ошибке отключите `Save temp images` или используйте PNG.

### `Threshold`

**Диапазон:** от `0` до `1` с шагом `0,05`. **По умолчанию:** `0,5`. Элемент показывается в обоих режимах.

Этот параметр задает порог допуска упрощения кривых -  большее значение допускает большее упрощение и меньше сегментов, но ценой точности.

## Невидимые, но важные правила преобразования

В коде задан фиксированный порог преобразования в чёрно-белое `0.5`; отдельного видимого ползунка для него нет. Также мелкие замкнутые области размером до двух единиц площади отбрасываются. Поэтому маленькие точки, пыль и тонкие изолированные детали могут исчезнуть.

Vector всегда записывает пути SVG как чёрную заливку. Фон в SVG не рисуется отдельным белым прямоугольником, поэтому в программе просмотра он обычно выглядит прозрачным. Это нормальное поведение, а не признак повреждённого файла.

## Режим 1: получить SVG из новой генерации

1. Настройте обычную генерацию Fooocus как обычно. Для логотипа, стикера или плоского рисунка можно выбрать в общем селекторе стилей вариант с Vector. Сам интерфейс Vector лишь рекомендует такие стили; это не обязательное условие.
2. Откройте `in generation`, затем вкладку `Vector`.
3. Включите `Enabled`.
4. Для первой попытки оставьте `Transparent PNG` и `Save temp images` выключенными. Значения `Noise Tolerance`, `Quantize` и `Threshold` в таком запуске не дадут полезной настройки результата.
5. Нажмите обычную кнопку генерации Fooocus.

SVG не заменяет сгенерированный PNG/JPEG и не добавляется в галерею. Код сохраняет его отдельно в папку вывода. Имя выглядит необычно: внутренний генератор имён по умолчанию создаёт имя с `.png`, а Vector добавляет `.svg`. Поэтому ожидайте файл наподобие `2026-10-12_16-56-00_1234.png.svg`.

Для текущего внутреннего задания Vector берёт последнее изображение из списка результатов. Не используйте этот режим как гарантированный экспортёр каждого варианта из сложного внутреннего пакета: проверяйте, что SVG появился для нужного изображения.

## Режим 2: обработать один файл в `modules` → `Vector`

1. Откройте раздел `modules` и вкладку `Vector`.
2. Убедитесь, что `Upload ZIP-file` выключен.
3. Перетащите один файл в поле `Drag (Select) 1 or more reference images`.
4. После загрузки интерфейс скрывает исходное поле и показывает `Reference image`. Это лишь удобный просмотр выбранного файла; обрабатывается тот же загруженный файл.
5. Для простого первого запуска оставьте `Transparent PNG` и `Save temp images` выключенными.
6. Нажмите `Start vectorization`.
7. Если создан ровно один файл результата, в области `Download a ZIP file` появится прямое скачивание SVG, несмотря на название поля. Скачайте файл до следующего запуска.

В окне `Process preview` показывается исходная картинка, которую модуль в данный момент взял в работу. Это не интерактивный просмотр SVG и не окончательная векторная картинка.

## Несколько изображений

1. Оставьте `Upload ZIP-file` выключенным.
2. Загрузите два или больше изображений в то же поле.
3. Настройте прозрачность только если у всех изображений похожий, простой фон. Настройки применяются одинаково ко всем файлам пакета.
4. Нажмите `Start vectorization`.
5. Модуль сортирует имена файлов по алфавиту и обрабатывает их в этом порядке. После окончания он соберёт результаты в ZIP.

Нет отдельного режима папки: интерфейс поддерживает несколько выбранных файлов или один ZIP, но не выбор локальной папки. Если нужно обработать содержимое папки, выделите её файлы в диалоге выбора или создайте плоский ZIP-архив по правилам ниже.

## ZIP-архив: как подготовить и обработать

Режим ZIP есть и предназначен для пакетной обработки.

1. Создайте **плоский ZIP-архив**: изображения должны лежать прямо в корне архива.
2. Не добавляйте в архив не-изображения, системные файлы и служебные папки. Не кладите картинки во вложенные папки.
3. Откройте `modules` → `Vector`.
4. Включите `Upload ZIP-file`. Поле изображений скроется, вместо него появится `Upload a ZIP file`.
5. Загрузите один файл `.zip`.
6. Выберите нужные настройки и нажмите `Start vectorization`.
7. Скачайте выданный ZIP с результатами.

Почему нужен именно плоский архив: код распаковывает ZIP целиком, но затем Vector перебирает только обычные файлы в верхнем уровне своей временной папки. Вложенные каталоги не обходятся и их изображения будут пропущены. Файлы в корне не фильтруются: каждый из них передаётся как картинка. Один текстовый файл или иной служебный файл в корне может остановить обработку ошибкой.

Используйте уникальные базовые имена. Например, `logo.png` и `logo.jpg` оба пытаются создать `logo.svg`; один результат способен перезаписать другой. То же касается двух загрузок с одинаковым именем: во временной папке останется только последний скопированный файл.

## Где лежат результаты и что вы скачаете

### В `modules` → `Vector`

Для входного `flower.png` без прозрачности создаётся `flower.svg`. Если включены и `Transparent PNG`, и `Save temp images`, дополнительно создаётся `flower_trans.png`.

Когда в работе один файл, интерфейс выдаёт этот файл напрямую. Когда файлов два или больше, он создаёт ZIP. Поэтому один исходник с включённым сохранением промежуточного изображения обычно вернётся ZIP-архивом из двух файлов.

### В `in generation` → `Vector`

SVG пишется в обычную папку результатов Fooocus, а не в временную папку пакетного модуля. Исходная растровая генерация сохраняется обычным способом. При включённом `Save temp images` добавляется ещё и промежуточная растровая версия после удаления фона.

## Практические сценарии

### Чёрный логотип на белом фоне

Откройте `modules` → `Vector`, загрузите PNG логотипа, оставьте `Transparent PNG` выключенным и нажмите `Start vectorization`. Начните с этого варианта: SVG и так не рисует белый фон, а лишняя предварительная обработка может повредить светлые детали логотипа.

### Наклейка на однотонном фоне

Загрузите PNG в `modules` → `Vector`. Включите `Transparent PNG`, оставьте `Noise Tolerance = 16` и `Quantize = 16`. В первый раз можно включить `Save temp images`, чтобы посмотреть `*_trans` и понять, верно ли удалён фон. Если остаются пятна фона, немного увеличьте `Noise Tolerance`; если исчезают светлые края наклейки, уменьшите его.

### SVG для собственной генерации Fooocus

Сделайте изображение с простой композицией и высоким контрастом. В `in generation` → `Vector` включите только `Enabled`. Сначала найдите SVG в папке `outputs` и оцените форму. Лишь после этого пробуйте `Transparent PNG` для картинок со специально подготовленным ровным фоном.

## Решение проблем

**После генерации виден PNG, но SVG нет в галерее.** Это ожидаемо. SVG не добавляется в список изображений результата. Ищите файл с окончанием `.png.svg` в папке вывода Fooocus за текущую дату.

**После запуска в модуле скачивается файл, а не ZIP.** Это нормальный сценарий: если итоговый временный каталог содержит ровно один файл, интерфейс отдаёт его напрямую.

**В ZIP не попали изображения из подпапок.** В этой версии вложенные папки не обходятся. Пересоздайте архив так, чтобы все изображения лежали в его корне.

**Пакетная обработка останавливается на архиве.** Вероятная причина — в корне ZIP есть не-изображение: код пытается открыть каждый верхнеуровневый файл как картинку. Удалите текстовые файлы, `.DS_Store`, другие служебные файлы и повторите запуск.

**Удаляется часть объекта вместе с фоном.** Предварительная обработка считает верхний левый пиксель фоном. Поместите в этот угол однородный фон, обрежьте изображение или выключите `Transparent PNG`. Затем при необходимости уменьшите `Noise Tolerance`.

**Ошибка возникает только при `Transparent PNG` + `Save temp images` для JPG.** Промежуточная картинка имеет прозрачность RGBA, а код сохраняет её с расширением исходного файла. JPEG не хранит прозрачность. Отключите `Save temp images` либо обработайте PNG.

**Часть результатов пропала или перезаписалась.** Проверьте имена. Файлы с одинаковым именем при загрузке могут перезаписаться, а разные расширения с одним базовым именем создают один и тот же SVG. Переименуйте входные файлы уникально.

**Вектор получился слишком простым, потерялись мелкие точки.** Это может быть нормальным результатом встроенного подавления мелких изолированных областей. Подготовьте более чистую и крупную исходную картинку; в интерфейсе отдельной настройки для этого нет.