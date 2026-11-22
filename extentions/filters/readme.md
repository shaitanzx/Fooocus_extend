## What the extension does
([Читать на русском языке](#Что-делает-расширение))

Filters applies a set of mathematical effects to images, mimicking styles such as pixel art, CRT, LCD, VHS, and those of vintage gaming systems. Processing is performed directly on the image.

Each filter takes an RGB image as input, performs a single operation, and passes the result to the next filter. Consequently, filters can be used individually or combined into a sequential chain:

Input → Pixelize → Palette Quantize → Scanlines → Bloom → VHS → Output

The current implementation includes 12 filters:

1. Pixelize
2. Palette Quantize
3. Color Depth
4. Dither
5. Scanlines
6. Subpixel Mask
7. Bloom
8. Chromatic Aberration
9. Curvature
10. Vignette
11. VHS
12. LCD Dot Matrix

## Interface Overview

### Filter chain (selection order)

This multi-select dropdown menu defines the processing sequence. The order in which items are selected determines the order in which the filters are applied.

For example:

Pixelize → Scanlines → Bloom

means:

1. The image first passes through Pixelize;
2. The output of Pixelize is passed to Scanlines;
3. The output of Scanlines is passed to Bloom;
4. The output of Bloom is saved as the final result.

### Save result after each filter

This checkbox enables the saving of intermediate processing stages.

If the chain contains only one filter, no intermediate file is created; the result is saved only as the final output.

If the chain contains multiple filters, all stages preceding the final filter are saved. The final stage is always saved exclusively as the final output file.

For instance, a chain of three filters will generate:

### Filter to configure

This dropdown menu allows you to select a specific filter to view and adjust its settings. Only one set of configuration options is displayed at any given time. Parameters are independent of whether a filter is actually included in the chain. You can configure all filters in advance and then select the ones you want to use in the `Filter chain (selection order)` section.

If a filter is not included in the chain, its settings are ignored during the current execution.

# Detailed Filter Settings

## 1. Pixelize

The `Pixelize` filter reduces the spatial detail of an image by breaking it down into square cells, then scaling it back to the original resolution using nearest-neighbor interpolation. The final image dimensions remain unchanged, but the content within each cell is rendered as a uniform color.

This filter is particularly useful at the beginning of a chain—for instance, before a palette effect, a CRT mask, or an LCD dot matrix overlay.

### Pixel size

- Type: Integer.
- Interface range: `1–128`.
- Step: `1`.
- Default value: `8`.
- A value of `1` results in detail nearly identical to the original.
- Higher values ​​produce larger pixels and less fine detail.

### Sampling

Available options:

- `center` — Uses the pixel at the center of each cell. This is the fastest method and creates a rigid grid without averaging.
- `mean` — Calculates the average color value within each cell, resulting in a smoother appearance.
- `median` — Uses the median color value across channels. This method is effective at suppressing isolated, extremely bright or dark pixels.
- `mode` — Selects the most frequently occurring color value. This is the default setting and is generally best suited for large areas of authentic pixel art.

### Pre-blur

- Range: `0–10`.
- Step: `0.1`.
- Default value: `0`.
- Unit: Sigma (for Gaussian blur). A pre-blur is applied before the image is modified. This reduces fine, high-frequency details and can help mitigate moiré patterns that may arise after pixelization.

A value of `0` preserves a sharp, pixel-art aesthetic. Higher values ​​soften the boundaries within the pixel cells before the final color selection takes place.

### Practical Examples

For a distinct pixel-art effect:
Pixel size: 8
Sampling: mode
Pre-blur: 0

For a photograph containing many fine details, you might try:
Pixel size: 6–12
Sampling: mean
Pre-blur: 0.5–1.5


## 2. Palette Quantize

`Palette Quantize` replaces each color with the closest match from a selected, limited retro palette. The matching process is based on distance within the CIELAB color space, rather than a simple, independent comparison of RGB channels.

This filter is designed to emulate the color limitations of various retro systems, such as PICO-8, Game Boy, C64, CGA, ZX Spectrum, and others.

### Built-in Palettes

The current UI includes the following palettes:
- `dmg_4` — The four-color green palette of the Game Boy (DMG).
- `pico8_16` — The standard 16-color PICO-8 palette (default setting).
- `c64_16` — The 16-color Commodore 64 palette.
- `cga_16` — The 16-color IBM CGA palette.
- `zx_spectrum_15` — The 15-color ZX Spectrum palette.
- `ega_64` — The 64-color EGA palette.
- `nes` — The 55-color palette of the Nintendo Entertainment System (2C02).
- `gray_4` — Four shades of gray. - `mono_1bit` — black and white.
- `sweetie_16` — the Sweetie16 palette.

Default value: `pico8_16`.

### Custom palette

This is a multi-line field for a custom palette. Colors are entered in a six-character hex format:
#0f380f, #306230, #8bac0f, #9bbc0f

Allowed delimiters:
- comma;
- semicolon;
- space;
- line break.

Entries with or without the `#` prefix are supported.

Shorthand notation like `#fff` is not supported. Each entry must consist of exactly six hex digits.

If the `Custom palette` field contains a value, it completely replaces the selected `Built-in palette`. In this case, the `Built-in palette` setting is ignored.

If the `Custom palette` field is left empty, the selected built-in palette is used instead.

### Dithering

In the Filters section, this component is called `Dithering`, while in the source node, it is referred to as `dither`. Available methods include:

- `none` — no dithering;
- `bayer2` — ordered Bayer dithering using a 2×2 matrix;
- `bayer4` — 4×4 Bayer dithering;
- `bayer8` — 8×8 Bayer dithering;
- `floyd-steinberg` — Floyd–Steinberg error diffusion;
- `atkinson` — Atkinson error diffusion.

`bayer2`, `bayer4`, and `bayer8` are fast. `floyd-steinberg` and `atkinson` process the image row by row, which may take significantly longer for large images.

### Dither strength

- Range: `0–1`.
- Step: `0.05`.
- Default value: `1`.

For ordered dithering methods, this parameter adjusts the amplitude of the threshold offset. For error-diffusion methods, it controls the magnitude of the error propagated to neighboring pixels.

- `0`: dithering is effectively disabled.
- `1`: the selected method is applied at full strength.

### Serpentine diffusion

- Default value: enabled.

When enabled, the processing direction alternates between rows. Odd-numbered rows are processed from right to left, with the error-diffusion kernel mirrored accordingly. This reduces directional banding, particularly when using the Floyd–Steinberg or Atkinson methods.

For Bayer-based methods, this parameter does not play a significant role, as they do not rely on sequential error diffusion.

### Practical Examples

For PICO-8:

Built-in palette: pico8_16
Custom palette: empty
Dithering: none or bayer4
Dither strength: 1
Serpentine diffusion: enabled


For a monochrome, Game Boy-style result:

Built-in palette: dmg_4
Dithering: floyd-steinberg
Dither strength: 0.5–1
Serpentine diffusion: enabled


## 3. Color Depth

`Color Depth` performs pure posterization—it uniformly reduces the number of levels for each RGB channel. Unlike `Dither`, this filter does not introduce any dithering.

### Interpret value as

The `Interpret value as` component offers two modes:

- `bits` — the value is interpreted as bit depth;
- `levels` — the value is interpreted as the exact number of levels.

### Bits / levels

The UI features a single slider with a range of `1–64`, a step of `1`, and a default value of `4`.

The valid range depends on the selected mode:
- In `bits` mode, values ​​from `1–8` are permitted;
- In `levels` mode, values ​​from `2–64` are permitted.

In `bits` mode, the number of levels is calculated as follows:
levels = 2 ** value

Examples:
- `1 bit` → 2 levels per channel;
- `2 bits` → 4 levels;
- `4 bits` → 16 levels;
- `8 bits` → 256 levels.

In `levels` mode, the value is used directly. For instance, `5` means exactly five levels per channel. If a value greater than `8` is provided in `bits` mode, the engine will throw an error. Similarly, if a value less than `2` is provided in `levels` mode, the engine will also throw an error. Therefore, it is recommended to manually keep the value within the range of `1–8` for `bits` mode, and `2–64` for `levels` mode.


## 4. Dither

`Dither` reduces the number of color levels while simultaneously applying dithering. Unlike `Palette Quantize`, it does not map the image to a fixed retro palette.

### Levels per channel

- Range: `2–32`.
- Step: `1`.
- Default value: `4`.

This represents the number of quantization levels for each channel.
A low value results in coarser posterization, while a high value preserves more shades.

### Mode

Available values:
- `color` — each RGB channel is processed independently;
- `mono` — the image is first converted to luminance using the Rec. 601 standard: Y = 0.299 R + 0.587 G + 0.114 B.

The resulting luminance channel is then copied into all three channels and dithered, producing a grayscale image.

Default value: `color`.

### Method

Available options in the UI:
- `bayer2`;
- `bayer4`;
- `bayer8`;
- `floyd-steinberg`;
- `atkinson`.

Default value: `bayer4`.

The `none` method is intentionally omitted from this filter; for posterization without dithering, use `Color Depth` instead.

### Strength

- Range: `0–1`.
- Step: `0.05`.
- Default value: `1`. `0` disables the effect of dithering, while `1` applies the algorithm at full strength.

### Serpentine diffusion

- Default: enabled.

Primarily used by error diffusion methods. Alternating the direction of the rows reduces regular, directional artifacts.

## 5. Scanlines

`Scanlines` simulates the horizontal darkened lines found on CRT displays. Within each period, some lines are darkened, while others remain unchanged.

### Pitch

- Range: `1–32`.
- Step: `1`.
- Default value in the current UI: `2`.

`Pitch` defines the repetition period of the pattern across the image rows.

- `1`: darkening is applied to every row;
- `2`: the pattern repeats every two rows;
- higher values: the lines are spaced further apart.

### Thickness

- Range: `0–1`.
- Step: `0.05`.
- Default value:value: `0.5`.

This represents the fraction of the period that corresponds to the darkened portion.

- `0`: no darkened area;
- `0.5`: half of each period is darkened;
- `1`: the entire period is darkened.

### Strength

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.35`.

This defines the maximum intensity of the line darkening.
- `0`: no effect;
- increasing the value makes the stripes darker.

This parameter does not add white lines; instead, it reduces the brightness of the selected portion of the period.

### Offset

- Range: `-64 – +64`.
- Step: `1`.
- Default value: `0`.

Shifts the phase of the pattern vertically. This parameter is useful for aligning scanlines with an existing pixel grid.

### Profile

Possible values:

- `hard` — a sharp transition between the normal and darkened portions;
- `soft` — a smooth, raised-cosine transition.

Default value: `hard`.


## 6. Subpixel Mask

The `Subpixel Mask` simulates the phosphor mask structure of a CRT or the pixel layout of a display. The red, green, and blue channels are passed through distinct vertical stripes.

To compensate for the fact that each channel occupies approximately one-third of the total area, the brightness is multiplied by three. Consequently, highly saturated areas may be subject to clipping.

### Mask

Available types:

- `aperture_grille` — straight vertical RGB stripes;
- `shadow_mask` — a staggered (checkerboard-like) row structure, resembling the holes in a physical CRT mask;
- `slot_mask` — an aperture grille with every second horizontal stripe darkened. Default value: `aperture_grille`.

### Pitch

- Range: `1–16`.
- Step: `1`.
- Default value: `1`.

`Pitch` defines the width of a single color subpixel stripe. The full RGB triad period is equal to:
3 × pitch

For alignment with Pixelize, it is recommended to choose a `Pixel size` that is a multiple of `3 × Pitch`.

For example:
Pixelize Pixel size: 6
Subpixel Mask Pitch: 2

---

## 7. Bloom

`Bloom` extracts bright areas, blurs them, and overlays the resulting glow onto the original image.

First, brightness is calculated according to Rec. 601; then, pixels exceeding a certain threshold are added to a "bright mask." This mask is blurred using a Gaussian blur and blended with the original image.

### Threshold

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.7`.

This is the brightness threshold above which pixels contribute to the glow effect.

- Low threshold: more of the image glows;
- High threshold: only the brightest areas glow;
- `1.0`: the bright mask is empty, and the image remains unchanged.

### Radius

- Range: `0.5–100`.
- Step: `0.5`.
- Default value: `8`.

This represents the sigma value for the Gaussian blur, measured in pixels.

- Small value: a compact glow;
- Large value: a broad, soft glow.

### Intensity

- Range: `0–4`.
- Step: `0.05`.
- Default value: `0.5`.

Scales the contribution of the blurred "bright mask."

- `0`: no visible bloom; - values ​​around `0.5`: moderate glow;
- higher values: intense glow with a risk of overexposure (clipping).

### Blend

Possible values:

- `screen` — a soft, additive-like blending mode based on the "Screen" formula;
- `add` — direct addition of brightness values.

Default value: `screen`.

`screen` generally produces a softer look, whereas `add` causes bright areas to reach clipping more quickly.


## 8. Chromatic Aberration

`Chromatic Aberration` shifts the red and blue color channels relative to the green channel, which remains fixed. This creates colored red-and-blue fringes, mimicking the optical aberrations found in vintage lenses or CRT displays.

### Amount

- Range: `0–20`.
- Step: `0.5`.
- Default value: `2`.

The unit represents the displacement in pixels.

- `0`: no change (original image);
- increasing the value: results in wider colored fringes.

### Mode

- `radial` — the displacement depends on the position relative to the center of the image;
- `linear` — the red and blue channels are shifted horizontally in opposite directions.

Default value: `radial`.

In `linear` mode, the red channel shifts to the right, while the blue channel shifts to the left. In `radial` mode, the colored fringes are oriented relative to the center of the image.


## 9. Curvature

`Curvature` simulates the convex, barrel-like geometry of a CRT screen. Image coordinates are remapped outward from the center, with the center point remaining stationary.

The corners may extend beyond the boundaries of the original image, resulting in black areas. The midpoints along the edges are compensated for via scaling, so standard black bars should not appear along the centers of the edges.

### Amount

- Range: `0–1`. - Step: `0.05`.
- Default value: `0.15`.

The strength of the geometric distortion.

- `0`: original geometry;
- increasing the value: the edges bulge outward more, and the corner areas become darker.

## Edge softness

- Range: `0–32`.
- Step: `0.5`.
- Default value: `0`.

The width of the smooth transition to the black area at the edges (in pixels).

- `0`: a hard edge;
- increasing the value: a soft fade to black.

The image within the area is not blurred; only the transition at the edge of the distortion is affected.

## 10. Vignette

`Vignette` darkens the image from the center toward the edges. Internally, the engine calculates the normalized distance from the center to the current point and then applies a power curve.
### Strength

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.4`.

The maximum degree of darkening.

- `0`: effect disabled;
- increasing the value: outer areas become darker.

### Radius

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.8`.

The normalized distance at which the darkening begins.

- small value: darkening begins closer to the center;
- large value: the center remains unchanged over a larger area;
- `1`: effect absent.

### Softness

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.5`.

Controls the steepness of the transition.

Higher values ​​result in a sharper transition closer to the edges.

---

## 11. VHS

`VHS` applies four analog effects in sequence:

1. chroma bleed;
2. horizontal line jitter;
3. additive noise;
4. vertical brightening band.

Each sub-effect is skipped if its strength is set to zero.

### Noise

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.15`.

The amplitude of Gaussian noise. The internal sigma is calculated as:
sigma = noise × 40

`0` disables the noise.

### Chroma bleed

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.5`.

Blurs the U/V color-difference components horizontally while preserving the luminance component. This simulates color bleeding typical of VHS tapes. The kernel width depends on the value and is constrained to a range of 3–15 pixels; the width is forced to an odd number.

`0` disables chroma bleed.

### Jitter

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.3`.

Each row of the image is shifted by a random horizontal offset. The maximum amplitude is:
jitter × 6 pixels

Edge replication is used, so the image does not wrap around to the opposite side.

### Band

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.3`.

Adds a vertical brightness band with a Gaussian profile. The band only brightens the image; it does not darken it. Its position is chosen randomly.

### Seed

- Range: `0–2147483647`.
- Step: `1`.
- Default value: `0`.

- `0`: new randomness for every run;
- a value greater than `0`: repeatable randomness for a given seed.

This is particularly important for batch processing and comparing results. If you need the same result upon subsequent runs, set a non-zero seed.

### Noise mode

- `mono` — a single, shared noise plane for all RGB channels;
- `color` — independent noise for each channel.

Default value: `color`.

`mono` typically resembles luminance-based television noise. `color` creates chromatic noise variations across the channels.

---

## 12. LCD Dot Matrix

`LCD Dot Matrix` overlays a regular LCD dot-matrix structure onto the image. The filter operates at the current resolution and does not alter the image dimensions. ### Cell size

- Range: `1–64`.
- Step: `1`.
- Default value: `8`.

The side length of a single cell, in pixels.

To align the LCD grid with the Pixelize effect, it is common to use matching values:
Pixelize Pixel size = 8
LCD Dot Matrix Cell size = 8

### Gap ratio

- Range: `0–0.5`.
- Step: `0.05`.
- Default value: `0.15`.

The proportion of the gap surrounding the active area of ​​each cell.

- `0`: no gap; the effect is nearly identical to the original image.
- Increasing the value: the active dots become smaller.
- `0.5`: the maximum gap size available via the interface.

### Grid dark

- Range: `0–1`.
- Step: `0.05`.
- Default value: `0.6`.

The darkness level of the inactive area (the gaps).

- `0`: the gaps are not darkened.
- `1`: the gaps are completely black.

### Shape

- `square` — a square active area within each cell.
- `round` — a circular dot.

Default value: `square`.

---

# How to build a processing chain

## Pixel-art chain

Pixelize → Palette Quantize

Recommended settings:
Pixelize:
Pixel size: 8
Sampling: mode
Pre-blur: 0

Palette Quantize:
Built-in palette: pico8_16
Dithering: none or bayer4
Dither strength: 1

For a smoother-looking image, you can add a `Pre-blur` value in the range of `0.5–1.5`. ## CRT Monitor
Pixelize → Scanlines → Subpixel Mask → Bloom → Curvature → Vignette

Starting configuration:
Pixelize:
Pixel size: 6–9
Sampling: mode
Scanlines:
Pitch: 2
Thickness: 0.5
Strength: 0.35
Profile: hard
Subpixel Mask:
Mask: aperture_grille
Pitch: 1
Bloom:
Threshold: 0.7
Radius: 8
Intensity: 0.5
Blend: screen
Curvature:
Amount: 0.15
Edge softness: 0–4
Vignette:
Strength: 0.4
Radius: 0.8
Softness: 0.5

If the mask pattern appears too dense, increase `Subpixel Mask → Pitch`. If colored stripes become too noticeable, decrease `Pixelize → Pixel size` or remove the Subpixel Mask from the chain.

## VHS Tape

VHS → Bloom

Starting parameters for VHS:
Noise: 0.15
Chroma bleed: 0.5
Jitter: 0.3
Band: 0.3
Seed: 0
Noise mode: color

For a reproducible result, set the `Seed` to a specific value (e.g., `12345`). To generate a new noise pattern each time you run it, leave it set to `0`.

## Black and White Image with Dithering

Dither

Recommended parameters:

Levels per channel: 4
Mode: mono
Method: floyd-steinberg
Strength: 1.0
gth: 1
Serpentine diffusion: enabled

## LCD Screen

Pixelize → LCD Dot Matrix

For a matching grid pitch:

Pixelize Pixel size: 8
LCD Dot Matrix Cell size: 8
LCD Dot Matrix Gap ratio: 0.15–0.25
LCD Dot Matrix Grid dark: 0.6
LCD Dot Matrix Shape: square or round


# Input Files and Execution

## Multiple Images

The upload section features a field to "Drag (Select) 1 or more reference images."

You can select one or more images. Once selected, click: Start.

The chosen processing chain is applied to each input file.

## ZIP File

When "Upload ZIP-file" is enabled, the interface switches to allow for a ZIP file upload. The archive is extracted to:

## Execution Order

Before processing:

1. `batch_post` is cleared;
2. `batch_temp` is cleared;
3. Standard images are copied—or a ZIP file is extracted—into `batch_post`;
4. The extension reads the input files;
5. Each filter in the chain is applied sequentially;
6. Results are saved to `batch_temp`;
7. If there are multiple results, a ZIP file is created.

---

# Results and Filenames

Filenames consist of the original name (without the extension), the date and time of execution, and the suffix `filters`.

If intermediate saving is enabled and the chain contains multiple filters, the intermediate files are assigned an additional suffix, `chain`, followed by the filter's position in the sequence.

## Preview, Image Output, and ZIP

During processing, the result is temporarily displayed in the "Process preview" section.

If there is a single output file, it can be presented directly as an image. If there are multiple files—for instance, due to multiple input images or intermediate saves—a ZIP file is generated for download.

Intermediate files are also included in the ZIP archive. When a single input image is processed with intermediate saving enabled, multiple files are generated. Following this process, a collapsed `compare` accordion appears, allowing you to compare the images from before and after processing.

## Что делает расширение
([Read in English](#What-the-extension-does))

Filters применяет к изображениям набор математических эффектов в стиле pixel art, CRT, LCD, VHS и старых игровых систем. Обработка выполняется непосредственно над изображением.

Каждый фильтр получает изображение в формате RGB, выполняет одну операцию и передаёт результат следующему фильтру. Поэтому фильтры можно использовать по одному или собрать в последовательную цепочку:

Input → Pixelize → Palette Quantize → Scanlines → Bloom → VHS → Output

Текущая реализация содержит 12 фильтров:

1. Pixelize
2. Palette Quantize
3. Color Depth
4. Dither
5. Scanlines
6. Subpixel Mask
7. Bloom
8. Chromatic Aberration
9. Curvature
10. Vignette
11. VHS
12. LCD Dot Matrix

## Как устроен интерфейс

### Filter chain (selection order)

В этом выпадающем поле с множественным выбором задаётся порядок обработки. Порядок выбранных элементов является порядком применения фильтров.

Например:

Pixelize → Scanlines → Bloom

означает:

1. изображение сначала проходит через Pixelize;
2. результат Pixelize поступает в Scanlines;
3. результат Scanlines поступает в Bloom;
4. результат Bloom сохраняется как финальный.

### Save result after each filter

Этот чекбокс сохраняет результаты промежуточных этапов.

Если в цепочке один фильтр, промежуточного файла нет: результат сохраняется только как финальный.

Если в цепочке несколько фильтров, сохраняются только этапы до последнего фильтра. Последний этап всегда записывается только в финальный файл.

Например, для цепочки из трёх фильтров создаются:

### Filter to configure

Этот выпадающий список выбирает фильтр, настройки которого нужно показать и изменить. В каждый момент отображается одна группа настроек.

Параметры не привязаны к факту присутствия фильтра в цепочке. Можно заранее настроить все фильтры, затем выбрать нужные фильтры в `Filter chain (selection order)`.

Если фильтр не включён в цепочку, его настройки не используются в текущем запуске.

# Подробные настройки фильтров

## 1. Pixelize

`Pixelize` уменьшает пространственную детализацию изображения, разбивая его на квадратные ячейки, а затем возвращает изображение к исходному разрешению с nearest-neighbor интерполяцией. Размер итогового изображения не меняется, но внутри ячеек появляется общий цвет.

Фильтр особенно полезен в начале цепочки перед палитрой, CRT-маской или LCD Dot Matrix.

### Pixel size

- Тип: целое число.
- Диапазон интерфейса: `1–128`.
- Шаг: `1`.
- Значение по умолчанию в Filters: `8`.
- Значение `1`: практически идентичная исходная детализация.
- Большие значения: более крупные пиксели и меньше мелких деталей.

### Sampling

Доступные значения:

- `center` — берётся пиксель в центре ячейки. Это самый быстрый метод, который создаёт жёсткую сетку без усреднения.
- `mean` — вычисляется среднее значение цветов внутри ячейки. Результат обычно более гладкий.
- `median` — используется медианный цвет по каналам. Метод лучше подавляет одиночные очень светлые или очень тёмные точки.
- `mode` — выбирается наиболее часто встречающийся цветовой бин. Это значение по умолчанию; оно обычно лучше всего подходит для крупных областей настоящего pixel-art.

### Pre-blur

- Диапазон: `0–10`.
- Шаг: `0.1`.
- Значение по умолчанию: `0`.
- Единица: sigma Gaussian blur.

Перед изменением изображения применяется предварительное размытие. Оно уменьшает мелкие высокочастотные детали и может снизить муар после пикселизации.

`0` сохраняет жёсткий pixel-art эффект. Более высокие значения делают границы внутри ячеек менее резкими перед выбором цвета.

### Практический пример

Для заметного pixel-art эффекта:
Pixel size: 8
Sampling: mode
Pre-blur: 0

Для фотографии с большим количеством мелких деталей можно попробовать:
Pixel size: 6–12
Sampling: mean
Pre-blur: 0.5–1.5


## 2. Palette Quantize

`Palette Quantize` заменяет каждый цвет ближайшим цветом выбранной ограниченной ретро-палитры. Внутреннее сопоставление выполняется по расстоянию в CIELAB, а не простым независимым сравнением RGB-каналов.

Фильтр предназначен для имитации ограниченных цветовых систем: PICO-8, Game Boy, C64, CGA, ZX Spectrum и других.

### Built-in palette

Текущий UI содержит следующие палитры:
- `dmg_4` — четырёхцветная зелёная палитра Game Boy DMG.
- `pico8_16` — стандартная 16-цветная палитра PICO-8. Значение по умолчанию.
- `c64_16` — 16-цветная палитра Commodore 64.
- `cga_16` — 16-цветная палитра IBM CGA.
- `zx_spectrum_15` — 15-цветная палитра ZX Spectrum.
- `ega_64` — 64-цветная EGA-палитра.
- `nes` — 55 цветов палитры Nintendo Entertainment System 2C02.
- `gray_4` — четыре градации серого.
- `mono_1bit` — чёрный и белый.
- `sweetie_16` — палитра Sweetie16.

Значение по умолчанию: `pico8_16`.

### Custom palette

Это многострочное поле для пользовательской палитры. Цвета вводятся в шестизначном hex-формате:
#0f380f, #306230, #8bac0f, #9bbc0f

Разрешены разделители:
- запятая;
- точка с запятой;
- пробел;
- перенос строки.

Поддерживаются записи с `#` и без `#`.

Трёхзначная сокращённая запись вроде `#fff` не поддерживается. В каждой записи должно быть ровно шесть hex-цифр.

Если `Custom palette` заполнено непустым значением, оно полностью заменяет выбранную `Built-in palette`. Поле `Built-in palette` в этом случае игнорируется.

Пустое поле `Custom palette` означает использование выбранной встроенной палитры.

### Dithering

В Filters компонент называется `Dithering`, а в исходном узле — `dither`. Возможные методы:

- `none` — без дизеринга;
- `bayer2` — упорядоченный Bayer-дизеринг с матрицей 2×2;
- `bayer4` — Bayer-дизеринг 4×4;
- `bayer8` — Bayer-дизеринг 8×8;
- `floyd-steinberg` — диффузия ошибки Floyd–Steinberg;
- `atkinson` — диффузия ошибки Atkinson.

`bayer2`, `bayer4` и `bayer8` работают быстро. `floyd-steinberg` и `atkinson` используют последовательный проход по строкам и могут обрабатываться заметно дольше на больших изображениях.

### Dither strength

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `1`.

Для упорядоченных методов параметр регулирует амплитуду порогового смещения. Для error-diffusion методов он регулирует величину ошибки, передаваемой соседним пикселям.

- `0`: дизеринг практически отключён.
- `1`: полная сила выбранного метода.

### Serpentine diffusion

- Значение по умолчанию: включён.

При включении направление обработки чередуется между строками. Нечётные строки обрабатываются справа налево с зеркальными смещениями ядра ошибки. Это уменьшает направленные полосы, особенно у методов Floyd–Steinberg и Atkinson.

Для Bayer-методов этот параметр существенной роли не играет, потому что они не используют последовательное распространение ошибки.

### Практические примеры

Для PICO-8:

Built-in palette: pico8_16
Custom palette: пусто
Dithering: none или bayer4
Dither strength: 1
Serpentine diffusion: включён


Для монохромного Game Boy-подобного результата:

Built-in palette: dmg_4
Dithering: floyd-steinberg
Dither strength: 0.5–1
Serpentine diffusion: включён


## 3. Color Depth

`Color Depth` выполняет чистое posterization — равномерно уменьшает количество уровней каждого RGB-канала. В отличие от `Dither`, этот фильтр не добавляет дизеринг.

### Interpret value as

Текущий компонент `Interpret value as` содержит два режима:

- `bits` — значение интерпретируется как битовая глубина;
- `levels` — значение интерпретируется как точное количество уровней.

### Bits / levels

Текущий UI использует один Slider с диапазоном `1–64`, шагом `1`, значением по умолчанию `4`.

Фактическая допустимость зависит от выбранного режима:
- в режиме `bits` разрешены значения `1–8`;
- в режиме `levels` разрешены значения `2–64`.

В режиме `bits` число уровней вычисляется так:
levels = 2 ** value

Примеры:
- `1 bit` → 2 уровня на канал;
- `2 bits` → 4 уровня;
- `4 bits` → 16 уровней;
- `8 bits` → 256 уровней.

В режиме `levels` значение используется напрямую. Например, `5` означает ровно пять уровней на канал.

Если в режиме `bits` передать значение выше `8`, движок выдаёт ошибку. Если в режиме `levels` передать значение ниже `2`, движок также выдаёт ошибку. Поэтому для режима `bits` рекомендуется вручную держать в диапазоне `1–8`, а для режима `levels` — `2–64`.


## 4. Dither

`Dither` уменьшает количество уровней цвета и одновременно может добавить дизеринг. В отличие от `Palette Quantize`, он не сопоставляет изображение с фиксированной ретро-палитрой.

### Levels per channel

- Диапазон: `2–32`.
- Шаг: `1`.
- Значение по умолчанию: `4`.

Это количество уровней квантования для каждого канала.
Малое значение создаёт более грубую постеризацию. Большое значение сохраняет больше оттенков.

### Mode

Доступные значения:
- `color` — каждый RGB-канал обрабатывается независимо;
- `mono` — изображение сначала переводится в яркость по Rec. 601: Y = 0.299 R + 0.587 G + 0.114 B

Затем один канал яркости копируется в три канала и дизерится. Результат является серым изображением.

Значение по умолчанию: `color`.

### Method

В UI доступны:
- `bayer2`;
- `bayer4`;
- `bayer8`;
- `floyd-steinberg`;
- `atkinson`.

Значение по умолчанию: `bayer4`.

Метод `none` намеренно не показывается в этом фильтре. Для постеризации без дизеринга предназначен `Color Depth`.

### Strength

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `1`.

`0` отключает влияние дизеринга, `1` использует полную силу алгоритма.

### Serpentine diffusion

- По умолчанию: включён.

Используется прежде всего методами error diffusion. Чередование направления строк уменьшает регулярные направленные артефакты.


## 5. Scanlines

`Scanlines` имитирует горизонтальные затемнённые строки CRT. В каждом периоде несколько строк затемняются, а остальные проходят без изменений.

### Pitch

- Диапазон: `1–32`.
- Шаг: `1`.
- Значение по умолчанию в текущем UI: `2`.

`Pitch` — период повторения рисунка в строках изображения.

- `1`: затемнение применяется к каждой строке;
- `2`: рисунок повторяется каждые две строки;
- большие значения: строки расположены реже.

### Thickness

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.5`.

Это доля периода, которая относится к затемнённой части.

- `0`: затемнённая область отсутствует;
- `0.5`: затемняется половина каждого периода;
- `1`: затемняется весь период.

### Strength

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.35`.

Это максимальная степень затемнения строк.
- `0`: эффект отсутствует;
- увеличение значения делает полосы темнее.

Параметр не добавляет белые линии: он уменьшает яркость выбранной части периода.

### Offset

- Диапазон: `-64 – +64`.
- Шаг: `1`.
- Значение по умолчанию: `0`.

Сдвигает фазу рисунка по вертикали. Параметр полезен, если нужно совместить scanlines с уже существующей пиксельной сеткой.

### Profile

Возможные значения:

- `hard` — резкий переход между обычной и затемнённой частью;
- `soft` — плавный raised-cosine переход.

Значение по умолчанию: `hard`.


## 6. Subpixel Mask

`Subpixel Mask` имитирует структуру люминофорной маски CRT или пиксельную структуру дисплея. Красный, зелёный и синий каналы пропускаются через разные вертикальные полосы.

Для компенсации того, что каждый канал занимает примерно одну треть площади, яркость умножается на три. Поэтому насыщенные участки могут выйти в clipping.

### Mask

Доступные типы:

- `aperture_grille` — прямые вертикальные RGB-полосы;
- `shadow_mask` — шахматно смещённая структура строк, похожая на отверстия реальной CRT-маски;
- `slot_mask` — апертурная решётка с затемнением каждой второй горизонтальной полосы.

Значение по умолчанию: `aperture_grille`.

### Pitch

- Диапазон: `1–16`.
- Шаг: `1`.
- Значение по умолчанию: `1`.

`Pitch` задаёт ширину полосы одного цветового субпикселя. Полный RGB-триадный период равен:
3 × pitch

Для согласования с Pixelize рекомендуется выбирать `Pixel size`, кратный `3 × Pitch`.

Например:
Pixelize Pixel size: 6
Subpixel Mask Pitch: 2

---

## 7. Bloom

`Bloom` извлекает яркие области, размывает их и возвращает свечение поверх исходного изображения.

Сначала рассчитывается яркость по Rec. 601, затем пиксели выше порога попадают в bright mask. Эта маска размывается Gaussian blur и смешивается с исходным изображением.

### Threshold

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.7`.

Это порог яркости, выше которого пиксели участвуют в свечении.

- низкий порог: светится больше изображения;
- высокий порог: светятся только самые яркие области;
- `1.0`: bright mask пустая, изображение возвращается без изменений.

### Radius

- Диапазон: `0.5–100`.
- Шаг: `0.5`.
- Значение по умолчанию: `8`.

Это sigma Gaussian blur в пикселях.

- маленькое значение: компактное свечение;
- большое значение: широкое мягкое свечение.

### Intensity

- Диапазон: `0–4`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.5`.

Масштабирует вклад размытой светящейся маски.

- `0`: видимый bloom отсутствует;
- значения около `0.5`: умеренное свечение;
- большие значения: сильное свечение и риск пересвета.

### Blend

Возможные значения:

- `screen` — мягкое additive-like смешивание по формуле Screen;
- `add` — прямое сложение яркости.

Значение по умолчанию: `screen`.

`screen` обычно выглядит мягче, а `add` быстрее приводит яркие области к clipping.


## 8. Chromatic Aberration

`Chromatic Aberration` смещает красный и синий каналы относительно зелёного. Зелёный канал остаётся неподвижным. Это создаёт цветные красно-синие края, похожие на оптическую аберрацию старого объектива или CRT.

### Amount

- Диапазон: `0–20`.
- Шаг: `0.5`.
- Значение по умолчанию: `2`.

Единица — пиксели смещения.

- `0`: идентичный результат;
- увеличение значения: более широкая цветная кайма.

### Mode

- `radial` — смещение зависит от положения относительно центра изображения;
- `linear` — красный и синий каналы сдвигаются горизонтально в противоположные стороны.

Значение по умолчанию: `radial`.

В linear-режиме красный канал сдвигается вправо, а синий влево. В radial-режиме цветные края ориентированы относительно центра изображения.


## 9. Curvature

`Curvature` имитирует выпуклую бочкообразную геометрию CRT-экрана. Координаты изображения ремапятся от центра, при этом центр остаётся неподвижным.

Углы могут выйти за границы исходного изображения и стать чёрными. Средние точки краёв компенсируются масштабированием, поэтому обычные чёрные полосы по центру краёв не должны появляться.

### Amount

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.15`.

Сила геометрического искажения.

- `0`: исходная геометрия;
- увеличение значения: сильнее выпирают края и темнее становятся угловые зоны.

### Edge softness

- Диапазон: `0–32`.
- Шаг: `0.5`.
- Значение по умолчанию: `0`.

Ширина плавного перехода к чёрной области по краям в пикселях.

- `0`: жёсткая граница;
- увеличение значения: мягкий fade к чёрному.

Изображение внутри области не размывается. Меняется только переход на границе искажения.


## 10. Vignette

`Vignette` затемняет изображение от центра к краям. Внутри движка вычисляется нормализованное расстояние от центра до текущей точки, а затем применяется степенная кривая.

### Strength

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.4`.

Максимальная степень затемнения.

- `0`: эффект отключён;
- увеличение значения: сильнее затемняются внешние области.

### Radius

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.8`.

Нормализованное расстояние, с которого начинается затемнение.

- маленькое значение: затемнение начинается ближе к центру;
- большое значение: центр остаётся неизменным на большей площади;
- `1`: эффект отсутствует.

### Softness

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.5`.

Управляет крутизной перехода.

Большие значения дают более резкий переход ближе к краям.

---

## 11. VHS

`VHS` применяет четыре аналоговых эффекта последовательно:

1. chroma bleed;
2. горизонтальный jitter строк;
3. additive noise;
4. вертикальная brightening band.

Каждый подэффект пропускается, если его сила равна нулю.

### Noise

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.15`.

Амплитуда Gaussian noise. Внутренняя sigma рассчитывается как:
sigma = noise × 40

`0` отключает шум.

### Chroma bleed

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.5`.

Размывает цветоразностные U/V-компоненты по горизонтали, сохраняя яркостную составляющую. Это имитирует растекание цвета на VHS.

Ширина ядра зависит от значения и ограничивается диапазоном 3–15 пикселей; ширина принудительно делается нечётной.

`0` отключает chroma bleed.

### Jitter

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.3`.

Каждая строка изображения сдвигается на случайное горизонтальное смещение. Максимальная амплитуда равна:
jitter × 6 пикселей

Используется edge replication, поэтому изображение не заворачивается по кругу на противоположный край.

### Band

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.3`.

Добавляет вертикальную полосу яркости с Gaussian-профилем. Полоса только осветляет изображение и не затемняет его. Её положение выбирается случайно.

### Seed

- Диапазон: `0–2147483647`.
- Шаг: `1`.
- Значение по умолчанию: `0`.

- `0`: новая случайность при каждом запуске;
- значение больше `0`: повторяемая случайность для заданного seed.

Это особенно важно при пакетной обработке и сравнении результатов. Если нужен одинаковый результат при повторном запуске, задайте ненулевой seed.

### Noise mode

- `mono` — одна общая шумовая плоскость для всех RGB-каналов;
- `color` — независимый шум для каждого канала.

Значение по умолчанию: `color`.

`mono` обычно выглядит как яркостный телевизионный шум. `color` создаёт цветные шумовые различия между каналами.

---

## 12. LCD Dot Matrix

`LCD Dot Matrix` накладывает на изображение регулярную структуру точек LCD-дисплея. Фильтр работает на текущем разрешении и не меняет размеры изображения.

### Cell size

- Диапазон: `1–64`.
- Шаг: `1`.
- Значение по умолчанию: `8`.

Размер стороны одной ячейки в пикселях.

Чтобы согласовать LCD-сетку с Pixelize, обычно выбирают одинаковые значения:
Pixelize Pixel size = 8
LCD Dot Matrix Cell size = 8

### Gap ratio

- Диапазон: `0–0.5`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.15`.

Доля зазора вокруг активной области ячейки.

- `0`: зазора нет, эффект практически идентичен исходному;
- увеличение значения: активные точки становятся меньше;
- `0.5`: максимальный зазор, доступный интерфейсом.

### Grid dark

- Диапазон: `0–1`.
- Шаг: `0.05`.
- Значение по умолчанию: `0.6`.

Сила затемнения неактивной области.

- `0`: зазор не затемняется;
- `1`: зазор полностью чёрный.

### Shape

- `square` — квадратная активная область внутри ячейки;
- `round` — круглая точка.

Значение по умолчанию: `square`.

---

# Как собрать цепочку

## Pixel-art цепочка

Pixelize → Palette Quantize

Рекомендуемые значения:
Pixelize:
  Pixel size: 8
  Sampling: mode
  Pre-blur: 0

Palette Quantize:
  Built-in palette: pico8_16
  Dithering: none или bayer4
  Dither strength: 1


Для более гладкой фотографии можно добавить `Pre-blur` в диапазоне `0.5–1.5`.

## CRT Monitor
Pixelize → Scanlines → Subpixel Mask → Bloom → Curvature → Vignette

Стартовый вариант:
Pixelize:
  Pixel size: 6–9
  Sampling: mode
Scanlines:
  Pitch: 2
  Thickness: 0.5
  Strength: 0.35
  Profile: hard
Subpixel Mask:
  Mask: aperture_grille
  Pitch: 1
Bloom:
  Threshold: 0.7
  Radius: 8
  Intensity: 0.5
  Blend: screen
Curvature:
  Amount: 0.15
  Edge softness: 0–4
Vignette:
  Strength: 0.4
  Radius: 0.8
  Softness: 0.5

Если маска выглядит слишком частой, увеличьте `Subpixel Mask → Pitch`. Если стали слишком заметны цветные полосы, уменьшите `Pixelize → Pixel size` или уберите Subpixel Mask из цепочки.

## VHS Tape

VHS → Bloom

Стартовые параметры VHS:
Noise: 0.15
Chroma bleed: 0.5
Jitter: 0.3
Band: 0.3
Seed: 0
Noise mode: color

Для повторяемого результата установите, например, `Seed: 12345`. Для нового варианта шума при каждом запуске оставьте `0`.

## Чёрно-белое изображение с дизерингом

Dither

Рекомендуемые параметры:

Levels per channel: 4
Mode: mono
Method: floyd-steinberg
Strength: 1
Serpentine diffusion: включён

## LCD-экран

Pixelize → LCD Dot Matrix

Для совпадающего шага сетки:

Pixelize Pixel size: 8
LCD Dot Matrix Cell size: 8
LCD Dot Matrix Gap ratio: 0.15–0.25
LCD Dot Matrix Grid dark: 0.6
LCD Dot Matrix Shape: square или round


# Входные файлы и запуск

## Несколько изображений

В блоке загрузки используется поле Drag (Select) 1 or more reference images

Можно выбрать одно или несколько изображений. После этого нажмите: Start

Для каждого входного файла выполняется одна и та же выбранная цепочка.

## ZIP-файл

При включении Upload ZIP-file интерфейс переключается на загрузку ZIP. Архив распаковывается в:

## Порядок выполнения

Перед обработкой:

1. очищается `batch_post`;
2. очищается `batch_temp`;
3. обычные изображения копируются либо ZIP распаковывается в `batch_post`;
4. расширение читает входные файлы;
5. каждый фильтр цепочки применяется последовательно;
6. результаты записываются в `batch_temp`;
7. для нескольких результатов создаётся ZIP.

---

# Результаты и имена файлов

В имени используются исходное имя без расширения, дата и время запуска, затем суффикс `filters`.

Если включено промежуточное сохранение и в цепочке несколько фильтров, то промежуточные файлы получают дополнительный суффика `chain` c номером фильтра в цепочке

## Preview, Image output и ZIP

Во время обработки результат временно показывается в `Process preview`.

Когда на выходе один файл, он может быть выдан напрямую как изображение. Когда файлов несколько, например из-за нескольких входных изображений или промежуточных сохранений, то создаёт ZIP для скачивания.

Промежуточные файлы также входят в ZIP.

При одном входном изображении и включённом промежуточном сохранении оказывается несколько файлов. После такой обработки воявляется закрытый аккордеон `compare`, в котором можно сравнить изображения до и после обработки.

