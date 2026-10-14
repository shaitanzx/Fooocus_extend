## What Roller Does
([Читать на русском языке](#что-делает-roller))

`Roller` is a tool for **cyclically shifting a single tile image** after generation in seamless texture mode. It does not generate new pixels, does not redraw edges, and does not change the image size. Instead, pixels that "move off" one edge appear on the opposite side. Therefore, in an already seamless tile only the arrangement of details changes, while seamlessness is preserved. The goal of the module is to rearrange the tile for a more aesthetic look without losing seamlessness.

This is useful when the seam of a repeating texture is too noticeable at the edges. Shifting by half the width and/or height moves the original edges to the center of the image: that makes it easier to inspect the seams or place them in a less noticeable part of the tile.

`Roller` **does not make a regular image seamless**. If the edges of the source image did not match, after the shift the mismatch will simply move to another place.

Roller has no model loading and no separate button to start processing. The shift is performed when, after moving the slider, the user releases it.

## Interface and All Settings

Before an image is loaded, only the **`Upload file of tile`** area is visible. After loading, two sliders and a save button appear.

**`Upload file of tile`** - uploading an image.
**`Y Shift`** - cyclically shifts the image vertically.
**`X Shift`** - cyclically shifts the image horizontally.
**`Save to OUTPUT FOLDER`** - writes the currently displayed version of the image with the `roll` prefix to the OUTPUT folder.

### How `X Shift` and `Y Shift` Work

Both sliders are applied **simultaneously**, not sequentially to an already shifted preview. Therefore you can first adjust `X Shift`, then change `Y Shift`: the horizontal shift will not accumulate or "break", and the result will always be calculated as a combination of the current X and Y values applied to the original.

This gives two simple ways to restore the original appearance:

- set `X Shift = 0` and `Y Shift = 0`;
- or upload the original file again.

The sliders have no separate "Apply" button. While the slider is being dragged, the result may not update; release the mouse or finger and wait for the image to change.

### The Range Depends on the Image Size

For a **1024 × 1024 px** tile, both ranges are available from `−512` to `+512`. For a **2048 × 1024 px** file, `X Shift` will range from `−1024` to `+1024`, and `Y Shift` — from `−512` to `+512`. Roller takes the width and height from the loaded image.

## Step by Step: Processing a Single Image

1. Open **`Extention` → `tools` → `Roller`**.
2. In **`Upload file of tile`**, drag and drop or select a single texture file. After successful loading, `Y Shift`, `X Shift`, and `Save to OUTPUT FOLDER` will appear.
3. First assess which seam needs to be moved. If a vertical boundary on the left/right is noticeable, start with `X Shift`. If a horizontal boundary on the top/bottom is noticeable, start with `Y Shift`.
4. Move the slider and **release it**. The left upload area will show the new result. You can change both sliders in any order.
5. Check that you see exactly the result you need. If necessary, reset one of the parameters to zero or adjust the second one.
6. Press **`Save to OUTPUT FOLDER`** to save the result.
7. Open the results folder described below and take the generated PNG.

## Practical Scenarios

### Move All Original Seams to the Center of the Tile

For a 1024 × 1024 px image:

1. Load the file.
2. Set `X Shift = 512` and release the slider.
3. Set `Y Shift = 512` and release the slider.
4. The original left/right and top/bottom boundaries will end up roughly in the center of the new image.
5. Save the PNG if this arrangement of details looks better.

The sign does not matter when shifting by exactly half, but it is important for arbitrary values.

### Check the Vertical Seam

If you only need to inspect the transition between the left and right boundaries:

1. Leave `Y Shift = 0`.
2. Set `X Shift` close to half the width, for example `512` for a tile 1024 px wide.
3. Look at the vertical line in the center. The seam of the original left and right edges will be there.
4. If sharp transitions of color, shape, or lighting are noticeable, the source texture needs to be fixed: Roller only relocates this seam.

### Move a Detail to the Right and Up

To move an element approximately 160 px to the right and 80 px up, set `X Shift = 160`, `Y Shift = 80`. Remember: a positive `X` is to the right, a positive `Y` is up. Pixels that leave the boundary will appear on the opposite side.

### Restore the Original Appearance Without Re-uploading

Set both values to `0` and release any slider. Roller always takes the hidden original copy and applies the current pair of values, so a zero pair restores the original pixel arrangement.

## Troubleshooting

- **The slider moves, but the image does not change.** Roller updates the result after the slider is released, not while it is being dragged. Release it and wait for the update to finish.

- **`X Shift`, `Y Shift`, or the save button are not visible.** They are deliberately hidden until an image appears. Load a single file into `Upload file of tile`; if the image was cleared, upload it again.

- **After adjusting X, the Y effect disappeared, or vice versa.** Check the current values of both sliders. Roller does not accumulate shifts one after another, but recalculates the image from the original copy using the current X/Y pair. This is expected behavior.

- **The saved file did not appear in the browser.** This is normal: Roller writes the PNG to `outputs/<date>/` rather than providing a download link.

- **The image still does not look seamless after the shift.** Roller does not blend or retouch edges. The shift helps you see where the seam falls in the center, but the seam itself must be fixed in the source texture or in another editor.

## Что делает Roller
([Read in English](#what-roller-does))

`Roller` — это инструмент для **циклического сдвига одного изображения-тайла** после генерации в режиме бесшовных текстур. Он не генерирует новые пиксели, не дорисовывает края и не меняет размер изображения. Вместо этого пиксели, которые «уезжают» за один край, появляются с противоположной стороны. Поэтому у уже бесшовного тайла меняется только расположение деталей, а бесшовность сохраняется. Цель модуля -  перестановка тайла для более эстетичного вида без потери бесшовности.

Это полезно, когда стык повторяющейся текстуры слишком заметен по краям. Сдвиг на половину ширины и/или высоты переносит исходные края в центр изображения: так проще проверить стыки или разместить их в менее заметной части тайла.

`Roller` **не делает обычную картинку бесшовной**. Если края исходного изображения не совпадали, после сдвига несоответствие просто переместится в другое место.

В Roller нет загрузки моделей и нет отдельной кнопки запуска обработки. Сдвиг выполняется, когда после перемещения ползунка пользователь отпускает его.

## Интерфейс и все настройки

До загрузки картинки видна только область **`Upload file of tile`**. После загрузки появляются два ползунка и кнопка сохранения.

**`Upload file of tile`** - загрузка изображения.
**`Y Shift`** - Циклически сдвигает изображение по вертикали.
**`X Shift`** - Циклически сдвигает изображение по горизонтали.
**`Save to OUTPUT FOLDER`** - Записывает текущую показанную версию изображения с префиксом `roll` в папку OUTPUT

### Как работают `X Shift` и `Y Shift`

Оба ползунка применяются **одновременно**, а не последовательно к уже сдвинутому предпросмотру. Поэтому можно сначала подобрать `X Shift`, затем изменить `Y Shift`: горизонтальный сдвиг не накопится и не «сломается», а итог всегда будет рассчитан как комбинация текущих X и Y от оригинала.

Это даёт два простых способа вернуть исходный вид:

- установите `X Shift = 0` и `Y Shift = 0`;
- либо снова загрузите исходный файл.

У ползунков нет отдельной кнопки «Применить». Пока ползунок перетаскивается, результат может не обновляться; отпустите мышь или палец и дождитесь изменения изображения.

### Диапазон зависит от размера изображения

Для тайла **1024 × 1024 px** доступны оба диапазона от `−512` до `+512`. Для файла **2048 × 1024 px** `X Shift` будет от `−1024` до `+1024`, а `Y Shift` — от `−512` до `+512`. Roller берёт ширину и высоту из загруженного изображения.

## Пошагово: обработка одного изображения

1. Откройте **`Extention` → `tools` → `Roller`**.
2. В **`Upload file of tile`** перетащите или выберите один файл текстуры. После успешной загрузки появятся `Y Shift`, `X Shift` и `Save to OUTPUT FOLDER`.
3. Сначала оцените, какой стык нужно перенести. Если заметна вертикальная граница слева/справа, начните с `X Shift`. Если заметна горизонтальная граница сверху/снизу, начните с `Y Shift`.
4. Передвиньте ползунок и **отпустите его**. Левая область загрузки покажет новый результат. Можно менять оба ползунка в любом порядке.
5. Проверьте, что видите именно нужный результат. При необходимости верните один из параметров к нулю или отрегулируйте второй.
6. Нажмите **`Save to OUTPUT FOLDER`** для сохранения результата
7. Откройте папку результатов, описанную ниже, и возьмите созданный PNG.

## Практические сценарии

### Перенести все исходные стыки в центр тайла

Для изображения 1024 × 1024 px:

1. Загрузите файл.
2. Установите `X Shift = 512` и отпустите ползунок.
3. Установите `Y Shift = 512` и отпустите ползунок.
4. Исходные границы слева/справа и сверху/снизу окажутся примерно в центре нового изображения.
5. Сохраните PNG, если такое расположение деталей выглядит лучше.

Знак при сдвиге ровно наполовину не влияет на расположение шва в центре, но он важен для произвольных значений.

### Проверить вертикальный стык

Если нужен только контроль перехода между левой и правой границей:

1. Оставьте `Y Shift = 0`.
2. Установите `X Shift` близко к половине ширины, например `512` для тайла шириной 1024 px.
3. Посмотрите на вертикальную линию в центре. Там окажется стык исходных левого и правого краёв.
4. Если заметны резкие переходы цвета, формы или света, исправлять нужно исходную текстуру: Roller лишь переносит этот шов.

### Переместить деталь вправо и вверх

Чтобы переместить элемент примерно на 160 px вправо и 80 px вверх, задайте `X Shift = 160`, `Y Shift = 80`. Помните: положительный `X` — вправо, положительный `Y` — вверх. Пиксели, покинувшие границу, появятся на противоположной стороне.

### Вернуть оригинальный вид без повторной загрузки

Установите оба значения в `0` и отпустите любой ползунок. Roller каждый раз берёт скрытую исходную копию и применяет текущую пару значений, поэтому нулевая пара возвращает исходное расположение пикселей.

## Устранение неполадок

- **Ползунок двигается, а картинка не меняется.** Roller обновляет результат после отпускания ползунка, а не во время перетаскивания. Отпустите его и подождите завершения обновления.

- **Не видны `X Shift`, `Y Shift` или кнопка сохранения.** Они специально скрыты до появления изображения. Загрузите один файл в `Upload file of tile`; если картинка была очищена, загрузите её заново.

- **После настройки X исчез эффект Y, или наоборот.** Проверьте текущие значения обоих ползунков. Roller не накапливает сдвиги по очереди, а заново вычисляет изображение из исходной копии с текущей парой X/Y. Это ожидаемое поведение.

- **Сохранённый файл не появился в браузере.** Это нормально: Roller пишет PNG в `outputs/<дата>/`, а не выдаёт ссылку на скачивание.

- **Изображение после сдвига всё равно не выглядит бесшовным.** Roller не смешивает и не ретуширует края. Сдвиг помогает увидеть место стыка в центре, но сам стык нужно исправлять в исходной текстуре или другом редакторе.