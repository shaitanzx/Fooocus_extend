## What TextMask Does
([Читать на русском языке](#что-делает-textmask))

`TextMask` is a built-in text editor in Fooocus Extend for overlaying text on a single image. It helps quickly create two related results:

- a finished image with text;
- a black-and-white mask where the text is turned into white areas on a black background.

The finished image can be used as a regular picture. The mask can be loaded into Inpaint or used with ControlNet to stylize, replace, or draw over the text area.

## What TextMask Can and Cannot Do

TextMask is designed for a simple scenario: load one picture, add text blocks to it, edit the style, and download the image or mask.

It supports up to **five text objects** simultaneously. For text, fonts, size, color, transparency, bold and italic styling, alignment, movement, scaling, rotation, deletion, and duplication are available.

## Quick First Launch

1. Open **tools → TextMask**.
2. Click **Click to upload image** or drag and drop an image into the **Drop image here** area.
3. Click **Add text**.
4. Enter the text in the **Text** field.
5. Drag the text with the mouse to the desired location on the image.
6. If necessary, change the font, size, color, transparency, boldness, italics, and alignment.
7. To get the finished picture, click **Generate image**.
8. To get the mask, click **Generate mask**.
9. Save the desired result using the **Save** button under the corresponding preview.

It is recommended to save both files first: `text-image.png` and `text-mask.png`. This way, you can use the image and mask separately in subsequent Fooocus steps.

## Uploading an Image

### **Drop image here**

This is the image drag-and-drop area. Drag the file into the dashed area. The code checks the type only when dragging: if the MIME type does not start with `image.`, the message **Please select an image!** appears.

Upon successful upload, the original image becomes a static canvas background. It is not selected as a regular object and is not edited with text tools. Text objects are placed on top of it.

### **Click to upload image**

The button opens a standard file selection dialog.

## Canvas and Display Scale

After uploading, the image becomes the canvas background. The interface adjusts the displayed size while maintaining proportions.

This is only the viewing scale. Internal object coordinates are then recalculated during export, so the text should not become small just because the image was reduced on the screen.

When resizing the TextMask window, TextMask readjusts the visible canvas. If the selected text looks unusual after resizing, deselect it and select it again before exporting.

## Adding and Selecting Text

### **Add text**

The button adds a new text object if an image is already loaded. Without an image, the message **Upload an image first!** appears.

The new object is created with the following values:

- text: `Your text`;
- font: `Montserrat`;
- size: `36 px`;
- color: white `#ffffff`;
- transparency: 100%;
- alignment: center;
- boldness: normal;
- italics: off;
- shadow: black semi-transparent shadow with a blur of 5 and an offset of 2 horizontally and vertically;
- position: approximately in the center of the image.

The text object is created as an editable object. After adding, it becomes selected.

You can add text up to a maximum of five objects. When the number of objects reaches five, **Add text** is blocked. If you delete one object, the button becomes available again.

### Selecting an Object

Click on the text on the canvas. Control elements will appear around it. The selected object can be:

- dragged with the mouse;
- scaled using corner markers;
- rotated using the rotation marker;
- edited by double-clicking or using the keyboard.

The **Edit text** panel appears only after selecting a text object. If you click on an empty space, the panel hides because the image background is not considered editable text.

### **Text**

The field becomes available after selecting a text object. Changes are applied immediately upon typing.

By default, a new object contains `Your text`. Replace this phrase with your own inscription. Standard Unicode characters are supported, so you can enter Cyrillic, Latin, hieroglyphs, and other characters, provided the selected font contains the required glyphs.

The text field changes the content of the selected object. It does not create a new object or change other inscriptions.

## Text Style

All settings below work only with the **selected text object**. If no object is selected, clicking does not change the text. After changing the color, size, font, transparency, or styling, previously created export caches are reset, so the next export is built from scratch.

### **Bold**

Toggles bold styling. On the first click, the selected text becomes `bold`, on the next it returns to `normal`.

Boldness affects the finished image. When generating a mask, the text still forcibly becomes white and opaque, but the shape of the letters, taking boldness into account, is preserved.

### **Italic**

Toggles italic styling. The button is active as long as the selected object has the `italic` style.

Italics are also taken into account in the mask: color and transparency are removed, but the geometry of the characters remains italic.

### **Align Left**, **Align Center**, **Align Right**

These buttons change the alignment property of the selected text object.

For a short single-line inscription, the difference is often unnoticeable because the object's width is close to the text's width. It becomes more noticeable after adding line breaks or when working with a text object that has a width greater than its content.

Alignment does not align the text relative to the entire image. It aligns the text within the selected text object. The position of the object itself is changed separately by dragging.

### **Popular fonts**

The **Popular fonts** block shows quick font buttons:

- `Montserrat`;
- `Pacifico`;
- `Oswald`;
- `Lobster`;
- `Caveat`;
- `Bebas Neue`.

Clicking a name immediately selects this font for the selected object and synchronizes the regular **Font** list.

Popular fonts are not a separate set with different parameters. They are quick shortcuts for selecting the same fonts that are in the list.

### **Font**

`Montserrat` is selected by default. The list contains the following options:

`Amatic SC`, `Bad Script`, `Bebas Neue`, `Caveat`, `Comfortaa`, `Cormorant`, `El Messiri`, `Gochi Hand`, `Inter`, `Jost`, `Kelly Slab`, `Lobster`, `Marck Script`, `Montserrat`, `Neucha`, `Noto Sans`, `Open Sans`, `Oswald`, `PT Sans`, `Pacifico`, `Pangolin`, `Philosopher`, `Playfair Display`, `Podkova`, `Poiret One`, `Press Start 2P`, `Raleway`, `Roboto`, `Rubik Mono One`, `Russo One`, `Ruslan Display`, `Unbounded`, `Yeseva One`.

Changing the font applies only to the selected inscription. TextMask loads these families via Google Fonts.

If the font displays as standard or does not change, wait for the font to finish loading and check the browser's access to `fonts.googleapis.com` and `fonts.gstatic.com`.

### **Font Preview**

This is a small preview field of the selected font. It shows the current text, color, transparency, boldness, and italics of the selected object.

Font Preview is not a separate export. It only helps to evaluate the style before placing or saving the text.

### **Font size**

The **Font size** slider has a range from `12` to `300` pixels and a default value of `36` pixels.

The value changes the letter size of the selected object. The number to the right of the label shows the current value. The size refers to the size of the original text object, not its screen display: if the object itself is additionally scaled by a corner marker, the final size will depend on this scale as well.

For a large heading, first select the size, then if necessary, adjust the position and overall scale of the object.

### **Text color**

The default color is white `#ffffff`. Choose another color via the palette.

The color is used in **Generate image** and in the preview. In **Generate mask**, the color is forcibly replaced with white, because the mask must show the text as a white area regardless of the chosen color.

### **Transparency**

The slider has a range from `10` to `100%`, default value `100%`.

Sets opacity: `100%` means fully visible text, while `10%` means almost transparent text.

## Object Management

### **Delete**

Deletes the selected inscription. After deletion, the **Edit text** panel hides, and the limit of five objects is freed up.

Deletion does not delete the background or cancel the image upload.

### **Duplicate**

Creates a copy of the selected inscription. The copy gets the same text, font, size, color, styling, transparency, and other properties, but is shifted approximately 20 pixels to the right and down.

Duplication is available as long as the number of text objects is less than five. After creating the fifth object, **Add text** and further duplication are blocked until one object is deleted.

### Movement, Resizing, and Rotation

These actions are performed directly on the canvas. Select the object, then:

- drag it to the desired location;
- pull the corner marker to change the scale;
- use the rotation marker to rotate the object.

Rotation and scaling apply to the specific object. When creating the image and mask, TextMask recalculates the position and scale, taking into account that the background image might have been reduced for display.

## Exporting the Finished Image

### **Export → Image with text**

This block is intended for creating an image with the original background and all texts.

### **Generate image**

Click the button after uploading an image and adding at least one text.

If the image is not uploaded, **Upload an image first!** will appear. If there are no text objects, **Add at least one text object!** will appear.

The preview appears in the **Image with text:** block.

Export includes:

- original image;
- text inscriptions;
- chosen colors;
- transparency;
- boldness and italics;
- alignment;
- position, scale, and rotation;
- shadow created when adding the text object.

The button generates only the image in the browser's memory. To get the file, click **Save** under the preview.

### Finished Image Cache

After generation, the result is saved in the internal cache. If you click **Generate image** again, TextMask may show the already created result without recalculating.

Any change in text, style, color, or position resets the cache. If you changed an object and still see the old picture, click **Generate image** again after completing the change.

### **Save** under the Image

The button saves the finished result with the name `text-image.png`.

If the image is not yet generated, the button triggers generation and after a short delay tries to save the result. It is more reliable to first click **Generate image**, wait for the preview, and then click **Save**.

## Exporting the Mask

### **Export → Mask for ControlNet or Inpaint**

This block creates a separate PNG image for subsequent use in ControlNet or Inpaint.

### **Generate mask**

Click the button after adding at least one text object.

When creating a mask, TextMask:

1. creates a temporary copy of the canvas;
2. sets a black background;
3. removes the original image from the copy;
4. leaves the text objects;
5. makes each text white;
6. sets it to full opacity;
7. removes shadows;
8. exports the result to PNG.

The final mask contains white text on a black background. The text color chosen in the editor, transparency, and shadow do not affect the brightness of the mask. The geometry of the letters, position, size, rotation, boldness, italics, and alignment are preserved.

This is important for Inpaint: the black background means the area outside the text, and the white inscription is the area you want to use as a mask.

### **Save** under the Mask

The button saves the mask with the name `text-mask.png`.

If the mask is not yet created, the button tries to launch **Generate mask**, wait a short time, and save the result. For reliability, first click **Generate mask**, wait for the preview, and then click **Save**.

## Using Results in Fooocus

### Finished Image

`text-image.png` is a regular PNG picture with text already drawn on it. It can be used as a source image in Fooocus tools, in an image editor, or as a visual example.

If you want the generation to change only the text design, do not use the finished picture as the only control: also add a mask and choose a suitable Inpaint or ControlNet mode.

### Mask

`text-mask.png` is intended for highlighting text areas. It can be used as a mask for Inpaint or fed into ControlNet along with the original image, if the chosen preprocessor and mode expect such a structure.

### Practical Scenario for Styling an Inscription

1. In TextMask, upload the original image.
2. Add text and place it where it should be.
3. Choose a suitable font and size.
4. Click **Generate mask** and save `text-mask.png`.
5. Click **Generate image** and save `text-image.png`.
6. In Fooocus, load the original or finished image into the desired mode.
7. Connect the mask according to the chosen Inpaint or ControlNet mode.
8. Check a small section and only then launch the final generation.

TextMask does not generate a new artistic style on its own. It prepares the text layer and mask, and the subsequent Fooocus mode performs the stylization.

## Clearing the Project

### **Clear project**

The button clears the project not by a separate undo operation, but by reloading the page.

Before reloading, a confirmation appears: **Clear project? All unsaved changes will be lost.** If confirmed, the current project disappears: the uploaded picture, texts, styles, and unsaved previews will be lost.

## Working with Cyrillic and Hieroglyphs

If Cyrillic displays as empty squares or is replaced by a standard font, choose a font with the required character set, such as `Noto Sans`, `Open Sans`, `Roboto`, `Montserrat`, or `PT Sans`, and ensure the font has loaded. Decorative families may have an incomplete character set.

Check the characters in TextMask itself before exporting. If they are incorrect on the canvas, they may turn out the same in the image and mask.

## Size and Quality Limitations

The size slider is limited to a range of 12–300 pixels. This is an interface limit for font size, not a mandatory limit for the final text width: the object can be additionally scaled on the canvas.

Export goes to PNG.

## Troubleshooting

- **The Add text button shows Upload an image first!** First, upload an image. TextMask does not allow creating text objects on an empty canvas.
- **Multiple files are visible in the selection field, but only one is loaded.** TextMask does not support batch processing.
- **The Edit text panel disappeared.** Select the text on the canvas again.
- **Changing settings does not change anything.** Check that the text object is selected. After changing, click the corresponding generation button again.
- **A shadow is visible in the finished image, but not in the mask.** This is a normal difference. When creating a regular picture, the shadow of the new text object is preserved.
- **The text looks brighter in the mask than in the image.** This is also expected. The color and transparency of the image are needed for visual design, while the mask must have a clear white area. TextMask forcibly sets white color and 100% opacity.
- **The font did not apply.** Check the internet connection and Google Fonts loading. Try selecting another font from the list and wait for the preview to update. If a decorative font does not contain the required characters, use a more universal font.
- **After rotation or scaling, the export looks shifted.** Deselect and select the text again, then generate the result again. Also, check the export on an original file of moderate size. TextMask recalculates the screen scale when creating a temporary canvas, so an unusual canvas state after resizing the window may affect the display.
- **Save reports that there is nothing to save.** First click **Generate image** or **Generate mask** depending on the required file. The recommended order is to generate the result, wait for the preview, then click **Save**.
- **Texts disappeared after Clear project.** This is by design. **Clear project** reloads the page and deletes all unsaved changes. Create the project again or save the files in advance.

## Recommended Workflow

For a neat result, use this order:

1. Upload an image of medium or large resolution.
2. Add one inscription.
3. Immediately replace `Your text` with the desired text.
4. Choose a font, size, and color.
5. Place the inscription and rotate it if necessary.
6. Add the remaining inscriptions, not exceeding five objects.
7. First, generate the image and check the text position.
8. Then generate the mask.
9. Save both PNGs.
10. Only after that, clear the project or load the next picture.

Do not start with five text objects. One object is easier to check, and duplication helps maintain the same style for repeated inscriptions.

## Quick Reference

- TextMask works with **one image at a time**.
- Maximum — **5 text objects**.
- New text default: `Your text`, font `Montserrat`, size `36 px`, white color, and 100% opacity.
- Font size: `12–300 px`.
- Interface transparency actually sets opacity: 100% — opaque, 10% — almost transparent.
- **Generate image** creates a PNG with colored text and its styling.
- **Generate mask** creates a PNG with white text on a black background without color and shadow.
- Files are saved as `text-image.png` and `text-mask.png`.
- Before starting a new project, save the necessary results.

## Что делает TextMask
([Read in English](#what-textmask-does))

`TextMask` — это встроенный в Fooocus Extend редактор текста поверх одного изображения. Он помогает быстро сделать два связанных результата:

- готовое изображение с текстом;
- чёрно-белую маску, где текст превращён в белые области на чёрном фоне.

Готовое изображение можно использовать как обычную картинку. Маску можно загрузить в Inpaint или использовать с ControlNet, чтобы стилизовать, заменить или дорисовать текстовую область.

## Что TextMask умеет и чего не умеет

TextMask рассчитан на простой сценарий: загрузить одну картинку, добавить на неё текстовые блоки, отредактировать стиль и скачать изображение или маску.

Он поддерживает до **пяти текстовых объектов** одновременно. Для текста доступны шрифты, размер, цвет, прозрачность, жирное и курсивное начертание, выравнивание, перемещение, масштабирование, поворот, удаление и дублирование.

## Быстрый первый запуск

1. Откройте **tools → TextMask**.
2. Нажмите **Click to upload image** или перетащите изображение в область **Drop image here**.
3. Нажмите **Add text**.
4. Введите надпись в поле **Text**.
5. Перетащите надпись мышью в нужное место на изображении.
6. При необходимости измените шрифт, размер, цвет, прозрачность, жирность, курсив и выравнивание.
7. Для получения готовой картинки нажмите **Generate image**.
8. Для получения маски нажмите **Generate mask**.
9. Сохраните нужный результат кнопкой **Save** под соответствующим предпросмотром.

Сначала рекомендуется сохранить оба файла: `text-image.png` и `text-mask.png`. Так вы сможете отдельно использовать картинку и маску в последующих шагах Fooocus.

## Загрузка изображения

### **Drop image here**

Это область перетаскивания изображения. Перетащите файл в пунктирную область. Код проверяет тип только при перетаскивании: если MIME-тип не начинается с `image.`, появляется сообщение **Please select an image!**.

При успешной загрузке исходная картинка становится неподвижным фоном холста. Она не выбирается как обычный объект и не редактируется инструментами текста. Текстовые объекты размещаются поверх неё.

### **Click to upload image**

Кнопка открывает обычный диалог выбора файла. 

## Холст и масштаб отображения

После загрузки изображение становится фоном холста. Интерфейс подбирает отображаемый размер с сохранением пропорций.

Это только масштаб просмотра. Внутренние координаты объектов затем пересчитываются при экспорте, поэтому текст не должен становиться маленьким только из-за того, что картинка была уменьшена на экране.

При изменении размера окна TextMask снова подстраивает видимый холст. Если после изменения размера окна выделенный текст выглядит необычно, снимите выделение и выберите его снова перед экспортом.

## Добавление и выбор текста

### **Add text**

Кнопка добавляет новый текстовый объект, если изображение уже загружено. Без изображения появляется сообщение **Upload an image first!**.

Новый объект создаётся со значениями:

- текст: `Your text`;
- шрифт: `Montserrat`;
- размер: `36 px`;
- цвет: белый `#ffffff`;
- прозрачность: 100%;
- выравнивание: по центру;
- жирность: обычная;
- курсив: выключен;
- тень: чёрная полупрозрачная тень с размытием 5 и смещением 2 по горизонтали и вертикали;
- положение: примерно в центре изображения.

Текстовый объект создаётся как редактируемый объект. После добавления он становится выбранным.

Добавлять текст можно максимум до пяти объектов. Когда количество объектов достигает пяти, **Add text** блокируется. Если удалить один объект, кнопка снова становится доступной.

### Выбор объекта

Щёлкните по тексту на холсте. Вокруг него появятся элементы управления. Выбранный объект можно:

- перетащить мышью;
- масштабировать за угловые маркеры;
- повернуть через маркер поворота;
- редактировать двойным щелчком или клавиатурой.

Панель **Edit text** появляется только после выбора текстового объекта. Если щёлкнуть по пустому месту, панель скрывается, потому что фон изображения не считается редактируемым текстом.

### **Text**

Поле становится доступным после выбора текстового объекта. Изменения применяются сразу при вводе.

По умолчанию новый объект содержит `Your text`. Замените эту фразу своей надписью. Поддерживаются обычные Unicode-символы, поэтому можно вводить кириллицу, латиницу, иероглифы и другие символы, если выбранный шрифт содержит нужные глифы.

Текстовое поле меняет содержимое выбранного объекта. Оно не создаёт новый объект и не меняет остальные надписи.

## Стиль текста

Все настройки ниже работают только с **выбранным текстовым объектом**. Если объект не выбран, нажатие не изменяет текст. После изменения цвета, размера, шрифта, прозрачности или начертания ранее созданные кэши экспорта сбрасываются, поэтому следующий экспорт строится заново.

### **Bold**

Переключает жирное начертание. При первом нажатии выбранный текст становится `bold`, при следующем возвращается к `normal`.

Жирность влияет на готовое изображение. При генерации маски текст всё равно принудительно становится белым и непрозрачным, но форма букв с учётом жирности сохраняется.

### **Italic**

Переключает курсивное начертание. Кнопка активна, пока выбранный объект имеет стиль `italic`.

Курсив также учитывается в маске: цвет и прозрачность убираются, но геометрия символов остаётся курсивной.

### **Align Left**, **Align Center**, **Align Right**

Эти кнопки меняют свойство выравнивания выбранного текстового объекта.

Для короткой однострочной надписи разница часто незаметна, потому что ширина объекта близка к ширине текста. Она становится заметнее после добавления переносов строк или при работе с текстовым объектом, который имеет ширину больше содержимого.

Выравнивание не выравнивает текст относительно всего изображения. Оно выравнивает текст внутри выбранного текстового объекта. Положение самого объекта меняется отдельно перетаскиванием.

### **Popular fonts**

В блоке **Popular fonts** показаны быстрые кнопки-шрифты:

- `Montserrat`;
- `Pacifico`;
- `Oswald`;
- `Lobster`;
- `Caveat`;
- `Bebas Neue`.

Щелчок по названию сразу выбирает этот шрифт для выделенного объекта и синхронизирует обычный список **Font**.

Популярные шрифты не являются отдельным набором с другими параметрами. Это быстрые ярлыки для выбора тех же шрифтов, которые есть в списке.

### **Font**

По умолчанию выбран `Montserrat`. Список содержит следующие варианты:

`Amatic SC`, `Bad Script`, `Bebas Neue`, `Caveat`, `Comfortaa`, `Cormorant`, `El Messiri`, `Gochi Hand`, `Inter`, `Jost`, `Kelly Slab`, `Lobster`, `Marck Script`, `Montserrat`, `Neucha`, `Noto Sans`, `Open Sans`, `Oswald`, `PT Sans`, `Pacifico`, `Pangolin`, `Philosopher`, `Playfair Display`, `Podkova`, `Poiret One`, `Press Start 2P`, `Raleway`, `Roboto`, `Rubik Mono One`, `Russo One`, `Ruslan Display`, `Unbounded`, `Yeseva One`.

Смена шрифта применяется только к выбранной надписи. TextMask загружает эти семейства через Google Fonts.

Если шрифт отображается как стандартный или не меняется, дождитесь окончания загрузки шрифта и проверьте доступ браузера к `fonts.googleapis.com` и `fonts.gstatic.com`.

### **Font Preview**

Это небольшое поле предпросмотра выбранного шрифта. Оно показывает текущий текст, цвет, прозрачность, жирность и курсив выбранного объекта.

Font Preview не является отдельным экспортом. Он только помогает оценить стиль перед размещением или сохранением текста.

### **Font size**

Ползунок **Font size** имеет диапазон от `12` до `300` пикселей и стандартное значение `36` пикселей.

Значение меняет размер букв выбранного объекта. Число справа от подписи показывает текущее значение. Размер относится к размеру исходного текстового объекта, а не к его экранному отображению: если сам объект дополнительно масштабирован угловым маркером, итоговый размер будет зависеть и от этого масштаба.

Для крупного заголовка сначала выберите размер, затем при необходимости подправьте положение и общий масштаб объекта.

### **Text color**

Стандартный цвет — белый `#ffffff`. Выберите другой цвет через палитру.

Цвет используется в **Generate image** и в предпросмотре. В **Generate mask** цвет принудительно заменяется на белый, потому что маска должна показывать текст как белую область независимо от выбранного цвета.

### **Transparency**

Ползунок имеет диапазон от `10` до `100%`, стандартное значение `100%`.

Задаёт непрозрачность: `100%` означает полностью видимый текст, а `10%` — почти прозрачный текст. 

## Управление объектами

### **Delete**

Удаляет выбранную надпись. После удаления панель **Edit text** скрывается, а лимит пяти объектов освобождается.

Удаление не удаляет фон и не отменяет загрузку изображения.

### **Duplicate**

Создаёт копию выбранной надписи. Копия получает те же текст, шрифт, размер, цвет, начертание, прозрачность и другие свойства, но смещается примерно на 20 пикселей вправо и вниз.

Дублирование доступно, пока количество текстовых объектов меньше пяти. После создания пятого объекта **Add text** и дальнейшее дублирование блокируются до удаления одного объекта.

### Перемещение, изменение размера и поворот

Эти действия выполняются непосредственно на холсте. Выберите объект, затем:

- перетащите его в нужное место;
- потяните за угловой маркер, чтобы изменить масштаб;
- используйте маркер поворота, чтобы повернуть объект.

Поворот и масштабирование относятся к конкретному объекту. При создании изображения и маски TextMask пересчитывает положение и масштаб с учётом того, что фоновое изображение могло быть уменьшено для отображения.

## Экспорт готового изображения

### **Export → Image with text**

Этот блок предназначен для создания изображения с исходным фоном и всеми текстами.

### **Generate image**

Нажмите кнопку после загрузки изображения и добавления хотя бы одного текста.

Если изображение не загружено, появится **Upload an image first!**. Если текстовых объектов нет, появится **Add at least one text object!**.

Предпросмотр появляется в блоке **Image with text:**.

Экспорт включает:

- исходное изображение;
- текстовые надписи;
- выбранные цвета;
- прозрачность;
- жирность и курсив;
- выравнивание;
- положение, масштаб и поворот;
- тень, созданную при добавлении текстового объекта.

Кнопка генерирует только изображение в памяти браузера. Чтобы получить файл, нажмите **Save** под предпросмотром.

### Кэш готового изображения

После генерации результат сохраняется во внутреннем кэше. Если нажать **Generate image** ещё раз, TextMask может показать уже созданный результат без повторного пересчёта.

Любое изменение текста, стиля, цвета или положения сбрасывает кэш. Если вы изменили объект и всё ещё видите старую картинку, снова нажмите **Generate image** после завершения изменения.

### **Save** под изображением

Кнопка сохраняет готовый результат с именем `text-image.png`.

Если изображение ещё не сгенерировано, кнопка запускает генерацию и через короткую задержку пытается сохранить результат. Надёжнее сначала нажать **Generate image**, дождаться предпросмотра, а затем нажать **Save**.

## Экспорт маски

### **Export → Mask for ControlNet or Inpaint**

Этот блок создаёт отдельное PNG-изображение для последующего использования в ControlNet или Inpaint.

### **Generate mask**

Нажмите кнопку после добавления хотя бы одного текстового объекта.

При создании маски TextMask:

1. создаёт временную копию холста;
2. устанавливает чёрный фон;
3. удаляет из копии исходное изображение;
4. оставляет текстовые объекты;
5. делает каждый текст белым;
6. устанавливает ему полную непрозрачность;
7. убирает тени;
8. экспортирует результат в PNG.

Итоговая маска содержит белый текст на чёрном фоне. Цвет текста, выбранный в редакторе, прозрачность и тень не влияют на яркость маски. Геометрия букв, положение, размер, поворот, жирность, курсив и выравнивание сохраняются.

Это важно для Inpaint: чёрный фон означает область вне текста, а белая надпись — область, которую вы хотите использовать как маску.

### **Save** под маской

Кнопка сохраняет маску с именем `text-mask.png`.

Если маска ещё не создана, кнопка пытается запустить **Generate mask**, подождать короткое время и сохранить результат. Для надёжности сначала нажмите **Generate mask**, дождитесь предпросмотра и затем нажмите **Save**.

## Использование результатов в Fooocus

### Готовое изображение

`text-image.png` — обычная PNG-картинка с уже нарисованным текстом. Её можно использовать как исходное изображение в инструментах Fooocus, в редакторе изображений или как визуальный пример.

Если вы хотите, чтобы генерация изменила только оформление текста, не используйте готовую картинку как единственный контроль: добавьте также маску и выберите подходящий режим Inpaint или ControlNet.

### Маска

`text-mask.png` предназначена для выделения областей текста. Её можно использовать как маску для Inpaint или подать в ControlNet вместе с исходным изображением, если выбранный препроцессор и режим ожидают такую структуру.

### Практический сценарий стилизации надписи

1. В TextMask загрузите исходное изображение.
2. Добавьте текст и разместите его там, где он должен находиться.
3. Выберите подходящий шрифт и размер.
4. Нажмите **Generate mask** и сохраните `text-mask.png`.
5. Нажмите **Generate image** и сохраните `text-image.png`.
6. В Fooocus загрузите исходное или готовое изображение в нужный режим.
7. Подключите маску согласно выбранному режиму Inpaint или ControlNet.
8. Проверьте небольшой участок и только затем запускайте финальную генерацию.

TextMask не генерирует новый художественный стиль самостоятельно. Он подготавливает текстовый слой и маску, а стилизацию выполняет последующий режим Fooocus.

## Очистка проекта

### **Clear project**

Кнопка очищает проект не отдельной операцией отмены, а перезагрузкой страницы.

Перед перезагрузкой появляется подтверждение **Clear project? All unsaved changes will be lost.** Если подтвердить, текущий проект исчезает: загруженная картинка, тексты, стили и несохранённые предпросмотры будут потеряны.

## Работа с кириллицей и иероглифами

Если кириллица отображается пустыми квадратами или заменяется стандартным шрифтом, выберите шрифт с нужным набором символов, например `Noto Sans`, `Open Sans`, `Roboto`, `Montserrat` или `PT Sans`, и убедитесь, что шрифт загрузился. Декоративные семейства могут иметь неполный набор символов.

Проверьте символы в самом TextMask до экспорта. Если они неверны на холсте, такими же они могут оказаться в изображении и маске.

## Ограничения размера и качества

Ползунок размера ограничен диапазоном 12–300 пикселей. Это ограничение интерфейса размера шрифта, а не обязательный предел итоговой ширины текста: объект можно дополнительно масштабировать на холсте.

Экспорт идёт в PNG.

## Устранение проблем

### Кнопка Add text показывает Upload an image first!

Сначала загрузите изображение. TextMask не позволяет создавать текстовые объекты на пустом холсте.

### В поле выбора видно несколько файлов, но загружен только один

TextMask не поддерживает пакетную работу.

### Панель Edit text исчезла

Выберите текст на холсте ещё раз.

### Изменение настроек ничего не меняет

Проверьте, что текстовый объект выделен. После изменения нажмите соответствующую кнопку генерации ещё раз.

### В готовом изображении видна тень, а в маске её нет

Это нормальное различие. При создании обычной картинки сохраняется тень нового текстового объекта.

### В маске текст выглядит ярче, чем в изображении

Это также ожидаемо. Цвет и прозрачность изображения нужны для визуального оформления, а маска должна иметь чёткую белую область. TextMask принудительно устанавливает белый цвет и 100% непрозрачность.

### Шрифт не применился

Проверьте интернет-соединение и загрузку Google Fonts. Попробуйте выбрать другой шрифт из списка и дождаться обновления предпросмотра. Если декоративный шрифт не содержит нужные символы, используйте более универсальный шрифт.

### После поворота или масштабирования экспорт выглядит смещённым

Снимите выделение и выберите текст заново, затем снова сгенерируйте результат. Также проверьте экспорт на исходном файле умеренного размера. TextMask пересчитывает экранный масштаб при создании временного холста, поэтому нестандартное состояние холста после изменения размера окна может влиять на отображение.

### Save сообщает, что нечего сохранять

Сначала нажмите **Generate image** или **Generate mask** в зависимости от нужного файла. Рекомендуемый порядок — сгенерировать результат, дождаться предпросмотра, затем нажать **Save**.

### После Clear project пропали тексты

Так и задумано. **Clear project** перезагружает страницу и удаляет все несохранённые изменения. Создайте проект заново или заранее сохраните файлы.

## Рекомендуемый порядок работы

Для аккуратного результата используйте такой порядок:

1. Загрузите изображение среднего или большого разрешения.
2. Добавьте одну надпись.
3. Сразу замените `Your text` на нужный текст.
4. Выберите шрифт, размер и цвет.
5. Разместите надпись и при необходимости поверните её.
6. Добавьте остальные надписи, не превышая пять объектов.
7. Сначала сгенерируйте изображение и проверьте положение текста.
8. Затем сгенерируйте маску.
9. Сохраните оба PNG.
10. Только после этого очистите проект или загружайте следующую картинку.

Не начинайте с пяти текстовых объектов. Один объект проще проверить, а дублирование помогает сохранить одинаковый стиль для повторяющихся надписей.

## Краткая памятка

- TextMask работает с **одним изображением за раз**.
- Максимум — **5 текстовых объектов**.
- Новый текст по умолчанию: `Your text`, шрифт `Montserrat`, размер `36 px`, белый цвет и 100% непрозрачность.
- Размер шрифта: `12–300 px`.
- Прозрачность интерфейса фактически задаёт opacity: 100% — непрозрачно, 10% — почти прозрачно.
- **Generate image** создаёт PNG с цветным текстом и его оформлением.
- **Generate mask** создаёт PNG с белым текстом на чёрном фоне без цвета и тени.
- Файлы сохраняются как `text-image.png` и `text-mask.png`.
- Перед новым проектом сохраните нужные результаты.
