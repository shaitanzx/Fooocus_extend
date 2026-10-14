## What Cleaner Does
([Читать на русском языке](#что-делает-cleaner))

`Cleaner` is designed to remove small unwanted objects from images and videos. The user marks the object with a brush on the first image or the first frame of a video, and the **LaMa** model restores the marked area based on the surrounding content. A ready-made mask can also be uploaded. The drawn and uploaded masks are merged, meaning the ready-made mask does not replace the drawn one but adds to it.

Inside, there are two tabs:

- **Image** — processing one or more images, including a ZIP archive;
- **Video** — processing one or more video files.

## What to Prepare

For the first launch, take a copy of an image where the unwanted object is small and surrounded by a fairly uniform background: for example, a wire on a wall, a spot, a small logo, or a foreign object. Cleaner is not a universal replacement for Photoshop: the larger the area to be removed and the more complex the background, the higher the chance of noticeable artifacts.

On the first launch, Cleaner downloads the model. Because of this, the first processing may take longer due to downloading and loading the model.

## Quick First Launch (Image)

1. Open **Cleaner → Image**.
2. Upload a single file to the **Drag (Select) 1 or more reference images** field.
3. After uploading, the **First Image** field with a brush editor will appear.
4. Paint over the object you want to remove with a white brush. Paint over the object itself and, if necessary, a small margin around its boundary.
5. Leave **Add Mask** disabled if you do not have a ready-made mask.
6. Click **Clean Up**.
7. Wait for the processing to finish and download the result from **Output image**.

The white area in the editor becomes the deletion mask. The result is saved with the suffix `_clean` appended to the original filename.

## Image Mode Interface

### **Drag (Select) 1 or more reference images**

This field accepts one or more image files. The "1 or more" value means you can select a batch of files in the dialog or drag and drop multiple images.

When exactly one file is selected, the interface hides the list and shows the **Reference image** field. This is not a separate second file; it is a convenient display of the single selected image. When two or more files are selected, the list remains visible, and the handler later iterates through all files in sorted order.

For a standard launch, select one file. For a batch, use multiple files or a ZIP, but first check the mask on a single image: the same mask will be applied to the entire batch.

### **Reference image**

This field appears after selecting exactly one image. It displays the first and only selected file.

### **Upload ZIP-file**

Standard value: disabled.

When the checkbox is disabled, multiple separate image uploads are used. When enabled:

- the **Upload a ZIP file** field becomes visible;
- the **Drag (Select) 1 or more reference images** list is hidden;
- the **Reference image** field is hidden.

Select a single ZIP archive containing images, then upload a mask on the first image and click **Clean Up**. Before processing, the archive is unpacked into a temporary folder.

This field is active only when **Upload ZIP-file** is enabled. It accepts a single file with the `.zip` extension.

It is important to consider the actual behavior of the current implementation:

- only graphic files located directly in the root of the archive are processed;
- the first element of the ZIP is used for preview without filtering by extension. If the first element happens to be a folder or a non-image file, the preview may fail;
- it is better to place images directly in the root of the archive, without subfolders or unnecessary files;
- filenames should ideally be simple and unique.

### **First Image**

This is the editor for the first image with a brush. It appears after uploading regular files or a ZIP.

You can draw a mask here. In Image mode, the mask is created only on the first image and then applied to every file in the batch. If the images have different sizes or the object is in different locations, one mask will not fit all files. For such files, it is better to run them in separate groups or separate launches, drawing the mask anew each time.

### **Add Mask**

Standard value: disabled. When enabled, the **Add mask** field appears.

This checkbox is useful if you already have a ready-made mask. It does not disable drawing in **First Image**. Upon launch, the code merges the drawn mask and the one uploaded via `Add mask` using a pixel-wise maximum. Therefore, the deletion area equals the union of the two masks.

To use *only* the ready-made mask, do not draw additional areas in **First Image**. If a mask is uploaded and you accidentally painted over something extra on the first image, that extra area will also be processed.

The ready-made mask must match the source image in size and position. If the dimensions do not match, merging the arrays may result in an error or an incorrect outcome.

### **Clean Up**

This button starts the image processing. At the beginning, it becomes inactive, the previous ZIP and previous single result are hidden, then temporary folders are cleared, input files are loaded, processing is launched, and finally, the output is generated.

During operation:

- **Process preview** shows the image currently being processed;
- Gradio informational messages show the element number and filename.

After completion, the button becomes active again, and the results gallery is displayed.

### **Process preview**

This is a temporary preview of the currently processing image. It appears during batch processing and is hidden after completion.

This is not a separate result or a saved mask. You should download from **Clean result** or **Output image** after completion.

### **Clean result**

This is a gallery of processed images. It appears after processing is complete. A separate result is created for each file, and the built-in download button allows saving images one by one.

If there are multiple files, a combined archive appears next to the gallery in **Download a ZIP file**.

### **Download a ZIP file**

In Image mode, this field is used only when more than one image has been processed. For a single result, **Output image** is shown instead of a ZIP.

### **Output image**

This field appears for a single processed image. You can download the single final file here.

If two or more files are processed, this field is hidden, and the output switches to **Download a ZIP file**.

## How the Mask Works

Cleaner uses a mask to indicate which area needs to be restored. The mask is not an image that will be inserted in place of the object. It only tells the model which pixels to consider as deletable.

When **Add Mask** is disabled, the mask drawn in **First Image** is used. When **Add Mask** is enabled, the code converts both masks into RGB arrays and merges them via a pixel-wise maximum. Therefore, the white area of either mask becomes the deletion area.

### Limitation of the Shared Mask

A single mask drawn on the first image is used for all files in the batch. The code does not create a separate mask for each image, nor does it adjust the deletion area coordinates to the file size.

Therefore, batch processing is well-suited for a series of frames or photographs taken with an identical composition. For scattered photographs with objects in different locations, the result may be incorrect.

Practical rules:

- paint over the entire object, not just its thin central line;
- add a small margin around the edge if the object has a shadow or contour;
- do not paint over a large portion of the background unless necessary;
- for very thin objects, increase the mask slightly, but check if the model starts damaging the surrounding background;
- do not use one mask for different images if the object is located elsewhere.

## Processing a Single Image

1. Ensure **Upload ZIP-file** is disabled.
2. Upload a single file.
3. Draw the mask in **First Image**.
4. If necessary, enable **Add Mask** and upload a ready-made mask.
5. Click **Clean Up**.
6. Wait for **Output image** to appear.
7. Download the file before the next launch.

The result receives a name following the scheme `original_name_clean.extension`. For example, `photo.jpg` becomes `photo_clean.jpg`. The code opens the source, converts it to RGB before inference, and then saves the result with the original extension.

Important consequence: if the source was a JPEG, the result is also saved as a JPEG. Transparency is not used here; Cleaner restores the image area, it does not remove the background.

## Processing Multiple Images

1. Leave **Upload ZIP-file** disabled.
2. Select multiple images in **Drag (Select) 1 or more reference images**.
3. After uploading, check the first image in **First Image**.
4. Draw the mask on the first image.
5. Click **Clean Up**.
6. Observe the **Process preview** and the messages `Cleaner Batch: start element generation ...`.
7. After completion, download individual files from **Clean result** or the combined archive from **Download a ZIP file**.

Files are copied to a temporary folder with their original names, then processed in sorted order by name. A file with the `_clean` suffix is created for each.

## Processing a ZIP Archive of Images

1. Prepare the ZIP.
2. Place all images directly in the root of the archive.
3. Remove subfolders, text files, hidden service records, and other extraneous elements.
4. Enable **Upload ZIP-file**.
5. Upload the archive to **Upload a ZIP file**.
6. Ensure the first image of the archive correctly appears in **First Image**.
7. Draw the mask on this image.
8. If necessary, enable **Add Mask** and upload an additional mask.
9. Click **Clean Up**.
10. Download individual results from the gallery or a single combined ZIP.

## Video Mode Interface

### **Drag (Select) 1 or more video files**

This field accepts one or more video files.

### **First video**

After uploading a video, the **First video** window appears. It shows the first selected video file. This window is intended for preview, not for selecting a specific frame from the entire sequence.

### **First Frame**

After uploading the video, the code reads the first frame of the first video and displays it in the brush editor. The user draws the mask on this frame.

The mask from **First Frame** is then applied to all frames of all uploaded videos. It does not change from frame to frame and does not track moving objects.

### **Add Mask** and **Add mask** in Video

These work the same as in Image mode. When enabled, the **Add mask** field appears, and the uploaded mask is merged with the drawn mask via pixel-wise maximum.

The final mask is built once before video processing and then used for all frames. The uploaded mask must match the size and position of the area in the first frame.

### **Clean Up** in Video

The button becomes visible after uploading a video. Clicking it starts the processing of all selected video files sequentially.

During operation, the **Process** field shows progress by frames and files, for example, `Processed 25 of 300 (1 of 2)`. After completion, a list of individual processed videos appears in **Download a videofiles**.

### **Download a videofiles**

Video results are provided as separate files. A combined ZIP archive for videos is not created.

Each file receives a name following the scheme `original_name_clean.mp4`. Even if the source file had a different container, the OpenCV Writer in the current code creates the result with the `.mp4` extension and `mp4v` codec.

## Processing a Single Video

1. Open **Cleaner → Video**.
2. Upload a single video file.
3. Wait for **First Frame** to appear.
4. Paint over the object on the first frame.
5. Leave **Add Mask** disabled or add a ready-made mask.
6. Click **Clean Up**.
7. Monitor the progress.
8. Download the file from **Download a videofiles**.

The video is read via OpenCV. The original FPS, width, height, and frame count are preserved as much as the reading of the source file and the creation of the output video allow.

## Processing Multiple Videos

1. Select multiple files in **Drag (Select) 1 or more video files**.
2. Check the first frame of the first video.
3. Draw a single mask.
4. Ensure that the same object is located in approximately the same place and has the same size in all videos.
5. Click **Clean Up**.
6. Wait for all files to finish processing.
7. Download each video file separately.

One mask is applied to all frames of all videos. This is convenient for clips with identical resolution and composition. For different scenes, launch the videos separately.

## How the Video Mask Works

Cleaner does not perform object tracking. The mask does not follow a moving object, person, or camera. If the object moves, the fixed mask will start deleting the wrong area.

This mode is best suited for a static spot, a logo in one place, or a stationary object. For a moving object, prepare frames with identical composition or use a specialized editor with mask animation.

With corrupted or non-standard video, processing may end in an error instead of gracefully skipping the last frame.

## What Happens to Audio and Video Format

The video is written via OpenCV `VideoWriter` with the `mp4v` codec. The Cleaner code does not have a separate step to transfer the audio track, subtitles, or metadata from the source file. Therefore, the processed MP4 does not contain the original sound. If audio is important, save the original and, after processing, merge the audio with the new video in an external video editor.

## Practical Scenarios

### Remove a small logo from a photograph

Upload a single file in Image mode, paint over the logo in **First Image**, add a small margin around the edges, and click **Clean Up**. If a blurry halo appears, try reducing the mask area first, rather than running it again with an even larger area.

### Remove an object from a series of frames

Select multiple images if they have the same resolution and the object is in the same position. Draw the mask on the first frame. Check the result of several files in the gallery, because the exact same mask is applied to the entire series.

### Process a folder via ZIP

Assemble the ZIP so that all images lie directly in the root. Do not add subfolders. Enable **Upload ZIP-file**, upload the archive, and use the mask of the first image. After processing, download the combined archive.

### Remove a stationary object from a video

Upload the video, paint over the object on **First Frame**, and start processing. Use this method for an object that does not move relative to the frame. If the object moves, the fixed mask will not follow it.

## Troubleshooting

### First Image does not appear after uploading an image

Check that a regular image file is uploaded, not a ZIP (unless the checkbox is enabled). For a ZIP, enable **Upload ZIP-file** and upload exactly one `.zip`. If the ZIP starts with a folder or a non-image file, the preview may not open because the code takes the first element of the archive without filtering.

### Only the first file from the ZIP was processed

Check the archive structure. Files must be located in its root.

### Different images have the wrong deletion area

This is an expected limitation of the shared mask. In Image mode, the mask is drawn only on the first image and used for all files. Group images with identical compositions or process them one by one.

### Ready-made mask does not work

Check that the **Add Mask** checkbox is enabled and the file is uploaded to the appearing **Add mask** field. If the drawn mask on the first image also contains an area, it will be merged with the ready-made one. Ensure the dimensions of the ready-made mask match the source image.

### Result is empty or the area was not restored

Check that the mask actually paints over the object. Try processing a small test file and painting the area slightly wider. If the model did not load, check the Fooocus console and ensure CUDA is available and `big-lama.safetensors` has been downloaded.

### Background around the object looks unnatural

LaMa restores the area based on the surrounding context. If the mask is too large, the model is forced to invent more details. Reduce the mask to the object and a small necessary margin. For repeating patterns or text, it is better to process in small sections.

### Video is processing, but the object moves in the result

Cleaner uses a single fixed mask for all frames. It does not track the object. For a moving object, separate mask tracking or processing short fragments with different masks is required.

### No sound in the processed video

The code writes a new video stream via OpenCV and does not copy the audio. Use an external video editor or FFmpeg to add the audio track from the original to the processed video.

## What to Download After Processing

For a single image, download the file from **Output image**. For multiple images, download individual items from **Clean result** or the combined **Download a ZIP file**.

For one or more videos, download the separate MP4 files from **Download a videofiles**. The interface does not create a video archive.

Cleaner results are not saved in the standard Fooocus History Log.

## Short Safe Algorithm

1. First, process a single copy of a single file.
2. Draw the mask only on the object and a small margin around it.
3. Check the result at 100% scale.
4. For a batch, use only files with identical composition.
5. For a ZIP, place images in the root of the archive.
6. For video, ensure the object is stationary relative to the frame.
7. Download results before the next launch.
8. For important videos, save the original audio separately.

## Что делает Cleaner
([Read in English](#what-cleaner-does))

`Cleaner` предназначен для удаления небольших лишних объектов с изображений и видео. Пользователь отмечает объект кистью на первом изображении или на первом кадре видео, а модель **LaMa** восстанавливает отмеченную область по окружающему содержимому. Можно также загрузить готовую маску. Нарисованная и загруженная маски объединяются, поэтому готовая маска не заменяет нарисованную, а добавляет к ней области.

Внутри есть две вкладки:

- **Image** — обработка одного или нескольких изображений, включая ZIP-архив;
- **Video** — обработка одного или нескольких видеофайлов.

## Что нужно подготовить

Для первого запуска возьмите копию изображения, где лишний объект небольшой и окружён достаточно однородным фоном: например, провод на стене, пятно, маленький логотип или посторонний предмет. Cleaner не является универсальной заменой Photoshop: чем больше удаляемая область и чем сложнее фон, тем выше вероятность заметных артефактов.

При первом запуске Cleaner загружает модель. Из-за этого первая обработка может быть дольше из-за скачивания и загрузки модели.

## Быстрый первый запуск изображения

1. Откройте **Cleaner → Image**.
2. Загрузите один файл в поле **Drag (Select) 1 or more reference images**.
3. После загрузки появится поле **First Image** с редактором кисти.
4. Закрасьте белой кистью объект, который нужно удалить. Закрашивайте сам объект и, если требуется, небольшой запас вокруг его границы.
5. Оставьте **Add Mask** выключенным, если готовой маски нет.
6. Нажмите **Clean Up**.
7. Дождитесь обработки и скачайте результат из **Output image**.

Белая область редактора становится маской удаления. Результат сохраняется с суффиксом `_clean`, что и исходный файл.

## Интерфейс режима Image

### **Drag (Select) 1 or more reference images**

Это поле принимает один или несколько файлов изображений. Значение «1 or more» означает, что можно выбрать пакет файлов в диалоге или перетащить несколько изображений.

При выборе ровно одного файла интерфейс скрывает список и показывает поле **Reference image**. Это не отдельный второй файл: это удобное отображение выбранного единственного изображения. При выборе двух и более файлов список остаётся видимым, а обработчик позже перебирает все файлы в сортированном порядке.

Для обычного запуска выберите один файл. Для пакета используйте несколько файлов или ZIP, но сначала проверьте маску на одном изображении: одна и та же маска будет применена ко всему пакету.

### **Reference image**

Поле появляется после выбора ровно одного изображения. Оно показывает первый и единственный выбранный файл.

### **Upload ZIP-file**

Стандартное значение — выключено.

При выключенной галочке используется множественная загрузка отдельных изображений. При включении:

- поле **Upload a ZIP file** становится видимым;
- список **Drag (Select) 1 or more reference images** скрывается;
- поле **Reference image** скрывается.

Выберите один ZIP-архив, содержащий изображения, затем загрузите маску на первом изображении и нажмите **Clean Up**. Перед обработкой архив распаковывается во временную папку.

Это поле активно только при включённом **Upload ZIP-file**. Оно принимает один файл с расширением `.zip`.

Важно учитывать фактическое поведение текущей реализации:

- в обработку попадают только графические файлы непосредственно в корне архива;
- первый элемент ZIP используется для предпросмотра без фильтрации по расширению. Если первым элементом окажется папка или посторонний файл, предпросмотр может завершиться ошибкой;
- лучше класть изображения прямо в корень архива, без подпапок и без лишних файлов;
- имена файлов желательно делать простыми и уникальными.

### **First Image**

Это редактор первого изображения с кистью. Он появляется после загрузки обычных файлов или ZIP.

В нём можно рисовать маску. В режиме Image маска создаётся только на первом изображении и затем применяется к каждому файлу пакета. Если изображения имеют разный размер или объект находится в разных местах, одна маска не подойдёт всем файлам. Для таких файлов лучше запускать их отдельными группами или отдельными запусками, каждый раз рисуя маску заново.

### **Add Mask**

Стандартное значение — выключено. При включении появляется поле **Add mask**.

Галочка полезна, если у вас уже есть готовая маска. Она не отключает рисование в **First Image**. При запуске код объединяет нарисованную маску и загруженную через `Add mask` по принципу максимального значения пикселей. Поэтому область удаления равна объединению двух масок.

Чтобы использовать только готовую маску, не рисуйте дополнительные области в **First Image**. Если маска загружена и вы случайно закрасили лишнее на первом изображении, лишняя область тоже попадёт в обработку.

Готовая маска должна совпадать с исходным изображением по размеру и положению. Если размеры не совпадают, объединение массивов может завершиться ошибкой или дать некорректный результат.

### **Clean Up**

Кнопка запускает обработку изображений. В начале она становится неактивной, скрываются предыдущий ZIP и предыдущий одиночный результат, затем очищаются временные папки, загружаются входные файлы, запускается обработка и после неё формируется итоговая выдача.

Во время работы:

- **Process preview** показывает изображение, которое обрабатывается сейчас;
- в информационных сообщениях Gradio показывается номер элемента и имя файла.

После завершения кнопка снова становится активной, а галерея результатов показывается.

### **Process preview**

Это временный просмотр текущего обрабатываемого изображения. Он появляется во время пакетной обработки и скрывается после её завершения.

Это не отдельный результат и не сохранённая маска. Скачивать нужно из **Clean result** или **Output image** после завершения.

### **Clean result**

Это галерея обработанных изображений. Она появляется после завершения обработки. Для каждого файла создаётся отдельный результат, а встроенная кнопка скачивания позволяет сохранить изображения по одному.

Если файлов несколько, рядом с галереей появляется общий архив в **Download a ZIP file**.

### **Download a ZIP file**

В режиме Image поле используется только тогда, когда обработано больше одного изображения. Для одного результата вместо ZIP показывается **Output image**.

### **Output image**

Это поле появляется для единственного обработанного изображения. В нём можно скачать один итоговый файл.

Если обработано два и более файла, поле скрывается, а выдача переключается на **Download a ZIP file**.

## Как работает маска

Cleaner использует маску, чтобы указать, какую область нужно восстановить. Маска не является изображением, которое будет вставлено вместо объекта. Она только говорит модели, какие пиксели считать удаляемыми.

При выключенном **Add Mask** используется маска, нарисованная в **First Image**. При включённом **Add Mask** код преобразует обе маски в RGB-массивы и объединяет их через поэлементный максимум. Поэтому белая область любой из двух масок становится областью удаления.

### Ограничение общей маски

Одна маска, нарисованная на первом изображении, используется для всех файлов пакета. Код не создаёт отдельную маску для каждого изображения и не подстраивает координаты области удаления под размер файла.

Поэтому пакетная обработка хорошо подходит для серии кадров или фотографий, сделанных с одинаковой компоновкой. Для разрозненных фотографий с объектами в разных местах результат может быть неправильным.

Практические правила:

- закрашивайте весь объект, а не только его тонкую центральную линию;
- добавляйте небольшой запас по краю, если объект имеет тень или контур;
- не закрашивайте большую часть фона без необходимости;
- для очень тонких объектов увеличьте маску немного, но проверьте, не начинает ли модель портить окружающий фон;
- не используйте одну маску для разных изображений, если объект расположен в другом месте.

## Обработка одного изображения

1. Убедитесь, что **Upload ZIP-file** выключен.
2. Загрузите один файл.
3. В **First Image** нарисуйте маску.
4. При необходимости включите **Add Mask** и загрузите готовую маску.
5. Нажмите **Clean Up**.
6. Дождитесь, пока появится **Output image**.
7. Скачайте файл до следующего запуска.

Результат получает имя по схеме `исходное_имя_clean.расширение`. Например, `photo.jpg` превращается в `photo_clean.jpg`. Код открывает исходник, переводит его в RGB перед инференсом, а затем сохраняет результат с исходным расширением.

Важное следствие: если исходник был JPEG, результат также сохраняется как JPEG. Прозрачность здесь не используется; Cleaner восстанавливает область изображения, а не удаляет фон.

## Обработка нескольких изображений

1. Оставьте **Upload ZIP-file** выключенным.
2. Выберите несколько изображений в **Drag (Select) 1 or more reference images**.
3. После загрузки проверьте первое изображение в **First Image**.
4. Нарисуйте маску на первом изображении.
5. Нажмите **Clean Up**.
6. Наблюдайте за **Process preview** и сообщениями `Cleaner Batch: start element generation ...`.
7. После завершения скачайте отдельные файлы из **Clean result** или общий архив из **Download a ZIP file**.

Файлы копируются во временную папку с исходными именами, затем обрабатываются в сортированном порядке по имени. Для каждого создаётся файл с суффиксом `_clean`.

## Обработка ZIP-архива изображений

1. Подготовьте ZIP.
2. Положите все изображения непосредственно в корень архива.
3. Уберите подпапки, текстовые файлы, скрытые служебные записи и другие посторонние элементы.
4. Включите **Upload ZIP-file**.
5. Загрузите архив в **Upload a ZIP file**.
6. Убедитесь, что первое изображение архива корректно появилось в **First Image**.
7. Нарисуйте маску на этом изображении.
8. При необходимости включите **Add Mask** и загрузите дополнительную маску.
9. Нажмите **Clean Up**.
10. Скачайте отдельные результаты из галереи или один общий ZIP.

## Интерфейс режима Video

### **Drag (Select) 1 or more video files**

Это поле принимает один или несколько видеофайлов.

### **First video**

После загрузки видео появляется окно **First video**. Оно показывает первый выбранный видеофайл. Это окно предназначено для предварительного просмотра, а не для выбора отдельного кадра из всей последовательности.

### **First Frame**

После загрузки видео код читает первый кадр первого видео и показывает его в редакторе кисти. Пользователь рисует маску на этом кадре.

Маска из **First Frame** затем применяется ко всем кадрам всех загруженных видео. Она не меняется от кадра к кадру и не отслеживает движущийся объект.

### **Add Mask** и **Add mask** в Video

Работают так же, как в Image. При включении появляется поле **Add mask**, а загруженная маска объединяется с нарисованной маской по максимуму пикселей.

Итоговая маска строится один раз до обработки видео и затем используется для всех кадров. Загруженная маска должна соответствовать размеру и расположению области первого кадра.

### **Clean Up** в Video

Кнопка становится видимой после загрузки видео. Нажатие запускает обработку всех выбранных видеофайлов по очереди.

Во время работы поле **Process** показывает прогресс по кадрам и файлам, например `Processed 25 of 300 (1 of 2)`. После завершения в **Download a videofiles** появляется список отдельных обработанных видео.

### **Download a videofiles**

Результаты видео выдаются отдельными файлами. Общий ZIP-архив для видео не создаётся.

Каждый файл получает имя по схеме `исходное_имя_clean.mp4`. Даже если исходный файл имел другой контейнер, Writer OpenCV в текущем коде создаёт результат с расширением `.mp4` и кодеком `mp4v`.

## Обработка одного видео

1. Откройте **Cleaner → Video**.
2. Загрузите один видеофайл.
3. Дождитесь появления **First Frame**.
4. На первом кадре закрасьте объект.
5. Оставьте **Add Mask** выключенным или добавьте готовую маску.
6. Нажмите **Clean Up**.
7. Следите за прогрессом.
8. Скачайте файл из **Download a videofiles**.

Видео читается через OpenCV. Сохраняются исходные FPS, ширина, высота и количество кадров, насколько это позволяет чтение исходного файла и создание выходного видео.

## Обработка нескольких видео

1. Выберите несколько файлов в **Drag (Select) 1 or more video files**.
2. Проверьте первый кадр первого видео.
3. Нарисуйте одну маску.
4. Убедитесь, что один и тот же объект находится примерно в одном месте и имеет одинаковый размер во всех видео.
5. Нажмите **Clean Up**.
6. Дождитесь обработки всех файлов.
7. Скачайте каждый видеофайл отдельно.

Одна маска применяется ко всем кадрам всех видео. Это удобно для одинаковых по разрешению и композиции клипов. Для разных сцен запускайте видео отдельно.

## Как работает видео-маска

Cleaner не выполняет трекинг объекта. Маска не следует за движущимся предметом, человеком или камерой. Если объект перемещается, фиксированная маска начнёт удалять не ту область.

Для статичного пятна, логотипа в одном месте или неподвижного объекта режим подходит лучше. Для движущегося объекта подготовьте кадры с одинаковой компоновкой либо используйте специализированный редактор с анимацией маски.

При повреждённом или нестандартном видео обработка может завершиться ошибкой вместо аккуратного пропуска последнего кадра.

## Что происходит с аудио и форматом видео

Видео записывается через OpenCV `VideoWriter` с кодеком `mp4v`. В коде Cleaner нет отдельного шага переноса аудиодорожки, субтитров или метаданных из исходного файла. Поэтому обработанный MP4 не содержит исходный звук. Если звук важен, сохраните оригинал и после обработки объедините аудио с новым видео во внешнем видеоредакторе. 

## Практические сценарии

### Удалить маленький логотип с фотографии

Загрузите один файл в Image, закрасьте логотип в **First Image**, добавьте небольшой запас вокруг краёв и нажмите **Clean Up**. Если появился размытый ореол, попробуйте сначала уменьшить область маски, а не запускать повторно с ещё большей областью.

### Удалить объект с серии кадров

Выберите несколько изображений, если они имеют одинаковое разрешение и объект находится в одной позиции. Нарисуйте маску на первом кадре. Проверьте результат нескольких файлов в галерее, потому что одна и та же маска применяется ко всей серии.

### Обработать папку через ZIP

Соберите ZIP так, чтобы все изображения лежали непосредственно в корне. Не добавляйте подпапки. Включите **Upload ZIP-file**, загрузите архив и используйте маску первого изображения. После обработки скачайте общий архив.

### Удалить неподвижный объект из видео

Загрузите видео, на **First Frame** закрасьте объект и запускайте обработку. Используйте этот способ для объекта, который не двигается относительно кадра. Если объект перемещается, фиксированная маска не будет следовать за ним.

## Устранение проблем

### После загрузки изображения не появляется First Image

Проверьте, что загружен обычный файл изображения, а не ZIP при выключенной галочке. Для ZIP включите **Upload ZIP-file** и загрузите именно один `.zip`. Если ZIP начинается с папки или постороннего файла, предпросмотр может не открыться, потому что код берёт первый элемент архива без фильтрации.

### Обработался только первый файл из ZIP

Проверьте структуру архива. Файлы должны лежать в его корне.

### У разных изображений неправильная область удаления

Это ожидаемое ограничение общей маски. В Image маска рисуется только на первом изображении и используется для всех файлов. Разделите изображения на группы с одинаковой компоновкой или обрабатывайте их по одному.

### Готовая маска не работает

Проверьте, что включена именно галочка **Add Mask**, а файл загружен в появившееся поле **Add mask**. Если нарисованная маска на первом изображении тоже содержит область, она будет объединена с готовой. Убедитесь, что размеры готовой маски совпадают с исходным изображением.

### Результат пустой или область не восстановилась

Проверьте, что маска действительно закрашивает объект. Попробуйте обработать небольшой тестовый файл и закрасить область чуть шире. Если модель не загрузилась, посмотрите консоль Fooocus и убедитесь, что доступна CUDA и скачалась `big-lama.safetensors`.

### Фон вокруг объекта выглядит неестественно

LaMa восстанавливает область по окружающему контексту. Если маска слишком большая, модель вынуждена придумывать больше деталей. Уменьшите маску до объекта и небольшого необходимого запаса. Для повторяющегося узора или текста лучше обрабатывать небольшими участками.

### Видео обрабатывается, но объект движется в результате

Cleaner использует одну фиксированную маску для всех кадров. Он не отслеживает объект. Для движущегося объекта нужен отдельный трекинг маски или обработка коротких фрагментов с разными масками.

### В обработанном видео нет звука

Код записывает новый видеопоток через OpenCV и не копирует аудио. Используйте внешний видеоредактор или FFmpeg, чтобы добавить звуковую дорожку из оригинала к обработанному видео.

## Что скачать после обработки

Для одного изображения скачайте файл из **Output image**. Для нескольких изображений скачайте отдельные элементы из **Clean result** или общий **Download a ZIP file**.

Для одного или нескольких видео скачивайте отдельные MP4-файлы из **Download a videofiles**. Архив видео интерфейс не создаёт.

Результаты Cleaner не сохраняются в стандартный History Log Fooocus. 

## Короткий безопасный алгоритм

1. Сначала обработайте одну копию одного файла.
2. Нарисуйте маску только на объекте и небольшом запасе вокруг него.
3. Проверьте результат на 100% масштабе.
4. Для пакета используйте только файлы с одинаковой компоновкой.
5. Для ZIP кладите изображения в корень архива.
6. Для видео проверяйте, что объект неподвижен относительно кадра.
7. Скачивайте результаты перед следующим запуском.
8. Для важных видео отдельно сохраните оригинальный звук.
