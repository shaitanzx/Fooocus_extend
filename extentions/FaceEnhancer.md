## What FaceEnhancer Does
([Читать на русском языке](#что-делает-faceenhancer))

This extension can perform three actions on a finished image: enhance detected faces, upscale and restore the entire background, and swap faces. Within a single run, the order is as follows: first, if necessary, face swapping is performed; then the entire image is upscaled; after that, the restored faces are pasted back in.

The extension does not create an image from a text description. It works either with the result of Fooocus generation or with already uploaded files.

## Where to Find FaceEnhancer

Open the **Extensions** panel. It contains two separate tabs named **FaceEnhancer**.

- In the **in generation** group, FaceEnhancer processes the result immediately after generation. A toggle switch **Enabled** appears here. While it is turned off, the FaceEnhancer settings do not participate in the generation.
- In the **modules** group, FaceEnhancer works as a standalone processor for uploaded images. There is no **Enabled** toggle here: instead, there are upload fields and a **Start FaceEnhancer** button.

## Quick First Launch

For the first test, use a single regular portrait photograph. This helps you see the effect without wasting time on a large batch.

1. Open **Extensions → modules → FaceEnhancer**.
2. Drag and drop a single file into the **Drag (Select) 1 or more reference images** area.
3. Leave the standard values: **GFPGANv1.4.pth**, **SRVGG, realesr-general-x4v3.pth**, **retinaface_resnet50**, scale factor **4**.
4. Do not enable **Enable Face Swap**.
5. Click **Start FaceEnhancer**.
6. Wait for completion. For a single result, an **Output image** field with a PNG file will appear. Download it immediately.

On first use, the selected models are downloaded. Therefore, the first launch may be noticeably longer and requires internet access.

> Before processing the face of another person, make sure you have the right to use their image and show the result. This is especially important for Face Swap.

## What All the Settings Mean

Below are the working settings of FaceEnhancer, their standard values, and the modes where they are actually applied.

### **Enabled**

Standard value: disabled. This checkbox exists **only** in **in generation** mode. It enables FaceEnhancer processing after generation. In the standalone module, this checkbox is absent: clicking **Start FaceEnhancer** already starts the processing.

### **Face Restoration version**

Standard value: **GFPGANv1.4.pth**. This is the model that finds and restores faces on the already prepared image. It is available in both modes.

The list contains nine models:

- **GFPGANv1.4.pth** — the standard option.
- **RestoreFormer++.ckpt**.
- **CodeFormer.pth**.
- **GPEN-BFR-512.pth**.
- **GPEN-BFR-1024.pt**.
- **GPEN-BFR-2048.pt**.
- **GFPGANv1.3.pth** — marked in the source code as deprecated.
- **GFPGANv1.2.pth** — marked in the source code as deprecated.
- **RestoreFormer.ckpt** — a deprecated version of RestoreFormer.

For a start, leave **GFPGANv1.4.pth**. The **About models** section contains descriptions and links to the original sources of all models.

The dropdown list also has a `None` item, which corresponds to skipping this action.

### **UpScale version**

Standard value: **SRVGG, realesr-general-x4v3.pth**. This is the model for upscaling and restoring the **entire image**, including the background. It is available in both modes and can be used together with Face Restoration.

In the item name, the internal model type is specified before the comma. The interface marks **SRVGG** type models as fast; **DAT**, **HAT**, **DRCT**, **ATD**, and **SRFormer** as slow; other types as normal speed. This is a guideline, not a guarantee of the same time on any computer.

The current list contains 60 models:

- **SRVGG:** `realesr-general-x4v3.pth`, `realesr-animevideov3.pth`, `4xLSDIRCompact.pth`, `4xLSDIRCompactC.pth`, `4xLSDIRCompactR.pth`, `4xLSDIRCompactN.pth`, `4xLSDIRCompactC3.pth`, `4xLSDIRCompactR3.pth`, `4xLSDIRCompactCR3.pth`, `2xParimgCompact.pth`, `1xExposureCorrection_compact.pth`, `1xUnderExposureCorrection_compact.pth`, `1xOverExposureCorrection_compact.pth`, `2x-sudo-UltraCompact.pth`, `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth`, `2x_AnimeJaNai_HD_V3_UltraCompact.pth`, `2x_AnimeJaNai_HD_V3_Compact.pth`.
- **RRDB:** `RealESRGAN_x4plus_anime_6B.pth`, `RealESRGAN_x2plus.pth`, `RealESRNet_x4plus.pth`, `RealESRGAN_x4plus.pth`.
- **ESRGAN:** `4x-AnimeSharp.pth`, `4x_IllustrationJaNai_V1_ESRGAN_135k.pth`, `2x-sudo-RealESRGAN.pth`, `2x-sudo-RealESRGAN-Dropout.pth`, `4xNomos2_otf_esrgan.pth`, `4xNomosWebPhoto_esrgan.pth`, `4x_foolhardy_Remacri.pth`.
- **DAT:** `4xNomos8kDAT.pth`, `4x-DWTP-DS-dat2-v3.pth`, `4xBHI_dat2_real.pth`, `4xBHI_dat2_otf.pth`, `4xBHI_dat2_multiblur.pth`, `4xBHI_dat2_multiblurjpg.pth`, `4x_IllustrationJaNai_V1_DAT2_190k.pth`, `4x-PBRify_UpscalerDAT2_V1.pth`, `4xBHI_dat2_otf_nn.pth`.
- **HAT:** `4xNomos8kSCHAT-L.pth`, `4xNomos8kSCHAT-S.pth`, `4xNomos8kHAT-L_otf.pth`, `4xBHI_small_hat-l.pth`.
- **RealPLKSR_dysample:** `4xHFA2k_ludvae_realplksr_dysample.pth`, `4xArtFaces_realplksr_dysample.pth`, `4x-PBRify_RPLKSRd_V3.pth`, `4xNomos2_realplksr_dysample.pth`.
- **RealPLKSR:** `2x-AnimeSharpV2_RPLKSR_Sharp.pth`, `2x-AnimeSharpV2_RPLKSR_Soft.pth`, `4xPurePhoto-RealPLSKR.pth`, `2x_Text2HD_v.1-RealPLKSR.pth`, `2xVHS2HD-RealPLKSR.pth`, `4xNomosWebPhoto_RealPLKSR.pth`.
- **DRCT:** `4xNomos2_hq_drct-l.pth`.
- **ATD:** `4xNomos2_hq_atd.pth`.
- **MoSR:** `4xNomos2_hq_mosr.pth`, `2x-AnimeSharpV2_MoSR_Sharp.pth`, `2x-AnimeSharpV2_MoSR_Soft.pth`.
- **SRFormer:** `4xNomos8kSCSRFormer.pth`, `4xFrankendataFullDegradation_SRFormer460K_g.pth`, `4xFrankendataPretrainer_SRFormer400K_g.pth`, `1xFrankenfixer_SRFormerLight_g.pth`.

For a regular photograph, start with the standard `realesr-general-x4v3.pth`. For anime and drawings, choose models whose descriptions mention **Anime**, **Cartoon**, or **Manga**. For JPEG and web images with artifacts, models marked **Compression Removal** or **JPEG** are useful. For text, `2x_Text2HD_v.1-RealPLKSR.pth` is intended. Look at the description of the specific model in **About models**, rather than relying solely on its name.

### **Face detection only center**

Standard value: disabled. If enabled, during the **Face Restoration** stage, the extension will leave only the face closest to the center of the image for restoration. This is convenient for a single main portrait against a crowd background. When disabled, all faces that pass the size filter are restored.

This setting **does not control Face Swap**. For face swapping, the extension always uses its own detector, and this checkbox is passed only to the subsequent face restoration module.

### **Face eye dist threshold**

Standard value: **10**. This is the distance threshold between eyes for filtering small or heavily turned faces during **Face Restoration**. A higher threshold filters out more small faces; a lower threshold allows attempting to process them, but does not guarantee a high-quality result.

Like the previous setting, this threshold does not select faces for Face Swap. Leave it at `10` unless there is a clear reason to change it. If small faces disappear in a group shot, try slightly decreasing the number and repeat the test on a copy of the file.

### **Face Detection type**

Standard value: **retinaface_resnet50**. Other options are **YOLOv5l** and **YOLOv5n**. The choice is passed to the Face Restoration module as the face detection model. It is available in both modes.

This list also relates to face restoration, not Face Swap: swapping uses `buffalo_l` via the CPU provider. Therefore, face swapping may be slower than expected from GPU performance.

### **Rescaling factor**

Standard value: **4**. This is the required final scale factor: a 1000 × 1000 pixel image with a value of `4` will strive to become approximately 4000 × 4000 pixels. The value is passed to both the background upscaler and the face restoration model.

A larger factor sharply increases file size, time, and memory consumption. For testing, start with `2` or a small source, and leave `4` for the final output. The model file name (`2x`, `4x`, `1x`) describes its base specialization; do not combine it with an unusual factor without testing.

For large images, the extension automatically divides the picture into overlapping tiles if it is larger than approximately 2048 × 2048 pixels or if a CUDA out-of-memory error occurs, and then assembles the result. This helps, but does not make processing instantaneous.

### **Save input image**

Standard value: disabled. This setting is visible **only in in generation mode**. This setting allows saving the input image for this module to the output folder during generation.

Therefore, do not use **Save input image** as the only way to save the original. If the original generation is important, download it separately or work with a copy.

### **Output image files name with model name (if saved *.zip)**

Standard value: enabled. This checkbox exists **only in the standalone module**, not in generation mode. It adds the selected Face Restoration and UpScale models to the name of each PNG. When disabled, the result gets a shorter name.

For example, without face swapping, the name is created using the scheme `source_name_face-model_upscale-model.png`; with face swapping — `target-name_source-name_face-model_upscale-model.png`. All results of the standalone module are saved as PNG, even if the input was a JPEG.

### **Enable Face Swap**

Standard value: disabled. The setting is available in both modes. After enabling, fields for the source face and indices appear.

Face Swap replaces the face in the target image with the face from the source image. After swapping, the selected upscale and face restoration are still applied to the image. If you only need to improve a photo, do not enable this checkbox.

### **Source Face** and source loading methods

These fields are visible only after enabling **Enable Face Swap**.

- In **in generation** mode, upload a single image to the **Source Face** field. It will be the source for each generated result of the current run.
- In the standalone module, you have a separate **Drag (Select) 1 or more Source Face images** area and a separate **Upload ZIP-file** checkbox for an archive of source faces. Target images and source images are uploaded separately.

For a stable result, use a photograph where the face is large, not covered by hair, a mask, or a hand, and well-lit.

### **Source Face Index** and **Target Face Index**

Both fields have a standard value of **0** and work only in Face Swap.

The index is the face number, starting from zero. In this version, faces are sorted only by horizontal position, meaning usually **left to right**. Do not rely on a "top to bottom" order, even if faces are at different levels of the frame.

- `0` and `0` means: replace the first face found on the left in the source with the first face found on the left in the target.
- You can specify multiple indices separated by commas. For example, `0,1` in the source and `1,0` in the target will swap the pairs crosswise: the first source face goes to the second target face, the second to the first.
- If the lists are of different lengths, only pairs up to the length of the shorter list are used.
- `-1` in **Target Face Index**, when there is exactly one face in the source image, means: replace **all** faces in the target with this single face.
- In other cases, `-1` expands to a list of all detected faces. If there are multiple faces in both source and target, they are matched in order up to the length of the shorter list.

Enter only `-1` or non-negative numbers. Other negative numbers are not processed. An index greater than the number of found faces is skipped.

## Working in **in generation** Mode

This mode is needed when you want FaceEnhancer to automatically process the image immediately after Fooocus generation.

1. Open **Extensions → in generation → FaceEnhancer**.
2. Enable **Enabled**.
3. Select the face model, upscale model, detector, and scale factor.
4. For normal enhancement, leave **Enable Face Swap** disabled.
5. For face swapping, enable **Enable Face Swap**, upload a **Source Face**, and specify the indices.
6. Start the generation as usual.

FaceEnhancer receives the last result from the current generation list and returns the processed version. The parameters of the face model, upscaler, scale, detector, center face, and threshold are recorded in the task metadata when FaceEnhancer is enabled.

## Standalone Processing of a Single Image

1. Open **Extensions → modules → FaceEnhancer**.
2. Leave **Upload ZIP-file** disabled.
3. Drag and drop a single file into **Drag (Select) 1 or more reference images**.
4. After uploading, the interface hides the list and shows the **Reference image** field. This is normal behavior for a single file.
5. Configure Face Restoration, UpScale, detector, and scale.
6. Click **Start FaceEnhancer**.
7. Monitor the **Process preview**. When processing finishes, the single result is shown in **Output image**.
8. Download the PNG file.

For this standard path, the code clears its temporary folders and the results folder before launching. The next standard run may delete the temporary result of the previous one, so do not delay downloading.

## Processing Multiple Images

1. Leave **Upload ZIP-file** disabled.
2. Drag and drop multiple images into **Drag (Select) 1 or more reference images**.
3. With multiple files, the interface keeps the list visible instead of the single image field.
4. Specify the settings and click **Start FaceEnhancer**.
5. The **Process preview** will sequentially show the files being processed.
6. With two or more results, a **Download a ZIP file** button will appear. Download the created archive.

Files are iterated in alphabetical order by name. Each standard input file yields one PNG result.

## Processing a ZIP Archive

The archive mode is available only in the standalone module.

1. Place the images **in the root of the ZIP archive**.
2. Do not add subfolders, text files, videos, or other extraneous files.
3. Give the files different, simple names.
4. Enable **Upload ZIP-file**.
5. Instead of the image list, an **Upload a ZIP file** field will appear. Upload a single `.zip` file.
6. Configure the processing and click **Start FaceEnhancer**.
7. Download a single PNG for a single result, or the archive from **Download a ZIP file** for multiple results.

The interface accepts a ZIP as a container, but after extraction, the processor takes only regular files from the root of the temporary folder. Files in subfolders do not enter the queue. There is no additional extension filtering in the archive content: if a non-image file lies next to the photos, the processor may try to open it as an image and end with an error.

## Face Swap with a Single Image

1. Open the standalone FaceEnhancer and upload the **target** image — the one in which the face needs to be replaced.
2. Enable **Enable Face Swap**.
3. In the source area, upload the image with the desired face.
4. If there is one face on both images, leave `0` in **Source Face Index** and **Target Face Index**.
5. Click **Start FaceEnhancer**.
6. Check the result: the face is swapped first, then the image is upscaled and restored.

If you need to put one face on all faces of a group target, use a single face in the source photo and specify `-1` in **Target Face Index**.

## Face Swap with Multiple Files or Archives

In the standalone module, you can upload multiple targets and multiple sources, including separate ZIP archives for both sides. However, this is **not** pairwise processing by matching names.

The extension runs each source image for each target image. For example, 3 targets and 2 sources create up to 6 results. This is a Cartesian product, not "first with first, second with second". The result names include both source names.

The same ZIP rules apply to sources: images must lie in the root, without additional files and subfolders.

## How to Choose Starting Settings

For the first test with a regular photo, use:

- **Face Restoration version:** `GFPGANv1.4.pth`.
- **UpScale version:** `SRVGG, realesr-general-x4v3.pth`.
- **Face detection only center:** disabled for a single face; enabled if there are many people in the photo, but only the central portrait is needed.
- **Face eye dist threshold:** `10`.
- **Face Detection type:** `retinaface_resnet50`.
- **Rescaling factor:** `2` for testing and `4` for the final file, if memory and space allow.
- **Enable Face Swap:** disabled until you have tested simple enhancement.

For JPEGs with noticeable blocks and noise, look in **About models** for options marked `Compression Removal` and `JPEG`, such as the `4xLSDIRCompactC*` or `4xBHI_*` families. For anime, choose a model marked `Anime` or `Cartoon`. These designations are taken from the model descriptions inside the extension.

## What You Will Get and Where to Find the Result

In standalone mode, the result is saved in the FaceEnhancer temporary folder as a PNG.

- If exactly one file was processed, the interface shows this PNG in **Output image**.
- If multiple files were processed, the interface packages the results into a ZIP and shows it in **Download a ZIP file**.
- During normal batch processing, the results working folder is cleared before the next run.
- With the model name addition enabled, the PNG names get a suffix with the selected models.

In generation mode, the processed picture is returned to the standard Fooocus results stream, not to the ZIP download window.

## Troubleshooting

### The face was not swapped, but the file still appeared
Face Swap is wrapped in separate error handling. If the source or target does not contain a detected face, or the index is incorrect, the error is output to the console, but further upscaling and Face Restoration may continue. As a result, you will get an enhanced, but not swapped, picture. Check that the faces are large and clearly visible, and that the index values correspond to their left-to-right order.

### The wrong face was swapped
Start with `0` and `0` on images with a single face. In group shots, remember that the order is determined horizontally, not top to bottom. For explicit matching, use comma-separated lists, for example, source `0,1` and target `1,0`.

### Too many files were generated
In batch Face Swap, the number of runs equals the number of targets multiplied by the number of sources. Reduce one of the sets to a single file or process pairs in separate runs. If the number is still higher than expected, take into account the known feature of uncleared Face Swap temporary folders.

### The ZIP downloaded, but the images were not processed
Check the archive structure: images must be directly in its root. Remove subfolders and any non-graphic files. The interface checks the extension of the uploaded container (`.zip`), but does not filter its contents before attempting to open each file.

### Processing is very slow or ends with a memory error
First, decrease the **Rescaling factor**, select a single input file, and try a faster **SRVGG** type. For large pictures, the code tries to automatically split the image into tiles, but with a very heavy combination of model, scale, and resolution, this is not always enough.

## Что делает FaceEnhancer
([Read in English](#what-faceenhancer-does))

Это расширение, которое может выполнять три действия с готовой картинкой: улучшать обнаруженные лица, увеличивать и восстанавливать весь фон, а также подменять лица. Внутри одного запуска порядок такой: сначала, при необходимости, выполняется подмена лица; затем увеличивается вся картинка; после этого восстановленные лица вставляются обратно.

Расширение не создаёт изображение по текстовому описанию. Оно работает либо с результатом генерации Fooocus, либо с уже загруженными файлами.

## Где находится FaceEnhancer

Откройте панель **Extensions**. В ней есть две отдельные вкладки с названием **FaceEnhancer**.

- В группе **in generation** FaceEnhancer обрабатывает результат сразу после генерации. Здесь появляется переключатель **Enabled**. Пока он выключен, настройки FaceEnhancer не участвуют в генерации.
- В группе **modules** FaceEnhancer работает как самостоятельный обработчик загруженных изображений. Здесь нет переключателя **Enabled**: вместо него есть поля загрузки и кнопка **Start FaceEnhancer**.

## Быстрый первый запуск

Для первой проверки используйте одну обычную портретную фотографию. Это помогает увидеть эффект и не тратить время на большой пакет.

1. Откройте **Extensions → modules → FaceEnhancer**.
2. Перетащите один файл в область **Drag (Select) 1 or more reference images**.
3. Оставьте стандартные значения: **GFPGANv1.4.pth**, **SRVGG, realesr-general-x4v3.pth**, **retinaface_resnet50**, коэффициент **4**.
4. Не включайте **Enable Face Swap**.
5. Нажмите **Start FaceEnhancer**.
6. Дождитесь завершения. Для одного результата появится поле **Output image** с PNG-файлом. Скачайте его сразу.

При первом использовании скачиваются выбранные модели. Поэтому первый запуск может быть заметно дольше и требует доступа к сети.

> Перед обработкой лиц другого человека убедитесь, что у вас есть право использовать его изображение и показывать результат. Это особенно важно при Face Swap.

## Что означают все настройки

Ниже перечислены рабочие настройки FaceEnhancer, их стандартные значения и режимы, где они действительно применяются.

### **Enabled**

Стандартное значение — выключено. Эта галочка есть **только** в режиме **in generation**. Она включает обработку FaceEnhancer после генерации. В самостоятельном модуле этой галочки нет: нажатие **Start FaceEnhancer** уже запускает обработку.

### **Face Restoration version**

Стандартное значение — **GFPGANv1.4.pth**. Это модель, которая ищет и восстанавливает лица на уже подготовленном изображении. Она доступна в обоих режимах.

В списке есть девять моделей:

- **GFPGANv1.4.pth** — стандартный вариант.
- **RestoreFormer++.ckpt**.
- **CodeFormer.pth**.
- **GPEN-BFR-512.pth**.
- **GPEN-BFR-1024.pt**.
- **GPEN-BFR-2048.pt**.
- **GFPGANv1.3.pth** — помечена в исходниках как устаревшая.
- **GFPGANv1.2.pth** — помечена в исходниках как устаревшая.
- **RestoreFormer.ckpt** — устаревший вариант RestoreFormer.

Для начала оставьте **GFPGANv1.4.pth**. Раздел **About models** содержит описания и ссылки на первоисточники всех моделей.

В выпадающем списке также есть пункт `None`, чир соответствет пропуску данного действия.

### **UpScale version**

Стандартное значение — **SRVGG, realesr-general-x4v3.pth**. Это модель для увеличения и восстановления **всего изображения**, включая фон. Она доступна в обоих режимах и может использоваться вместе с Face Restoration.

В имени пункта перед запятой указан внутренний тип модели. Интерфейс помечает модели типа **SRVGG** как быстрые; **DAT**, **HAT**, **DRCT**, **ATD** и **SRFormer** — как медленные; остальные типы — как обычные по скорости. Это ориентир, а не гарантия одинакового времени на любом компьютере.

В текущем списке 60 моделей.

- **SRVGG:** `realesr-general-x4v3.pth`, `realesr-animevideov3.pth`, `4xLSDIRCompact.pth`, `4xLSDIRCompactC.pth`, `4xLSDIRCompactR.pth`, `4xLSDIRCompactN.pth`, `4xLSDIRCompactC3.pth`, `4xLSDIRCompactR3.pth`, `4xLSDIRCompactCR3.pth`, `2xParimgCompact.pth`, `1xExposureCorrection_compact.pth`, `1xUnderExposureCorrection_compact.pth`, `1xOverExposureCorrection_compact.pth`, `2x-sudo-UltraCompact.pth`, `2x_AnimeJaNai_HD_V3_SuperUltraCompact.pth`, `2x_AnimeJaNai_HD_V3_UltraCompact.pth`, `2x_AnimeJaNai_HD_V3_Compact.pth`.
- **RRDB:** `RealESRGAN_x4plus_anime_6B.pth`, `RealESRGAN_x2plus.pth`, `RealESRNet_x4plus.pth`, `RealESRGAN_x4plus.pth`.
- **ESRGAN:** `4x-AnimeSharp.pth`, `4x_IllustrationJaNai_V1_ESRGAN_135k.pth`, `2x-sudo-RealESRGAN.pth`, `2x-sudo-RealESRGAN-Dropout.pth`, `4xNomos2_otf_esrgan.pth`, `4xNomosWebPhoto_esrgan.pth`, `4x_foolhardy_Remacri.pth`.
- **DAT:** `4xNomos8kDAT.pth`, `4x-DWTP-DS-dat2-v3.pth`, `4xBHI_dat2_real.pth`, `4xBHI_dat2_otf.pth`, `4xBHI_dat2_multiblur.pth`, `4xBHI_dat2_multiblurjpg.pth`, `4x_IllustrationJaNai_V1_DAT2_190k.pth`, `4x-PBRify_UpscalerDAT2_V1.pth`, `4xBHI_dat2_otf_nn.pth`.
- **HAT:** `4xNomos8kSCHAT-L.pth`, `4xNomos8kSCHAT-S.pth`, `4xNomos8kHAT-L_otf.pth`, `4xBHI_small_hat-l.pth`.
- **RealPLKSR_dysample:** `4xHFA2k_ludvae_realplksr_dysample.pth`, `4xArtFaces_realplksr_dysample.pth`, `4x-PBRify_RPLKSRd_V3.pth`, `4xNomos2_realplksr_dysample.pth`.
- **RealPLKSR:** `2x-AnimeSharpV2_RPLKSR_Sharp.pth`, `2x-AnimeSharpV2_RPLKSR_Soft.pth`, `4xPurePhoto-RealPLSKR.pth`, `2x_Text2HD_v.1-RealPLKSR.pth`, `2xVHS2HD-RealPLKSR.pth`, `4xNomosWebPhoto_RealPLKSR.pth`.
- **DRCT:** `4xNomos2_hq_drct-l.pth`.
- **ATD:** `4xNomos2_hq_atd.pth`.
- **MoSR:** `4xNomos2_hq_mosr.pth`, `2x-AnimeSharpV2_MoSR_Sharp.pth`, `2x-AnimeSharpV2_MoSR_Soft.pth`.
- **SRFormer:** `4xNomos8kSCSRFormer.pth`, `4xFrankendataFullDegradation_SRFormer460K_g.pth`, `4xFrankendataPretrainer_SRFormer400K_g.pth`, `1xFrankenfixer_SRFormerLight_g.pth`.

Для обычной фотографии начните со стандартного `realesr-general-x4v3.pth`. Для аниме и рисунков выбирайте модели, в описании которых указаны **Anime**, **Cartoon** или **Manga**. Для JPEG и веб-изображений с артефактами полезны модели с пометками **Compression Removal** или **JPEG**. Для текста предназначена `2x_Text2HD_v.1-RealPLKSR.pth`. Смотрите описание конкретной модели в **About models**, а не ориентируйтесь только на её название.

### **Face detection only center**

Стандартное значение — выключено. Если включить, при этапе **Face Restoration** расширение оставит для восстановления только лицо, находящееся ближе всего к центру изображения. Это удобно для одного главного портрета на фоне толпы. При выключенной галочке восстанавливаются все лица, которые прошли фильтр размера.

Настройка **не управляет Face Swap**. Для подмены лиц расширение всегда использует собственный детектор, а эта галочка передаётся только в модуль последующего восстановления лица.

### **Face eye dist threshold**

Стандартное значение — **10**. Это порог расстояния между глазами для фильтрации мелких или сильно повернутых лиц во время **Face Restoration**. Более высокий порог отсекает больше маленьких лиц; более низкий позволяет пытаться обработать их, но не гарантирует качественный результат.

Как и предыдущая настройка, этот порог не выбирает лица для Face Swap. Оставьте `10`, если нет понятной причины менять его. Если на групповом снимке пропадают небольшие лица, попробуйте немного уменьшить число и повторите тест на копии файла.

### **Face Detection type**

Стандартное значение — **retinaface_resnet50**. Другие варианты — **YOLOv5l** и **YOLOv5n**. Выбор передаётся в модуль Face Restoration как модель детекции лиц. Он доступен в обоих режимах.

Этот список также относится к восстановлению лица, а не к Face Swap: подмена использует `buffalo_l` через CPU-провайдер. Поэтому подмена лица может быть медленнее, чем ожидается от производительности видеокарты.

### **Rescaling factor**

Стандартное значение — **4**. Это требуемый коэффициент итогового масштаба: изображение 1000 × 1000 пикселей при значении `4` стремится стать примерно 4000 × 4000 пикселей. Значение передаётся и фоновому апскейлеру, и модели восстановления лица.

Больший коэффициент резко увеличивает размер файла, время и расход памяти. Для проверки начните с `2` или с небольшого исходника, а `4` оставьте для финала. Название файла модели (`2x`, `4x`, `1x`) описывает её базовую специализацию; не стоит без теста сочетать его с необычным коэффициентом.

Для крупных изображений расширение автоматически делит картинку на перекрывающиеся части, если она больше примерно 2048 × 2048 пикселей или если возникает ошибка нехватки памяти CUDA, а затем собирает результат. Это помогает, но не делает обработку мгновенной.

### **Save input image**

Стандартное значение — выключено. Настройка видна **только в режиме in generation**. Данная настройка позволяет сохранить в выходную папку входное изображение для этого модуля во время генерации.

Поэтому не используйте **Save input image** как единственный способ сохранить оригинал. Если исходная генерация важна, скачайте её отдельно или работайте с копией.

### **Output image files name with model name (if saved *.zip)**

Стандартное значение — включено. Эта галочка есть **только в самостоятельном модуле**, не в режиме генерации. Она добавляет к имени каждого PNG выбранные Face Restoration и UpScale модели. При выключении результат получает более короткое имя.

Например, без подмены лица имя создаётся по схеме `имя_исходника_модель-лица_модель-апскейла.png`; при подмене лица — `имя-цели_имя-источника_модель-лица_модель-апскейла.png`. Все результаты самостоятельного модуля сохраняются как PNG, даже если на входе был JPEG.

### **Enable Face Swap**

Стандартное значение — выключено. Настройка доступна в обоих режимах. После включения появляются поля для исходного лица и индексов. 

Face Swap заменяет лицо в целевой картинке лицом из изображения-источника. После подмены к изображению всё равно применяются выбранные апскейл и восстановление лица. Если нужно только улучшить фото, не включайте эту галочку.

### **Source Face** и способы загрузки источника

Эти поля видны только после включения **Enable Face Swap**.

- В режиме **in generation** загрузите одно изображение в поле **Source Face**. Оно будет источником для каждого сгенерированного результата текущего запуска.
- В самостоятельном модуле у вас есть отдельная область **Drag (Select) 1 or more Source Face images** и отдельная галочка **Upload ZIP-file** для архива лиц-источников. Целевые изображения и изображения-источники загружаются раздельно.

Для стабильного результата используйте фотографию, где лицо крупное, не закрыто волосами, маской или рукой и хорошо освещено.

### **Source Face Index** и **Target Face Index**

Оба поля имеют стандартное значение **0** и работают только в Face Swap.

Индекс — это номер лица, начиная с нуля. В этой версии лица сортируются только по горизонтальному положению, то есть обычно **слева направо**. Не полагайтесь на порядок «сверху вниз», даже если лица находятся на разных уровнях кадра.

- `0` и `0` означает: первое найденное слева лицо источника заменить первым найденным слева лицом цели.
- Можно указать несколько индексов через запятую. Например, `0,1` в источнике и `1,0` в цели поменяет пары крест-накрест: первое лицо источника пойдёт ко второму лицу цели, второе — к первому.
- Если списки разной длины, используются только пары до длины более короткого списка.
- `-1` в **Target Face Index**, когда на изображении-источнике ровно одно лицо, означает: заменить этим одним лицом **все** лица в цели.
- В остальных случаях `-1` разворачивается в список всех обнаруженных лиц. Если в источнике и цели по несколько лиц, они сопоставляются по порядку до длины более короткого списка.

Вводите только `-1` либо неотрицательные номера. Другие отрицательные числа не обраатываются. Индекс больше числа найденных лиц пропускается.

## Работа в режиме **in generation**

Этот режим нужен, когда вы хотите, чтобы FaceEnhancer автоматически обработал изображение сразу после генерации Fooocus.

1. Откройте **Extensions → in generation → FaceEnhancer**.
2. Включите **Enabled**.
3. Выберите модель лица, модель апскейла, детектор и коэффициент масштаба.
4. Для обычного улучшения оставьте **Enable Face Swap** выключенным.
5. Для подмены лица включите **Enable Face Swap**, загрузите **Source Face** и укажите индексы.
6. Запустите генерацию как обычно.

FaceEnhancer получает последний результат из текущего списка генерации и возвращает обработанную версию. Параметры модели лица, апскейлера, масштаба, детектора, центрального лица и порога записываются в метаданные задачи, когда FaceEnhancer включён.

## Самостоятельная обработка одного изображения

1. Откройте **Extensions → modules → FaceEnhancer**.
2. Оставьте **Upload ZIP-file** выключенным.
3. Перетащите один файл в **Drag (Select) 1 or more reference images**.
4. После загрузки интерфейс скрывает список и показывает поле **Reference image**. Это нормальное поведение для одного файла.
5. Настройте Face Restoration, UpScale, детектор и масштаб.
6. Нажмите **Start FaceEnhancer**.
7. Следите за **Process preview**. Когда обработка закончится, единственный результат показывается в **Output image**.
8. Скачайте PNG-файл.

Для этого обычного пути код перед запуском очищает свои временные папки и папку результатов. Следующий обычный запуск может удалить временный результат предыдущего, поэтому не откладывайте скачивание.

## Обработка нескольких изображений

1. Оставьте **Upload ZIP-file** выключенным.
2. Перетащите несколько картинок в **Drag (Select) 1 or more reference images**.
3. При нескольких файлах интерфейс оставляет видимым список вместо поля одиночного изображения.
4. Укажите настройки и нажмите **Start FaceEnhancer**.
5. В **Process preview** последовательно показываются обрабатываемые файлы.
6. При двух и более результатах появится **Download a ZIP file**. Скачайте созданный архив.

Файлы перебираются в отсортированном по имени порядке. Каждый обычный входной файл даёт один PNG-результат.

## Обработка ZIP-архива

Архивный режим доступен только в самостоятельном модуле.

1. Положите изображения **в корень ZIP-архива**.
2. Не добавляйте подпапки, текстовые файлы, видео или другие посторонние файлы.
3. Дайте файлам разные простые имена.
4. Включите **Upload ZIP-file**.
5. Вместо списка изображений появится поле **Upload a ZIP file**. Загрузите один файл `.zip`.
6. Настройте обработку и нажмите **Start FaceEnhancer**.
7. Скачайте один PNG при единственном результате либо архив из **Download a ZIP file** при нескольких.

Интерфейс принимает ZIP как контейнер, но после распаковки обработчик берёт только обычные файлы из корня временной папки. Файлы в подпапках не попадают в очередь. В содержимом архива нет дополнительного фильтра по расширению: если рядом с фотографиями лежит посторонний файл, обработчик может попытаться открыть его как картинку и завершиться ошибкой.

## Face Swap с одним изображением

1. Откройте самостоятельный FaceEnhancer и загрузите **целевую** картинку — ту, в которой нужно заменить лицо.
2. Включите **Enable Face Swap**.
3. В области источника загрузите картинку с нужным лицом.
4. Если на обоих изображениях по одному лицу, оставьте `0` в **Source Face Index** и **Target Face Index**.
5. Нажмите **Start FaceEnhancer**.
6. Проверьте результат: лицо сначала подменяется, затем изображение увеличивается и восстанавливается.

Если нужно поставить одно лицо на все лица групповой цели, используйте одно лицо в исходном фото и укажите `-1` в **Target Face Index**.

## Face Swap с несколькими файлами или архивами

В самостоятельном модуле можно загрузить несколько целей и несколько источников, в том числе отдельные ZIP-архивы для обеих сторон. Но это **не** попарная обработка по одинаковым именам.

Расширение запускает каждое изображение-источник для каждого целевого изображения. Например, 3 цели и 2 источника создают до 6 результатов. Это декартово произведение, а не «первый с первым, второй со вторым». Имена результатов включают оба исходных имени.

Для источников действуют те же правила ZIP: картинки должны лежать в корне, без дополнительных файлов и подпапок.

## Как выбрать стартовые настройки

Для первого теста с обычным фото используйте:

- **Face Restoration version:** `GFPGANv1.4.pth`.
- **UpScale version:** `SRVGG, realesr-general-x4v3.pth`.
- **Face detection only center:** выключено для одного лица; включено, если на фото много людей, но нужен только центральный портрет.
- **Face eye dist threshold:** `10`.
- **Face Detection type:** `retinaface_resnet50`.
- **Rescaling factor:** `2` для проверки и `4` для финального файла, если хватает памяти и места.
- **Enable Face Swap:** выключено, пока вы не проверили простое улучшение.

Для JPEG с заметными блоками и шумом посмотрите в **About models** варианты с пометками `Compression Removal` и `JPEG`, например семейство `4xLSDIRCompactC*` или `4xBHI_*`. Для аниме выберите модель с пометкой `Anime` либо `Cartoon`. Эти обозначения берутся из описаний моделей внутри расширения.

## Что вы получите и где искать результат

В самостоятельном режиме результат сохраняется во временной папке FaceEnhancer как PNG.

- Если обработан ровно один файл, интерфейс показывает этот PNG в **Output image**.
- Если обработано несколько файлов, интерфейс упаковывает результаты в ZIP и показывает его в **Download a ZIP file**.
- При обычной пакетной обработке перед очередным запуском очищается рабочая папка результатов.
- При включённом добавлении имени модели названия PNG получают суффикс с выбранными моделями.

В режиме генерации обработанная картинка возвращается в стандартный поток результатов Fooocus, а не в окно загрузки ZIP.

## Если что-то пошло не так

### Лицо не заменилось, но файл всё же появился

Face Swap обёрнут в отдельную обработку ошибок. Если источник или цель не содержит найденного лица, либо индекс неверен, ошибка выводится в консоль, но дальнейший апскейл и Face Restoration могут продолжиться. В результате получится улучшенная, но не подменённая картинка. Проверьте, что лица крупные и хорошо видны, а значения индексов соответствуют их порядку слева направо.

### Заменилось не то лицо

Начните с `0` и `0` на изображениях с одним лицом. На групповых кадрах помните, что порядок определяется горизонтально, а не сверху вниз. Для явного сопоставления используйте списки через запятую, например источник `0,1` и цель `1,0`.

### Получилось слишком много файлов

При пакетном Face Swap число запусков равно числу целей, умноженному на число источников. Сократите один из наборов до одного файла либо обработайте пары отдельными запусками. Если количество всё равно больше ожидаемого, учитывайте известную особенность с неочищаемыми временными папками Face Swap.

### ZIP загрузился, но изображения не обработались

Проверьте структуру архива: изображения должны быть непосредственно в его корне. Уберите подпапки и любые не-графические файлы. Интерфейс проверяет расширение самого загружаемого контейнера `.zip`, но не фильтрует его содержимое перед попыткой открыть каждый файл.

### Обработка очень долгая или заканчивается ошибкой памяти

Сначала уменьшите **Rescaling factor**, выберите один входной файл и попробуйте более быстрый тип **SRVGG**. Для больших картинок код пытается автоматически разбивать изображение на части, однако при очень тяжёлой комбинации модели, масштаба и разрешения это не всегда достаточно.
