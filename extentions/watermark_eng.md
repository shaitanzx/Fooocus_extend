## What Caption Does
([Читать на русском языке](watermark_rus.md))

`Logo` overlays either a single logo in a corner or a repeating watermark pattern onto images.

## What You Need

Prepare a source image (or set of images) and a separate logo/watermark image. A PNG with a transparent background is ideal for the logo/watermark, as only the mark itself will appear in the final image. If you upload a JPEG or a PNG with an opaque background, that background will be overlaid as well.

The tool accepts a single image, multiple images, or a ZIP archive containing images. The ZIP file must contain the image files directly at the root level, without subfolders. Standard formats like PNG and JPG are supported.

## Selecting a Mode

Two options are available under **Select Paste Mode**:

- **Logo** — places a single logo in one of the image corners. The tool attempts to account for detected faces and may add a background layer behind the logo if the image background is complex.
- **Watermark** — overlays a single large mark in the center or places multiple repeating marks across the entire image.

The selected mode determines which setting groups are visible and active.

## Uploading Source Images

### Single Image

Leave the **Upload ZIP-file** option disabled and add a single file to the **Drag (Select) 1 or more Source images** field. If exactly one file is uploaded, the interface hides the file list and displays the image in the **Source Image** field.

Upload the logo file and click **Start paste logo/watermark** to process it.

### Multiple Individual Images

Leave the **Upload ZIP-file** option disabled and add two or more images to the **Drag (Select) 1 or more Source images** field. After uploading multiple files, the interface keeps the file list visible. During processing, filenames are sorted alphabetically. Use unique filenames: if files from different folders share the same name, one file may overwrite another when copied to the temporary folder.

### ZIP Archive

1. Place the images directly in the root of the ZIP archive. Do not pack them into a subfolder.
2. Ensure there are no documents or other non-image files in the root.
3. Select the **Upload ZIP-file** checkbox.
4. In the **Upload a ZIP file of Source Images** field that appears, select a single `.zip` archive.


When the checkbox is selected, the ZIP field appears in place of the individual file list. The processor extracts the archive but only scans files located directly in the extraction folder; images in subdirectories will not be processed.

### Logo or Watermark Image

Upload the mark to the **Source Logo/Watermark** field. This is a separate input, distinct from the ZIP archive containing the source images. The field passes the image in RGBA format, allowing the use of PNG alpha channels for transparency.

The same uploaded mark is used for all source images. If no mark is selected or the file is corrupt, the process may fail.

## Logo Mode Settings

In **Logo** mode, size and placement settings are visible, along with the **Corner priority** group.

### Size Ratio

Range: `0.00–1.00`; step: `0.01`; default value: `0.20`.

This parameter sets the logo size relative to the **shorter side of the source image** while maintaining the logo's aspect ratio. For example, on a `1000 × 700` image, a value of `0.10` sets a reference size of about `70` pixels for the logo's long edge; `0.20` corresponds to about `140` pixels.

The default value of `0.20` might appear too large. For a small logo, try a range of `0.08–0.12` and evaluate the result. Do not set this to `0`: the code does not interpret zero as a command to "disable the logo," and the resulting logo size might become invalid.

### Margin Ratio

Range: `0.00–1.00`; step: `0.01`; default: `0.02`.

Sets the margin from the edge: the horizontal margin is calculated based on the image width, and the vertical margin based on the height. For example, `0.02` represents about 2% of the corresponding image dimension. A value of `0.02–0.05` is usually sufficient. Very high values ​​may push the logo off the edge or cause it to be cropped.

### Minimal complexity for background

Range: `0.00–1.00`; step: `0.01`; default: `0.70`.

This is the threshold for adding a background overlay behind the logo. The tool assesses the degree of color variation in the area near the selected corner. If the background complexity score exceeds the threshold, a semi-transparent colored overlay—matched to the surrounding background—is placed behind the logo.

- Lower threshold: the overlay appears more frequently.
- Higher threshold: the overlay appears only against more complex backgrounds.

This parameter **does not select the least cluttered corner**. It simply determines whether to add an overlay in the corner that has already been chosen. ### Corner priority

The group contains four lists: **Priority 1**, **Priority 2**, **Priority 3**, and **Priority 4**. For each, you can select `top-left`, `top-right`, `bottom-left`, or `bottom-right`. The default order is: top-left, top-right, bottom-left, bottom-right.

The algorithm checks the corners in this order and attempts to skip any that overlap with a detected face. A small additional margin around the recognized face is taken into account during this check. If no safe corner is found, the tool uses the corner from **Priority 1** regardless. Therefore, face detection is merely a heuristic, not a guarantee that the sign will never obscure a face or an important detail.

Try to specify different angles across the four lists. The interface does not prevent repetitions, but a duplicate angle will simply be checked again without adding a new placement option. Background complexity does not, in itself, alter the priority order.

## Watermark Mode Settings

In **Watermark** mode, the **Opacity**, **Scale**, **Rotate angle**, **Offset X**, and **Offset Y** settings become visible. Logo settings are hidden and not used in this mode.

### Opacity (%)

Range: `0–100`; step: `0.5`; default value: `85`.

Sets the final opacity of the mark, taking its original alpha channel into account. `100` preserves the mark image's original transparency, `50` makes it roughly twice as transparent, and `0` renders it invisible. This control works only in **Watermark** mode. There is no separate transparency setting in **Logo** mode; if necessary, prepare a PNG with the desired transparency beforehand.

### Scale (%)

Range: `0–100`; step: `0.05`; default value: `20`.

Sets the watermark size relative to the maximum size that fits on the source image, accounting for the rotation angle.

- At values ​​**below 95%**, the code repeats the mark across the image, creating a pattern.
- At a value of **95% or higher**, the code places a single large mark in the center instead of a repeating pattern. For a repeating pattern, start with `10–30%`. Do not use a value of `0`: this can result in an invalid tile size and cause processing to fail.

### Rotate angle (degrees)

Range: `-180°…180°`; step: `1°`; default value: `0°`.

Rotates the mark before overlaying it. `0°` keeps it horizontal. Try values ​​like `30°` or `45°` if you need a diagonal pattern.

### Offset X (%) and Offset Y (%)

Both parameters have a range of `0–100%`, a step of `0.5%`, and a default value of `50%`.

These parameters determine the spacing between repetitions:

- **Offset X** sets the horizontal interval between marks. If the value is greater than zero, every second row is additionally shifted by approximately half a step, creating a staggered row pattern.
- **Offset Y** sets the vertical interval between rows.

Higher values ​​result in less frequent repetition of the mark. This is not simply shifting the entire watermark layer left or right. For example, `50%` represents a gap equal to about half the size of a single mark along the corresponding axis.

## Processing workflow and results

1. Select **Logo** or **Watermark**.
2. Upload a single image, multiple images, or a ZIP archive.
3. Upload the mark in the **Source Logo/Watermark** section.
4. Configure only the parameters for the selected mode.
5. Click **Start paste logo/watermark**.
6. Monitor the **Process preview** and notifications. For each file, a message appears in the format: `Logo/Watermark Batch: start element generation X/Y. Filename: filename`.
7. Once processing is complete, save the result from the output area that appears. While running, **Process preview** displays the source image currently being processed rather than the final result with the overlay applied. The list of input files is processed in alphabetical order.

Each result is saved as a PNG file with the suffix `_logo`. For example, `portrait.jpg` becomes `portrait_logo.png`. This suffix is ​​also used in **Watermark** mode. The output is saved in RGB format, so transparency is not preserved in the final result. [2]

If **exactly one file** is processed, the finished image appears in a separate output display area, and the **Download a ZIP file** option is hidden. A ZIP archive is not automatically created for a single result.

If **multiple files** are processed, the **Download a ZIP file** option provides an archive containing the results. The ZIP file contains the finished PNGs (with the `_logo` suffix), not the original source images.

Temporary folders containing working input and output files are cleared whenever the process is restarted. Therefore, be sure to save any desired results before restarting, and download the archive via **Download a ZIP file** if processing multiple images.

## Configuration examples

### Small logo in the corner

1. Select **Logo**.
2. Upload the source image and a PNG with a transparent background.
3. Set the **Size Ratio** to approximately `0.10`.
4. Leave the **Margin Ratio** at `0.02` or try `0.04`.
5. Initially, leave **Minimal complexity for background** at `0.70`.
6. Set your preferred corner as the first choice in **Corner priority**; assign the other corners to the remaining positions.
7. Process a single test image and check whether the logo obscures any important details.

### Repeating semi-transparent watermark

1. Select **Watermark**. 
2. Upload the source image and the mark, preferably with a transparent background.
3. Set the **Opacity** to approximately `20–35%`.
4. For a repeating pattern, set the **Scale** below `95%`, for example, `20%`.
5. Set the **Rotate angle** to `30°` or `45°` if a tilt is required.
6. Start with **Offset X** and **Offset Y** values ​​around `50%`; increase them if the pattern is too dense.
7. Test the result on a single image first.

## Troubleshooting

- **No result appeared.** Check that the source images and the separate watermark image have been uploaded. For ZIP files, ensure the images are located directly at the archive's root and that there are no other files inside.
- **Not everything was processed from the ZIP.** Images in subfolders are not processed. Move them to the ZIP root and try again.
- **One file went missing when uploading multiple separate files.** Check for duplicate filenames. Rename them to ensure each name is unique.
- **The logo is covering a face.** Change the order in **Corner priority**. The algorithm might fail to detect a face or default to the first corner if all priority positions overlap with detected faces.
- **The background appears too often or hardly ever.** Adjust the **Minimal complexity for background** setting: lowering the threshold makes the background more likely to appear, while raising it makes it less likely.
- **The watermark is not visible.** Check the **Opacity** setting, the source watermark's alpha channel, and ensure the **Watermark** mode is selected. The watermark is invisible if opacity is set to zero.
- **The watermark looks too dense.** Increase **Offset X/Y** to widen the gaps between repetitions. You can also increase the **Scale** (keeping it below `95%`); at `95%` or higher, the mode switches to displaying a single central watermark.
- **An error occurs at the minimum size.** Do not use `0` for **Size Ratio** or **Scale**. These fields are not switches, and the code does not handle a zero size separately.