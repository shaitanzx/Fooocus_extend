## What Adetailer Does
([Читать на русском языке](#что-делает-adetailer))

`Adetailer` automatically finds faces, eyes, hands, people, or other objects in an image, builds masks for them, and runs local redrawing. Thanks to this, you can fix a face or hands after generation without redrawing the entire picture.

The working principle consists of three stages:

1. Fooocus creates an image or accepts a ready-made image.
2. The Adetailer detector finds objects and forms masks.
3. Fooocus performs Inpaint on the found masks with separate text and settings.

In Fooocus Extend, Adetailer is combined with the **Enhance** logic. Therefore, in the interface, it uses some parameters of upscaling and variation, and the image selection tab is named **ADetailer**.

## What is Important to Know in Advance

Adetailer supports two practical scenarios:

- processing a result that Fooocus has just generated;
- processing **one** uploaded image via the **Use with Adetailer, skips image generation** field.

In the second scenario, Fooocus skips the standard generation and proceeds directly to detection and Inpaint.

## How to Enable Adetailer

On the main page of Fooocus Extend, find the **Adetailer** toggle next to **Input Image** and **Enhance**.

By default, the Adetailer toggle is off. After enabling it, a block with the **ADetailer** tab and extension settings appears.

Inside the block, you will find:

- a tab for selecting the source image;
- the **Upscale or Variation** tab;
- several tabs for individual detectors;
- additional debug parameters in the advanced settings group.

By default, the configuration creates five detector tabs. The first tab is enabled, the rest are disabled.

## Quick First Launch After Generation

This is the easiest way to check that the extension is working.

1. Leave the **Adetailer** toggle off until you have filled in the main prompt and negative prompt.
2. Enable **Adetailer**.
3. Open the first detector tab.
4. Ensure the **Enable this tab (1)** checkbox is checked.
5. In the **ADetailer detector** field, select a face model, for example, `face_yolov8n.pt` or `face_yolov8s.pt`.
6. Leave the other parameters at their default values.
7. Start the generation.
8. After the main image is created, Fooocus will show the stage **Processing ADetailer ...**.
9. Wait for it to finish. The detector will find the face, after which Fooocus will redraw the found area.

For the first test, it is better to use one object and one active tab. If you enable multiple detectors at once, processing will take longer, and it will be harder to determine the source of the changes.

## Processing a Single Ready-Made Image

In the input image block, there is the **ADetailer** tab. It contains the field:

> **Use with Adetailer, skips image generation**

Upload a single image there. When launched, Fooocus will skip the base generation, use the uploaded picture as the source, and proceed to Adetailer.

This mode is convenient when an image was created earlier, but you need to fix a face, hands, or another object on it.

Practical order:

1. Enable **Adetailer**.
2. Open the **ADetailer** tab among the input image fields.
3. Upload one image.
4. Fill in the detector settings.
5. If necessary, set a separate prompt for the object being corrected.
6. Start processing.

In this mode, the main prompt is used as a fallback text if the separate Adetailer prompt field is left empty.

## General Logic of Detector Tabs

Each detector tab is an independent processing pass. You can enable multiple tabs and, for example, fix a face first, and then hands.

Only tabs where both of the following conditions are met simultaneously are included in active processing:

- the **Enable this tab (...)** checkbox is checked;
- a model other than `None` is selected in the **ADetailer detector** field.

Inactive tabs are completely skipped. If the detector does not find an object, this tab does not perform Inpaint and moves on.

The order of the tabs matters. Fooocus processes them sequentially. The result of the first tab becomes the input for the next one.

## Top-Level Adetailer Modes

### **Only detect**

The **Only detect** checkbox runs detection and shows a preview result without subsequent Inpaint.

When this mode is enabled, the upscaling method is disabled, but the detector still builds masks and outputs a preview image with the markup of found objects. The actual correction of the face, hand, or other object is not performed.

Use this mode if you need to check:

- whether the model finds the desired object;
- whether it captures the background;
- how many objects will be processed;
- whether the chosen confidence threshold is suitable.

If the detection looks correct, disable **Only detect** and run the processing again.

### **Upscale or Variation** Tab

This is a general additional pass that can be executed along with detectors. It is not a mask detector, but alters the image using upscaling or variation methods.

Parameters of this tab:

- **Upscale or Variation**;
- **Order of Processing**;
- **Prompt**.

### **Upscale or Variation:**

The list contains the following options:

- `Disabled`;
- `Vary (Subtle)`;
- `Vary (Strong)`;
- `Upscale (1.5x)`;
- `Upscale (2x)`;
- `Final Upscale`.

The standard value is `Disabled`. While this option is selected, a separate upscaling or variation pass is not launched.

`Vary (Subtle)` and `Vary (Strong)` create variants of the source image with different degrees of change. `Upscale (1.5x)` and `Upscale (2x)` increase the image size. `Final Upscale` uses the final Fooocus upscaling mode.

Do not confuse this parameter with mask-based correction. Detectors perform local Inpaint on found areas, while Upscale or Variation works as a separate pass over the image.

### **Order of Processing**

Two options are available:

- `Before First Enhancement` is useful when you need to upscale small details first;
- `After Last Enhancement` is useful when you need to upscale already large areas after local corrections.

The standard value is `Before First Enhancement`.

If `Before First Enhancement` is selected, the Upscale or Variation pass is executed before the detector tabs. If `After Last Enhancement` is selected, it is executed after the last active detector.

### **Prompt**

This field appears when `After Last Enhancement` is selected for **Order of Processing**.

Available options:

- `Original Prompts`;
- `Last Filled Enhancement Prompts`.

`Original Prompts` uses the initial prompt and negative prompt. `Last Filled Enhancement Prompts` uses the last filled prompt settings of the processing.

If the processing order is set to `Before First Enhancement`, the **Prompt** field is hidden and its value does not affect the current run.

## Detector Tab Settings

Below are the settings located inside each numbered tab. The suffix at the end of the name depends on the tab: for example, the first tab contains `(1)`, the second contains `(2)`.

### **Enable this tab (1)**

Enables a specific tab. For the second tab, the label will be **Enable this tab (2)**, and so on.

Standard state:

- the first tab is enabled;
- the other tabs are disabled.

Even if a model is selected, it will not be processed if the tab's checkbox is unchecked. Similarly, a checked checkbox will not activate processing if the model is set to `None`.

Usually, one tab is enough. Enable a second one if you need to sequentially process a different type of object, for example, a face and hands.

### **ADetailer detector**

This is the model responsible for finding objects:

- `deepfashion2_yolov8s-seg.pt` — clothing and fashion objects;
- `face_yolov8m.pt` — faces;
- `face_yolov8n.pt` — lightweight face model;
- `face_yolov8n_v2.pt` — variant of the face model;
- `face_yolov8s.pt` — larger face model;
- `hand_yolov8n.pt` — lightweight hand model;
- `hand_yolov8s.pt` — larger hand model;
- `person_yolov8m-seg.pt`, `person_yolov8n-seg.pt`, `person_yolov8s-seg.pt` — people;
- `yolov8l-worldv2.pt`, `yolov8m-worldv2.pt`, `yolov8s-worldv2.pt`, `yolov8x-worldv2.pt` — YOLO World with class selection;
- `mediapipe_face_full.pt`;
- `mediapipe_face_short.pt`;
- `mediapipe_face_mesh.pt`;
- `mediapipe_face_mesh_eyes_only.pt`.

Choose the model based on the task:

- for faces, use `face_yolov8...` or MediaPipe face;
- for hands, use `hand_yolov8...`;
- for the whole person — `person_yolov8...-seg.pt`;
- for free text classes — YOLO World;
- for eyes — `mediapipe_face_mesh_eyes_only.pt`.

The value `None` means "do not process this tab". The internal check skips the tab if the model equals `None` or the tab's checkbox is unchecked.

### Difference between YOLO and MediaPipe

For YOLO, the code calls Ultralytics and gets a bounding box, confidence, and, if it is a segmentation model, a more precise mask. If there is no segmentation mask, a rectangular mask based on the bounding box is used.

For MediaPipe, special functions are used:

- `mediapipe_face_short` — short face detection model;
- `mediapipe_face_full` — full face detection model;
- `mediapipe_face_mesh` — mask along the face contour;
- `mediapipe_face_mesh_eyes_only` — eye masks.

MediaPipe Mesh does not use a standard face rectangle: the code builds a polygon based on landmarks. For eyes, separate areas are formed along the contour of the left and right eye.

### **ADetailer detector classes**

This field is hidden for standard detectors and becomes available for YOLO World models.

Enter classes separated by commas. For example:
```
person, bicycle, dog
```


If the field is left empty, the model's standard set of classes is used. For regular face, hand, and person models, this field does not change the detector.

### **ad_prompt (1)**

This is the prompt applied during the Inpaint of the found area. The field is empty by default.

If left empty, the main Fooocus prompt is used.

If you only need to correct a face, you can write, for example:
```
highly detailed face, natural eyes, realistic skin, symmetrical features
```

For hands:
```
anatomically correct hands, five fingers, natural pose, detailed skin
```


Do not describe the entire scene in this field if only local correction is required. A short prompt related to the object being corrected is usually easier to control.

### **ad_negative_prompt (1)**

This is the negative prompt for local Inpaint. The field is empty by default.

If left empty, the main Fooocus negative prompt is used.

Example for a face:
```
blurry face, asymmetry, deformed eyes, duplicate features, extra facial features
```

Example for hands:
```
extra fingers, fused fingers, missing fingers, malformed hands, blurry hands
```

### The **[PROMPT]** Token

You can use the special token `[PROMPT]` in the local prompt. It is replaced by the main Fooocus prompt.

For example:
```
[PROMPT], detailed realistic face, natural skin
```

In the negative prompt, `[PROMPT]` is replaced by the main negative prompt.

This allows you to keep the description of the entire scene and add local requirements for the object.

### The **[SEP]** Token

If the prompt has multiple parts separated by `[SEP]`, they are distributed to the found masks in order.

For example:
```
young woman face [SEP] old man face
```

If two faces are found, the first part will be applied to the first mask, and the second to the second. If there are fewer parts, the last part is repeated for the remaining masks. If there are more parts, the extra parts are not used.

The same rule applies to the negative prompt. If the prompt is empty, the main prompt is used.

### The **[SKIP]** Token

If the prompt for a specific mask is `[SKIP]`, this mask does not undergo Inpaint.

This is convenient when the detector has found multiple objects, but not all of them need to be corrected.

Important: `[SKIP]` does not forbid detection. It only skips the subsequent Inpaint for the selected mask.

## Detection Section

Open the **Detection** accordion to configure the filtering of found objects.

### **Detection model confidence threshold**

Range: `0.0` to `1.0`. Step: `0.01`. Standard value: `0.3`.

The detector returns a confidence score for each object. Objects with a score below the set threshold do not proceed further.

A lower value allows finding more objects but increases the risk of false positives. A higher value leaves only more confident detections but may miss a small, turned, or partially hidden face.

Start with the value `0.3`. If a face is not found, try `0.2`. If the detector captures extra items, increase the threshold to `0.4–0.6`.

### **Method to filter top k masks by (confidence or area)**

Two options are available:

- `Area`;
- `Confidence`.

This parameter is used only in conjunction with **Mask only the top k**.

`Area` keeps the largest bounding boxes. This is convenient when you need to process the largest faces or hands.

`Confidence` keeps objects with the highest detector confidence. This is convenient when there are multiple objects in the frame, but priority should be given to the most obvious ones.

The standard value is `Area`.

### **Mask only the top k (0 to disable)**

Range: `0` to `10`. Standard value: `0`.

`0` disables the limit: all found masks that passed the other filters are used.

The value `1` leaves only one object. The value `2` leaves two objects, and so on.

Which specific objects remain depends on the **Method to filter top k masks by**. With `Area`, the code selects the largest bounding boxes. With `Confidence`, it selects the most confident ones.

For one main face, it is often useful to set `1`. For a group portrait, leave `0` or set the number equal to the expected number of faces.

### **Mask min area ratio**

Range: `0.0–1.0`. Step: `0.001`. Standard value: `0.0`.

This parameter compares the area of the object's bounding box to the area of the entire image.

`0.0` does not discard small objects based on this criterion. If you set, for example, `0.01`, objects occupying less than 1% of the image area will be discarded.

This is useful for excluding tiny faces or random distant objects in the background. For a large portrait, leave it at `0.0`.

### **Mask max area ratio**

Range: `0.0–1.0`. Step: `0.001`. Standard value: `1.0`.

This parameter discards objects that occupy too large a proportion of the image.

For example, the value `0.5` will keep only objects with a bounding box area of no more than half the image area. The standard `1.0` practically does not limit the size.

This filter is useful if the person model mistakenly captures almost the entire frame, but for normal face correction, it is better not to change it.

Min/max filtering is performed before the top k limit.

## Mask Preprocessing Section

This section alters masks after detection and before Inpaint.

The order of application in the code is as follows:

1. x and y offset;
2. mask expansion or shrinking;
3. merging and/or inverting masks.

### **Mask x(→) offset**

Range: `-200` to `200`. Step: `1`. Standard value: `0`.

Shifts the mask horizontally. A positive value moves it to the right, a negative value to the left.

Use a small shift if the detector systematically captures the area next to the object or, conversely, fails to cover the required side. Large values can move the mask completely off the object.

### **Mask y(↑) offset**

Range: `-200` to `200`. Step: `1`. Standard value: `0`.

In the user label, the arrow points upward for a positive value. 

Practically, a positive value should be checked as an upward shift, and a negative value as a downward shift. Start with small values, for example, `±5` or `±10`.

### **Mask erosion (-) / dilation (+)**

Range: `-128` to `128`. Step: `4`. Standard value: `4`.

This is the expansion or shrinking of the mask:

- a positive value `+` expands the mask;
- a negative value `-` shrinks the mask;
- `0` leaves the size unchanged.

An expanded mask captures more surrounding pixels. This can help if the detector cuts off the chin, fingers, or hair edges, but too much expansion can alter the background.

Shrinking reduces interference with the surrounding area but may leave part of the object uncorrected.

### **Mask merge mode**

Available options:

- `None`;
- `Merge`;
- `Merge and Invert`.

#### `None`

Each mask is processed separately. If the detector found three faces, Fooocus performs three sequential local Inpaints with the same selected prompt or with different parts separated by `[SEP]`.

This is the main option if objects need different prompt settings.

#### `Merge`

All masks are merged into one. Fooocus performs a single Inpaint over the combined area.

This is useful when several close areas should be changed together. However, separate objects can no longer be processed with different prompts via independent passes.

#### `Merge and Invert`

First, all masks are merged, then inverted. As a result, the processing area becomes the opposite of the found objects.

Use this option with caution. It is not needed for normal face correction: it forces Inpaint to work *around* objects, not inside them.

## Inpainting Section

The settings in this section control the local redrawing of found masks.

### **Method**

Three modes are available in the list:

- `Inpaint or Outpaint (default)`;
- `Improve Detail (face, hand, eyes, etc.)`;
- `Modify Content (add objects, change background, etc.)`.

The standard mode is determined by the general Fooocus setting.

#### `Inpaint or Outpaint (default)`

Universal mode for local correction. Use it if you need to replace or redraw a found face, hand, or other area.

#### `Improve Detail (face, hand, eyes, etc.)`

Mode for improving details of faces, hands, and eyes. When selected, the UI automatically changes related parameters:

- disables **ADetailer Disable initial latent**;
- sets **Inpaint Engine** to `None`;
- sets denoising strength to `0.5`;
- sets **Inpaint Respective Field** to `0.0`.

The value `0.0` means processing only the mask area.

#### `Modify Content (add objects, change background, etc.)`

Mode for stronger content changes. When selected, the UI automatically:

- enables **ADetailer Disable initial latent**;
- returns the selected **Inpaint Engine** version;
- sets denoising strength to `1.0`;
- sets **Inpaint Respective Field** to `0.0`.

If you need to carefully restore a face, start with `Improve Detail`. If you need to replace the content of an area, use `Modify Content`.

### **ADetailer Disable initial latent**

Standard value: off.

When enabled, the initial latent is not used in local Inpaint. This can provide a more independent redrawing of the area, but sometimes reduces resemblance to the original fragment.

The value changes automatically when selecting **Method**:

- `Improve Detail` sets it to `False`;
- `Modify Content` sets it to `True`;
- normal mode returns it to `False`.

### **Inpaint Engine**

Available:

- `None`;
- `v1`;
- `v2.5`;
- `v2.6`.

Standard value: `v2.6`.

This is the Fooocus inpaint model version. It is recommended to use `Quality` or `Speed` performance when choosing an engine, because performance LoRAs with this mode are not applied optimally.

For `Improve Detail`, the UI automatically selects `None`. For normal mode and `Modify Content`, the currently selected engine version is used unless changed by the user.

### **Inpaint denoising strength**

Range: `0.0–1.0`. Step: `0.01`. In the visible interface, the standard value is `1`.

This parameter determines how strongly the local area will be changed relative to the source image.

- a lower value better preserves original details;
- a higher value gives the generator more freedom to change the area.

For careful face correction, it is usually worth starting with `0.3–0.5`. For full content replacement, use a higher value.

Since Adetailer usually processes a local mask, the main meaning of this parameter is the strength of change in the found area.

### **Inpaint Respective Field**

Range: `0.0–1.0`. Step: `0.01`. Standard value is `0.618` in normal mode.

This parameter determines what area around the mask participates in Inpaint:

- `0.0` corresponds to processing only the mask;
- `1.0` corresponds to processing the entire image;
- an intermediate value processes the mask and some surrounding area.

A large value gives the generator more context but may change the surrounding background. A small value better limits interference to the object area.

For a face, start with `0.0–0.4` if the background must remain unchanged. For complex transitions of hair, clothing, or lighting, try `0.5–0.7`.

## Separate CFG Scale

### **ADetailer use separate CFG scale**

Standard value: off.

When the checkbox is off, Adetailer uses the general Fooocus CFG scale. When enabled, a separate **ADetailer CFG scale** slider becomes available.

### **ADetailer CFG scale**

Range: `0.0–30.0`. Step: `0.5`. Standard value: `7.0`.

This parameter affects how strictly the local Inpaint follows the prompt.

A higher value usually pushes the image more toward the text description but can make the result less similar to the original area. A lower value softer preserves the original features.

A separate value is applied only when **ADetailer use separate CFG scale** is enabled. After the tab finishes, the previous general CFG scale is restored for subsequent operations.

## Separate Checkpoint Model

### **Use separate checkpoint**

Standard value: off.

When the checkbox is off, local Inpaint uses the same checkpoint as the main run.

When enabled, the **ADetailer checkpoint** list becomes available.

### **ADetailer checkpoint**

The first item in the list is `Use same checkpoint`, followed by available Fooocus checkpoint files.

If the **Use separate checkpoint** checkbox is checked, but `Use same checkpoint` is left in the list, the working branch does not change the base model. If a specific checkpoint is selected, it is used for the current Adetailer tab, after which the original checkpoint is restored.

This is useful if the main checkpoint is good at creating the scene, but another is better at drawing faces or hands. However, loading an additional model increases time and VRAM consumption.

## Separate Sampler and Scheduler

### **Use separate sampler** and **Use separate scheduler**

Standard value: off.

When enabled, **ADetailer sampler** and **ADetailer scheduler** become interactive.

### **ADetailer sampler**

The first option is `Use same sampler`, followed by Fooocus samplers, including standard options like `euler`, `dpmpp_2m`, and others present in the current build.

If `Use same sampler` is selected, the main sampler is preserved. If a specific sampler is selected, it applies only to the local Adetailer processing.

### **ADetailer scheduler**

The first option is `Use same scheduler`, followed by Fooocus schedulers, for example, `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `lcm`, `turbo`, `align_your_steps`, `tcd`, `edm_playground_v2.5`, and `beta57`.

The scheduler is applied separately only when **Use separate sampler** is enabled. If `Use same scheduler` is selected, the working code leaves the main scheduler.

For the first launch, leave the general sampler and scheduler. Separate values are needed if you already know that local correction works better with a different combination.

## Mask Debugging

In the advanced Fooocus settings, there is a checkbox:

> **Debug ADetailer Masks**

It is not located inside a specific detector tab. When enabled, Fooocus shows a preview of detection and masks during processing, and saves intermediate results according to general output settings.

Use it if the result looks incorrect:

- the mask does not hit the face;
- the detector captures the background;
- the wrong object is selected;
- the face is found but not redrawn.

Combined with **Only detect**, this mode allows checking detection without Inpaint.

## How to Configure Adetailer for Faces

### Soft correction of one face

1. Enable **Adetailer**.
2. In the first tab, enable **Enable this tab (1)**.
3. Select `face_yolov8n.pt`.
4. Set confidence to around `0.3`.
5. Leave **Mask only the top k** at `0` if there are multiple faces in the image.
6. Select **Improve Detail (face, hand, eyes, etc.)**.
7. Start with a denoising strength of around `0.4–0.5`.
8. Set **Inpaint Respective Field** to around `0.0–0.3`.
9. In the prompt, add `[PROMPT], detailed realistic face, natural eyes` or leave the field empty to use the main prompt.
10. Start generation.

### Processing only the largest face

Use:

- **Method to filter top k masks by:** `Area`;
- **Mask only the top k:** `1`.

Thus, Adetailer will keep the largest bounding box. This mode is convenient for portraits where small faces might be in the background.

### Processing the most confidently found face

Use:

- **Method to filter top k masks by:** `Confidence`;
- **Mask only the top k:** `1`.

Thus, priority will be given to the detection with the highest confidence, not necessarily the largest one.

### Correcting a group of faces

Leave **Mask only the top k** at `0`. If all faces need the same prompt, you can leave **Mask merge mode** at `None` and process them sequentially. If the faces are closely located and should change together, try `Merge`.

## How to Configure Adetailer for Hands

1. Enable a separate tab.
2. Select `hand_yolov8n.pt` for lighter processing or `hand_yolov8s.pt` for a larger detector.
3. Start with confidence `0.3`.
4. If fingers are cut off, increase **Mask erosion (-) / dilation (+)** from `4` to `8–16`.
5. Use a prompt like `anatomically correct hands, five fingers, natural pose`.
6. For strong hand replacement, increase denoising strength, but check a small result first.

The hand detector can find multiple areas. If you need to fix only one, use top k or separate the prompt via `[SEP]`.

## How to Configure Eyes

For narrow eye correction, select `mediapipe_face_mesh_eyes_only.pt`.

This detector builds masks along the eye contours, not the entire face. Therefore, when correcting eye color or shape, it is better to use a small **Inpaint Respective Field** and moderate denoising strength.

If the mask is too small, do not start with strong expansion. Try a small positive value for **Mask erosion (-) / dilation (+)** and enable **Debug ADetailer Masks** to see the actual area.

## How to Use YOLO World

Select one of the `yolov8...-worldv2.pt` detectors. After this, the **ADetailer detector classes** field will appear.

Enter classes separated by commas in English, for example:
```
hat, glasses, backpack
```


YOLO World is convenient when you need to fix an object for which there is no separate standard model. Accuracy depends on the model quality and class formulation, so always check the result via **Only detect** or **Debug ADetailer Masks**.

## How to Tell if the Wrong Mask is Selected

Symptoms of an incorrect mask:

- the background around the face changes;
- only part of the object is corrected;
- the face or hand does not change;
- multiple unwanted objects change at once;
- the result looks like a large new insert.

Safe check order:

1. Enable **Only detect**.
2. Enable **Debug ADetailer Masks**.
3. Raise or lower the confidence.
4. Use **Mask min area ratio** to remove objects that are too small.
5. Use **Mask max area ratio** to exclude overly large erroneous areas.
6. Limit the number of objects via **Mask only the top k**.
7. Adjust dilation or offsets.
8. Only after this, return to normal Inpaint mode.

## What Happens with Multiple Active Detectors

Fooocus processes each active detector for each image sequentially.

If face and hands are enabled, the order is roughly:

1. face detection and Inpaint;
2. the result with the corrected face becomes the input;
3. hand detection and Inpaint;
4. the final result is returned to the user.

If Upscale or Variation is enabled, it is added as a separate pass before the first or after the last tab according to **Order of Processing**.

Processing time grows with the number of images, active tabs, and found masks. If one detector finds five faces, local Inpaint may be executed for each mask separately in `None` mode.

## Model and Detector Download

On the first launch, model download may take some time. This is normal. If the model did not download, check your internet connection and available disk space.

## Results and Saving

Adetailer does not create a separate gallery or ZIP archive of results. It alters the image in the current Fooocus workflow.

After normal generation, the result is displayed in the standard Fooocus gallery. When using an uploaded image, the outcome is also returned via the normal generation result.

Intermediate images may be shown if **Debug ADetailer Masks** or **Only detect** are enabled. The exact set of displayed intermediate results depends on general Fooocus settings, including disabling intermediate results and saving only the final image.

If multiple tabs are enabled and the user has a custom setting to save only the final enhanced image, intermediate passes may not be saved in the gallery. The last result remains the outcome of the chain.

Adetailer does not create a separate file with a fixed suffix for each mask, nor does it pack results into a ZIP. Therefore, do not look for the result in a separate batch processing folder.

## Recommended Settings by Task

### Face on a portrait

Use `face_yolov8n.pt` or `face_yolov8s.pt`, confidence `0.3`, `Mask only the top k = 1`, mode `Improve Detail`, denoising strength `0.4–0.5`, and Respective Field `0.0–0.3`.

### All faces on a group photo

Use a face model, leave top k at `0`, do not invert the mask, and leave the main prompt empty if the main prompt already describes the people.

### Hands

Use `hand_yolov8n.pt` or `hand_yolov8s.pt`, a prompt with anatomical constraints, and a small positive dilation if finger boundaries are cut off.

### Eyes

Use `mediapipe_face_mesh_eyes_only.pt`, moderate denoising strength, and a small Respective Field. Check the mask via debugging before running strong Inpaint.

### Replacing area content

Use `Modify Content`, denoising strength `1.0`, and a separate prompt. This mode changes the object more strongly and may affect the surroundings, so increase mask limitation only gradually.

### Correcting only the found area

Use `Improve Detail` or normal Inpaint, Respective Field closer to `0.0`, without mask inversion and without large dilation.

## Troubleshooting

### Adetailer does not start

Check three conditions:

- the general **Adetailer** toggle is enabled;
- at least one **Enable this tab** is enabled;
- a model is selected in this tab, not `None`.

### Image generates, but the face is not corrected

Probable causes:

- the detector did not find the face;
- confidence is too high;
- the mask was filtered by ratio parameters;
- **Only detect** is enabled;
- the prompt equals `[SKIP]`;
- the mask became empty after erosion.

Enable **Only detect** and **Debug ADetailer Masks**, lower confidence to `0.2–0.3`, and check the preview.

### The wrong face is corrected

Set **Mask only the top k = 1** and choose the method `Area` or `Confidence`. If you need to select a face by position, this version of the user sorting toggle does not have it: the code uses fixed left-to-right sorting before processing. Therefore, for precise selection, use top k, ratio filters, or `[SEP]` and `[SKIP]`.

### Detector captures the background

Increase confidence. For small random objects, use **Mask min area ratio**, for example, `0.01`. For too large erroneous masks, limit **Mask max area ratio**.

### Mask is too small

Increase **Mask erosion (-) / dilation (+)** to `8` or `12`. If the object is offset, try a small x/y offset.

### Mask is too large

Decrease dilation or set a negative value, for example, `-4`. Decrease Respective Field so Inpaint refers less to the surrounding area.

### Mask shifted to the side

Check **Mask x(→) offset** and **Mask y(↑) offset**. Return both values to `0`, then change only one parameter at a time by 5–10 points.

### All found areas merge unexpectedly

Check **Mask merge mode**. With `Merge`, all masks turn into one. With `Merge and Invert`, the processing area is inverted. For independent correction, select `None`.

### Local area changed too much

Decrease denoising strength, Respective Field, and CFG. Check if `Modify Content` is selected, which by code automatically sets the change strength to `1.0`.

### Local area barely changed

Increase denoising strength or use a more suitable prompt. Ensure the mask actually covers the object and `[SKIP]` is not enabled.

### Prompt tab disappeared

This is expected if **Order of Processing** is set to `Before First Enhancement`. The field is shown only for `After Last Enhancement`.

### Separate CFG does not work

Check if **ADetailer use separate CFG scale** is enabled. The slider itself is inactive by default and applies only after checking the box.

### Separate checkpoint does not work

Check if **Use separate checkpoint** is enabled, and if a specific checkpoint is selected instead of `Use same checkpoint`.

### Processing takes too long

Decrease the number of active tabs, disable unnecessary Upscale or Variation, limit top_k, and use a light model like `face_yolov8n.pt` or `hand_yolov8n.pt`.

### Not enough VRAM

Disable separate checkpoint if it is not needed. Decrease the number of active detectors and do not enable multiple heavy models simultaneously. For the first test, use one light model and one object.

## Safe Configuration Order

To avoid changing too many parameters at once, use the following sequence:

1. Select one active tab.
2. Check the model via **Only detect**.
3. Enable **Debug ADetailer Masks**.
4. Adjust confidence.
5. If necessary, limit top k.
6. Check min/max area ratio.
7. Adjust dilation.
8. Select Method.
9. Adjust denoising strength and Respective Field.
10. Add a local prompt.
11. Run normal Inpaint.
12. Only after a successful result, add a second tab or a separate model.

## Quick Reference

- Adetailer automatically detects objects and corrects them via local Inpaint.
- Main application areas: faces, eyes, hands, people, and custom YOLO World classes.
- Processing is performed after normal generation or for **one** uploaded image.
- The first detector tab is enabled by default, the rest are disabled.
- Only a tab with **Enable this tab** checked and a selected model is active simultaneously.
- `Only detect` shows detection without Inpaint.
- Empty prompt and negative prompt inherit the main Fooocus fields.
- `[PROMPT]` adds the main prompt, `[SEP]` separates prompts by masks, `[SKIP]` skips a specific mask.
- Confidence regulates the detector's confidence filter.
- Top k limits the number of objects.
- Area ratio discards areas that are too small or too large.
- Dilation expands the mask, erosion shrinks it.
- `Merge` combines masks, `Merge and Invert` additionally inverts them.
- **Improve Detail** is suitable for faces, hands, and eyes.
- **Modify Content** is intended for strong content replacement.
- Separate CFG, checkpoint, and sampler work only after enabling the corresponding checkboxes.
- Adetailer does not create a separate ZIP with results: the outcome appears in the standard Fooocus gallery.

## Что делает Adetailer
([Read in English](#what-adetailer-does))

`Adetailer` автоматически находит лица, глаза, руки, людей или другие объекты на изображении, строит для них маски и запускает локальную перерисовку. Благодаря этому можно исправить лицо или руки после генерации, не перерисовывая всю картинку.

Принцип работы состоит из трёх этапов:

1. Fooocus создаёт изображение или принимает одно готовое изображение.
2. Детектор Adetailer находит объекты и формирует маски.
3. Fooocus выполняет Inpaint по найденным маскам с отдельным текстом и настройками.

В Fooocus Extend Adetailer объединён с логикой **Enhance**. Поэтому в интерфейсе он использует часть параметров увеличения и вариации, а вкладка выбора изображения называется **ADetailer**.

## Что важно знать заранее

Adetailer поддерживает два практических сценария:

- обработка результата, который Fooocus только что сгенерировал;
- обработка **одного** загруженного изображения через поле **Use with Adetailer, skips image generation**.

Во втором сценарии Fooocus пропускает обычную генерацию и сразу переходит к детекции и Inpaint.

## Как включить Adetailer

На главной странице Fooocus Extend найдите переключатель **Adetailer** рядом с **Input Image** и **Enhance**.

По умолчанию переключатель Adetailer выключен. После включения появляется блок с вкладкой **ADetailer** и настройками расширения.

Внутри блока находятся:

- вкладка выбора исходного изображения;
- вкладка **Upscale or Variation**;
- несколько вкладок отдельных детекторов;
- дополнительные параметры отладки в группе расширенных настроек.

По умолчанию конфигурация создаёт пять вкладок детектора. Первая вкладка включена, остальные выключены.

## Быстрый первый запуск после генерации

Это самый простой способ проверить, что расширение работает.

1. Оставьте переключатель **Adetailer** выключенным, пока не заполните основной prompt и negative prompt.
2. Включите **Adetailer**.
3. Откройте первую вкладку детектора.
4. Убедитесь, что включён флажок **Enable this tab (1)**.
5. В поле **ADetailer detector** выберите модель лица, например `face_yolov8n.pt` или `face_yolov8s.pt`.
6. Оставьте остальные параметры по умолчанию.
7. Запустите генерацию.
8. После создания основного изображения Fooocus покажет этап **Processing ADetailer ...**.
9. Дождитесь завершения. Детектор найдёт лицо, после чего Fooocus перерисует найденную область.

Для первого теста лучше использовать один объект и одну активную вкладку. Если сразу включить несколько детекторов, обработка станет дольше, а определить источник изменений будет сложнее.

## Обработка одного готового изображения

В блоке входных изображений есть вкладка **ADetailer**. В ней находится поле:

> **Use with Adetailer, skips image generation**

Загрузите туда одно изображение. При запуске Fooocus пропустит базовую генерацию, использует загруженную картинку как исходную и перейдёт к Adetailer.

Этот режим удобен, когда изображение уже создано раньше, но на нём нужно поправить лицо, руки или другой объект.

Практический порядок:

1. Включите **Adetailer**.
2. Откройте вкладку **ADetailer** среди полей входного изображения.
3. Загрузите одну картинку.
4. Заполните настройки детектора.
5. При необходимости задайте отдельный prompt для исправляемого объекта.
6. Запустите обработку.

В этом режиме основной prompt используется как запасной текст, если отдельное поле Adetailer prompt оставлено пустым.

## Общая логика вкладок детектора

Каждая вкладка детектора — самостоятельный проход обработки. Можно включить несколько вкладок и, например, сначала исправить лицо, а затем руки.

В активную обработку попадают только вкладки, где одновременно выполнены два условия:

- включён флажок **Enable this tab (...);**
- в поле **ADetailer detector** выбрана модель, отличная от `None`.

Неактивные вкладки полностью пропускаются. Если детектор не нашёл объект, эта вкладка не выполняет Inpaint и переходит дальше.

Порядок вкладок имеет значение. Fooocus обрабатывает их последовательно. Результат первой вкладки становится входом для следующей.

## Верхние режимы Adetailer

### **Only detect**

Флажок **Only detect** запускает детекцию и показывает предварительный результат без последующего Inpaint.

При включении этого режима метод увеличения отключается, детектор всё равно строит маски и выводит изображение предпросмотра с разметкой найденных объектов. Само исправление лица, руки или другого объекта не выполняется.

Используйте этот режим, если нужно проверить:

- находит ли модель нужный объект;
- не захватывает ли она фон;
- сколько объектов будет обработано;
- подходит ли выбранный порог уверенности.

Если детекция выглядит правильно, отключите **Only detect** и запустите обработку повторно.

### Вкладка **Upscale or Variation**

Это общий дополнительный проход, который может выполняться вместе с детекторами. Он не является детектором масок, а изменяет изображение методом увеличения или вариации.

Параметры этой вкладки:

- **Upscale or Variation:**
- **Order of Processing**;
- **Prompt**.

### **Upscale or Variation:**

Список содержит следующие варианты:

- `Disabled`;
- `Vary (Subtle)`;
- `Vary (Strong)`;
- `Upscale (1.5x)`;
- `Upscale (2x)`;
- `Final Upscale`.

Стандартное значение — `Disabled`. Пока выбран этот вариант, отдельный проход увеличения или вариации не запускается.

`Vary (Subtle)` и `Vary (Strong)` создают варианты исходного изображения с разной степенью изменения. `Upscale (1.5x)` и `Upscale (2x)` увеличивают изображение. `Final Upscale` использует финальный режим увеличения Fooocus.

Не путайте этот параметр с исправлением по маске. Детекторы выполняют локальный Inpaint найденных областей, а Upscale or Variation работает как отдельный проход над изображением.

### **Order of Processing**

Доступны два варианта:

- `Before First Enhancement` полезен, когда нужно сначала увеличить мелкие детали;;
- `After Last Enhancement` полезен, когда нужно увеличить уже крупные области после локальных исправлений.

Стандартное значение — `Before First Enhancement`.

Если выбран `Before First Enhancement`, проход Upscale or Variation выполняется до вкладок детектора. Если выбран `After Last Enhancement`, он выполняется после последнего активного детектора.

### **Prompt**

Поле появляется, когда для **Order of Processing** выбран вариант `After Last Enhancement`.

Доступны варианты:

- `Original Prompts`;
- `Last Filled Enhancement Prompts`.

`Original Prompts` использует исходные prompt и negative prompt. `Last Filled Enhancement Prompts` использует последние заполненные prompt-настройки обработки.

Если порядок обработки установлен в `Before First Enhancement`, поле **Prompt** скрывается и его значение не влияет на текущий запуск.

## Настройки вкладки детектора

Ниже описаны настройки, расположенные внутри каждой вкладки с номером. Суффикс в конце названия зависит от вкладки: например, первая вкладка содержит `(1)`, вторая — `(2)`.

### **Enable this tab (1)**

Включает конкретную вкладку. Для второй вкладки надпись будет **Enable this tab (2)** и так далее.

Стандартное состояние:

- первая вкладка включена;
- остальные вкладки выключены.

Даже выбранная модель не будет обработана, если флажок вкладки выключен. Аналогично, включённый флажок не активирует обработку, если модель установлена в `None`.

Обычно достаточно одной вкладки. Вторую включайте, если нужно последовательно обработать другой тип объекта, например лицо и руки.

### **ADetailer detector**

Это модель, которая отвечает за поиск объектов:

- `deepfashion2_yolov8s-seg.pt` — объекты одежды и моды;
- `face_yolov8m.pt` — лица;
- `face_yolov8n.pt` — лёгкая модель лиц;
- `face_yolov8n_v2.pt` — вариант модели лиц;
- `face_yolov8s.pt` — более крупная модель лиц;
- `hand_yolov8n.pt` — лёгкая модель рук;
- `hand_yolov8s.pt` — более крупная модель рук;
- `person_yolov8m-seg.pt`, `person_yolov8n-seg.pt`, `person_yolov8s-seg.pt` — люди;
- `yolov8l-worldv2.pt`, `yolov8m-worldv2.pt`, `yolov8s-worldv2.pt`, `yolov8x-worldv2.pt` — YOLO World с выбором классов;
- `mediapipe_face_full.pt`;
- `mediapipe_face_short.pt`;
- `mediapipe_face_mesh.pt`;
- `mediapipe_face_mesh_eyes_only.pt`.

Выбирайте модель по задаче:

- для лиц используйте `face_yolov8...` или MediaPipe face;
- для рук используйте `hand_yolov8...`;
- для всего человека — `person_yolov8...-seg.pt`;
- для свободного текстового класса — YOLO World;
- для глаз — `mediapipe_face_mesh_eyes_only.pt`.

Значение `None` означает «не обрабатывать эту вкладку». Внутренняя проверка пропускает вкладку, если модель равна `None` или флажок вкладки выключен.

### Разница между YOLO и MediaPipe

Для YOLO код вызывает Ultralytics и получает bounding box, уверенность и, если модель сегментационная, более точную маску. Если сегментационной маски нет, используется прямоугольная маска по bounding box. 

Для MediaPipe используются специальные функции:

- `mediapipe_face_short` — короткая модель обнаружения лица;
- `mediapipe_face_full` — полная модель обнаружения лица;
- `mediapipe_face_mesh` — маска по контуру лица;
- `mediapipe_face_mesh_eyes_only` — маски глаз.

MediaPipe Mesh не использует обычный прямоугольник лица: код строит полигон по landmarks. Для глаз формируются отдельные области по контуру левого и правого глаза.

### **ADetailer detector classes**

Это поле скрыто для обычных детекторов и становится доступным для моделей YOLO World.

Введите классы через запятую. Например:

```
person, bicycle, dog
```

Если оставить поле пустым, используется стандартный набор классов модели. Для обычных face, hand и person-моделей это поле не изменяет детектор.

### **ad_prompt (1)**

Это prompt, который применяется при Inpaint найденной области. Поле пустое по умолчанию.

Если оставить его пустым, используется основной prompt Fooocus.

Если нужно исправлять только лицо, можно написать, например:

```
highly detailed face, natural eyes, realistic skin, symmetrical features
```

Для рук:

```
anatomically correct hands, five fingers, natural pose, detailed skin
```

Не описывайте в этом поле всю сцену, если требуется только локальная коррекция. Короткий prompt, относящийся к исправляемому объекту, обычно проще контролировать.

### **ad_negative_prompt (1)**

Это negative prompt для локального Inpaint. Поле пустое по умолчанию.

Если его оставить пустым, используется основной negative prompt Fooocus.

Пример для лица:

```
blurry face, asymmetry, deformed eyes, duplicate features, extra facial features
```

Пример для рук:

```
extra fingers, fused fingers, missing fingers, malformed hands, blurry hands
```

### Токен **[PROMPT]**

В локальном prompt можно использовать специальный токен `[PROMPT]`. Он заменяется основным prompt Fooocus.

Например:

```
[PROMPT], detailed realistic face, natural skin
```

В negative prompt `[PROMPT]` заменяется основным negative prompt.

Это позволяет сохранить описание всей сцены и добавить локальные требования к объекту.

### Токен **[SEP]**

Если в prompt несколько частей разделены `[SEP]`, они распределяются по найденным маскам по порядку.

Например:

```
young woman face [SEP] old man face
```

При двух найденных лицах первая часть будет применена к первой маске, вторая — ко второй. Если частей меньше, последняя часть повторяется для оставшихся масок. Если частей больше, лишние части не используются.

То же правило действует для negative prompt. Если prompt пустой, используется основной prompt.

### Токен **[SKIP]**

Если prompt для конкретной маски равен `[SKIP]`, эта маска не проходит Inpaint.

Это удобно, когда детектор нашёл несколько объектов, но исправлять нужно не все.

Важно: `[SKIP]` не запрещает детекцию. Он пропускает только последующий Inpaint для выбранной маски.

## Раздел **Detection**

Откройте accordion **Detection**, чтобы настроить отбор найденных объектов.

### **Detection model confidence threshold**

Диапазон: от `0.0` до `1.0`. Шаг: `0.01`. Стандартное значение: `0.3`.

Детектор возвращает оценку уверенности для каждого объекта. Объекты с оценкой ниже установленного порога не проходят дальше.

Меньшее значение позволяет найти больше объектов, но повышает риск ложных срабатываний. Большое значение оставляет только более уверенные детекции, но может пропустить маленькое, повернутое или частично закрытое лицо.

Начните со значения `0.3`. Если лицо не находится, попробуйте `0.2`. Если детектор захватывает лишние предметы, увеличьте порог до `0.4–0.6`.

### **Method to filter top k masks by (confidence or area)**

Доступны два варианта:

- `Area`;
- `Confidence`.

Параметр используется только вместе с **Mask only the top k**.

`Area` оставляет крупнейшие bounding box. Это удобно, когда нужно обрабатывать самые большие лица или руки.

`Confidence` оставляет объекты с наибольшей уверенностью детектора. Это удобно, когда в кадре есть несколько объектов, но приоритет нужно отдать наиболее очевидным.

Стандартное значение — `Area`.

### **Mask only the top k (0 to disable)**

Диапазон: от `0` до `10`. Стандартное значение: `0`.

`0` отключает ограничение: используются все найденные маски, которые прошли остальные фильтры.

Значение `1` оставляет только один объект. Значение `2` — два объекта и так далее.

Какие именно объекты останутся, зависит от **Method to filter top k masks by**. При `Area` код выбирает самые большие bounding box. При `Confidence` — самые уверенные.

Для одного главного лица часто полезно поставить `1`. Для группового портрета оставьте `0` или поставьте число, равное ожидаемому количеству лиц.

### **Mask min area ratio**

Диапазон: `0.0–1.0`. Шаг: `0.001`. Стандартное значение: `0.0`.

Параметр сравнивает площадь bounding box объекта с площадью всего изображения.

`0.0` не отбрасывает маленькие объекты по этому критерию. Если поставить, например, `0.01`, объекты, занимающие менее 1% площади изображения, будут отброшены.

Это полезно для исключения маленьких лиц или случайных далёких объектов на фоне. Для крупного портрета оставляйте `0.0`.

### **Mask max area ratio**

Диапазон: `0.0–1.0`. Шаг: `0.001`. Стандартное значение: `1.0`.

Параметр отбрасывает объекты, которые занимают слишком большую долю изображения.

Например, значение `0.5` оставит только объекты с площадью bounding box не более половины площади изображения. Стандартное `1.0` практически не ограничивает размер.

Этот фильтр полезен, если модель человека ошибочно захватывает почти весь кадр, но для обычной коррекции лица его лучше не менять.

Фильтрация по min/max выполняется до ограничения top k.

## Раздел **Mask Preprocessing**

Этот раздел изменяет маски после детекции и до Inpaint.

Порядок применения в коде такой:

1. смещение по x и y;
2. расширение или сужение маски;
3. объединение и/или инверсия масок.

### **Mask x(→) offset**

Диапазон: от `-200` до `200`. Шаг: `1`. Стандартное значение: `0`.

Сдвигает маску по горизонтали. Положительное значение перемещает её вправо, отрицательное — влево.

Используйте небольшой сдвиг, если детектор систематически захватывает область рядом с объектом или, наоборот, не покрывает нужную сторону. Большие значения могут полностью увести маску с объекта.

### **Mask y(↑) offset**

Диапазон: от `-200` до `200`. Шаг: `1`. Стандартное значение: `0`.

В пользовательской подписи стрелка показывает направление вверх для положительного значения. 

Практически положительное значение следует проверять как смещение вверх, а отрицательное — вниз. Начинайте с небольших значений, например `±5` или `±10`.

### **Mask erosion (-) / dilation (+)**

Диапазон: от `-128` до `128`. Шаг: `4`. Стандартное значение: `4`.

Это расширение или сужение маски:

- положительное значение `+` расширяет маску;
- отрицательное значение `-` сужает маску;
- `0` оставляет размер без изменения.

Расширенная маска захватывает больше окружающих пикселей. Это может помочь, если детектор обрезает подбородок, пальцы или края волос, но слишком большое расширение способно изменить фон.

Сужение уменьшает вмешательство в окружающую область, но может оставить часть объекта неисправленной.

### **Mask merge mode**

Доступны варианты:

- `None`;
- `Merge`;
- `Merge and Invert`.

#### `None`

Каждая маска обрабатывается отдельно. Если детектор нашёл три лица, Fooocus выполняет три последовательных локальных Inpaint с одним и тем же выбранным prompt или с разными частями, разделёнными `[SEP]`.

Это основной вариант, если объектам нужны разные prompt-настройки.

#### `Merge`

Все маски объединяются в одну. Fooocus выполняет один Inpaint по общей области.

Это полезно, когда несколько близких областей должны изменяться совместно. Однако отдельные объекты уже нельзя обработать разными prompt через независимые проходы.

#### `Merge and Invert`

Сначала все маски объединяются, затем инвертируются. В результате область обработки становится противоположной найденным объектам.

Используйте этот вариант осторожно. Для обычного исправления лица он не нужен: он заставляет Inpaint работать вокруг объектов, а не внутри них.

## Раздел **Inpainting**

Настройки этого раздела управляют локальной перерисовкой найденных масок.

### **Method**

В списке доступны три режима:

- `Inpaint or Outpaint (default)`;
- `Improve Detail (face, hand, eyes, etc.)`;
- `Modify Content (add objects, change background, etc.)`.

Стандартный режим определяется общей настройкой Fooocus.

#### `Inpaint or Outpaint (default)`

Универсальный режим для локального исправления. Используйте его, если нужно заменить или перерисовать найденное лицо, руку или другую область.

#### `Improve Detail (face, hand, eyes, etc.)`

Режим для улучшения деталей лица, рук и глаз. При выборе UI автоматически меняет связанные параметры:

- отключает **ADetailer Disable initial latent**;
- устанавливает **Inpaint Engine** в `None`;
- устанавливает denoising strength в `0.5`;
- устанавливает **Inpaint Respective Field** в `0.0`.

Значение `0.0` означает обработку только области маски.

#### `Modify Content (add objects, change background, etc.)`

Режим для более сильного изменения содержимого. При его выборе UI автоматически:

- включает **ADetailer Disable initial latent**;
- возвращает выбранную версию **Inpaint Engine**;
- устанавливает denoising strength в `1.0`;
- устанавливает **Inpaint Respective Field** в `0.0`.

Если нужно именно аккуратно восстановить лицо, начинайте с `Improve Detail`. Если нужно заменить содержимое области, используйте `Modify Content`.

### **ADetailer Disable initial latent**

Стандартное значение — выключено.

При включении начальный latent не используется в локальном Inpaint. Это может дать более независимую перерисовку области, но иногда уменьшает сходство с исходным фрагментом.

Значение автоматически меняется при выборе **Method**:

- `Improve Detail` устанавливает его в `False`;
- `Modify Content` устанавливает его в `True`;
- обычный режим возвращает его в `False`.

### **Inpaint Engine**

Доступны:

- `None`;
- `v1`;
- `v2.5`;
- `v2.6`.

Стандартное значение — `v2.6`.

Это версия inpaint-модели Fooocus. Рекомендуется при выборе движка использовать производительность `Quality` или `Speed`, потому что performance LoRA с этим режимом не применяются оптимальным образом.

Для `Improve Detail` UI автоматически выбирает `None`. Для обычного режима и `Modify Content` используется текущая выбранная версия движка, если её не изменил пользователь.

### **Inpaint denoising strength**

Диапазон: `0.0–1.0`. Шаг: `0.01`. В видимом интерфейсе стандартное значение — `1`.

Параметр определяет, насколько сильно локальная область будет изменена относительно исходного изображения.

- меньшее значение лучше сохраняет исходные детали;
- большее значение даёт генератору больше свободы изменить область.

Для аккуратной коррекции лица обычно стоит начать с `0.3–0.5`. Для полной замены содержимого используйте более высокое значение.

Для Adetailer обычно обрабатывается локальная маска, поэтому основной смысл параметра — сила изменения найденной области.

### **Inpaint Respective Field**

Диапазон: `0.0–1.0`. Шаг: `0.01`. Стандартное значение — `0.618` в обычном режиме.

Параметр определяет, какая область вокруг маски участвует в Inpaint:

- `0.0` соответствует обработке только маски;
- `1.0` соответствует обработке всего изображения;
- промежуточное значение обрабатывает маску и некоторую окружающую область.

Большое значение даёт генератору больше контекста, но может изменить окружающий фон. Малое значение лучше ограничивает вмешательство областью объекта.

Для лица начните с `0.0–0.4`, если фон должен остаться неизменным. Для сложного перехода волос, одежды или освещения попробуйте `0.5–0.7`.

## Отдельный CFG scale

### **ADetailer use separate CFG scale**

Стандартное значение — выключено.

Когда флажок выключен, Adetailer использует общий CFG scale Fooocus. Когда включён, становится доступным отдельный ползунок **ADetailer CFG scale**.

### **ADetailer CFG scale**

Диапазон: `0.0–30.0`. Шаг: `0.5`. Стандартное значение: `7.0`.

Параметр влияет на то, насколько строго локальный Inpaint следует prompt.

Большое значение обычно сильнее подталкивает изображение к текстовому описанию, но может сделать результат менее похожим на исходную область. Малое значение мягче сохраняет исходные признаки.

Отдельное значение применяется только при включённом **ADetailer use separate CFG scale**. После завершения вкладки прежний общий CFG scale восстанавливается для следующих операций.

## Отдельная модель checkpoint

### **Use separate checkpoint**

Стандартное значение — выключено.

Когда флажок выключен, локальный Inpaint использует тот же checkpoint, что и основной запуск.

При включении становится доступен список **ADetailer checkpoint**.

### **ADetailer checkpoint**

Первый пункт списка — `Use same checkpoint`, затем идут доступные checkpoint-файлы Fooocus.

Если установлен флажок **Use separate checkpoint**, но в списке оставлено `Use same checkpoint`, рабочая ветка не меняет базовую модель. Если выбран конкретный checkpoint, он используется для текущей вкладки Adetailer, после чего исходный checkpoint восстанавливается.

Это полезно, если основной checkpoint хорошо создаёт сцену, а другой лучше рисует лица или руки. Однако загрузка дополнительной модели увеличивает время и расход видеопамяти.

## Отдельный sampler и scheduler

### **Use separate sampler** и **Use separate scheduler**

Стандартное значение — выключено.

При включении становятся интерактивными **ADetailer sampler** и **ADetailer scheduler**.

### **ADetailer sampler**

Первый вариант — `Use same sampler`, после него доступны sampler-ы Fooocus, включая стандартные варианты вроде `euler`, `dpmpp_2m` и другие, присутствующие в текущей сборке.

Если выбран `Use same sampler`, основной sampler сохраняется. Если выбран конкретный sampler, он применяется только к локальной обработке Adetailer.

### **ADetailer scheduler**

Первый вариант — `Use same scheduler`, после него доступны scheduler-ы Fooocus, например `normal`, `karras`, `exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `lcm`, `turbo`, `align_your_steps`, `tcd`, `edm_playground_v2.5` и `beta57`

Scheduler применяется отдельно только при включённом **Use separate sampler**. Если выбран `Use same scheduler`, рабочий код оставляет основной scheduler.

Для первого запуска оставьте общий sampler и scheduler. Отдельные значения нужны, если вы уже знаете, что локальная коррекция лучше работает с другим сочетанием.

## Отладка масок

В расширенных настройках Fooocus есть флажок:

> **Debug ADetailer Masks**

Он не находится внутри конкретной вкладки детектора. При включении Fooocus показывает предпросмотр детекции и масок в ходе обработки, а также сохраняет промежуточные результаты согласно общим настройкам вывода.

Используйте его, если результат выглядит неправильно:

- маска не попадает на лицо;
- детектор захватывает фон;
- выбирается не тот объект;
- лицо найдено, но не перерисовано.

В сочетании с **Only detect** этот режим позволяет проверить детекцию без Inpaint.

## Как настроить Adetailer для лица

### Мягкое исправление одного лица

1. Включите **Adetailer**.
2. В первой вкладке включите **Enable this tab (1)**.
3. Выберите `face_yolov8n.pt`.
4. Установите confidence около `0.3`.
5. Оставьте **Mask only the top k** равным `0`, если на изображении несколько лиц.
6. Выберите **Improve Detail (face, hand, eyes, etc.)**.
7. Начните с denoising strength около `0.4–0.5`.
8. Установите **Inpaint Respective Field** около `0.0–0.3`.
9. В prompt добавьте `[PROMPT], detailed realistic face, natural eyes` или оставьте поле пустым, чтобы использовать основной prompt.
10. Запустите генерацию.

### Обработка только самого крупного лица

Используйте:

- **Method to filter top k masks by:** `Area`;
- **Mask only the top k:** `1`.

Так Adetailer оставит крупнейший bounding box. Этот режим удобен для портрета, где на фоне могут быть маленькие лица.

### Обработка самого уверенно найденного лица

Используйте:

- **Method to filter top k masks by:** `Confidence`;
- **Mask only the top k:** `1`.

Так приоритет получит детекция с самой высокой уверенностью, а не обязательно самая крупная.

### Исправление группы лиц

Оставьте **Mask only the top k** равным `0`. Если всем лицам нужен один и тот же prompt, можно оставить **Mask merge mode** равным `None` и обработать их последовательно. Если лица тесно расположены и должны изменяться вместе, попробуйте `Merge`.

## Как настроить Adetailer для рук

1. Включите отдельную вкладку.
2. Выберите `hand_yolov8n.pt` для более лёгкой обработки или `hand_yolov8s.pt` для более крупного детектора.
3. Начните с confidence `0.3`.
4. Если пальцы обрезаются, увеличьте **Mask erosion (-) / dilation (+)** с `4` до `8–16`.
5. Используйте prompt вроде `anatomically correct hands, five fingers, natural pose`.
6. Для сильной замены рук увеличьте denoising strength, но сначала проверьте небольшой результат.

Детектор рук может находить несколько областей. Если нужно исправить только одну, используйте top k или разделите prompt через `[SEP]`.

## Как настроить глаза

Для узкой коррекции глаз выберите `mediapipe_face_mesh_eyes_only.pt`.

Этот детектор строит маски по контурам глаз, а не по всему лицу. Поэтому при исправлении цвета или формы глаз лучше использовать небольшой **Inpaint Respective Field** и умеренный denoising strength.

Если маска слишком мала, не начинайте с сильного расширения. Попробуйте небольшое положительное значение **Mask erosion (-) / dilation (+)** и включите **Debug ADetailer Masks**, чтобы увидеть фактическую область.

## Как использовать YOLO World

Выберите один из детекторов `yolov8...-worldv2.pt`. После этого появится поле **ADetailer detector classes**.

Введите классы через запятую на английском, например:

```
hat, glasses, backpack
```

YOLO World удобен, когда нужно исправить объект, для которого нет отдельной стандартной модели. Точность зависит от качества модели и формулировки класса, поэтому обязательно проверяйте результат через **Only detect** или **Debug ADetailer Masks**.

## Как понять, что маска выбрана неправильно

Симптомы неправильной маски:

- изменяется фон вокруг лица;
- исправляется только часть объекта;
- лицо или рука не меняются;
- меняются сразу несколько нежелательных объектов;
- результат выглядит как большая новая вставка.

Порядок безопасной проверки:

1. Включите **Only detect**.
2. Включите **Debug ADetailer Masks**.
3. Поднимите или опустите confidence.
4. Используйте **Mask min area ratio** для удаления слишком маленьких объектов.
5. Используйте **Mask max area ratio** для исключения слишком крупных ошибочных областей.
6. Ограничьте число объектов через **Mask only the top k**.
7. Подправьте dilation или offsets.
8. Только после этого верните обычный режим Inpaint.

## Что происходит при нескольких активных детекторах

Fooocus обрабатывает каждый активный детектор для каждого изображения последовательно.

Если включены лицо и руки, порядок примерно такой:

1. детекция и Inpaint лица;
2. результат с исправленным лицом становится входом;
3. детекция и Inpaint рук;
4. итоговый результат возвращается пользователю.

Если включён Upscale or Variation, он добавляется как отдельный проход до первой или после последней вкладки согласно **Order of Processing**.

Время работы растёт с количеством изображений, активных вкладок и найденных масок. Если один детектор находит пять лиц, локальный Inpaint может выполняться для каждой маски отдельно при режиме `None`.

## Модель и загрузка детектора

При первом запуске загрузка модели может занять время. Это нормально. Если модель не скачалась, проверьте интернет-соединение и наличие свободного места.

## Результаты и сохранение

Adetailer не создаёт отдельную галерею или ZIP-архив результатов. Он изменяет изображение в текущем рабочем процессе Fooocus.

После обычной генерации результат отображается в стандартной галерее Fooocus. При использовании загруженного изображения итог также возвращается через обычный результат генерации.

Промежуточные изображения могут показываться, если включены **Debug ADetailer Masks** или **Only detect**. Точный набор отображаемых промежуточных результатов зависит от общих настроек Fooocus, включая отключение промежуточных результатов и сохранение только финального изображения.

Если включены несколько вкладок и пользовательская настройка сохранения только финального enhanced-изображения, промежуточные проходы могут не сохраняться в галерее. Последний результат остаётся итогом цепочки.

Adetailer не создаёт для каждой маски отдельный файл с фиксированным суффиксом и не упаковывает результаты в ZIP. Поэтому не ищите результат в отдельной папке пакетной обработки.

## Рекомендуемые настройки по задачам

### Лицо на портрете

Используйте `face_yolov8n.pt` или `face_yolov8s.pt`, confidence `0.3`, `Mask only the top k = 1`, режим `Improve Detail`, denoising strength `0.4–0.5` и Respective Field `0.0–0.3`.

### Все лица на групповой фотографии

Используйте face-модель, оставьте top k равным `0`, не инвертируйте маску и оставьте основной prompt пустым, если основной prompt уже описывает людей.

### Руки

Используйте `hand_yolov8n.pt` или `hand_yolov8s.pt`, prompt с анатомическими ограничениями и небольшое положительное dilation, если границы пальцев обрезаются.

### Глаза

Используйте `mediapipe_face_mesh_eyes_only.pt`, умеренный denoising strength и малый Respective Field. Проверяйте маску через отладку до запуска сильного Inpaint.

### Замена содержимого области

Используйте `Modify Content`, denoising strength `1.0` и отдельный prompt. Этот режим сильнее меняет объект и может затронуть окружение, поэтому увеличивайте ограничение маски только постепенно.

### Исправление только найденной области

Используйте `Improve Detail` или обычный Inpaint, Respective Field ближе к `0.0`, без инверсии маски и без большого dilation.

## Устранение проблем

### Adetailer не запускается

Проверьте три условия:

- включён общий флажок **Adetailer**;
- включён хотя бы один **Enable this tab**;
- в этой вкладке выбрана модель, а не `None`.

### Изображение генерируется, но лицо не исправляется

Вероятные причины:

- детектор не нашёл лицо;
- confidence слишком высокий;
- маска отфильтрована ratio-параметрами;
- включён **Only detect**;
- prompt равен `[SKIP]`;
- маска после erosion стала пустой.

Включите **Only detect** и **Debug ADetailer Masks**, снизьте confidence до `0.2–0.3` и проверьте предпросмотр.

### Исправляется не то лицо

Поставьте **Mask only the top k = 1** и выберите метод `Area` или `Confidence`. Если нужно выбирать лицо по позиции, в этой версии пользовательского переключателя сортировки нет: код использует фиксированную сортировку слева направо перед обработкой. Поэтому для точного выбора используйте top k, ratio-фильтры или `[SEP]` и `[SKIP]`.

### Детектор захватывает фон

Увеличьте confidence. Для мелких случайных объектов используйте **Mask min area ratio**, например `0.01`. Для слишком большой ошибочной маски ограничьте **Mask max area ratio**.

### Маска слишком маленькая

Увеличьте **Mask erosion (-) / dilation (+)** до `8` или `12`. Если объект находится со смещением, попробуйте небольшой x/y offset.

### Маска слишком большая

Уменьшите dilation или установите отрицательное значение, например `-4`. Уменьшите Respective Field, чтобы Inpaint меньше обращался к окружающей области.

### Маска ушла в сторону

Проверьте **Mask x(→) offset** и **Mask y(↑) offset**. Верните оба значения в `0`, затем меняйте только один параметр за раз на 5–10 пунктов.

### Все найденные области объединяются неожиданно

Проверьте **Mask merge mode**. При `Merge` все маски превращаются в одну. При `Merge and Invert` область обработки инвертируется. Для независимой коррекции выберите `None`.

### Локальная область слишком сильно изменилась

Уменьшите denoising strength, Respective Field и CFG. Проверьте, не выбран ли `Modify Content`, который по коду автоматически выставляет силу изменения `1.0`.

### Локальная область почти не изменилась

Увеличьте denoising strength или используйте более подходящий prompt. Убедитесь, что маска действительно покрывает объект и не включён `[SKIP]`.

### Вкладка Prompt исчезла

Это ожидаемо, если **Order of Processing** установлен в `Before First Enhancement`. Поле показывается только для `After Last Enhancement`.

### Отдельный CFG не действует

Проверьте, включён ли **ADetailer use separate CFG scale**. Сам ползунок по умолчанию неактивен и применяется только после включения флажка.

### Отдельный checkpoint не действует

Проверьте, включён ли **Use separate checkpoint**, и выбран ли конкретный checkpoint вместо `Use same checkpoint`.

### Обработка работает слишком долго

Уменьшите число активных вкладок, отключите лишний Upscale or Variation, ограничьте top_k и используйте лёгкую модель `face_yolov8n.pt` или `hand_yolov8n.pt`.

### Не хватает видеопамяти

Отключите отдельный checkpoint, если он не нужен. Уменьшите количество активных детекторов и не включайте одновременно несколько тяжёлых моделей. Для первого теста используйте одну лёгкую модель и один объект.

## Безопасный порядок настройки

Чтобы не менять сразу слишком много параметров, используйте такую последовательность:

1. Выберите одну активную вкладку.
2. Проверьте модель через **Only detect**.
3. Включите **Debug ADetailer Masks**.
4. Настройте confidence.
5. При необходимости ограничьте top k.
6. Проверьте min/max area ratio.
7. Подкорректируйте dilation.
8. Выберите Method.
9. Настройте denoising strength и Respective Field.
10. Добавьте локальный prompt.
11. Запустите обычный Inpaint.
12. Только после успешного результата добавляйте вторую вкладку или отдельную модель.

## Краткая памятка

- Adetailer автоматически детектирует объекты и исправляет их через локальный Inpaint.
- Основные области применения — лица, глаза, руки, люди и пользовательские классы YOLO World.
- Обработка выполняется после обычной генерации или для **одного** загруженного изображения.
- Первая вкладка детектора включена по умолчанию, остальные выключены.
- Одновременно активна только вкладка с включённым **Enable this tab** и выбранной моделью.
- `Only detect` показывает детекцию без Inpaint.
- Пустые prompt и negative prompt наследуют основные поля Fooocus.
- `[PROMPT]` добавляет основной prompt, `[SEP]` разделяет prompt по маскам, `[SKIP]` пропускает отдельную маску.
- Confidence регулирует фильтр уверенности детектора.
- Top k ограничивает количество объектов.
- Area ratio отбрасывает слишком маленькие или большие области.
- Dilation расширяет маску, erosion сужает её.
- `Merge` объединяет маски, `Merge and Invert` дополнительно инвертирует их.
- **Improve Detail** подходит для лица, рук и глаз.
- **Modify Content** предназначен для сильной замены содержимого.
- Отдельные CFG, checkpoint и sampler работают только после включения соответствующих флажков.
- Adetailer не создаёт отдельный ZIP с результатами: итог появляется в стандартной галерее Fooocus.
