# BEN2

BEN2 (Background Erase Network) introduces a novel approach to foreground segmentation through its innovative Confidence Guided Matting (CGM) pipeline. The architecture employs a refiner network that targets and processes pixels where the base model exhibits lower confidence levels, resulting in more precise and reliable matting results

## Usage

### Single Image

```python
from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image


model = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2").to("cuda")

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240401162056.png"
)

foreground = model(image)[0]
foreground.save("foreground.png")
```

### Single image plus mask

```python
from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image


model = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2").to("cuda")

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/20240401162056.png"
)

foreground, mask = model(image, return_mask=True)
foreground[0].save("foreground.png")
mask[0].save("mask.png")
```

### Multiple Images

Note: The images need to be the same size to be able to use batches with them.

```python
from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image


model = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2").to("cuda")

image_one = load_image("C:/Users/Ozzy/Pictures/diffusers/chroma/20250715053400_4020590469.png")
image_two = load_image("C:/Users/Ozzy/Pictures/diffusers/chroma/20250715053811_709755595.png")

output = model([image_one, image_two])

for i, image in enumerate(output):
    image.save(f"./image_{i}.png")

```

### Multiple images plus masks

Note: The images need to be the same size to be able to use batches with them.

```python
from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image


model = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2").to("cuda")

image_one = load_image("C:/Users/Ozzy/Pictures/diffusers/chroma/20250715053400_4020590469.png")
image_two = load_image("C:/Users/Ozzy/Pictures/diffusers/chroma/20250715053811_709755595.png")

foregrounds, masks = model([image_one, image_two], return_mask=True)

for i, (image, mask) in enumerate(zip(foregrounds, masks)):
    image.save(f"./new_foreground_{i}.png")
    mask.save(f"./mask_{i}.png")
```

## Additional resources

* [Project page](https://github.com/PramaLLC/BEN2/)
* [Paper](https://huggingface.co/papers/2501.06230)
* [BEN](https://github.com/PramaLLC/BEN)