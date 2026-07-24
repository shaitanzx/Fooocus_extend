# RIFE - Real-Time Intermediate Flow Estimation for Video Frame Interpolation

Real-time video frame interpolation (VFI) is very useful in video processing, media players, and display devices. We propose RIFE, a Real-time Intermediate Flow Estimation algorithm for VFI. To realize a high-quality flow-based VFI method, RIFE uses a neural network named IFNet that can estimate the intermediate flows end-to-end with much faster speed. A privileged distillation scheme is designed for stable IFNet training and improve the overall performance. RIFE does not rely on pre-trained optical flow models and can support arbitrary-timestep frame interpolation with the temporal encoding input. Experiments demonstrate that RIFE achieves state-of-the-art performance on several public benchmarks. Compared with the popular SuperSlomo and DAIN methods, RIFE is 4--27 times faster and produces better results. Furthermore, RIFE can be extended to wider applications thanks to temporal encoding.

## Usage

### Images

```py
from image_gen_aux import RIFEFrameInterpolator
# using from_pretrained
model=RIFEFrameInterpolator.from_pretrained("1himan/RIFE")

# using load_model:
# model=RIFEFrameInterpolator("./")

outputs = model.interpolate_image("images/img0.png", "images/img1.png", exp=4)
print(f"Generated {len(outputs)} interpolated images")
```

### Video

```py
from image_gen_aux import RIFEFrameInterpolator

model=RIFEFrameInterpolator.from_pretrained("1himan/RIFE")

output_video = model.interpolate_video(
    # provide the correct video path, according to how you compiled this project
    video_path="videos/spider-man-electro.gif",
    output_path="videos/output_video.mp4",
    exp=2,
    scale=1.0,
    transfer_audio=True,
    # montage=True
)
print(f"Video saved to: {output_video}")
```

## Additional resources

* [Project page](https://github.com/hzwer/ECCV2022-RIFE)
* [Paper](https://huggingface.co/papers/2011.06294)