"""
From the CodeFormer project by author sczhou.
"""
import torch
import sys
sys.path.insert(0,'./extras.facexlib_custom/detection/yolov5face')
model = torch.load('extras.facexlib_custom/detection/yolov5face/yolov5n-face.pt', map_location='cpu')['model']
torch.save(model.state_dict(),'extras.facexlib_custom/weights/yolov5n-face.pth')