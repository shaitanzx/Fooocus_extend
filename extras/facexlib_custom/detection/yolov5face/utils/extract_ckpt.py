"""
From the CodeFormer project by author sczhou.
"""
import torch
import sys
sys.path.insert(0,'./facexlib/detection/yolov5face')
model = torch.load('facexlib/detection/yolov5face/yolov5n-face.pt', map_location='cpu')['model']
torch.save(model.state_dict(),'facexlib/weights/yolov5n-face.pth')