from torch import nn
from mtcnn_modules import PNet, RNet, ONet
from detect_face import detect_face, postprocess_face, check_image_and_box
from PIL import Image
import numpy as np
import torch

class MTCNN(nn.Module):
    def __init__(self) -> None:
        super(MTCNN, self).__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, full_image: Image.Image, device) -> Image.Image:
        x = torch.from_numpy(np.array(full_image)).to(device)
        y, _ = detect_face(x, self.pnet, self.rnet, self.onet)
        boxes, probs = postprocess_face(y)
        res = check_image_and_box(full_image, boxes, probs)
        if res != 0:
            return res
        box = boxes[0]
        face = full_image.crop(box)
        return face