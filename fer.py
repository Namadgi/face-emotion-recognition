import torch.nn as nn

from mtcnn import MTCNN
# import onnxruntime as ort
import torchvision as tv
from fer_utils import postprocess_face, check_image_and_box
import torchvision.transforms.functional as F
import timm

class FER(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mtcnn = MTCNN()
        self.efnet = timm.create_model('efficientnet_b0')
        self.efnet.classifier = nn.Sequential(nn.Linear(1280, 7))
        self.image_processing = nn.Sequential(
            tv.transforms.Resize((224, 224)),
            # tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    def forward(self, x):
        # x = torch.from_numpy(np.array(full_image)).to(device)
        # full_image = Image.fromarray(x.numpy())
        y, _ = self.mtcnn(x)
        boxes, probs = postprocess_face(y)
        res = check_image_and_box(x, boxes, probs)
        if res != 0:
            return -res
        box = boxes[-1]
        left, top, right, bottom = [i.item() for i in box.int()]
        face = x[top:bottom, left:right, :].permute(2, 0, 1)
        face = face / 255
        face = self.image_processing(face)
        face = face.unsqueeze(0)
    
        results_ort = self.efnet(face).squeeze()
        label = int(results_ort.argmax().item())

        return label


