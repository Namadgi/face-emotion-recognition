# import numpy as np
import torch
from typing import Tuple
# from PIL import Image
def postprocess_face(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Sort boxes by its area and extract probabilities.

        Parameters:
            boxes (np.ndarray): Coordinates of bounding boxes
        Returns:
            boxes (np.ndarray): Coordinates of bounding boxes
            probs (np.ndarray): Probabilities of bounding boxes
    '''
    b = boxes[0]
    b = b[torch.argsort((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
    boxes = b[:, :4].clone()
    probs = b[:,  4].clone()
    return boxes, probs

def check_image_and_box(img: torch.Tensor, boxes: torch.Tensor, score: torch.Tensor) -> int:
    '''
    Check image and face coordinates for errors.

        Parameters:
            img   (torch.Tensor): Whole input image
            boxes (torch.Tensor): Coordinates of bounding boxes
            score (torch.Tensor): Probabilities of each bounding box
        Returns:
            result (int): Code result of checking 
    '''
    h, w, _ = img.shape

    # if (h < w):
    #     return 5
    
    if len(boxes) == 0:
        return 2

    # Filter by scores.

    # idx = score > 0.8
    # boxes  = boxes[idx]
    
    # Filter by area.
    areas = torch.stack([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    img_area = h * w
    area_idx = (areas / img_area) > 0.1
    boxes  = boxes[area_idx]
    
    if len(boxes) < 1:
        return 2 # Zero faces or too small.
    
    if len(boxes) > 1:
        return 3 # More than one face on image.
    
    # Get the first element from the list.
    box   = boxes[0]
    
    x0, y0, x1, y1 = [i.item() for i in box] # Get coordinates.
    
    # Check coordinates to align with image's dimensions.
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return 4 # Face is not centered.
    
    return 0