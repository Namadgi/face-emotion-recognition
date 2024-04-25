import torch
from typing import Tuple

def bbreg(boxes: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    '''
    Tune bounding boxes' coordinates from regression results.
        
        Parameters:
            boxes (np.ndarray): Coordinates of bounding boxes
        Returns:
            boxes (np.ndarray): Coordinates of bounding boxes
    '''
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w  = boxes[:, 2] - boxes[:, 0] + 1
    h  = boxes[:, 3] - boxes[:, 1] + 1
    b1 = boxes[:, 0] + reg[:, 0] * w
    b2 = boxes[:, 1] + reg[:, 1] * h
    b3 = boxes[:, 2] + reg[:, 2] * w
    b4 = boxes[:, 3] + reg[:, 3] * h
    boxes[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boxes


def generate_bounding_box(
            reg: torch.Tensor, probs: torch.Tensor, scale: float, thresh: float
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Generate bounding box from results of PNet.

        Parameters:
            reg (torch.Tensor): Regression results of PNet
            probs (torch.Tensor): Probabilities of each bounding box
            scale (float): Specific scale from scale pyramid
            thresh (float): Score threshold for PNet only
        Returns:
            box (torch.Tensor): Coordinates of bounding boxes
            image_inds (torch.Tensor): Indeces of images in batch
    '''
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    box = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    
    return box, image_inds


def batched_nms(
            boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, 
            threshold: float, method: str
        ) -> torch.Tensor:
    '''
    Batched Non Maximum Suppression on Torch.
    
    Strategy: in order to perform NMS independently per class, we add an offset
    to all the boxes.     
    
    The offset is dependent only on the class idx, and is large enough so that 
    boxes from different classes do not overlap.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
            scores (torch.Tensor): Probabilities of each bounding box
            idxs (torch.Tensor): Indeces of images in batch
            threshold (float): Score threshold for PNet only
            method (str): Method for choosing the region
        Returns:
            keep (torch.Tensor): Indeces of selected bounding boxes.
    '''
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu()
    scores = scores.cpu()
    keep = nms(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def nms(
            boxes: torch.Tensor, scores: torch.Tensor, threshold: float, method: str
        ) -> torch.Tensor:
    '''
    Non Maximum Suppression algorithm implemented on Torch.

        Parameters:
            boxes (torch.Tensor): Regression results of PNet
            scores (torch.Tensor): Probabilities of each bounding box
            threshold (float): Score threshold for PNet only
            method (str): Method for choosing the region
        Returns:
            pick (torch.Tensor): Indeces of selected bounding boxes.
    '''
    if torch.numel(boxes) == 0:
        return torch.zeros((0, 3))

    x1 = boxes[:, 0].clone()
    y1 = boxes[:, 1].clone()
    x2 = boxes[:, 2].clone()
    y2 = boxes[:, 3].clone()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = torch.argsort(s)
    pick = torch.zeros_like(s, dtype=torch.int16)
    counter = 0
    while torch.numel(I) > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = torch.max(torch.tensor(x1[i]), x1[idx]).clone()
        yy1 = torch.max(y1[i], y1[idx]).clone()
        xx2 = torch.min(x2[i], x2[idx]).clone()
        yy2 = torch.min(y2[i], y2[idx]).clone()

        w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1).clone()
        h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1).clone()

        inter = w * h
        if method == 'Min':
            o = inter / torch.min(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        
        I = I[torch.nonzero(o <= threshold).squeeze()]

    pick = pick[:counter].clone()
    return pick


def pad(boxes: torch.Tensor, w: int, h: int) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Assure that bounding boxes' coordinates will not go out the image.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
            w (int): Width of input Image
            h (int): Height of input Image
        Returns:
            y  (np.ndarray): y0 coordinates of boxes
            ey (np.ndarray): y1 coordinates of boxes
            x  (np.ndarray): x0 coordinates of boxes
            ex (np.ndarray): x1 coordinates of boxes
    '''
    boxes = boxes.trunc().int().cpu()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(boxes: torch.Tensor) -> torch.Tensor:
    '''
    Make rectangle box more square-like.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
        Returns:
            boxes (torch.Tensor): Coordinates of bounding boxes
    '''
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    
    l = torch.max(w, h)
    boxes[:, 0]   = boxes[:, 0] + w * 0.5 - l * 0.5
    boxes[:, 1]   = boxes[:, 1] + h * 0.5 - l * 0.5
    boxes[:, 2:4] = boxes[:, :2] + l.repeat(2, 1).permute(1, 0)

    return boxes
