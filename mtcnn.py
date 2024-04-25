import math
import torch
from torch import nn
from mtcnn_modules import PNet, RNet, ONet
from detect_face import generate_bounding_box, rerec, bbreg, pad, batched_nms

from torch.nn import functional as F
from torchvision.ops import boxes as B

class MTCNN(nn.Module):
    def __init__(self) -> None:
        super(MTCNN, self).__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, img: torch.Tensor, device: torch.device = torch.device('cuda')):
        '''
        Return faces' coordinates and landmark points.

            Parameters:
                img (torch.Tensor): Input image
                pnet (torch.nn.Module): PNet model of MTCNN
                rnet (torch.nn.Module): RNet model of MTCNN
                onet (torch.nn.Module): ONet model of MTCNN
                minsize (int): Minimal size of scale pyramid
                threshold (list[float]): Score thresholds for P-, R- and ONet
                factor (float): Scaling factor of scale pyramid
                device (torch.device): Device for inference
            Returns:
                batch_boxes (np.ndarray): Coordinates of bounding boxes
                batch_probs (np.ndarray): Probabilities of bounding boxes
        '''
        minsize: int = 20
        threshold: list[float] = [.6, .7, .7] 
        factor: float = 0.709

        imgs = img.unsqueeze(0) # When only one image.
        # model_dtype = next(self.pnet.parameters()).dtype
        imgs = imgs.permute(0, 3, 1, 2).float() # Convert from NHWC to NCHW.
        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        m = 12.0 / minsize
        minl = min(h, w) * m

        # Create scale pyramid.
        scales = [m * factor**i for i in range(int(math.log(12 / minl, factor)) + 1)]

        # First stage.
        boxes = []
        image_inds = []
        scale_picks = []
        offset = 0
        for scale in scales:
            # Scale the image.
            im_data = F.interpolate(
                imgs, [int(h * scale + 1), int(w * scale + 1)], mode='area'
            )
            im_data = (im_data - 127.5) * 0.0078125
            reg, probs = self.pnet(im_data)

            # Get bounding boxes.
            boxes_scale, image_inds_scale = generate_bounding_box(
                reg, probs[:, 1], scale, threshold[0]
            )
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)

            # NMS boxes.
            pick = B.batched_nms(
                boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5
            )
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]
        
        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0)
        scale_picks = torch.cat(scale_picks, dim=0)

        # NMS within each scale + image
        boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

        # NMS within each image
        pick = B.batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick]

        regw  = boxes[:, 2] - boxes[:, 0]
        regh  = boxes[:, 3] - boxes[:, 1]
        qq1   = boxes[:, 0] + boxes[:, 5] * regw
        qq2   = boxes[:, 1] + boxes[:, 6] * regh
        qq3   = boxes[:, 2] + boxes[:, 7] * regw
        qq4   = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = rerec(boxes)
        
        
        # Second stage
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h) # Pad boxes.
            # Scale the boxes.
            im_data = []
            for k in range(len(y)):
                if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
                    im_data.append(F.interpolate(
                            imgs[image_inds[k], :, y[k]-1:ey[k], x[k]-1:ex[k]].unsqueeze(0), 
                            (24, 24), mode='area',
                        ))

            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            out = self.rnet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > threshold[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = B.batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)

        # Third stage
        points = torch.zeros([0, 5, 2], device=device)
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h)
            im_data = []
            for k in range(len(y)):
                if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
                    im_data.append(F.interpolate(
                            imgs[image_inds[k], :, y[k]-1:ey[k], x[k]-1:ex[k]].unsqueeze(0), 
                            size=(48, 48), mode='area',
                        ))
                
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            
            # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
            out = self.onet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            points = out1
            ipass = score > threshold[2]
            points = points[:, ipass]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            w_i = boxes[:, 2] - boxes[:, 0] + 1
            h_i = boxes[:, 3] - boxes[:, 1] + 1
            points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
            points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
            points = torch.stack((points_x, points_y)).permute(2, 1, 0)
            boxes = bbreg(boxes, mv)

            # NMS within each image using "Min" strategy
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
            boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

        boxes = boxes.cpu()
        points = points.cpu()

        image_inds = image_inds.cpu()

        # batch_boxes = [
        #     boxes [torch.where(image_inds == b_i)] for b_i in range(batch_size)
        # ]

        # batch_points = [
        #     points[torch.where(image_inds == b_i)] for b_i in range(batch_size)
        # ]

        # batch_boxes, batch_points = torch.stack(boxes), torch.stack(points)

        return boxes.unsqueeze(0), points.unsqueeze(0)
