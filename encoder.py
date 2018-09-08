'''Encode object boxes and labels.'''
import math
import torch
import settings
from retinautils import meshgrid, box_iou, box_nms, change_box_order
import pdb

class DataEncoder:
    def __init__(self):
        #self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.anchor_areas = [14*14, 28*28, 56*56, 112*112, 224*224]
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()
        self.class_threshold = 0.2 #0.202
        self.nms_threshold = 0.5
        #self.anchor_boxes = self._get_anchor_boxes(torch.Tensor([settings.IMG_SZ, settings.IMG_SZ])).cuda()
        self.anchor_boxes_cpu = self._get_anchor_boxes(torch.Tensor([settings.IMG_SZ, settings.IMG_SZ]))

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            #pdb.set_trace()
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        #input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
        #             else torch.Tensor(input_size)
        
        anchor_boxes = self.anchor_boxes_cpu #self._get_anchor_boxes(input_size)
        if boxes.size()[0] == 0:
            loc_targets = torch.zeros_like(anchor_boxes)
            cls_targets = torch.zeros(anchor_boxes.size()[0]).long()
            return loc_targets, cls_targets
        
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = self.class_threshold #0.2
        NMS_THRESH = self.nms_threshold #0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self.anchor_boxes_cpu.cuda() #_get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        #print(cls_preds.size())
        #score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        score = cls_preds.sigmoid().squeeze()          # [#anchors,]
        print('score>>>', score.size())
        
        #labels += 1
        ids = score > CLS_THRESH
        print('ids size:', ids.size())
        print('BBOXES:', ids.sum().item())
        if ids.sum().item() < 0.5:
              return [], []
        ids = ids.nonzero().squeeze(1)             # [#obj,]
        if ids.sum().item() < 0.5:
            return [], []
        print(ids.size())
        print('>>>', boxes[ids].size())
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH).cuda()
        return boxes[ids][keep].cpu().tolist(), score[ids][keep].cpu().tolist()

def test():
    encoder = DataEncoder()
    boxes = torch.Tensor([[264, 152, 477, 531], [562, 152, 818, 605], [0, 0, 10, 10]])
    boxes = torch.Tensor([])
    labels = torch.LongTensor([1,1,1])
    b, c = encoder.encode(boxes, labels)
    print(b.size(), c.size())
    print(c)

if __name__ == '__main__':
    test()
