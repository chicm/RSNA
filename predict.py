import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import time
import os
import settings
import pandas as pd
import numpy as np

from dcom import get_val_ids, get_test_ids
from imgdataset import get_test_loader, get_train_loader

CKP_FILE = './ckps/best_1.pth'
batch_size = 4

def get_xywh(bbox):
    #bbox = [x / src_sz * tgt_sz for x in bbox]
    x_min = max(0.0, bbox[0])
    y_min = max(0.0, bbox[1])
    x_max = min(1024, bbox[2])
    y_max = min(1024, bbox[3])
    
    return x_min, y_min, x_max - x_min, y_max-y_min

def _get_prediction_string(bboxes, scores):
    prediction_list = []
    for bbox, score in zip(bboxes, scores):
        prediction_list.append(str(score))
        prediction_list.extend([str(i) for i in get_xywh(bbox)])
    prediction_string = " ".join(prediction_list)
    return prediction_string

def transform(image_ids, results):
    #self.decoder_dict = decoder_dict
    prediction_strings = []
    for bboxes, scores in results:
        prediction_strings.append(_get_prediction_string(bboxes, scores))
    submission = pd.DataFrame({'patientId': image_ids, 'PredictionString': prediction_strings})
    return {'submission': submission}


def predict():
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    print('==> Preparing data..')

    dloader = get_test_loader(get_test_ids(), img_dir=settings.TEST_IMG_DIR, batch_size=batch_size)
    print(dloader.num)

    # Model
    net = RetinaNet()
    net.load_state_dict(torch.load(CKP_FILE))
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    bgtime = time.time()
    encoder = DataEncoder()
    net.eval()
    prediction_strings = []
    for batch_idx, inputs in enumerate(dloader):
        inputs = Variable(inputs.cuda())
        
        _, loc_preds, cls_preds = net(inputs)
        print('{} / {}  {:.2f}'.format(batch_size*(batch_idx+1), dloader.num, (time.time() - bgtime)/60), end='\r')
        for i in range(len(loc_preds)):
            boxes, scores = encoder.decode(loc_preds[i].data, cls_preds[i].data, (settings.IMG_SZ, settings.IMG_SZ))
            prediction_strings.append(_get_prediction_string(boxes, scores))
    print(len(prediction_strings))
    print(prediction_strings[:3])
    submission = pd.DataFrame({'patientId': dloader.img_ids, 'PredictionString': prediction_strings})
    submission.to_csv('sub4.csv', index=False)

def evaluate_threshold(img_ids, cls_threshold, bbox_dict):

    dloader = get_test_loader(img_ids, img_dir=settings.IMG_DIR, batch_size=batch_size)
    
    # Model
    net = RetinaNet()
    net.load_state_dict(torch.load(CKP_FILE))
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    net.eval()

    bgtime = time.time()
    encoder = DataEncoder()
    encoder.class_threshold = cls_threshold
    true_objects_num = 0
    pred_objects_num = 0

    for batch_idx, inputs in enumerate(dloader):
        inputs = Variable(inputs.cuda())
        loc_preds, cls_preds = net(inputs)
        
        for i in range(len(loc_preds)):
            boxes, labels, scores = encoder.decode(loc_preds[i].data, cls_preds[i].data, (settings.IMG_SZ, settings.IMG_SZ))
            pred_objects_num += len(scores)

        for img_idx in range(len(inputs)):   
            img_id = dloader.img_ids[batch_idx*batch_size+img_idx]
            if img_id in bbox_dict:
                true_objects_num += len(bbox_dict[img_id])

        print('{} / {}, {} / {}, {:.4f},  {:.2f} min'.format(
            batch_size*(batch_idx+1), dloader.num,
            pred_objects_num, true_objects_num, cls_threshold,
            (time.time() - bgtime)/60), end='\r')

    print('\n')
    print('=>>> {}/{}, {}, {:.4f}\n'.format(pred_objects_num, true_objects_num, pred_objects_num - true_objects_num, cls_threshold))

def find_threshold():
    img_ids = np.random.permutation(get_val_ids()).tolist()[:2000]
    bbox_dict = load_bbox_dict()
    cls_threshold = 0.18
    for i in range(20):
        print('threshold: {:.4f}'.format(cls_threshold))
        evaluate_threshold(img_ids, cls_threshold, bbox_dict)
        cls_threshold -= 0.002

if __name__ == '__main__':
    predict()
    #find_threshold()

