import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import skimage

import settings
from loader import get_test_loader
from models import RSNAV1
from postprocessing import binarize, resize_image, split_mask, mask_to_bbox
from utils import run_length_encoding
from augmentation import tta_back_mask_np

min_box_area = 10000

# define function that extracts confidence and coordinates of boxes from a prediction mask
def parse_boxes(msk, threshold=0.5, connectivity=None):
    """
    :param msk: (torch.Tensor) CxWxH tensor representing the prediction mask
    :param threshold: threshold in the range 0-1 above which a pixel is considered a positive target
    :param connectivity: connectivity parameter for skimage.measure.label segmentation (can be None, 1, or 2)
                         http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    :returns: (list, list) predicted_boxes, confidences
    """
    # extract 2d array
    #msk = msk[0]
    # select pixels above threshold and mark them as positives (1) in an array of equal size as the input prediction mask
    #pos = np.zeros(msk.shape)
    pos = (msk > threshold).astype(np.uint8)
    #pos[msk>threshold] = 1.
    # label regions
    lbl = skimage.measure.label(pos, connectivity=connectivity)
    
    predicted_boxes = []
    confidences = []
    # iterate over regions and extract box coordinates
    for region in skimage.measure.regionprops(lbl):
        # retrieve x, y, height and width and add to prediction string
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
        # add control over box size (eliminate if too small)
        if w*h > min_box_area: 
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)
    
    return predicted_boxes, confidences

# # debug code for above function
# plt.imshow(dataset_train[3][1][0], cmap=mpl.cm.jet) 
# print(dataset_train[3][1].shape)
# print(parse_boxes(dataset_train[3][1]))

def do_tta_predict(args, model, ckp_path, tta_indices):
    '''
    return 18000x128x128 np array
    '''
    model.eval()
    preds = []
    cls_preds = []
    meta = None

    # i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
    for flip_index in tta_indices:
        print('flip_index:', flip_index)
        test_loader = get_test_loader(args.batch_size, index=flip_index, dev_mode=args.dev_mode, img_sz=args.img_sz)
        meta = test_loader.meta
        outputs = None
        cls_outputs = None
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                img = img.cuda()
                output, cls_output = model(img)
                output, cls_output = torch.sigmoid(output), torch.sigmoid(cls_output)
                if outputs is None:
                    outputs = output.squeeze().cpu()
                    cls_outputs = cls_output.squeeze().cpu()
                else:
                    outputs = torch.cat([outputs, output.squeeze().cpu()], 0)
                    cls_outputs = torch.cat([cls_outputs, cls_output.squeeze().cpu()])
                
                #cls_preds.extend(cls_output.squeeze().cpu().tolist())

                print('{} / {}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
        outputs = outputs.numpy()
        cls_outputs = cls_outputs.numpy()
        outputs = tta_back_mask_np(outputs, flip_index)
        preds.append(outputs)
        cls_preds.append(cls_outputs)
    
    parent_dir = ckp_path+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred_{}.npy'.format(''.join([str(x) for x in tta_indices])))
    np_file_cls = os.path.join(parent_dir, 'pred_cls_{}.npy'.format(''.join([str(x) for x in tta_indices])))

    model_pred_result = np.mean(preds, 0)
    model_cls_pred_result = np.mean(cls_preds, 0)

    np.save(np_file, model_pred_result)
    np.save(np_file_cls, model_cls_pred_result)

    return model_pred_result, model_cls_pred_result, meta


def predict(args, model, checkpoint, out_file):
    print('predicting {}...'.format(checkpoint))
    mask_outputs, _, meta = do_tta_predict(args, model, checkpoint, tta_indices=[0,1])

    df_test_cls_preds = pd.read_csv('test_cls_preds.csv')
    test_target = df_test_cls_preds['Target'].values.tolist()

    print(mask_outputs.shape)
    #print(len(cls_preds))
    #print(cls_preds)
    #print(meta.head(10))
    #y_pred_test = generate_preds(pred)
    #print(meta.shape)

    ship_list_dict = []
    for i, row in enumerate(meta.values):
        img_id = row[0]
        if test_target[i] < 0:
            ship_list_dict.append({'patientId': img_id,'PredictionString': np.nan})
        else:
            pred_str = generate_preds(args, mask_outputs[i])
            ship_list_dict.append({'patientId': img_id,'PredictionString': pred_str})

    pred_df = pd.DataFrame(ship_list_dict)
    pred_df.to_csv(args.sub_file, columns=['patientId', 'PredictionString'], index=False)
    #submission = create_submission(meta, y_pred_test)
    #submission.to_csv(out_file, index=None, encoding='utf-8')

def prediction_string(predicted_boxes, confidences):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    """
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]   

# # debug code for above function
# predicted_boxes, confidences = parse_boxes(dataset_train[3][1])
# print(predicted_boxes, confidences)
# print(prediction_string(predicted_boxes, confidences))

def generate_preds(args, output, target_size=(settings.ORIG_H, settings.ORIG_W), threshold=0.5):
    mask = resize_image(output, target_size=target_size)
    #pred = binarize(cropped, threshold)

    predicted_boxes, confidences = parse_boxes(mask) 

    pred_str = prediction_string(predicted_boxes, confidences)

    return pred_str

def predict_model(args):
    model = RSNAV1(34)
    
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    else:
        raise ValueError('model file not found: {}'.format(model_file))
    model = model.cuda()
    predict(args, model, model_file, args.sub_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--model_name', default='RSNAV1', type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--sub_file', default='sub_3.csv', type=str, help='submission file')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--img_sz', default=256, type=int, help='image size')
    #parser.add_argument('--bbox', action='store_true')


    args = parser.parse_args()
    print(args)

    predict_model(args)
    #ensemble_predict(args)
