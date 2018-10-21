import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import settings

df_labels = pd.read_csv(settings.STAGE1_TRAIN_LABLES)

def get_boxes_per_patient(pId):
    '''
    Given the dataset and one patient ID, 
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return: 
    array([[x1, y1, width1, height1],
           [x2, y2, width2, height2]])
    '''
    
    boxes = df_labels.loc[df_labels['patientId']==pId][['x', 'y', 'width', 'height']].astype('int').values.tolist()
    return boxes

def get_patient_mask(pid, target, mask_size):
    mask = np.zeros((mask_size, mask_size)).astype(np.uint8)
    if target > 0:
        for bb in get_boxes_per_patient(pid):
            x, y, w, h = map(lambda t: int(t / settings.ORIG_H * mask_size), bb)
            mask[y:y+h, x:x+w] = 1
    return mask

def get_train_val_meta(drop_empty=False):
    df = pd.read_csv(settings.STAGE1_TRAIN_LABLES)

    df = df.groupby('patientId')['Target'].apply(sum).reset_index()
    df = shuffle(df, random_state=1234)
    #print('unique:', df.shape)

    split_index = df.shape[0] - 2500

    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]

    df_train_true = df_train[df_train['Target'] > 0]
    df_train_false = shuffle(df_train[df_train['Target'] == 0])

    df_val_true = df_val[df_val['Target'] > 0]
    df_val_false = df_val[df_val['Target'] == 0]

    if drop_empty:
        df_train = df_train_true
        df_val = df_val_true
    else:
        df_train = shuffle(df_train_true.append(df_train_false), random_state=1234)
        df_val = shuffle(df_val_true.append(df_val_false), random_state=1234)

    print(df_train.shape, df_val.shape)

    return df_train, df_val

def get_test_meta():
    df = pd.read_csv(settings.STAGE1_SAMPLE_SUBMISSION)
    return df


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b
    
    return ' '.join(str(rle_item) for rle_item in rle)


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape((shape[1], shape[0])).T

if __name__ == '__main__':
    #get_train_val_meta(True)

    print(get_boxes_per_patient('261341d3-1df2-4cf4-b9a8-b1e02a124900'))
    print(get_boxes_per_patient('39cac594-7f2b-4505-b43c-eb3e76c1aedb'))

