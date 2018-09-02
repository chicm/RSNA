import os
import glob
import numpy as np
import cv2
import argparse
import pydicom
import settings

H, W = settings.IMG_SZ, settings.IMG_SZ

def resize(src_dir, tgt_dir):
    filenames = glob.glob(os.path.join(src_dir, '*.dcm'))
    print(len(filenames))
    for i, fn in enumerate(filenames):
        print('{:06d}/{} {}'.format(i, len(filenames), fn), end='\r')
        img = pydicom.read_file(fn).pixel_array
        img = cv2.resize(img, (H, W))
        img = np.expand_dims(img, -1)
        tgt_fn = os.path.join(tgt_dir, os.path.basename(fn).split('.')[0] + '.jpg')
        cv2.imwrite(tgt_fn, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default=settings.TRAIN_DCM_DIR, required=False)
    parser.add_argument('--tgt_dir', type=str, default=settings.TRAIN_IMG_DIR, required=False)
    args = parser.parse_args()

    resize(args.src_dir, args.tgt_dir)
    #resize(r'd:\kaggle\open-images\train_0\*.jpg', )