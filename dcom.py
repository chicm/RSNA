import os
import glob, pylab, pandas as pd
import pydicom, numpy as np
import cv2
from PIL import Image, ImageDraw
import json
import settings

LABEL_FILE = settings.LABEL_FILE
TRAIN_IMG_DIR = settings.TRAIN_IMG_DIR


def get_train_data():
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    df = pd.read_csv(LABEL_FILE)
    # --- Define lambda to extract coords in list [x1, y1, x2, y2]
    extract_box = lambda row: [int(row['x']), int(row['y']), int(row['x'])+int(row['width']), int(row['y'])+int(row['height'])]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': os.path.join(TRAIN_IMG_DIR, '{}.dcm'.format(pid)),
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

def draw_img(image, name = '', resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))
    cv2.waitKey(0)


def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    #pylab.imshow(im, cmap=pylab.cm.gist_gray)
    #pylab.axis('off')
    draw_img(im)

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Extract coordinates
    x1, y1, x2, y2 = box

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

def get_train_ids():
    df = pd.read_csv(LABEL_FILE)
    pids = list(set(df.values[:, 0].tolist()))
    print(len(pids))
    print(pids[:2])
    return pids
    #img_files = glob.glob(os.path.join(settings.TRAIN_IMG_DIR, '*.dcm'))
    #print(len(img_files))

def test():
    train_data = get_train_data()
    with open('train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    pids = get_train_ids()

    draw(train_data['00436515-870c-4b36-a041-de91049b9ab4'])

    for i in range(5):
        print('>>>')
        pid = pids[10+i]
        draw(train_data[pid])

    df_detailed = pd.read_csv(settings.CLASS_FILE)
    print(df_detailed.iloc[0])

    patientId = df_detailed['patientId'][0]
    draw(train_data[patientId])

    summary = {}
    for n, row in df_detailed.iterrows():
        if row['class'] not in summary:
            summary[row['class']] = 0
        summary[row['class']] += 1
        
    print(summary)

if __name__ == '__main__':
    test()
    #get_train_ids()