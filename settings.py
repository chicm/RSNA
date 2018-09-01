import os

DATA_DIR = '/media/chicm/NVME/rsna'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')

LABEL_FILE = os.path.join(DATA_DIR, 'stage_1_train_labels.csv')
CLASS_FILE =  os.path.join(DATA_DIR, 'stage_1_detailed_class_info.csv')