import os

#DATA_DIR = '/media/chicm/NVME/rsna'
DATA_DIR = r'G:\rsna'

TRAIN_DCM_DIR = os.path.join(DATA_DIR, 'train', 'dcm')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', '512')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

LABEL_FILE = os.path.join(DATA_DIR, 'stage_1_train_labels.csv')
CLASS_FILE =  os.path.join(DATA_DIR, 'stage_1_detailed_class_info.csv')

IMG_SZ = 512