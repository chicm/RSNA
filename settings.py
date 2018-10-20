import os
import local_settings

DATA_DIR = local_settings.DATA_DIR

ORIG_H, ORIG_W = 1024, 1024

MODEL_DIR = os.path.join(DATA_DIR, 'models')

TRAIN_DCM_DIR = os.path.join(DATA_DIR, 'train', 'dcm')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', '512')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test', '512')

STAGE1_TRAIN_LABLES = os.path.join(DATA_DIR, 'stage_1_train_labels.csv')
STAGE1_DETAILED_CLASS_INFO =  os.path.join(DATA_DIR, 'stage_1_detailed_class_info.csv')
STAGE1_SAMPLE_SUBMISSION =  os.path.join(DATA_DIR, 'stage_1_sample_submission.csv')

IMG_SZ = 512
