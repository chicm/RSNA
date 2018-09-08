import os, cv2, glob
import numpy as np
import pydicom
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from dcom import get_train_data, get_train_ids, get_test_ids
from encoder import DataEncoder
import settings

#IMG_DIR = settings.IMG_DIR

class ImageDataset(data.Dataset):
    def __init__(self, pids, img_dir, label_dict=None):
        self.input_size = settings.IMG_SZ
        self.pids = pids
        self.img_dir = img_dir
        self.num = len(pids)
        self.label_dict = label_dict
        self.transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Lambda(lambda x: x/255.)
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        if label_dict:
            for pid in self.pids:
                #print(label_dict[pid])
                self.labels.append(label_dict[pid]['label'])
                self.boxes.append(label_dict[pid]['boxes'])


    def __getitem__(self, index):
        fn = os.path.join(self.img_dir, '{}.jpg'.format(self.pids[index]))
        #img = np.expand_dims(pydicom.read_file(fn).pixel_array, -1)
        img = cv2.imread(fn)
        #print(img.shape)
        img = self.transform(img)
        #print(get_class_names(self.labels[index]))

        if self.label_dict:
            return img, torch.Tensor(self.boxes[index]), torch.LongTensor([self.labels[index]])
        else:
            return [img]

    def __len__(self):
        return self.num

    def collate_fn(self, batch):
        """Encode targets.

        Args:
          batch: (list) of images, ids

        Returns:
          images, stacked bbox_targets, stacked clf_targets.
        """
        imgs = [x[0] for x in batch]

        if self.label_dict:
            boxes = [x[1] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        labels = torch.zeros(num_imgs).long()

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            #print('1>>>')
            #print(boxes[i].size(), labels[i].size())
            if self.label_dict:
                loc_target, cls_target = self.encoder.encode(torch.Tensor(boxes[i]), torch.LongTensor([0]*len(boxes[i])))
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)

                labels[i] = batch[i][2]
        if self.label_dict:
            return inputs, labels, torch.stack(loc_targets), torch.stack(cls_targets)
        else:
            return inputs

def get_train_loader(pids, img_dir=settings.TRAIN_IMG_DIR, batch_size=8, shuffle = True):
    label_dict = get_train_data()

    dset = ImageDataset(pids, img_dir, label_dict)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def get_test_loader(img_ids, img_dir=settings.TEST_IMG_DIR, batch_size=16):
    dset = ImageDataset(img_ids, img_dir, None)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dset.collate_fn, drop_last=False)
    dloader.num = dset.num
    dloader.img_ids = img_ids
    return dloader

def get_small_train_loader(img_dir=settings.TRAIN_IMG_DIR, batch_size=4):
    pids = get_train_ids()[:4]
    label_dict = get_train_data()

    dset = ImageDataset(pids, img_dir, label_dict)
    dloader = data.DataLoader(dset, batch_size=4, shuffle=False, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def test_train_loader():
    #loader = ImageDataLoader(get_train_ids())
    train_data = get_train_data()
    img_ids =  get_train_ids()
    loader = get_train_loader(img_ids, shuffle=False)
    count = 0
    for i, data in enumerate(loader):
        imgs, img_labels, bbox, clfs = data
        for l in img_labels:
            assert(l.item() == train_data[img_ids[count]]['label'])
            count += 1
            print(count, end='\r')
        #print(imgs)
        #print(img_labels)
        #print(imgs.size(), bbox.size(), clfs.size())
        #print(torch.max(bbox))

def test_test_loader():
    loader = get_test_loader(get_test_ids())
    for i, data in enumerate(loader):
        print(data.size())
        if i > 10:
            break

if __name__ == '__main__':
    test_test_loader()
    #test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])
