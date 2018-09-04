import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import cv2
import os
import settings

from dcom import get_train_data, get_val_ids, get_train_ids

#bbox_dict, _ = load_small_train_ids()
train_data = get_train_data()

IMG_DIR = settings.TRAIN_IMG_DIR
#IMG_DIR = settings.TEST_IMG_DIR

val_ids = get_val_ids()[:100]
#val_ids = ['353d3a524120046c', '3576033ecd412822', '351e1266649364be', '357fe840ff5bf6d3', '89f6490612d9ae52', '35cc71035da293d0', '354cb68600c29cee', '8aa665e7810453b6', '3530be997286cf58', '37191563d6e5fc80']
#test_ids = ['0000048549557964', '0000071d71a0a6f6', '000018acd19b4ad3', '00001bcc92282a38', '0000201cd362f303', '000020780ccee28d', '000023aa04ab09ed', '0000253ea4ecbf19', '0000286a5c6a3eb5', '00002f4ff380c64c']
#test_ids = ['00000b4dcff7f799', '00001a21632de752', '0000d67245642c5f', '0001244aa8ed3099', '000172d1dd1adce0', '0001c8fbfb30d3a6', '0001dd930912683d', '0002c96937fae3b3', '0002f94fe2d2eb9f', '000305ba209270dc']
img_id = '00436515-870c-4b36-a041-de91049b9ab4'

print('Loading model..')
net = RetinaNet()
#net.load_state_dict(torch.load('./model/best_512.pth'))
net.load_state_dict(torch.load('./ckps/best_1.pth'))
net.eval()
net.cuda()

transform = transforms.Compose([
    transforms.ToTensor()
])

print('Loading image..')
img = cv2.imread(os.path.join(IMG_DIR, img_id+'.jpg'))  #Image.open(os.path.join(IMG_DIR, img_id+'.jpg'))
w = h = settings.IMG_SZ
#img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0).cuda()
x = Variable(x, volatile=True)
_, loc_preds, cls_preds = net(x)

print(loc_preds.size(), cls_preds.size())
print('Decoding..')
encoder = DataEncoder()
boxes,  scores = encoder.decode(loc_preds.data[0], cls_preds.data[0], (w,h))


#draw = ImageDraw.Draw(img)
print('>>Detected objects:', len(boxes))
print(boxes)
#print([i for i in zip(get_class_names(list(labels)), list(scores))])

#for box in boxes:
#    draw.rectangle(list(box), outline='red')


trueboxes = train_data[img_id]['boxes']   #torch.Tensor([b[1] for b in bbox_dict[img_id]])*settings.IMG_SZ
#truelabels = [b[0] for b in bbox_dict[img_id]]
print(trueboxes)
#print(get_class_names(truelabels))
#for box in trueboxes:
#    draw.rectangle(list(box), outline='blue')

#img.show()
