import torch
import torch.nn as nn
import numpy as np
from model import mobileNetv2
import cv2
import os
import sys

model = mobileNetv2().cuda()
model.load_state_dict(torch.load("PATH_TO_PTH"))

root = "DIR_TO_IMGS"
img_list = os.listdir(root)

model.eval()
pred_pos = 0
with torch.no_grad():
    for img_name in img_list:
        if ".jpg" in img_name:
            img = cv2.imread(os.path.join(root, img_name))
            img = cv2.resize(img, (224, 224)).T
            img = img / 255
            img = torch.from_numpy(np.expand_dims(img, axis=0))
            img = img.float().cuda()
            output = model(img)
            _, pred = torch.max(output, 1)
            if int(pred) == 1:
                pred_pos += 1

print(str(pred_pos) + "/" + str(len(img_list)))