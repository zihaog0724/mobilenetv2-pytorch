import os
import shutil
import random

train_file = open("train.txt", "w")
val_file = open("val.txt", "w")

pos = os.listdir("./pos")
neg = os.listdir("./neg")

random.shuffle(pos)
random.shuffle(neg)

train_percent = 0.8

pos_train = pos[:int(len(pos) * train_percent)]
pos_val = pos[int(len(pos) * train_percent):]
neg_train = neg[:int(len(neg) * train_percent)]
neg_val = neg[int(len(neg) * train_percent):]

for img_name in pos_train:
	if "jpg" in img_name:
		train_file.write("./pos/" + img_name + " 1" + "\n")

for img_name in neg_train:
	if "jpg" in img_name:
		train_file.write("./neg/" + img_name + " 0" + "\n")

for img_name in pos_val:
	if "jpg" in img_name:
		val_file.write("./pos/" + img_name + " 1" + "\n")

for img_name in neg_val:
	if "jpg" in img_name:
		val_file.write("./neg/" + img_name + " 0" + "\n")


