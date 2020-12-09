import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

class Images(Dataset):
	def __init__(self, path):
		f = open(path, "r")
		self.imgs = []
		for line in f.readlines():
			img_path = line.strip("\n").split(" ")[0]
			img_label = line.strip("\n").split(" ")[1]
			self.imgs.append((img_path, img_label))
			
	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		tup = self.imgs[index]
		img = cv2.imread(tup[0])
		img = cv2.resize(img, (224, 224)).T
		img = img / 255
		label = int(tup[1])
		return img, label