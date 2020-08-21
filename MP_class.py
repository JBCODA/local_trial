import os
import torch
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


class MP_data(Dataset):
	def __init__(self, mainpath, transform=None):
		self.images = [os.path.join(mainpath, "rawimg", raw) for raw in sorted(os.listdir(os.path.join(mainpath, "rawimg"))) if os.path.isfile(os.path.join(mainpath, fl))]
		self.masks = [os.path.join(mainpath, "'"labels"'", mask) for mask in sorted(os.listdir(os.path.join(mainpath, "'"labels"'")))]


		normalize = transforms.Normalize()

		self.rawTransform = transforms.Compose(transforms.ToTensor())
		self.maskTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
												  transforms.ToTensor()])

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		return self.Transform(Image.open(self.images[idx]).convert("'"RGB"'")), self.maskTransform(Image.open(self.masks[idx]))
