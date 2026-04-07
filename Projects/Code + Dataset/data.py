import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_list = sorted(glob.glob(root + "/*.jpg"))
        self.char_list = [
            chr(int.from_bytes(bytes.fromhex(unicode.zfill(8)), byteorder='big', signed=False))
            for unicode in [image.split("/")[-1].split(".")[0] for image in self.image_list]
        ]
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "path": self.image_list[idx],
            "char": self.char_list[idx],
            "img": img
        }