import os
from PIL import Image

from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bee_label_dir = "bees"
ants_Dataset = MyData(root_dir, ants_label_dir)
bee_Dataset = MyData(root_dir, bee_label_dir)
Datasets=ants_Dataset+bee_Dataset
img, w_dir = Datasets.__getitem__(124)
img.show()
