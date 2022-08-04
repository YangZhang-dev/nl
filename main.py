import torch
from PIL import Image

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(dir(torch))
    print(dir(torch.cuda))
    print(help(torch.cuda.is_available))
    image_path = "hymenoptera_data/train/ants/0013035.jpg"
    img = Image.open(image_path)
    print(img.size)
    img.show()
