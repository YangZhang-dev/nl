from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
img_path = "/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_arr = np.array(img_PIL)
writer.add_image("test", img_arr, 1, dataformats='HWC')
# y=x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
