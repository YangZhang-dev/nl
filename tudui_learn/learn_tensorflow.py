import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img1=cv2.imread(img_path) # narrays
img2=Image.open(img_path)


# ToTensor
tensor = transforms.ToTensor()
tensor_img1= tensor(img1)
tensor_img2= tensor(img2)


writer=SummaryWriter("learn_tensorflow/")
writer.add_image("opencv",tensor_img1)
writer.add_image("PIL",tensor_img2)


# Normalize

normalize= transforms.Normalize([1, 10, 1], [4, 0.1, 1])
img_nor=normalize(tensor_img1)
writer.add_image("opencv_normalize",img_nor)

# Resize

resize = transforms.Resize((512, 512))
img2_resize = resize(img2)
print(img2.size,img2_resize.size)


# Compose (提供了transform转换列表)
# PIL -> PIL -> tensor
compose = transforms.Compose([transforms.Resize((111, 111)), transforms.ToTensor])
after_compose_img = compose(img1)


writer.close()