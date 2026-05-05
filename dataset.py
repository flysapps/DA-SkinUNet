import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF



# ISIC 2018 数据集类
class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 转为灰度图


        #调整大小
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))


        if self.mode == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 转为 Tensor (此时形状为 [C, H, W]，像素值 0-1)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)


        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #二值化掩码(mask)，确保只有 0 和 1
        mask = (mask > 0.5).float()

        return image, mask