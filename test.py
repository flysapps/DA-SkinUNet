import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics import Dice, JaccardIndex

from model import AttentionUNet
from dataset import ISICDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


DATA_PATH = './data/ISIC2018'
VAL_IMG_DIR = os.path.join(DATA_PATH, 'val_images')
VAL_MASK_DIR = os.path.join(DATA_PATH, 'val_masks')
MODEL_PATH = 'best_attention_unet.pth'


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"未找到模型文件 {MODEL_PATH}，请先运行 train.py！")

val_dataset = ISICDataset(VAL_IMG_DIR, VAL_MASK_DIR, mode='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)


model = AttentionUNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("模型加载成功！")


dice_metric = Dice(task="binary").to(device)
iou_metric = JaccardIndex(task="binary").to(device)


print("\n开始评估......")
total_dice = 0.0
total_iou = 0.0

with torch.no_grad():
    for imgs, masks in tqdm(val_loader, desc="评估中"):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs) # 这里的 outputs 是 logits
        
        # 联动修改：将 logits 转为概率
        pred_probs = torch.sigmoid(outputs)
        
        total_dice += dice_metric(pred_probs, masks.int()).item()
        total_iou += iou_metric(pred_probs, masks.int()).item()

avg_dice = total_dice / len(val_loader)
avg_iou = total_iou / len(val_loader)
print(f"\n验证集平均指标:")
print(f"Dice: {avg_dice:.4f}")
print(f"IoU:  {avg_iou:.4f}")


print("\n正在生成可视化结果......")
os.makedirs("visual_results", exist_ok=True)

with torch.no_grad():
    for idx, (imgs, masks) in enumerate(val_loader):
        if idx >= 8:  # 只保存前8张
            break
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        
        # 反归一化用于显示 (针对 ImageNet 均值和方差)
        img = imgs[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        mask = masks[0].cpu().squeeze().numpy()
        
        # 联动修改：将 logits 转为概率后，再与 0.5 比较进行二值化
        pred = (torch.sigmoid(outputs)[0].cpu().squeeze().numpy() > 0.5).astype(np.float32)
        
        # 绘图
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        save_path = f"visual_results/result_{idx+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

print("\n全部完成！结果保存在visual_results文件夹。")