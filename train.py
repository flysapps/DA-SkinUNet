import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Dice, JaccardIndex
from model import AttentionUNet, DiceBCELoss
from dataset import ISICDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

DATA_PATH = './data/ISIC2018'
TRAIN_IMG_DIR = os.path.join(DATA_PATH, 'train_images')
TRAIN_MASK_DIR = os.path.join(DATA_PATH, 'train_masks')
VAL_IMG_DIR = os.path.join(DATA_PATH, 'val_images')
VAL_MASK_DIR = os.path.join(DATA_PATH, 'val_masks')

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc="训练中"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)  # 这里的 outputs 是 logits
        loss = criterion(outputs, masks)  # Loss 内部使用了 BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(train_dataset)


def validate():
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="验证中"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)  # 这里的 outputs 是 logits

            # 计算 Loss
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)

            # 联动修改：计算指标前必须先过 Sigmoid 变成概率分布
            pred_probs = torch.sigmoid(outputs)
            total_dice += dice_metric(pred_probs, masks.int()).item() * imgs.size(0)
            total_iou += iou_metric(pred_probs, masks.int()).item() * imgs.size(0)

    return (total_loss / len(val_dataset),
            total_dice / len(val_dataset),
            total_iou / len(val_dataset))


if __name__ == '__main__':
    train_dataset = ISICDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, mode='train')
    val_dataset = ISICDataset(VAL_IMG_DIR, VAL_MASK_DIR, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"训练集数量: {len(train_dataset)}")
    print(f"验证集数量: {len(val_dataset)}")

    model = AttentionUNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    dice_metric = Dice(task="binary").to(device)
    iou_metric = JaccardIndex(task="binary").to(device)

    best_dice = 0.0
    train_losses, val_losses, val_dices, val_ious = [], [], [], []

    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch()
        val_loss, val_dice, val_iou = validate()

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        # 学习率调整 (基于验证集 Loss)
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_attention_unet.pth')
            print(f"-> 最佳模型已保存 (Dice: {best_dice:.4f})")

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_dices, label='Val Dice', color='orange')
    plt.title('Dice Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_ious, label='Val IoU', color='green')
    plt.title('IoU Score')
    plt.legend()

    plt.savefig('training_curves.png')
    print(f"\n训练完成！最佳 Dice: {best_dice:.4f}")
    print("训练曲线已保存为 training_curves.png，模型已保存为 best_attention_unet.pth")