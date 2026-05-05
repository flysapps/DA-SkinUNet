# DA-SkinUNet: 基于双重注意力机制的医学图像分割
## 1 项目特点
    注意力机制创新：在经典U-Net的跳跃连接（Skip-connection）处引入了通道注意力（Channel Attention）和空间注意力（Spatial Attention）组成的双重注意力模块（CBAM变体），有效提升了模型对病灶边缘细节和关键特征的抓取能力。
    数值稳定的混合损失函数：结合了 BCEWithLogitsLoss与Dice Loss，在保证数值计算稳定性的同时，有效应对医学图像分割中常见的类别不平衡问题。
    健壮的数据处理管道：基于torchvision.transforms.functional实现了严谨的自定义 Dataset，确保图像与掩码在随机数据增强（如翻转）时的严格空间对齐，并完美兼容原图（.jpg）与掩码（.png）的自动化匹配。
## 2 目录结构
    ├── data/
    │   └── ISIC2018/
    │       ├── train_images/     # 训练集原图
    │       ├── train_masks/      # 训练集掩码
    │       ├── val_images/       # 验证/测试集原图
    │       └── val_masks/        # 验证/测试集掩码
    ├── visual_results/           # 自动生成的推理可视化结果目录
    ├── dataset.py                # 数据集加载与预处理模块
    ├── model.py                  # DA-SkinUNet网络架构与损失函数定义
    ├── train.py                  # 模型训练脚本
    ├── test.py                   # 模型评估与可视化脚本
    ├── best_attention_unet.pth   # 训练后生成的最佳模型权重
    ├── training_curves.png       # 训练过程的Loss、Dice、IoU曲线图
    └── README.md                 # 项目说明文档
## 3 环境配置
    Python == 3.8+
    PyTorch == 2.5.1+cu121
    Torchvision == 0.20.1+cu121
    Torchmetrics == 0.11.4
    Numpy == 2.0.2
## 4 快速开始
   ### 1 数据集准备
    本项目默认使用 ISIC 2018数据集。
      (1) 请前往 ISIC 官网或相关开源平台下载数据。
      (2) 将数据解压并严格按照上述目录结构放置在 data/ISIC2018文件夹中。
      (3) 注意图片格式类型
   ### 2 环境依赖
    pip install torch torchvision
    pip install pillow matplotlib numpy tqdm torchmetrics    
   ### 3 模型训练与测试  
    python train.py
    python test.py
