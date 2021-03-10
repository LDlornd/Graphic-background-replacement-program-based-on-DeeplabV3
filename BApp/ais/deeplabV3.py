from torchvision.models.segmentation import deeplabv3_resnet101
import torch
import numpy as np
import os

class DeeplabV3:

    def __init__(self):
        super().__init__()
        self.deeplabv3 = deeplabv3_resnet101(
            pretrained_backbone=False,
            num_classes=21,
            aux_loss=True # 辅助分类函数，必须设置为 True ，否则在加载模型时会出现奇奇怪怪的问题
        )

        # 加载预训练模型，并且忽略 Missing keys 和 Unexpected keys
        cur_path = __file__[:-13]
        model_path = os.path.join(cur_path, "models\\deeplabv3_resnet101_coco-586e9e4e.pth")
        self.state_dict = torch.load(model_path)
        self.deeplabv3.load_state_dict(self.state_dict, False)

        # 将模型转换为预测模式
        self.deeplabv3.eval()

        # 把模型加载进 GPU ，使用 GPU 进行运算
        self.deeplabv3.cuda()
    
    def segmentation(self, img):
        img = img.cuda()
        with torch.no_grad():
            res = self.deeplabv3(img)
        res = res["out"][0]
        res = torch.argmax(res, 0)
        return res

    def img_format(self, img):
        # GBR->RGB 通道转换
        img = img[:, :, ::-1].transpose(2, 0, 1)
        
        # 把图像的内存转换为连续内存，加快运行速度
        img = np.ascontiguousarray(img)

        # 转换为 tensor
        img = torch.from_numpy(img).float()
        
        # 图像归一化
        img /= 255.0

        # 把数组变成四维，以与模型的输入相匹配
        img = img.unsqueeze(0)
        return img
    
    def image_segmentation(self, img):
        img = self.img_format(img)
        img = self.segmentation(img)
        return img