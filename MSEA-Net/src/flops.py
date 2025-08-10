import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from unet import UNet

def calculate_params_and_flops():
    model = UNet(in_channels=3, num_classes=2, bilinear=False, base_c=32)
    input_tensor = torch.randn(4, 3, 512, 512)  # 假设输入尺寸为 (1, 3, 256, 256)

    macs, params = profile(model, inputs=(input_tensor,))
    macs, params = clever_format([macs, params], "%.3f")

    print(f"Number of parameters: {params}")
    print(f"FLOPs: {macs}")


if __name__ == "__main__":
    calculate_params_and_flops()
