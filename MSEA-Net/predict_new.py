import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet
import transforms as T
import distributed_utils

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "save_weights/model_unet_results20241116-181203.txt.pth"
    img_path = r"I:\pycharm\oval_512\compare_tn3k\tn3k_result\image\0010.jpg"
    mask_path = r"I:\pycharm\oval_512\compare_tn3k\dataset\val\masks\0011.png"
    palette_path = "palette.json"
    # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    mean=(0.68949017, 0.44734749, 0.30579927)
    std=(0.09792239, 0.11040133, 0.09589511)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')
    original_mask = Image.open(mask_path).convert('L')
    # from pil image to tensor and normalize
    data_transform = T.Compose([
            T.Resize(512),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    img,mask = data_transform(original_img,original_mask)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        # prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save(r"I:\pycharm\oval_512\compare_tn3k\tn3k_result\newnet_result\10.png")


if __name__ == '__main__':
    main()
