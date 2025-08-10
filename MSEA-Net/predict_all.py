import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet
import transforms as T

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def process_image(model, img_path, device, data_transform, pallette, output_dir):
    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    img, _ = data_transform(original_img, original_img)
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
        print(f"inference time for {img_path}: {t_end - t_start}")

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.png'))
        mask.save(output_path)

def main():
    classes = 1  # exclude background
    weights_path = "save_weights/model_unet_results20241115-225516.txt.pth"
    img_dir = r"I:\pycharm\oval_512\compare_tn3k\tn3k_result\image"
    palette_path = "palette.json"
    output_dir = r"I:\pycharm\oval_512\compare_tn3k\tn3k_result\newnet_result"
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_dir), f"image directory {img_dir} not found."
    assert os.path.exists(palette_path), f"palette file {palette_path} not found."

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    mean = (0.10304444, 0.10394574, 0.10565763)
    std = (0.17313279, 0.17340064, 0.17392088)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # define data transformation
    data_transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # process each image in the directory
    for img_name in os.listdir(img_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(img_dir, img_name)
            process_image(model, img_path, device, data_transform, pallette, output_dir)

if __name__ == '__main__':
    main()
