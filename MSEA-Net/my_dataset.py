import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "val"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "masks")) if i.endswith(".png")]
        self.manual = [os.path.join(data_root, "masks", i) for i in mask_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx])
        # manual = np.array(manual) / 1
        # # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # # roi_mask = 255 - np.array(roi_mask)
        # mask = np.clip(manual, a_min=0, a_max=255)
        #
        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        # mask = Image.fromarray(mask)
        img = Image.open(self.img_list[idx]).convert('RGB')

        mask_array = np.array(mask)[:, :, 0]  # 提取R通道

        # 将R通道中值为128的像素变为1
        mask_array[mask_array == 128] = 1

        # 转换回PIL图像，确保数据类型正确
        mask = Image.fromarray(mask_array.astype(np.uint8))

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

