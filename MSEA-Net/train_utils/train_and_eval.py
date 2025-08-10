import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target

from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充

        # 假设目标图像的类别标签存储在R通道

        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out']

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_targets = torch.tensor([], dtype=torch.long, device=device)
    all_preds = torch.tensor([], dtype=torch.float, device=device)

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            print(type(target), target.shape)
            print(type(output), output.shape)

            # 收集目标标签
            all_targets = torch.cat((all_targets, target.flatten()), dim=0)

            # 收集预测概率（假设输出是logits，需要转换为概率）
            probs = torch.softmax(output, dim=1)[:, 1]  # 只保留前景的概率
            all_preds = torch.cat((all_preds, probs.flatten()), dim=0)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

        # 将收集的数据转换为numpy数组
        all_targets = all_targets.cpu().numpy()
        all_preds = all_preds.cpu().numpy()

        # 初始化空列表来存储F1-Score曲线上的点
        f1_scores = []
        thresholds = np.linspace(0, 1, 100)  # 生成100个阈值

        for threshold in thresholds:
            y_pred = (all_preds >= threshold).astype(int)
            TP = np.sum((y_pred == 1) & (all_targets == 1))
            FP = np.sum((y_pred == 1) & (all_targets == 0))
            TN = np.sum((y_pred == 0) & (all_targets == 0))
            FN = np.sum((y_pred == 0) & (all_targets == 1))
            precision = TP / (TP + FP) if TP + FP > 0 else 1.0
            recall = TP / (TP + FN) if TP + FN > 0 else 1.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1_score)

        # 将thresholds和f1_scores转换为numpy数组
        thresholds = np.array(thresholds)
        f1_scores = np.array(f1_scores)

        # 找到最佳阈值（F1-Score最大时的阈值）
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1_score = np.max(f1_scores)

        # 获取当前脚本文件的路径
        script_path = os.path.abspath(__file__)

        # 获取上上级文件夹的路径
        parent_folder = os.path.dirname(os.path.dirname(script_path))

        # 获取上上级文件夹的名称
        folder_name = os.path.basename(parent_folder)

        # 指定文件夹路径
        folder_path = r"I:\pycharm\oval_512\compare\F1"

        # 确保文件夹存在，如果不存在则创建
        os.makedirs(folder_path, exist_ok=True)

        # 定义Excel文件名
        excel_file = os.path.join(folder_path, f"{folder_name}.xlsx")

        # 将数据写入Excel文件
        with pd.ExcelWriter(excel_file) as writer:
            # 写入Thresholds和F1-Score数据
            df_f1 = pd.DataFrame({'Threshold': thresholds, 'F1 Score': f1_scores})
            df_f1.to_excel(writer, sheet_name='F1 Curve', index=False)

            # 写入最佳阈值和F1-Score
            df_best = pd.DataFrame({'Best Threshold': [best_threshold], 'Best F1 Score': [best_f1_score]})
            df_best.to_excel(writer, sheet_name='Best F1', index=False)

        # 绘制F1-Score曲线
        plt.figure(figsize=(12, 5))
        plt.plot(thresholds, f1_scores, label=f'F1 Score curve (max = {best_f1_score:.10f} at threshold = {best_threshold:.4f})')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curve')
        plt.legend(loc='lower left')

        # 保存图片
        output_image_path = os.path.join(folder_path, 'f1_curves.png')
        plt.savefig(output_image_path)
        print(f"图片已保存到 {output_image_path}")

        # 显示图形
        plt.show()

    return confmat, dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
