import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from .utils import get_lr, adaptive_weights
from torch_sdaa.utils import cuda_migrate
import logging

# 日志
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

adaptive_weights_calculate = adaptive_weights(device="sdaa")

# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def train_epoch(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    model.train()
    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_index, (over_patch, under_patch) in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 载入批量图像
        over_patch, under_patch = over_patch.to(device), under_patch.to(device)
        # 调用 adaptive_weights_calculate.calculate
        # 计算信息保留度n
        weights_preserve = adaptive_weights_calculate.calculate(over_patch, under_patch)
        # print(f"{epoch}:{batch_index}-{weights_preserve}")
        # 前向传播
        over_patch, under_patch = over_patch.to('cpu'), under_patch.to('cpu')
        model_cpu = model.to('cpu')
        outputs = model_cpu(over_patch, under_patch)
        over_patch, under_patch, outputs = over_patch.to(device), under_patch.to(device), outputs.to(device)
        outputs = (outputs + 1) / 2  # [-1,1]->[0,1]
        # 计算损失
        pixel_loss_value = weights_preserve[:, 0] * criterion["mse_loss"](outputs, over_patch) \
                           + weights_preserve[:, 1] * criterion["mse_loss"](outputs, under_patch)
        pixel_loss_value = torch.mean(pixel_loss_value)
        ssim_loss_value = weights_preserve[:, 0] * (1 - criterion["ssim_loss"](outputs, over_patch, normalize=True)) \
                          + weights_preserve[:, 1] * (1 - criterion["ssim_loss"](outputs, under_patch, normalize=True))
        ssim_loss_value = torch.mean(ssim_loss_value)
        loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
        
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss["mse_loss"].append(pixel_loss_value.item())
        train_epoch_loss["ssim_loss"].append(ssim_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        # pbar.set_postfix(loss=loss.item(), train_acc)
        pbar.set_postfix(
            pixel_loss=pixel_loss_value.item(),
            ssim_loss=ssim_loss_value.item(),
            learning_rate=get_lr(optimizer),
        )
        # pbar.set_postfix(**{'loss': loss.item(),
        #                     'lr': get_lr(optimizer),
        #                     })
        
    # 记录每个 epoch 的损失到日志中
    logging.info(f'Epoch [{epoch + 1}/{num_Epoches}] - MSE Loss: {np.average(train_epoch_loss["mse_loss"]):.4f}, '
                 f'SSIM Loss: {np.average(train_epoch_loss["ssim_loss"]):.4f}, '
                 f'Total Loss: {np.average(train_epoch_loss["total_loss"]):.4f}')

    return {"mse_loss": np.average(train_epoch_loss["mse_loss"]),
            "ssim_loss": np.average(train_epoch_loss["ssim_loss"]),
            "total_loss": np.average(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   验证
# ----------------------------------------------------#
# def valid_epoch(model, device, valid_dataloader, criterion):
#     model.eval()
#     valid_epoch_loss = []
#     # valid_epoch_accuracy = []
#     pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
#     # for index, (inputs, targets) in enumerate(train_dataloader, start=1):
#     for index, image_batch in enumerate(pbar, start=1):
#         # 载入批量图像
#         inputs = image_batch.to(device)
#         # 复制图像作为标签
#         labels = image_batch.data.clone().to(device)
#         # 前向传播
#         outputs = model(inputs)
#         # 计算损失
#         pixel_loss_value = criterion["mse_loss"](outputs, labels)
#         ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
#         loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
#         valid_epoch_loss.append(loss.item())
#
#         pbar.set_description('valid')
#         pbar.set_postfix(
#             pixel_loss=pixel_loss_value.item(),
#             ssim_loss=ssim_loss_value.item(),
#         )
#     return np.average(valid_epoch_loss)


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'lr': lr_scheduler.state_dict(),
                   'best_loss': best_loss,
                   }
    checkpoints_name = '/epoch%03d-loss%.3f.pth' % (epoch, best_loss)
    save_path = checkpoints_path + checkpoints_name
    torch.save(checkpoints, save_path)


# ----------------------------------------------------#
#   tensorboard
# ----------------------------------------------------#
def tensorboard_load(writer, model, train_loss, test_image, device, epoch):
    with torch.no_grad():
        model = model.to('cpu')
        writer.add_scalar('pixel_loss', train_loss["mse_loss"].item(), global_step=epoch)
        writer.add_scalar('ssim_loss', train_loss["ssim_loss"].item(), global_step=epoch)
        writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)

        test_over, test_under = test_image
        test_over, test_under = test_over.to('cpu'), test_under.to('cpu')
        fused_img = model(test_over, test_under)

        img_grid_over = torchvision.utils.make_grid(test_over, normalize=True, nrow=4)
        img_grid_under = torchvision.utils.make_grid(test_under, normalize=True, nrow=4)
        img_grid_fuse = torchvision.utils.make_grid(fused_img, normalize=True, nrow=4)
        writer.add_image('test_over_patch', img_grid_over, global_step=1, dataformats='CHW')
        writer.add_image('test_under_patch', img_grid_under, global_step=1, dataformats='CHW')
        writer.add_image('fused_img', img_grid_fuse, global_step=epoch, dataformats='CHW')
