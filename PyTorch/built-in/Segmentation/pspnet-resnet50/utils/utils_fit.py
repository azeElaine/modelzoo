import os

import torch
from nets.pspnet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, aux_branch, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            if aux_branch:
                #----------------------#
                #   前向传播
                #----------------------#
                aux_outputs, outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    aux_loss  = Focal_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                    main_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    aux_loss  = CE_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                    main_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss      = aux_loss * 0.4 + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice * 0.4 + main_dice
            else:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                if aux_branch:
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    aux_outputs, outputs = model_train(imgs)
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    if focal_loss:
                        aux_loss  = Focal_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                        main_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                    else:
                        aux_loss  = CE_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                        main_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                    loss      = aux_loss * 0.4 + main_loss
                    if dice_loss:
                        aux_dice  = Dice_loss(aux_outputs, labels)
                        main_dice = Dice_loss(outputs, labels)
                        loss      = loss + aux_dice * 0.4 + main_dice
                else:
                    #----------------------#
                    #   前向传播
                    #----------------------#
                    outputs = model_train(imgs)
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    if focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        # Print iteration-level loss and f_score
        if local_rank == 0:
            print(f"Iteration {iteration + 1}/{epoch_step} - Loss: {loss.item():.4f} - F-Score: {_f_score.item():.4f}")
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            #-------------------------------#
            #   判断是否使用辅助分支
            #-------------------------------#
            if aux_branch:
                #----------------------#
                #   前向传播
                #----------------------#
                aux_outputs, outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    aux_loss  = Focal_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                    main_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    aux_loss  = CE_Loss(aux_outputs, pngs, weights, num_classes = num_classes)
                    main_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss      = aux_loss * 0.4 + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice * 0.4 + main_dice
            else:
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()

        # Print validation iteration loss and f_score
        if local_rank == 0:
            print(f"Validation Iteration {iteration + 1}/{epoch_step_val} - Loss: {loss.item():.4f} - F-Score: {_f_score.item():.4f}")
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))