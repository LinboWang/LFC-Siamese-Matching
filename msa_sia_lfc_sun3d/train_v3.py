import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test_v2 import valid
# from loss import MatchLoss
from loss_v2 import MatchLoss
from utils import tocuda


def train_step(step, optimizer, model, match_loss, data):
    model.train()

    res_logits, res_e_hat = model(data)
    loss = 0
    loss_val = []

    loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[0], res_e_hat[0], toggle=False)
    loss += loss_i
    loss_val += [geo_loss, cla_loss, l2_loss]
    for i in range(1, len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i],
                                                                   toggle=(i % 2 == 0))
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
    # for name, param in model.named_parameters():
    # if torch.any(torch.isnan(param.grad)):
    # print('skip because nan')
    # return loss_val

    optimizer.step()
    return loss_val


def train(model, train_loader, valid_loader, config):
    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # model = torch.nn.DataParallel(model, device_ids=[0])
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss'] * (config.iter_num + 2))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
    train_loader_iter = iter(train_loader)

    epoch = 0
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data)
        logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0

        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss, _, _, _ = valid(valid_loader, model, step, config)
            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            print("Saving best model with va_res = {}".format(va_res))
            best_acc = va_res
            epoch += 1
            index = str(epoch)
            model_name = 'model_best' + index + '.pth'

            torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(config.log_path, model_name))

        if b_save:
            torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
