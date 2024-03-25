import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR
import time
from tqdm import tqdm
from loguru import logger

from args import args
from stgcn import STGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np


def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.mean(loss)


def train(loader, model, optimizer, criterion, std, mean, device,A_hat):
    batch_rmse_loss = 0
    batch_mae_loss = 0
    batch_mape_loss = 0
    batch_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.train()
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs,A_hat) * std + mean
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        out_unnorm = outputs.detach().cpu().numpy()
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0.0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0.0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0.0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss
        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1), batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


@torch.no_grad()
def eval(loader, model, std, mean, device,A_hat):
    batch_rmse_loss = 0
    batch_mae_loss = 0
    batch_mape_loss = 0
    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs,A_hat)

        out_unnorm = output.detach().cpu().numpy() * std + mean
        target_unnorm = targets.detach().cpu().numpy()

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0.0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0.0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0.0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)
    data, mean, std,sp_matrix = read_data(args)
    train_loader, valid_loader, test_loader, train_mean, train_std, val_mean, val_std, test_mean, test_std = generate_dataset(
        data, args)
    print('mean,std: ', train_mean, train_std, val_mean, val_std)


    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_wave = get_normalized_adj(sp_matrix).to(device)

    net = STGCN(num_nodes=data.shape[1],
                 num_features=data.shape[2],
                 num_timesteps_input=args.his_length,
                 num_timesteps_output=args.pred_length,
                 A_hat=A_wave)

    net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    scheduler = MultiStepLR(optimizer=optimizer,
                            milestones=args.lr_decay_steps,
                            gamma=args.lr_decay_rate)

    val_mape_min = float('inf')
    wait = 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}=====".format(epoch))
        print('Training...')
        loss, train_rmse, train_mae, train_mape = train(train_loader, net, optimizer, criterion, train_std, train_mean,
                                                        device,A_wave)
        print('Evaluating...')
        valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, val_std, val_mean, device,A_wave)



        if valid_mape <= val_mape_min:
            print(f'\n##on train data## loss: {loss}, \n' +
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')

            print(f'save model to {args.save + "epoch_" + str(epoch) + "_" + str(round(valid_mape.item(), 2)) + "_best_model.pth"}\n')
            wait = 0
            val_mape_min = valid_mape
            best_model_wts = net.state_dict()
            torch.save(best_model_wts,
                       args.save + "epoch_" + str(epoch) + "_" + str(round(val_mape_min.item(), 2)) + "_best_model.pth")
            last_weight_add=args.save + "epoch_" + str(epoch) + "_" + str(round(val_mape_min.item(), 2)) + "_best_model.pth"
        else:
            print(f'\n##on train data## loss: {loss}, \n' +
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n' +
                        f'##on valid data## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}\n')

            wait += 1
            if wait==30:
                print(f'can not have better mape, so ends now\n')
                break

        scheduler.step()

    net.load_state_dict(torch.load(last_weight_add))
    test_rmse, test_mae, test_mape = eval(test_loader, net, test_std, test_mean, device,A_wave)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')

if __name__ == '__main__':
    main(args)
