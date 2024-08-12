import os
import time
import numpy as np
from models.BiTimelyGPT import BiTimelyGPT
import argparse
import torch.optim as optim
import torch
from torch import nn
import random
from data.data_pipeline import data_pipeline
from layers.optimization import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiTimelyGPT for pretraining (continuously) biosignals')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='BiTimelyGPT', help='model name, options: [BiTimelyGPT]')


    # dataset_small loader
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_folder', type=str, default='sleepEDF', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--n_output', type=int, default=1, help='number of variates')
    parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=128, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1024, help='prediction sequence length')

    # the hyperparameters for BiTimelyGPT
    # parser.add_argument('--patch_len', type=int, default=1, help='patch length')
    # parser.add_argument('--stride', type=int, default=1, help='stride')
    # parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--num_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--num_layers', type=int, default=4, help='num of decoder layers')
    parser.add_argument('--d_model', type=int, default=200, help='dimension of model')
    parser.add_argument('--qk_dim', type=int, default=200, help='dimension for Q and K, default to d_model')
    parser.add_argument('--v_dim', type=int, default=200, help='dimension for V, default to d_model * 2')
    parser.add_argument('--ffn_proj_size', type=int, default=800, help='dimension for feed-forward projection layer')
    parser.add_argument('--forward_impl', type=str, default='chunkwise',
                        help='forward implementation, options:[parallel, recurrent, chunkwise]')
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size')
    parser.add_argument('--d_ff', type=int, default=144*2, help='dimension of feed-forward layer')
    parser.add_argument('--use_bias_in_msr', type=bool, default=False, help='Use bias in MSR')
    parser.add_argument('--use_bias_in_mlp', type=bool, default=True, help='Use bias in feed-forward layer')
    parser.add_argument('--use_bias_in_msr_out', type=bool, default=False, help='Use bias in MSR output layer')
    parser.add_argument('--use_default_gamma', type=bool, default=False, help='Use the default gamma')
    parser.add_argument('--forecasting_method', type=str, default='oneforward',
                        help='forecasting implementation, options:[oneforward, recurrent]')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_retentions', action='store_true', help='whether to output retention in decoder',  default=False)

    # optimization
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input dataset_small')
    parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00003, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    parser.add_argument('--use_grad_accum', action='store_false', help='use gradient accumulation', default=True)
    parser.add_argument('--accum_steps', type=int, default=4, help='number of steps to accumulate gradients before updating weights')
    parser.add_argument('--use_grad_ckp', action='store_false', help='use gradient checkpointing', default=False)

    args = parser.parse_args()
    print('Args in experiment:')
    print(args)

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    device = torch.device('cuda:{}'.format(args.gpu))

    model = BiTimelyGPT(configs=args, head_type='pretrain').to(device=device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {}'.format(pytorch_total_params))

    model_optim = optim.Adam(model.parameters(), lr=0.000001)

    # Pre-training stage
    print("Pre-training Stage")

    # Concatenate root_path and data_folder to form the full data path
    data_path = os.path.join(args.root_path, args.data_folder)
    pretrain_data, pretrain_loader = data_pipeline(data_path, flag='pre_train')
    vali_data, vali_loader = data_pipeline(data_path, flag='val')

    train_epochs = 20
    num_training_steps = len(pretrain_loader) * train_epochs
    num_warmup_steps = num_training_steps * 0.05
    scheduler = get_linear_schedule_with_warmup(model_optim, num_warmup_steps, num_training_steps)

    train_loss = []
    valid_loss = []
    for epoch in range(train_epochs):
        print('Epoch: {}'.format(epoch))
        # pre-train the model
        model.train()
        for i, (batch_x, batch_y) in enumerate(pretrain_loader):
            print("mini-batch number: ", i, " of ", len(pretrain_loader), " mini-batches.")
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device) # no label for pre-training
            model_optim.zero_grad()
            loss = model(batch_x, batch_y, forward_impl=args.forward_impl, chunk_size=args.chunk_size)
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())
        print('epoch: {}, train_loss: {}'.format(epoch, np.mean(train_loss)))

        model.eval()
        valid_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                print("mini-batch number: ", i, " of ", len(vali_loader), " mini-batches.")
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                loss = model(batch_x, batch_y, forward_impl=args.forward_impl, chunk_size=args.chunk_size)
                valid_loss.append(loss.item())
            print('epoch: {}, valid_loss: {}'.format(epoch, np.mean(valid_loss)))

