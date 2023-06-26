import sys
import os
import torch
import numpy as np
import pickle
import shutil
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

# torch.manual_seed(0)
# np.random.seed(0)


sys.path.append('.')
from models.oformer import Encoder1D, PointWiseDecoder1D, OFormer1D
from models.fno import FNO1d
from models.deeponet import DeepONet1D
from utils import TransformerOperatorDataset

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt


def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    for i in range(ncols):
        ax[0][i].plot(y_train_true[i].reshape(100,).detach().cpu())
        ax[0][i].plot(y_train_pred[i].reshape(100,).detach().cpu())
        ax[1][i].plot(y_val_true[i].reshape(100,).detach().cpu())
        ax[1][i].plot(y_val_pred[i].reshape(100,).detach().cpu())

    fname = str(ep)
    while(len(fname) < 8):
        fname = '0' + fname
    if(seed is not None):
        plt.savefig("./{}/{}_{}.png".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.png".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(v.reshape(200,).detach().cpu())
            #print(preds[0].shape)
            #ax.plot(preds[0][idx,:,0,0].detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_1/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def train_plots(train_loader, seed=None):
    im_num = 0
    for vals in train_loader:
        #print(vals[2][0][1:] - vals[2][0][:-1])
        #raise
        for idx, v in tqdm(enumerate(vals[0])):

            fig, ax = plt.subplots(figsize=(8,6))
            for j in range(0, v.shape[0], 10):
                ax.plot(v[j].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./train_plots/{}_{}.png".format(seed, fname))
            plt.close()
            #raise

            im_num += 1


def get_model(model_name, config):
    if(model_name == "fno"):
        model = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step'], config['dropout'])
    elif(model_name == "oformer"):
        encoder = Encoder1D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], dropout=config['dropout'],
                            res=config['enc_res'])
        decoder = PointWiseDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     decoding_depth=config['decoding_depth'], scale=config['scale'], res=config['dec_res'])
        #decoder = STDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
        #                             decoding_depth=config['decoding_depth'], scale=config['scale'], res=config['dec_res'])
        model = OFormer1D(encoder, decoder)
    elif(model_name == "deeponet"):
        model = DeepONet1D(config['branch_net'], config['trunk_net'], config['activation'], config['kernel_initializer'])
    
    model.to(device)
    return model


def get_data(f, config):
    train_data = TransformerOperatorDataset(f, config['flnm'],
                            split="train",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    train_data.data = train_data.data.to(device)
    train_data.grid = train_data.grid.to(device)
    #train_data.time_included_tokens = train_data.time_included_tokens.to(device)
    val_data = TransformerOperatorDataset(f, config['flnm'],
                            split="val",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    #val_data.time_included_tokens = val_data.time_included_tokens.to(device)
    test_data = TransformerOperatorDataset(f, config['flnm'],
                            split="test",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    test_data.data = test_data.data.to(device)
    test_data.grid = test_data.grid.to(device)
    #test_data.time_included_tokens = test_data.time_included_tokens.to(device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))

    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)) | \
                bool(set(train_data.data_list) & \
                     set(test_data.data_list)) & \
                bool(set(val_data.data_list) & \
                     set(test_data.data_list)))

    return train_loader, val_loader, test_loader


def evaluate(test_loader, model, loss_fn):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        model.eval()
        for bn, (xx, yy, grid) in enumerate(test_loader):
            
            if(isinstance(model, (FNO1d, OFormer1D))):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid, 1, 2)
                im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid, 1, 2)
                im = model(x, grid)[...,0,0]
                loss = loss_fn(yy[:,0,:], im)

            test_l2_step += loss.item()
            test_l2_full += loss.item()
    return test_l2_full/(bn+1)
                

def run_training(model, config, prefix):
    
    ################################################################
    # load data
    ################################################################
    
    path = "{}{}_{}".format(train_args['results_dir'], config['model_name'], prefix)
    f = h5py.File("{}{}".format(config['base_path'], config['data_name']), 'r')
    model_name = config['flnm'] + '_{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    
    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    train_loader, val_loader, test_loader = get_data(f, config)
    
    ################################################################
    # training and evaluation
    ################################################################
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')
    # Use Adam for Hyena
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if(isinstance(model, OFormer1D)):
        print("\nUSING ONECYCLELER SCHEDULER\n")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                        steps_per_epoch=len(train_loader), epochs=config['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    #loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for bn, (xx, yy, grid) in enumerate(train_loader):
            
            # Each model handles input differnetly
            if(isinstance(model, (FNO1d, OFormer1D))):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid, 1, 2)
                im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
            elif(isinstance(model,DeepONet1D)):
                x = torch.swapaxes(xx, 1, 2)
                grid = torch.swapaxes(grid, 1, 2)
                im = model(x, grid)[...,0,0]
                loss = loss_fn(yy[:,0,:], im)

            # Guarantees we're able to plot at least a few from first batch
            if(bn == 0):
                y_train_true = yy[:,0,:].clone()
                y_train_pred = im.clone()

            train_l2_step += loss.item()
            train_l2_full += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(isinstance(model, OFormer1D)):
                scheduler.step()

        train_l2s.append(train_l2_full/(bn+1))
        bn1 = bn

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                model.eval()
                for bn, (xx, yy, grid) in enumerate(val_loader):

                    # Each model handles input differnetly
                    if(isinstance(model, (FNO1d, OFormer1D))):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid, 1, 2)
                        im, loss = model.get_loss(x, yy[:,0,:], grid, loss_fn)
                    elif(isinstance(model,DeepONet1D)):
                        x = torch.swapaxes(xx, 1, 2)
                        grid = torch.swapaxes(grid, 1, 2)
                        im = model(x, grid)[...,0,0]
                        loss = loss_fn(yy[:,0,:], im)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
                        y_val_true = yy[:,0,:].clone()
                        y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    best_ep = ep
                    best_model = model.state_dict()
                    best_optimizer = optimizer.state_dict()
                    best_loss_val_min = loss_val_min

                    # Save best
                    torch.save({
                        'epoch': best_ep,
                        'model_state_dict': best_model,
                        'optimizer_state_dict': best_optimizer,
                        'loss': best_loss_val_min
                    }, model_path)

        model.train()
        val_l2s.append(val_l2_full/(bn+1))
                
        t2 = default_timer()
        if(not isinstance(model, OFormer1D)):
            scheduler.step()
        if(ep%config['log_freq'] == 0):
            print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
                .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)

        if(ep%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)


    # Make sure to capture last
    print('epoch: {0}, loss: {1:.5f}, time: {2:.5f}s, trainL2: {3:.5f}, testL2: {4:.5f}'\
          .format(ep, loss.item(), t2 - t1, train_l2s[-1], val_l2s[-1]))
    np.save("./{}/train_l2s_{}.npy".format(path, seed), train_l2s)
    np.save("./{}/val_l2s_{}.npy".format(path, seed), val_l2s)
    progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    test_vals = []
    #model.eval()
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, model, loss_fn)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)
    model.train()
            
if __name__ == "__main__":

    try:
        model_name = sys.argv[1]
    except IndexError:
        print("Default model is FNO. Training FNO.")
        model_name = "fno"
    try:
        assert model_name in ['fno', 'oformer','deeponet']
    except AssertionError as e:
        print("\nModel must be one of: fno, oformer or deeponet. Model selected was: {}\n".format(model_name))
        raise

    # Load config
    with open("./configs/{}_config.yaml".format(model_name), 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    train_args['model_name'] = model_name
    device = train_args['device']
    prefix = train_args['flnm'] + "_" + train_args['data_name'].split("_")[0] + "_" + train_args['train_style']
    os.makedirs("{}{}_{}".format(train_args['results_dir'], model_name, prefix), exist_ok=True)
    shutil.copy("./configs/{}_config.yaml".format(model_name),
                "{}{}_{}/{}_config.yaml".format(train_args['results_dir'], model_name, prefix, model_name))
    shutil.copy("./plot_progress.py", "{}{}_{}/plot_progress.py".format(train_args['results_dir'], model_name, prefix))

    for seed in range(train_args.pop('num_seeds')):
        #if(seed in [1,2]):
        #    continue
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed

        model = get_model(model_name, train_args)
        run_training(model, train_args, prefix)
    print("Done.")
