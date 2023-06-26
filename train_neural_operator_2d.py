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
from models.oformer import SpatialTemporalEncoder2D, PointWiseDecoder2D, OFormer2D, STDecoder2D
from models.fno import FNO2d
from models.deeponet import DeepONet2D

from utils import TransformerOperatorDataset2D, ElectricTransformerOperatorDataset2D

import yaml
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt

def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
    ncols = 4
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    ax[0][0].imshow(y_train_true[0].detach().cpu())
    ax[0][1].imshow(y_train_true[1].detach().cpu())
    ax[0][2].imshow(y_val_true[0].detach().cpu())
    ax[0][3].imshow(y_val_true[1].detach().cpu())

    ax[1][0].imshow(y_train_pred[0].detach().cpu())
    ax[1][1].imshow(y_train_pred[1].detach().cpu())
    ax[1][2].imshow(y_val_pred[0].detach().cpu())
    ax[1][3].imshow(y_val_pred[1].detach().cpu())

    ax[0][0].set_xlabel("VALIDATION SET TRUE")
    ax[1][0].set_xlabel("VALIDATION SET PRED")
    #ax[0][2].set_title("VALIDATION SET PRED")
    #ax[0][3].set_title("VALIDATION SET PRED")
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


def train_plots(train_loader, split, seed=None):
    im_num = 0
    viscocities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    amplitudes = [0.01, 0.05, 0.1, 0.25, 0.5, 1.]
    for vals in train_loader:
        #print(vals[2][0][1:] - vals[2][0][:-1])
        #raise
        #for idx, v in tqdm(enumerate(vals[1])):
        for idx, v in enumerate(vals[1]):
            visc = viscocities[(im_num//100)//6]
            amp = amplitudes[(im_num//100)%6]
            print(im_num, (im_num//100), (im_num//100)//7, (im_num//100)%6)

            fig, ax = plt.subplots(figsize=(8,6))
            ax.imshow(v.detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            ax.set_title("{} Viscocity, {} Amplitude".format(visc, amp))
            plt.savefig("./{}_plots/{}_{}.png".format(split, seed, fname))
            plt.close()
            #raise

            im_num += 1


def get_model(model_name, config):
    if(model_name == "fno"):
        model = FNO2d(config['num_channels'], config['modes1'], config['modes2'], config['width'], config['initial_step'],
                      config['dropout'])
    elif(model_name == "oformer"):
        encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
                            #, dropout=config['dropout'],
                            #res=config['enc_res'])
        decoder = PointWiseDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     propagator_depth=config['decoding_depth'], scale=config['scale'], out_steps=1)
        #decoder = STDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'], out_steps=1,
        #                       propagator_depth=config['decoder_depth'], scale=config['scale'], res=config['dec_res'])
        model = OFormer2D(encoder, decoder, num_x=config['num_x'], num_y=config['num_y'])
    elif(model_name == "deeponet"):
        model = DeepONet2D(config['branch_net'], config['trunk_net'], config['activation'], config['kernel_initializer'])
    
    model.to(device)
    return model


def get_data(f, config):
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    print("\nTRAINING DATA")
    if('electric' in config['data_name']):
        train_data = ElectricTransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
        print("\nVALIDATION DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        val_data = ElectricTransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
        print("\nTEST DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        test_data = ElectricTransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                seed=config['seed'],
        )
    else:
        train_data = TransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )
        print("\nVALIDATION DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        val_data = TransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )
        print("\nTEST DATA")
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        test_data = TransformerOperatorDataset2D(f,
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
                                split_style=config['split_style'],
                                samples_per_equation=config['samples_per_equation'],
                                seed=config['seed'],
        )

    if(config['split_style'] == 'equation'):
        assert not (bool(set(train_data.data_list) & \
                         set(val_data.data_list)) | \
                    bool(set(train_data.data_list) & \
                         set(test_data.data_list)) & \
                    bool(set(val_data.data_list) & \
                         set(test_data.data_list)))
    elif(config['split_style'] == 'initial_condition'):
        assert not (bool(set(train_data.idxs) & \
                         set(val_data.idxs)) | \
                    bool(set(train_data.idxs) & \
                         set(test_data.idxs)) & \
                    bool(set(val_data.idxs) & \
                         set(test_data.idxs)))
    else:
        raise ValueError("Invalid splitting style. Select initial_condition or equation")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cuda'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=True,
                                             generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cuda'))
    return train_loader, val_loader, test_loader


def evaluate(test_loader, model, loss_fn, navier_stokes=True):
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        model.eval()
        for bn, (xx, yy, grid, tokens, t) in enumerate(test_loader):
            xx = xx.to(device).float()
            yy = yy.to(device).float()
            grid = grid.to(device).float()
            
            if(isinstance(model, (FNO2d, OFormer2D))):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
            elif(isinstance(model, DeepONet2D)):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im = model(x, grid)
                loss = loss_fn(yy[...,0], im)
    
            test_l2_step += loss.item()
            test_l2_full += loss.item()
    return test_l2_full/(bn+1)
                

def run_training(model, config, prefix):
    
    ################################################################
    # load data
    ################################################################
    
    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}".format(config['results_dir'], config['model_name'], prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_name = '{}'.format(config['model_name']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name
    navier_stokes = not('electric' in config['data_name'])
    
    print("Seed: {}\n".format(config['seed']))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    train_loader, val_loader, test_loader = get_data(f, config)
    #train_plots(train_loader, 'train', 0)
    #train_plots(val_loader, 'val', 0)
    #raise
    
    ################################################################
    # training and evaluation
    ################################################################
    
    if(config['return_text']):
        _data, _, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if(isinstance(model, OFormer2D)):
        print("\nUSING ONECYCLELER SCHEDULER\n")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                        steps_per_epoch=len(train_loader), epochs=config['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    
    loss_fn = nn.L1Loss(reduction="mean")
    loss_val_min = np.infty
    
    start_epoch = 0

    # TODO: Model restarting
    #if continue_training:
    #    print('Restoring model (that is the network\'s weights) from file...')
    #    checkpoint = torch.load(model_path, map_location=device)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    model.to(device)
    #    model.train()
    #    
    #    # Load optimizer state dict
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    for state in optimizer.state.values():
    #        for k, v in state.items():
    #            if isinstance(v, torch.Tensor):
    #                state[k] = v.to(device)
    #                
    #    start_epoch = checkpoint['epoch']
    #    loss_val_min = checkpoint['loss']
    
    train_l2s, val_l2s = [], []
    for ep in tqdm(range(start_epoch, config['epochs'])):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for bn, (xx, yy, grid, tokens, t) in enumerate(train_loader):
            
            # Put data on correct device
            xx = xx.to(device).float()
            yy = yy.to(device).float()
            grid = grid.to(device).float()
            
            # Each model handles input differnetly
            if(isinstance(model, (FNO2d, OFormer2D))):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
            elif(isinstance(model, DeepONet2D)):
                if(navier_stokes):
                    x = torch.swapaxes(xx, 1, 3)
                    x = torch.swapaxes(x, 1, 2)
                else:
                    x = xx
                im = model(x, grid)
                loss = loss_fn(yy[...,0], im)

            # Guarantees we're able to plot at least a few from first batch
            if(bn == 0):
                y_train_true = yy[...,0].clone()
                y_train_pred = im.clone()

            train_l2_step += loss.item()
            train_l2_full += loss.item()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(isinstance(model, OFormer2D)):
                scheduler.step()

        train_l2s.append(train_l2_full/(bn+1))
        bn1 = bn

        if ep % config['validate'] == 0:
            val_l2_step = 0
            val_l2_full = 0
            model.eval()
            with torch.no_grad():
                for bn, (xx, yy, grid, tokens, t) in enumerate(val_loader):

                    # Put data on correct device
                    xx = xx.to(device).float()
                    yy = yy.to(device).float()
                    grid = grid.to(device).float()
                    
                    # Each model handles input differnetly
                    if(isinstance(model, (FNO2d, OFormer2D))):
                        if(navier_stokes):
                            x = torch.swapaxes(xx, 1, 3)
                            x = torch.swapaxes(x, 1, 2)
                        else:
                            x = xx
                        im, loss = model.get_loss(x, yy[...,0], grid, loss_fn)
                    elif(isinstance(model, DeepONet2D)):
                        if(navier_stokes):
                            x = torch.swapaxes(xx, 1, 3)
                            x = torch.swapaxes(x, 1, 2)
                        else:
                            x = xx
                        im = model(x, grid)
                        loss = loss_fn(yy[...,0], im)

                    # Guarantees we're able to plot at least a few from first batch
                    if(bn == 0):
                        y_val_true = yy[...,0].clone()
                        y_val_pred = im.clone()

                    val_l2_step += loss.item()
                    val_l2_full += loss.item()
                
                if  val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_val_min
                        }, model_path)
        val_l2s.append(val_l2_full/(bn+1))
                
            
        t2 = default_timer()
        if(not isinstance(model, OFormer2D)):
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
    if(len(y_train_true) >= 4): 
        progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)

    test_vals = []
    test_value = evaluate(test_loader, model, loss_fn, navier_stokes)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, model, loss_fn, navier_stokes)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("./{}/test_vals_{}.npy".format(path, seed), test_vals)

            
if __name__ == "__main__":

    try:
        model_name = sys.argv[1]
    except IndexError:
        print("Default model is FNO. Training FNO.")
        model_name = "fno"
    try:
        assert model_name in ['fno', 'deeponet', 'oformer']
    except AssertionError as e:
        print("\nModel must be one of: fno, deeponet, or oformer. Model selected was: {}\n".format(model_name))
        raise

    # Load config
    with open("./configs/2d_{}_config.yaml".format(model_name), 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    train_args['model_name'] = model_name
    device = train_args['device']#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prefix = train_args['data_name'].split("_")[0] + "_" + train_args['train_style']
    if('electric' in train_args['data_name']):
        prefix = 'electric_' + prefix
    os.makedirs("{}{}_{}".format(train_args['results_dir'], model_name, prefix), exist_ok=True)
    shutil.copy("./configs/2d_{}_config.yaml".format(model_name),
                "{}{}_{}/2d_{}_config.yaml".format(train_args['results_dir'], model_name, prefix, model_name))
    shutil.copy("./plot_progress.py", "{}{}_{}/plot_progress.py".format(train_args['results_dir'], model_name, prefix))


    for seed in range(train_args.pop('num_seeds')):
        if(seed != 4):
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed

        model = get_model(model_name, train_args)
        run_training(model, train_args, prefix)
    print("Done.")


