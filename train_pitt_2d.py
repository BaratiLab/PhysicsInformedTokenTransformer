import torch
import torch.nn as nn
import yaml
import h5py
from utils import TransformerOperatorDataset2D, ElectricTransformerOperatorDataset2D
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil

from models.pitt import StandardPhysicsInformedTokenTransformer2D
from models.pitt import PhysicsInformedTokenTransformer2D

from models.oformer import OFormer2D, SpatialTemporalEncoder2D, STDecoder2D, PointWiseDecoder2D
from models.deeponet import DeepONet2D
from models.fno import FNO2d

import sys

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'


def custom_collate(batch):
    x0 = torch.empty((len(batch), batch[0][0].shape[0]))
    y = torch.empty((len(batch), batch[0][1].shape[0], 1))
    grid = torch.empty((len(batch), batch[0][2].shape[0]))
    tokens = torch.empty((len(batch), batch[0][3].shape[0]))
    forcing = []
    time = torch.empty(len(batch))
    for idx, b in enumerate(batch):
        x0[idx] = b[0]
        y[idx] = b[1]
        grid[idx] = b[2]
        tokens[idx] = b[3]
        forcing.append(b[4])
        time[idx] = b[5]
    return x0, y, grid, tokens, forcing, time


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

    ax[0][0].set_ylabel("VALIDATION SET TRUE")
    ax[1][0].set_ylabel("VALIDATION SET PRED")
    fname = str(ep)
    plt.tight_layout()
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
            ax.plot(v.reshape(100,).detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_2/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def evaluate(test_loader, transformer, loss_fn, config=None):
    #src_mask = generate_square_subsequent_mask(640).cuda()
    with torch.no_grad():
        transformer.eval()
        test_loss = 0
        for bn, (x0, y, grid, tokens, t) in enumerate(test_loader):
            # Forward pass: compute predictions by passing the input sequence
            # through the transformer.

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            tokens = tokens.to(device).float()
            t = t.to(device).float()
            grid = grid.to(device).float()

            # Rearrange data
            if(config is not None and not('electric' in config['data_name'])):
                x0 = torch.swapaxes(x0, 1, 3)
                x0 = torch.swapaxes(x0, 1, 2)

            y_pred = transformer(grid, tokens, x0, t)
            y = y[...,0].to(device=device)
    
            # Compute the loss.
            test_loss += loss_fn(y_pred, y).item()

    return test_loss/(bn+1)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_data(f, config):
    if('electric' in config['data_name']):
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        print("\nTRAINING DATA")
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
                                seed=config['seed']
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
                                seed=config['seed']
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
                                seed=config['seed']
        )
    else:
        f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
        print("\nTRAINING DATA")
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
                                seed=config['seed']
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
                                seed=config['seed']
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
                                seed=config['seed']
        )

    # Check against data leaks
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], generator=torch.Generator(device='cuda'),
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'], generator=torch.Generator(device='cuda'),
                                             num_workers=config['num_workers'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False)
    return train_loader, val_loader, test_loader


def get_neural_operator(model_name, config):
    if(model_name == "fno"):
        model = FNO2d(config['num_channels'], config['modes1'], config['modes2'], config['width'], config['initial_step'],
                      config['dropout'])
    elif(model_name == "unet"):
        model = UNet2d(in_channels=config['initial_step'], init_features=config['init_features'], dropout=config['dropout'])
    elif(model_name == "oformer"):
        encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
                            #, dropout=config['dropout'],
                            #res=config['enc_res'])
        decoder = PointWiseDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     propagator_depth=config['decoder_depth'], scale=config['scale'], out_steps=1)
        model = OFormer2D(encoder, decoder, num_x=config['num_x'], num_y=config['num_y'])
    elif(model_name == 'deeponet'):
        model = DeepONet2D(layer_sizes_branch=config['branch_net'], layer_sizes_trunk=config['trunk_net'],
                                activation=config['activation'], kernel_initializer=config['kernel_initializer'])

    model.to(device)
    return model


def get_transformer(model_name, config):
    # Create the transformer model.
    if(config['embedding'] == "standard"):
        print("\n USING STANDARD EMBEDDING")
        neural_operator = get_neural_operator(config['neural_operator'], config)
        transformer = StandardPhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
                                        output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
                                        neural_operator=neural_operator).to(device=device)
    elif(config['embedding'] == "novel"):
        print("\n USING NOVEL EMBEDDING")
        neural_operator = get_neural_operator(config['neural_operator'], config)
        transformer = PhysicsInformedTokenTransformer2D(100, config['hidden'], config['layers'], config['heads'],
                                        output_dim1=config['num_x'], output_dim2=config['num_y'], dropout=config['dropout'],
                                        neural_operator=neural_operator).to(device=device)
    else:
        raise ValueError("Invalid embedding choice.")
    return transformer


def run_training(config, prefix):

    #print(f'Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}    , scheduler gamma = {scheduler_gamma}')

    ################################################################
    # load data
    ################################################################

    #prefix = config['data_name'].split("_")[0]
    path = "{}{}_{}_{}".format(config['results_dir'], config['model'], config['neural_operator'], prefix)
    f = h5py.File("{}/{}".format(config['base_path'], config['data_name']), 'r')
    model_name = '{}'.format(config['model']) + "_{}.pt".format(seed)
    model_path = path + "/" + model_name

    # Create the transformer model.
    transformer = get_transformer(config['model'], config)

    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Get data as loaders
    train_loader, val_loader, test_loader = get_data(f, config)

    ################################################################
    # training and evaluation
    ################################################################

    if(config['return_text']):
        _data, _, _, _, _ = next(iter(val_loader))
    else:
        _data, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print('Spatial Dimension', dimensions - 3)

    
    # Use Adam as the optimizer.
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],# div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])
    
    # Use mean squared error as the loss function.
    loss_fn = nn.L1Loss(reduction='mean')
    
    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    #src_mask = generate_square_subsequent_mask(640).cuda()
    lrs = []
    shift = 0
    for epoch in tqdm(range(config['epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        times = []
        max_val = 0
        transformer.train()
        for bn, (x0, y, grid, tokens, t) in enumerate(train_loader):
            start = time.time()

            # Put data on correct device
            x0 = x0.to(device).float()
            y = y.to(device).float()
            tokens = tokens.to(device).float()
            t = t.to(device).float()
            grid = grid.to(device).float()

            # Rearrange data
            if(not('electric' in config['data_name'])):
                x0 = torch.swapaxes(x0, 1, 3)
                x0 = torch.swapaxes(x0, 1, 2)

            # Forward pass
            y_pred = transformer(grid, tokens, x0, t)

            y = y[...,0].to(device=device)#.cuda()

            # Compute the loss.
            loss = loss_fn(y_pred, y)

            # Backward pass: compute gradient of the loss with respect to model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

            train_loss += loss.item()
            if(bn == 0):
                y_train_true = y.clone()
                y_train_pred = y_pred.clone()

            scheduler.step()


            if(bn%100 == 0 and len(train_loader) >= 1000):
                print("Batch: {0}\tloss = {1:.4f}".format(bn, train_loss/(bn+1)))
                fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(22,15))
                im0 = ax[0][0].imshow(transformer.query_matrix.detach().cpu()[0], cmap='bwr')
                divider = make_axes_locatable(ax[0][0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im0, cax=cax, orientation='vertical')

                im1 = ax[0][1].imshow(transformer.key_matrix.detach().cpu()[0], cmap='bwr')
                divider = make_axes_locatable(ax[0][1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')

                prod_mat = torch.mul(transformer.query_matrix.detach().cpu()[0], transformer.key_matrix.detach().cpu()[0])
                im2 = ax[0][2].imshow(prod_mat, vmin=prod_mat.mean()-prod_mat.std(), vmax=prod_mat.mean()+prod_mat.std(), cmap='bwr')
                divider = make_axes_locatable(ax[0][2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im2, cax=cax, orientation='vertical')

                im3 = ax[1][0].imshow(transformer.q2_matrix.detach().cpu()[0], cmap='bwr')
                divider = make_axes_locatable(ax[1][0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im3, cax=cax, orientation='vertical')

                im4 = ax[1][1].imshow(transformer.k2_matrix.detach().cpu()[0], cmap='bwr')
                divider = make_axes_locatable(ax[1][1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im4, cax=cax, orientation='vertical')

                prod_mat2 = torch.mul(transformer.q2_matrix.detach().cpu()[0], transformer.k2_matrix.detach().cpu()[0])
                im5 = ax[1][2].imshow(prod_mat2, vmin=prod_mat2.mean()-prod_mat2.std(), vmax=prod_mat2.mean()+prod_mat2.std(), cmap='bwr')
                divider = make_axes_locatable(ax[1][2])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im5, cax=cax, orientation='vertical')
                #raise
                ax[0][0].set_title("Query Matrix", fontsize=20)
                ax[0][1].set_title("Key Matrix", fontsize=20)
                ax[0][2].set_title("Matrix Product", fontsize=20)

                ax[0][0].set_ylabel("First Order Operator Matrices", fontsize=20)
                ax[1][0].set_ylabel("Second Order Operator Matrices", fontsize=20)
                #ax[2].imshow(transformer.v_embedding_layer.weight.detach().cpu().T)
                plt.tight_layout()
                #plt.savefig("./{}/weight_matrices_{}_{}.png".format(path, seed, epoch))
                fname = str(bn)
                while(len(fname) < 8):
                    fname = '0' + fname
                plt.savefig("./{}/weight_matrices_{}_{}_{}.png".format(path, seed, epoch, fname))
                plt.close()
                #plt.show()



        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        with torch.no_grad():
            transformer.eval()
            val_loss = 0
            all_val_preds = []
            for bn, (x0, y, grid, tokens, t) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                # Put data on correct device
                x0 = x0.to(device).float()
                y = y.to(device).float()
                tokens = tokens.to(device).float()
                t = t.to(device).float()
                grid = grid.to(device).float()

                # Rearrange data
                if(not('electric' in config['data_name'])):
                    x0 = torch.swapaxes(x0, 1, 3)
                    x0 = torch.swapaxes(x0, 1, 2)

                y_pred = transformer(grid, tokens, x0, t)
                y = y[...,0].to(device=device)#.cuda()
                if(bn == 0):
                    y_val_true = y.clone()
                    y_val_pred = y_pred.clone()
                all_val_preds.append(y_pred.detach())
    
                # Compute the loss.
                val_loss += loss_fn(y_pred, y).item()

            if  val_loss < loss_val_min:
                loss_val_min = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val_min
                    }, model_path)

        val_loss /= (bn + 1)
        val_losses.append(val_loss)

        # Print the loss at the end of each epoch.
        if(epoch%config['log_freq'] == 0):
            np.save("./{}/train_l2s_{}.npy".format(path, seed), train_losses)
            np.save("./{}/val_l2s_{}.npy".format(path, seed), val_losses)
            np.save("./{}/lrs_{}.npy".format(path, seed), lrs)
            print(f"Epoch {epoch+1}: loss = {train_loss:.6f}\t val loss = {val_loss:.6f}")

        if(epoch%config['progress_plot_freq'] == 0 and len(y_train_true) >= 4):
            progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)


    try:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(22,15))
        im0 = ax[0][0].imshow(transformer.query_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[0][0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')

        im1 = ax[0][1].imshow(transformer.key_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[0][1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        prod_mat = torch.mul(transformer.query_matrix.detach().cpu()[0], transformer.key_matrix.detach().cpu()[0])
        im2 = ax[0][2].imshow(prod_mat, vmin=prod_mat.mean()-prod_mat.std(), vmax=prod_mat.mean()+prod_mat.std())
        divider = make_axes_locatable(ax[0][2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        im3 = ax[1][0].imshow(transformer.q2_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[1][0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        im4 = ax[1][1].imshow(transformer.k2_matrix.detach().cpu()[0])
        divider = make_axes_locatable(ax[1][1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')

        prod_mat2 = torch.mul(transformer.q2_matrix.detach().cpu()[0], transformer.k2_matrix.detach().cpu()[0])
        im5 = ax[1][2].imshow(prod_mat2, vmin=prod_mat2.mean()-prod_mat2.std(), vmax=prod_mat2.mean()+prod_mat2.std())
        divider = make_axes_locatable(ax[1][2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im5, cax=cax, orientation='vertical')
        #raise
        ax[0][0].set_title("Query Matrix", fontsize=20)
        ax[0][1].set_title("Key Matrix", fontsize=20)
        ax[0][2].set_title("Matrix Product", fontsize=20)

        ax[0][0].set_ylabel("First Order Operator Matrices", fontsize=20)
        ax[1][0].set_ylabel("Second Order Operator Matrices", fontsize=20)
        #ax[2].imshow(transformer.v_embedding_layer.weight.detach().cpu().T)
        plt.tight_layout()
        plt.savefig("./{}/weight_matrices_{}.png".format(path, seed))
        #plt.show()
    except AttributeError:
        pass
    #raise
    #progress_plots(epoch, y_train_true, y_train_pred, y_val_true, y_val_pred, path, seed=seed)
    #val_plots(epoch, val_loader, all_val_preds, seed=seed)

    test_vals = []
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    transformer.load_state_dict(torch.load(model_path)['model_state_dict'])
    test_value = evaluate(test_loader, transformer, loss_fn, config=config)
    test_vals.append(test_value)
    print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    np.save("{}{}_{}_{}/test_vals_{}.npy".format(config['results_dir'], config['model'], config['neural_operator'], prefix, seed), test_vals)


if __name__ == '__main__':
    # Create a transformer with an input dimension of 10, a hidden dimension
    # of 20, 2 transformer layers, and 8 attention heads.

    # Load config
    #raise
    with open("./configs/2d_pitt_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Get arguments and get rid of unnecessary ones
    train_args = config['args']
    prefix = train_args['data_name'].split("_")[0] + "_" + train_args['train_style'] + "_" + train_args['embedding']
    if('electric' in train_args['data_name']):
        prefix = "electric_" + prefix
    train_args['prefix'] = prefix
    os.makedirs("{}{}_{}_{}".format(train_args['results_dir'], train_args['model'], train_args['neural_operator'], prefix),
                exist_ok=True)
    shutil.copy("./configs/2d_pitt_config.yaml",
                "{}{}_{}_{}/2d_pitt_config.yaml".format(train_args['results_dir'],
                train_args['model'], train_args['neural_operator'], prefix))
    shutil.copy("./plot_progress.py", "{}{}_{}_{}/plot_progress.py".format(train_args['results_dir'],
                train_args['model'], train_args['neural_operator'], prefix))


    for seed in range(train_args.pop('num_seeds')):
        print("\nSEED: {}\n".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_args['seed'] = seed
        run_training(train_args, prefix)
    
