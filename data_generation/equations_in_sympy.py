from sympy import Function, pprint, Sum, sin, cos, log
from sympy.core.symbol import Symbol, symbols
from sympy.abc import x, t, alpha, beta, gamma, delta, A, omega, pi, l, L, phi, j, J, eta

import tokenize
from io import StringIO
import numpy as np
np.random.seed(137)

import torch
torch.random.manual_seed(137)
from typing import Callable, Tuple
from mp_code.PDEs import PDE, CE
from mp_code.solvers import *
import h5py
import time
import sys

import json
from tqdm import tqdm

WORDS = ['(', ')', '+', '-', '*', '/', 'Derivative', 'Sum', 'j', 'A_j', 'l_j', 'omega_j', 'phi_j', 'sin', 't', 'u', 'x',
         'dirichlet', 'neumann', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^', 'E', 'e', ',', '.', '&']
word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for i, w in enumerate(WORDS)}
#torch.cuda.set_device('cuda:0')

def initial_conditions(A: torch.Tensor,
                       omega: torch.Tensor,
                       phi: torch.Tensor,
                       l: torch.Tensor,
                       pde: PDE) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Return initial conditions for combined equation based on initial parameters
    Args:
        A (torch.Tensor): amplitude of different sine waves
        omega (torch.Tensor): time-dependent frequency
        phi (torch.Tensor): phase shift of different sine waves
        l (torch.Tensor): frequency of sine waves
    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function which initializes for chosen set of parameters
    """
    def fnc(x, t=0):
        u = torch.sum(A * torch.sin(omega*t + (2 * np.pi * l * x / pde.L) + phi), -1)
        return u
    return fnc


def generate_data_combined_equation(experiment: str, pde: dict, modes: list, num_samples: int, batch_size: int,
                                    alpha, beta, gamma, As, omegas, phis, ls, device: torch.cuda.device = "cpu") -> None:

    """
    Generate data for combined equation using different coefficients
    Args:
        experiment (str): experiment string
        pde (dict): dictionary for PDEs at different resolution
        mode (list): train, valid, test
        num_samples (int): number of trajectories to solve
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
        alpha (list): alpha parameter (low, high)
        beta (list): beta parameter (low, high)
        gamma (list): gamma parameter (low, high)
        alpha
        beta
        gamma
        As
        omegas
        phis
        ls
    Returns:
        None
    """
    print(f'Device: {device}')
    num_batches = num_samples // batch_size

    pde_string = str(pde[list(pde.keys())[0]])
    print(f'Equation: {experiment}')
    print(f'Mode: {modes}')
    print(f'Number of samples: {num_samples}')
    print(f'Batch size: {batch_size}')
    print(f'Number of batches: {num_batches}')

    #save_name = "data/" + "_".join([str(pde[list(pde.keys())[0]]), mode]) + "_" + experiment
    #save_name = "data/" + str(pde[list(pde.keys())[0]]) + "_" + experiment + "_" + mode
    save_name = "../data/" + str(pde[list(pde.keys())[0]]) + "_" + experiment #+ "_" + mode
    h5f = h5py.File("".join([save_name, '.h5']), 'a')

    for mode in modes:
        print(mode)
        raise
        dataset = h5f.create_group(mode)

        t = {}
        x = {}
        h5f_u = {}
        h5f_alpha = {}
        h5f_beta = {}
        h5f_gamma = {}

        for key in pde:
            t[key] = torch.linspace(pde[key].tmin, pde[key].tmax, pde[key].grid_size[0]).to(device)
            x[key] = torch.linspace(0, pde[key].L, pde[key].grid_size[1]).to(device)
            h5f_u[key] = dataset.create_dataset(key, (num_samples, pde[key].grid_size[0], pde[key].grid_size[1]), dtype=float)
            h5f[mode][key].attrs['dt'] = pde[key].dt
            h5f[mode][key].attrs['dx'] = pde[key].dx
            h5f[mode][key].attrs['nt'] = pde[key].grid_size[0]
            h5f[mode][key].attrs['nx'] = pde[key].grid_size[1]
            h5f[mode][key].attrs['tmin'] = pde[key].tmin
            h5f[mode][key].attrs['tmax'] = pde[key].tmax
            h5f[mode][key].attrs['x'] = x[key].cpu()
            h5f[mode][key].attrs['t'] = t[key].cpu()

        #print(h5f[mode][key].attrs)
        #raise
        h5f_alpha['alpha'] = dataset.create_dataset('alpha', (num_samples, ), dtype=float)
        h5f_beta['beta'] = dataset.create_dataset('beta', (num_samples, ), dtype=float)
        h5f_beta['gamma'] = dataset.create_dataset('gamma', (num_samples, ), dtype=float)
        #h5f['system_parameters'] = dataset.create_dataset('system_parameters', (num_samples, ), dtype=float)

        # torch.random.manual_seed(2)
        for idx in range(num_batches):

            # Time dependent force term
            def force(t):
                #return initial_conditions(A, omega, phi, l, pde[key])(x[key][:, None], t)[:, None]
                return initial_conditions(A, omega, phi, l, pde[key])(x[key][:, None], t)

            sol = {}
            for key in pde:
                # Initialize PDE parameters and get initial condition
                u0 = initial_conditions(As, omegas, phis, ls, pde[key])(x[key][:, None])
                if(len(u0.shape) == 1):
                    u0 = u0[np.newaxis]

                # The spatial method is the WENO reconstruction for uux and FD for the rest
                spatial_method = pde[key].WENO_reconstruction

                # Solving full trajectories and runtime measurement
                torch.cuda.synchronize()
                t1 = time.time()
                solver = Solver(RKSolver(Dopri45(), device=device), spatial_method)
                sol[key] = solver.solve(x0=u0[:, None].to(device), times=t[key][None, :].to(device))
                torch.cuda.synchronize()
                t2 = time.time()
                #print(f'{key}: {t2 - t1:.4f}s')

            # Save solutions
            for key in pde:
                h5f_u[key][idx * batch_size:(idx + 1) * batch_size, :, :] = \
                    sol[key].cpu().reshape(batch_size, pde[key].grid_size[0], -1)

            h5f_alpha['alpha'][idx * batch_size:(idx + 1) * batch_size] = alpha#.detach().cpu()
            h5f_beta['beta'][idx * batch_size:(idx + 1) * batch_size] = beta#.detach().cpu()
            h5f_beta['gamma'][idx * batch_size:(idx + 1) * batch_size] = gamma#.detach().cpu()

            #print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
            #print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))

            sys.stdout.flush()

    #print()

    print("Data saved")
    print()
    print()
    h5f.close()
    return save_name


def generate_data_single_equation(h5f, experiment: str, pde: dict, modes: list, num_samples: int, batch_size: int, alpha, beta, gamma, As, omegas, phis, ls, all_tokens, encoded_tokens, device: torch.cuda.device = "cpu") -> None:

    """
    Generate data for combined equation using different coefficients
    Args:
        experiment (str): experiment string
        pde (dict): dictionary for PDEs at different resolution
        mode (list): train, valid, test
        num_samples (int): number of trajectories to solve
        batch_size (int): batch size
        device (torch.cuda.device): device (cpu/gpu)
        alpha (list): alpha parameter (low, high)
        beta (list): beta parameter (low, high)
        gamma (list): gamma parameter (low, high)
        alpha
        beta
        gamma
        As
        omegas
        phis
        ls
    Returns:
        None
    """
    #print(f'Device: {device}')
    num_batches = num_samples // batch_size

    pde_string = str(pde[list(pde.keys())[0]])
    #print(f'Equation: {experiment}')
    #print(f'Mode: {modes}')
    #print(f'Number of samples: {num_samples}')
    #print(f'Batch size: {batch_size}')
    #print(f'Number of batches: {num_batches}')

    #save_name = "data/" + "_".join([str(pde[list(pde.keys())[0]]), mode]) + "_" + experiment
    #save_name = "data/" + str(pde[list(pde.keys())[0]]) + "_" + experiment + "_" + mode
    #save_name = "data/" + str(pde[list(pde.keys())[0]]) + "_" + experiment #+ "_" + mode
    #h5f = h5py.File("".join([save_name, '.h5']), 'a')

    for mode in modes:
        #print(mode)
        #raise
        try:
            dataset = h5f.create_group(experiment)
        except ValueError:
            return

        t = {}
        x = {}
        h5f_u = {}
        h5f_alpha = {}
        h5f_beta = {}
        h5f_gamma = {}

        for key in pde:
            t[key] = torch.linspace(pde[key].tmin, pde[key].tmax, pde[key].grid_size[0]).to(device)
            x[key] = torch.linspace(0, pde[key].L, pde[key].grid_size[1]).to(device)
            h5f_u[key] = dataset.create_dataset(key, (num_samples, pde[key].grid_size[0], pde[key].grid_size[1]), dtype=float)
            h5f[mode][key].attrs['dt'] = pde[key].dt
            h5f[mode][key].attrs['dx'] = pde[key].dx
            h5f[mode][key].attrs['nt'] = pde[key].grid_size[0]
            h5f[mode][key].attrs['nx'] = pde[key].grid_size[1]
            h5f[mode][key].attrs['tmin'] = pde[key].tmin
            h5f[mode][key].attrs['tmax'] = pde[key].tmax
            h5f[mode][key].attrs['x'] = x[key].cpu()
            h5f[mode][key].attrs['t'] = t[key].cpu()
            #print(type(all_tokens))
            #h5f[mode][key].attrs['all_tokens'] = np.array(all_tokens)
            h5f[mode][key].attrs['encoded_tokens'] = np.array(encoded_tokens)

        #print(h5f[mode][key].attrs)
        #raise
        h5f_alpha['alpha'] = dataset.create_dataset('alpha', (num_samples, ), dtype=float)
        h5f_beta['beta'] = dataset.create_dataset('beta', (num_samples, ), dtype=float)
        h5f_beta['gamma'] = dataset.create_dataset('gamma', (num_samples, ), dtype=float)
        #h5f['system_parameters'] = dataset.create_dataset('system_parameters', (num_samples, ), dtype=float)

        # torch.random.manual_seed(2)
        for idx in range(num_batches):

            sol = {}
            for key in pde:
                # Initialize PDE parameters and get initial condition
                u0 = initial_conditions(As, omegas, phis, ls, pde[key])(x[key][:, None])
                if(len(u0.shape) == 1):
                    u0 = u0[np.newaxis]

                # Time dependent force term
                def force_fn(t):
                    #return initial_conditions(As, omegas, phis, ls, pde[key])(x[key][:, None], t)[:, None]
                    return initial_conditions(As, omegas, phis, ls, pde[key])(x[key][:, None], t)

                pde[key].force = force_fn

                # The spatial method is the WENO reconstruction for uux and FD for the rest
                spatial_method = pde[key].WENO_reconstruction

                # Solving full trajectories and runtime measurement
                #torch.cuda.synchronize()
                t1 = time.time()
                solver = Solver(RKSolver(Dopri45(), device=device), spatial_method)
                sol[key] = solver.solve(x0=u0[:, None].to(device), times=t[key][None, :].to(device))
                #torch.cuda.synchronize()
                t2 = time.time()
                #print(f'{key}: {t2 - t1:.4f}s')

            # Save solutions
            for key in pde:
                h5f_u[key][idx * batch_size:(idx + 1) * batch_size, :, :] = \
                    sol[key].cpu().reshape(batch_size, pde[key].grid_size[0], -1)

            h5f_alpha['alpha'][idx * batch_size:(idx + 1) * batch_size] = alpha#.detach().cpu()
            h5f_beta['beta'][idx * batch_size:(idx + 1) * batch_size] = beta#.detach().cpu()
            h5f_beta['gamma'][idx * batch_size:(idx + 1) * batch_size] = gamma#.detach().cpu()

            #print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
            #print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))

            sys.stdout.flush()

    #print()

    #print("Data saved")
    #print()
    #print()
    #h5f.close()
    #return save_name


def get_all_tokens(pde_tokens, delta_tokens, ic_tokens, left_bc, right_bc, As, omegas, ls, phis):
    all_tokens = []
    all_tokens.extend(pde_tokens)
    all_tokens.extend("&")
    all_tokens.extend(delta_tokens)
    all_tokens.extend("&")
    all_tokens.extend(ic_tokens)
    all_tokens.extend("&")
    all_tokens.extend([left_bc])
    all_tokens.extend("&")
    all_tokens.extend([right_bc])
    all_tokens.extend("&")
    all_tokens.extend(list(As.cpu().numpy().astype(str)))
    all_tokens.extend("&")
    all_tokens.extend(list(omegas.cpu().numpy().astype(str)))
    all_tokens.extend("&")
    all_tokens.extend(list(ls.cpu().numpy().astype(str)))
    all_tokens.extend("&")

    all_tokens.extend(list(phis.cpu().numpy().astype(str)))

    return all_tokens


def encode_tokens(all_tokens):
    encoded_tokens = []
    num_concat = 0
    for i in range(len(all_tokens)):
        try: # All the operators, bcs, regular symbols
            encoded_tokens.append(word2id[all_tokens[i]])
            # Puts commas in l values, can remove correction in utils.py
            if(num_concat == 7 and all_tokens[i] != '&'):
                encoded_tokens.append(word2id[","])
            # 5 concatenations before we get to lists of sampled values
            if(all_tokens[i] == "&"):
                if(num_concat >= 5): # Remove extraneous comma
                    encoded_tokens = encoded_tokens[:-2]
                    encoded_tokens.append(word2id['&'])
                num_concat += 1
        except KeyError: # Numerical values
            if(isinstance(all_tokens[i], str)):
                for v in all_tokens[i]:
                    encoded_tokens.append(word2id[v])
                if(num_concat >= 5): # We're in a list of sampled parameters
                    encoded_tokens.append(word2id[","])
            else:
                raise KeyError("Unrecognized token: {}".format(all_tokens[i]))

    return encoded_tokens[:-1]


def decode_tokens(all_tokens):
    decoded_tokens = []
    num_concat = 0
    for i in range(len(all_tokens)):
        decoded_tokens.append(id2word[all_tokens[i]])
        #continue
        #try: # All the operators, bcs, regular symbols
        #    encoded_tokens.append(word2id[all_tokens[i]])
        #    if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
        #        num_concat += 1
        #except KeyError: # Numerical values
        #    if(isinstance(all_tokens[i], str)):
        #        for v in all_tokens[i]:
        #            encoded_tokens.append(word2id[v])
        #        if(num_concat >= 5): # We're in a list of sampled parameters
        #            encoded_tokens.append(word2id[","])
        #    else:
        #        raise KeyError("Unrecognized token: {}".format(all_tokens[i]))

    return decoded_tokens


def generate_heat_data(save_name="all_data"):
    exp_names, exp_tokens, identifiers = [], [], []

    save_name = "./pde_data/" + save_name
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    print("\nETA: {}\n".format(eta))

    all_identifiers = []
    for (n, alpha, bet, gamma) in [("Heat", 0, 0.2, 0)]:
        for beta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            print("\nALPHA: {}\tBETA: {}\n".format(alpha, beta))
            for i in tqdm(range(NUM_RUNS)):
                # Get save name
                name = n + "_" + str(i) + "_" + str(beta)
    
                # PDE
                left = ut + alpha*u_squared_x - beta*uxx + gamma*uxxx
                pde_tokens = []
                val_fn = StringIO(str(left)).readline
                for t in tokenize.generate_tokens(val_fn):
                    if(t.type not in [0, 4]):
                        pde_tokens.append(t.string)
    
                # Forcing term
                A, omega, l, phi, t = symbols("A_j omega_j l_j phi_j t")
                dlta = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                delta_tokens = []
                val_fn = StringIO(str(dlta)).readline
                for t in tokenize.generate_tokens(val_fn):
                    if(t.type not in [0, 4]):
                        delta_tokens.append(t.string)
    
                # Initial condition
                t = 0.
                ic = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                ic_tokens = []
                val_fn = StringIO(str(ic)).readline
                for t in tokenize.generate_tokens(val_fn):
                    if(t.type not in [0, 4]):
                        ic_tokens.append(t.string)
    
                # Sample variables
                As = torch.tensor(np.random.uniform(-0.5, 0.5, J)).cuda()
                omegas = torch.tensor(np.random.uniform(-0.4, 0.4, J)).cuda()
                ls = torch.tensor(np.random.choice(3, J) + 1).cuda()
                phis = torch.tensor(np.random.uniform(0., 2*np.pi, J)).cuda()
    
                # Create instances of PDE for each (nt, nx)
                pde = {}
                all_tokens = get_all_tokens(pde_tokens, delta_tokens, ic_tokens, left_bc, right_bc, As, omegas, ls, phis)
                encoded_tokens = encode_tokens(all_tokens)
                decoded_tokens = decode_tokens(encoded_tokens)
                exp_tokens.append(encoded_tokens)
    
                # Generate train, test, validation set for each set of parameters
                pde[f'pde_{nt}-{nx}'] = CE(starting_time, end_time, grid_size=(nt, nx), L=L, alpha=alpha, beta=beta, gamma=gamma, device="cuda")
                single_exp_names = []
                exp_name = generate_data_single_equation(h5f, name, pde, [name], 1, 1, alpha, beta, gamma, As, omegas, phis, ls,
                                                         all_tokens, encoded_tokens, 'cuda')
    
    print("Data saved")
    print()
    print()
    h5f.close()


def generate_burgers_data(save_name="all_data"):
    exp_names, exp_tokens, identifiers = [], [], []

    save_name = "./pde_data/" + save_name
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    print("\nETA: {}\n".format(eta))

    all_identifiers = []
    for (n, alph, bet, gamma) in [("Burgers", 0.5, 0.1, 0)]:
        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for beta in [0.01, 0.05, 0.1, 0.2, 0.5, 1.]:
                print("\nALPHA: {}\tBETA: {}\n".format(alpha, beta))
                for i in tqdm(range(NUM_RUNS)):
                    name = n + "_" + str(i) + "_" + str(alpha) + "_" + str(beta)
    
                    # PDE
                    left = ut + alpha*u_squared_x - beta*uxx + gamma*uxxx
                    pde_tokens = []
                    val_fn = StringIO(str(left)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            pde_tokens.append(t.string)
    
                    # Forcing term
                    A, omega, l, phi, t = symbols("A_j omega_j l_j phi_j t")
                    dlta = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                    delta_tokens = []
                    val_fn = StringIO(str(dlta)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            delta_tokens.append(t.string)
    
                    # Initial condition
                    t = 0.
                    ic = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                    ic_tokens = []
                    val_fn = StringIO(str(ic)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            ic_tokens.append(t.string)
    
                    # Sample variables
                    As = torch.tensor(np.random.uniform(-0.5, 0.5, J))#.cuda()
                    omegas = torch.tensor(np.random.uniform(-0.4, 0.4, J))#.cuda()
                    ls = torch.tensor(np.random.choice(3, J) + 1)#.cuda()
                    phis = torch.tensor(np.random.uniform(0., 2*np.pi, J))#.cuda()
    
                    # Create instances of PDE for each (nt, nx)
                    pde = {}
                    all_tokens = get_all_tokens(pde_tokens, delta_tokens, ic_tokens, left_bc, right_bc, As, omegas, ls, phis)
                    encoded_tokens = encode_tokens(all_tokens)
                    decoded_tokens = decode_tokens(encoded_tokens)
                    exp_tokens.append(encoded_tokens)
    
                    # Generate train, test, validation set for each set of parameters
                    pde[f'pde_{nt}-{nx}'] = CE(starting_time, end_time, grid_size=(nt, nx), L=L, alpha=alpha, beta=beta, gamma=gamma, device="cpu")
                    single_exp_names = []
                    exp_name = generate_data_single_equation(h5f, name, pde, [name], 1, 1, alpha, beta, gamma, As, omegas, phis, ls,
                                                             all_tokens, encoded_tokens, device='cpu')
    
    print("Data saved")
    print()
    print()
    h5f.close()


def generate_kdv_data(save_name="all_data"):
    exp_names, exp_tokens, identifiers = [], [], []

    save_name = "./pde_data/" + save_name
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    print("\nETA: {}\n".format(eta))

    all_identifiers = []
    for (n, alph, beta, gamm) in [("KdV", 6, 0, 1)]:
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            #for gamma in [2,4,6,8,10,12]:
            for gamma in [1,2,3,4,5,6]:
                print("\nALPHA: {}\tGAMMA: {}\n".format(alpha, gamma))
                for i in tqdm(range(NUM_RUNS)):
                    name = n + "_" + str(i) + "_" + str(alpha) + "_" + str(gamma)
    
                    # PDE
                    left = ut + alpha*u_squared_x - beta*uxx + gamma*uxxx
                    pde_tokens = []
                    val_fn = StringIO(str(left)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            pde_tokens.append(t.string)
    
                    # Forcing term
                    A, omega, l, phi, t = symbols("A_j omega_j l_j phi_j t")
                    dlta = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                    delta_tokens = []
                    val_fn = StringIO(str(dlta)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            delta_tokens.append(t.string)
    
                    # Initial condition
                    t = 0.
                    ic = Sum(A*sin(omega*t + (2*pi*l*x)/L + phi), (j, 1, J))
                    ic_tokens = []
                    val_fn = StringIO(str(ic)).readline
                    for t in tokenize.generate_tokens(val_fn):
                        if(t.type not in [0, 4]):
                            ic_tokens.append(t.string)
    
                    # Sample variables
                    As = torch.tensor(np.random.uniform(-0.5, 0.5, J)).cuda()
                    omegas = torch.tensor(np.random.uniform(-0.4, 0.4, J)).cuda()
                    ls = torch.tensor(np.random.choice(3, J) + 1).cuda()
                    phis = torch.tensor(np.random.uniform(0., 2*np.pi, J)).cuda()
    
                    # Create instances of PDE for each (nt, nx)
                    pde = {}
                    all_tokens = get_all_tokens(pde_tokens, delta_tokens, ic_tokens, left_bc, right_bc, As, omegas, ls, phis)
                    encoded_tokens = encode_tokens(all_tokens)
                    decoded_tokens = decode_tokens(encoded_tokens)
                    exp_tokens.append(encoded_tokens)
    
                    # Generate train, test, validation set for each set of parameters
                    pde[f'pde_{nt}-{nx}'] = CE(starting_time, end_time, grid_size=(nt, nx), L=L, alpha=alpha, beta=beta, gamma=gamma, device="cuda")
                    single_exp_names = []
                    exp_name = generate_data_single_equation(h5f, name, pde, [name], 1, 1, alpha, beta, gamma, As, omegas, phis, ls,
                                                             all_tokens, encoded_tokens, device='cuda')
    
    print("Data saved")
    print()
    print()
    h5f.close()


if __name__ == '__main__':
    #NUM_RUNS = 10000
    #raise
    NUM_RUNS = 250

    # Equation setup
    u = Function('u')
    u = u(t,x)
    
    delta = Function('delta')
    delta = delta(t, x)
    
    ut = u.diff(t)
    ux = u.diff(x)
    u_squared = u**2
    u_squared_x = u_squared.diff(x)
    
    uxx = ux.diff(x)
    uxxx = uxx.diff(x)
    pi = np.pi
    
    # System parameters
    L = 16
    J = 5
    starting_time = 0
    end_time = 4.0
    #eta = 0.2
    
    # Discretization
    nx, nt = 100, 100
    
    # Boundary conditions
    left_bc = "None"
    right_bc = "None"

    generate_heat_data("heat_250")
    generate_burgers_data("burgers_250")
    generate_kdv_data("kdv_250")
