# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
  Authors:  Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
import time
from tqdm import tqdm
import itertools
import random
import copy
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class TransformerOperatorDataset(Dataset):
    def __init__(self, f, filename,
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 rollout_length=10,
                 train_style='fixed_future',
                 ssl=False, forcing=False, seed=0,
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        #self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        self.return_text = return_text
        self.train_style = train_style
        self.ssl = ssl
        self.forcing = forcing
        
        # Extract list of seeds
        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        if(filename != "all"):
            data_list = []
            for key in f.keys():
                if(filename in key):
                    data_list.append(key)
            np.random.shuffle(data_list)
        else:
            #data_list = list([key for key in f.keys() if("KdV" not in key))
            data_list = [key for key in f.keys()]
            np.random.shuffle(data_list)

        self.data_list = data_list

        # Get target split. Seeding is required to make this reproducible.
        # This splits each run, lets try a better shuffle
        if(num_samples is not None):
            data_list = data_list[:num_samples]
        train_idx = int(len(data_list) * (1 - test_ratio - val_ratio))
        val_idx = int(len(data_list) * (1-test_ratio))
        #print(train_idx, val_idx)
        #raise

        # Make sure no data points occur in two splits
        assert not (bool(set(self.data_list[:train_idx]) & \
                         set(self.data_list[train_idx:val_idx])) | \
                    bool(set(self.data_list[val_idx:]) & \
                         set(self.data_list[train_idx:])) & \
                    bool(set(self.data_list[val_idx:]) & \
                         set(self.data_list[train_idx:val_idx])))

        if(split == "train"):
            #print("TRAINING DATA")
            self.data_list = np.array(data_list[:train_idx])
            #print(self.data_list)
        elif(split == "val"):
            #print("VALIDATION DATA")
            self.data_list = np.array(data_list[train_idx:val_idx])
            #print(self.data_list)
        elif(split == "test"):
            #print("TESTING DATA")
            self.data_list = np.array(data_list[val_idx:])
            #print(self.data_list)
        else:
            raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.data_list)
        #raise
        
        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.rollout_length = rollout_length

        self.WORDS = ['(', ')', '+', '-', '*', '/', 'Derivative', 'Sum', 'j', 'A_j', 'l_j',
                 'omega_j', 'phi_j'    , 'sin', 't', 'u', 'x', 'dirichlet', 'neumann',
                 "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
                 #'E', ',', '.', '&']
                 'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}
        self.num_t = num_t
        self.num_x = num_x
        self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        self.data = []
        self.grid = []
        self.time = []
        self.tokens = []
        self.available_idxs = []
        #print(len(self.data_list))
        #raise
        print("Gathering data...")
        for i in tqdm(range(len(self.data_list))):
            #print(self.data_list[i])
            seed_group = self.h5_file[self.data_list[i]]
            self.data.append(seed_group[self.name][0])

            if(self.train_style == 'next_step'):
                #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:]
                idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:self.sim_time]
            elif(self.train_style == 'arbitrary_step'):
                #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:self.sim_time]
                idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:self.sim_time+self.initial_step]
                #idxs = np.arange(0, len(seed_group[self.name][0]))[:self.sim_time+self.initial_step]
            
            elif(self.train_style == 'rollout'):
                length = len(seed_group[self.name][0])
                idxs = np.arange(0, length)[self.initial_step:length-self.rollout_length]
            elif(self.train_style == 'fixed_future'):
                idxs = np.array([i])
                #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:]
                #idxs = np.arange(0, len(seed_group[self.name][0]))[self.initial_step:self.sim_time]

            if(len(self.available_idxs) != 0 and self.train_style != 'fixed_future'):
                # Needs to make sure it wraps all the way back around...
                #TODO Make sure this is right
                #print(self.available_idxs[-1])
                idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                        self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
						self.available_idxs[-1] + 100 - self.sim_time#self.available_idxs[-1] + 1
            self.available_idxs.extend(idxs)

            self.grid.append(np.array(seed_group[self.name].attrs["x"], dtype='f'))
            if(self.return_text):
                self.tokens.append(list(torch.Tensor(seed_group[self.name].attrs['encoded_tokens'])))
                self.time.append(seed_group[self.name].attrs['t'])

        self.data = torch.Tensor(np.array(self.data)).to(device=device)#, dtype=torch.float).cuda()
        self.grid = torch.Tensor(np.array(self.grid)).to(device=device)#.cuda()
        self.h5_file.close()
        #print(self.available_idxs)
        #raise

        print("\nNUMBER OF SAMPLES: {}".format(len(self.available_idxs)))

        def forcing_term(x, t, As, ls, phis, omegas):
            return np.sum(As[i]*torch.sin(2*np.pi/16. * ls[i]*x + omegas[i]*t + phis[i]) for i in range(len(As)))
        
        # Not suitable for autoregressive training
        if(self.train_style == 'fixed_future'):
            time_included_tokens = []
            #self.all_tokens = torch.empty(len(self.available_idxs), 500).to(device=device)#.cuda()
            #print(self.data.shape)
            self.all_tokens = torch.empty(self.data.shape[0], self.data.shape[1], 500)
            #print(self.available_idxs)
            #raise
            for idx, token in tqdm(enumerate(self.tokens)):
                time_tokens = self._encode_tokens("&" + str(self.time[idx][self.sim_time]))
                while(len(time_tokens) + len(self.tokens[idx]) < 490): # Padding 
                    time_tokens.append(len(self.WORDS))
                time_included_tokens.append(np.append(self.tokens[idx], time_tokens))
            self.time_included_tokens = torch.Tensor(np.array(time_included_tokens)).to(device=device)#.cuda()#.int()
            self.all_tokens = torch.empty(len(self.available_idxs), 500).to(device=device)#.cuda()

            for idx, sim_idx in tqdm(enumerate(self.available_idxs)):
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                # I can precompute all of this... which would increase memory but decrease compute time
                #slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][sim_time]))
                slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][self.sim_time]))
                #print(slice_tokens)
                return_tokens = torch.Tensor(self.tokens[sim_num].copy())

	        # TODO: Maybe put this back
                return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens)))
                return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(490 - len(return_tokens)))))
                
                # Add commas to l values
                split_tokens = list(np.argwhere(return_tokens == 35)[0])
                insert_tokens = return_tokens[split_tokens[6]:split_tokens[7]+1]
                insert_tokens = torch.Tensor((insert_tokens[0],
                                             insert_tokens[1], torch.tensor(33),
                                             insert_tokens[2], torch.tensor(33),
                                             insert_tokens[3], torch.tensor(33),
                                             insert_tokens[4], torch.tensor(33),
                                             insert_tokens[5], torch.tensor(33))
                )    
                return_tokens = torch.cat((return_tokens[:split_tokens[6]], insert_tokens,
                                           return_tokens[split_tokens[7]:]))

                return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens)))))
                self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()

        elif(self.train_style in ['next_step', 'arbitrary_step'] and self.return_text):
            # Create array of all legal encodings, pdes, and data
            self.all_tokens = torch.empty(len(self.available_idxs), 500).to(device=device)#.cuda()

            if(self.forcing):
                self.forcing_terms = []
                self.times = torch.empty(len(self.available_idxs))

            print("Processing data...")
            #print(self.available_idxs)
            #raise
            for idx, sim_idx in tqdm(enumerate(self.available_idxs)):
                #sim_idx = self.available_idxs[idx]      # Get valid prestored index
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
                if(self.return_text):

                    slice_tokens = self._encode_tokens("&" + str(self.time[sim_num][sim_time]))
                    return_tokens = torch.Tensor(self.tokens[sim_num].copy()).cpu()

                    # TODO: Maybe put this back
                    return_tokens = torch.cat((return_tokens, torch.Tensor(slice_tokens).cpu())).cpu()
                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(490 - len(return_tokens))).cpu()))
                    
                    # Add commas to l values
                    split_tokens = list(np.argwhere(return_tokens.cpu() == 35)[0])
                    insert_tokens = return_tokens[split_tokens[6]:split_tokens[7]+1].cpu()
                    insert_tokens = torch.Tensor((insert_tokens[0],
                                                 insert_tokens[1], torch.tensor(33),
                                                 insert_tokens[2], torch.tensor(33),
                                                 insert_tokens[3], torch.tensor(33),
                                                 insert_tokens[4], torch.tensor(33),
                                                 insert_tokens[5], torch.tensor(33))
                    ).cpu()
                    return_tokens = torch.cat((return_tokens[:split_tokens[6]].cpu(), insert_tokens,
                                               return_tokens[split_tokens[7]:].cpu()))#.cuda()

                    # Recreate forcing term and save as a lambda function
                    if(self.forcing):
                        split_tokens = list(np.argwhere(return_tokens == 35)[0])
                        As = return_tokens[split_tokens[4]:split_tokens[5]][1:]
                        omegas = return_tokens[split_tokens[5]:split_tokens[6]][1:]
                        ls = return_tokens[split_tokens[6]:split_tokens[7]][1:]
                        phis = return_tokens[split_tokens[7]:split_tokens[8]][1:]

                        # Split by commas
                        A_splits = torch.cat((torch.tensor([0]), torch.argwhere(As == 33.)[:,0]))
                        A_vals = [As[s+1:s+16] if(s != 0) else As[s:s+15] for s in A_splits][:-1]

                        omega_splits = torch.cat((torch.tensor([0]), torch.argwhere(omegas == 33.)[:,0]))
                        omega_vals = [omegas[s+1:s+16] if(s != 0) else omegas[s:s+15] for s in omega_splits][:-1]

                        l_splits = torch.cat((torch.tensor([0]), torch.argwhere(ls == 33.)[:,0]))[:-1]
                        l_vals = [ls[s+1] if(s != 0) else ls[s] for s in l_splits]

                        phi_splits = torch.cat((torch.tensor([0]), torch.argwhere(phis == 33.)[:,0]))
                        phi_vals = [phis[s+1:s+16] if(s != 0) else phis[s:s+15] for s in phi_splits][:-1]

                        # Convert each one to a float
                        A_num = []
                        for A in A_vals:
                            if(A[-1] == 33):
                                A_num.append(float(''.join([self.id2word[int(w)] for w in A[:-1] if(w < len(self.WORDS))])))
                            else:
                                try:
                                    A_num.append(float(''.join([self.id2word[int(w)] for w in A if(w < len(self.WORDS))])))
                                except ValueError: # Catches like one case
                                    print("FOUND AN ERROR")
                                    A_num.append(float(''.join([self.id2word[int(w)] for w in A[:-2] if(w < len(self.WORDS))])))
                        omega_num = []
                        for omega in omega_vals:
                            if(omega[-1] == 33):
                                omega_num.append(float(''.join([self.id2word[int(w)] for w in omega[:-1] if(w < len(self.WORDS))])))
                            else:
                                omega_num.append(float(''.join([self.id2word[int(w)] for w in omega if(w < len(self.WORDS))])))
                        l_num = []
                        for l in l_vals:
                            l_num.append(float(''.join([self.id2word[int(w)] for w in [l] if(w < len(self.WORDS))])))
                        phi_num = []
                        for phi in phi_vals:
                            if(phi[-1] == 33):
                                phi_num.append(float(''.join([self.id2word[int(w)] for w in phi[:-1] if(w < len(self.WORDS))])))
                            else:
                                phi_num.append(float(''.join([self.id2word[int(w)] for w in phi if(w < len(self.WORDS))])))
                 
                        #def forcing_term(x, t, As, ls, phis, omegas):
                        ft = lambda x, t: forcing_term(x, t, A_num, l_num, phi_num, omega_num)
                        #print(ft)

                        self.forcing_terms.append(ft)
                        self.times[idx] = float(''.join([self.id2word[int(w)] for w in slice_tokens[1:] if(w < len(self.WORDS))]))

                    return_tokens = torch.cat((return_tokens, torch.Tensor([len(self.WORDS)]*(500 - len(return_tokens))).cpu()))
                    self.all_tokens[idx] = return_tokens.to(device=device)#.cuda()
                    #print(self.all_tokens[idx])


        if(self.return_text):
            self.all_tokens = self.all_tokens.to(device=device)#.cuda()
        self.time = torch.Tensor(self.time).to(device=device)
        self.data = self.data.cuda()
        self.grid = self.grid.cuda()

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style in ['next_step', 'arbitrary_step']):
            return len(self.available_idxs)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)
    
    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        #print("\n\nHERE\n\n")
        #print("\nHERE\n")
        # Everything is precomputed
        if(self.train_style == 'fixed_future'):
            if(self.return_text):
                return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time][...,np.newaxis], \
                       self.grid[idx], \
                       self.all_tokens[idx].to(device=device), \
                       self.time[idx][self.sim_time]
            else:
                return self.data[idx][...,:self.initial_step,:], \
                       self.data[idx][self.sim_time], \
                       self.grid[udx][self.sim_time]

        # Need to slice according to available data
        elif(self.train_style == 'next_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                        self.data[sim_num][sim_time][...,np.newaxis], \
                        self.grid[sim_num], \
                        self.all_tokens[idx].to(device=device), \
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                if(sim_time == 0):
                    raise ValueError("WHOOPSIE")
                return self.data[sim_num][sim_time - self.initial_step:sim_time], \
                               self.data[sim_num][sim_time][np.newaxis], \
                               self.grid[sim_num][np.newaxis]

        elif(self.train_style == 'arbitrary_step'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            #sim_idx = idx      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

            if(self.return_text):
                if(self.forcing):
                    return self.data[sim_num][0], \
                           self.data[sim_num][sim_time][...,np.newaxis], \
                           self.grid[sim_num], \
                           self.all_tokens[idx].to(device=device), \
                           self.forcing_terms[idx], \
                           self.times[idx]
                else:
                    if(self.ssl):
                        return self.data[sim_num][0], \
                           self.data[sim_num][sim_time][...,np.newaxis], \
                           self.grid[sim_num], \
                           self.all_tokens[idx].to(device=device), \
                           self.time[sim_num][sim_time], \
                           self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis]
                    else:
                        return self.data[sim_num][0], \
                           self.data[sim_num][sim_time][...,np.newaxis], \
                           self.grid[sim_num], \
                           self.all_tokens[idx].to(device=device), \
                           self.time[sim_num][sim_time]# - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                       self.data[sim_num][sim_time][...,np.newaxis], \
                       self.grid[sim_num][...,np.newaxis]

        # Need to slice according ot available data and rollout
        elif(self.train_style == 'rollout'):
            sim_idx = self.available_idxs[idx]      # Get valid prestored index
            sim_num = sim_idx // self.data.shape[1] # Get simulation number
            sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
            if(self.return_text):
                # Add additional times to text encoding.
                slice_times = self.time[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length] # Get times
                #print(sim_time, sim_time - self.initial_step, sim_time + self.rollout_length, self.initial_step, self.rollout_length)
                slice_tokens = torch.empty((len(slice_times), 15))
                for idx, st in enumerate(slice_times):
                    # Loses a very small amount of precision
                    # Need predefined tensor
                    slce = self._encode_tokens("&" + str(st))
                    if(len(slce) < 15):
                        slce.extend([20.]*(15-len(slce)))
                    slice_tokens[idx] = torch.Tensor(slce)[:15].to(device=device)#.cuda()

                # This goes into ssl training loop.
                return_tokens = self.tokens[sim_num].copy()
                return_tokens.extend([len(self.WORDS)]*(500 - len(return_tokens)))
                return_tokens = torch.Tensor(return_tokens)
                return_tokens = return_tokens.repeat(self.rollout_length, 1)
                slice_tokens = torch.swapaxes(slice_tokens.unfold(0, 10, 1)[:-1], 1, 2).reshape(self.rollout_length, -1)
                all_tokens = torch.cat((return_tokens, slice_tokens), dim=1)

                # Most processing happens in the training loop
                return self.data[sim_num][sim_time-self.initial_step:sim_time+self.rollout_length,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length][...,np.newaxis], \
                       self.grid[sim_num][...,np.newaxis], \
                       all_tokens
                       #return_tokens, slice_tokens
            else:
                return self.data[sim_num][sim_time-self.initial_step:sim_time,...][...,np.newaxis], \
                       self.data[sim_num][sim_time:sim_time+self.rollout_length], \
                       self.grid[sim_num][...,np.newaxis]


class TransformerOperatorDataset2D(Dataset):
    def __init__(self, f, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 samples_per_equation=111,
                 seed=0
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        self.samples_per_equation = samples_per_equation
        
        # Extract list of seeds
        self.data_list = list(f.keys())

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.WORDS = ['(', ')', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
                      'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10^',
                      'E', 'e', ',', '.', '&']
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x
        #self.name = "pde_{}-{}".format(self.num_t, self.num_x)

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        sample_num = 0
        # Get all indices
        idxs = []
        #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?
        for i in range(len(self.data_list)):
            seed_group = self.h5_file[self.data_list[i]]
            samples_per_sim = seed_group['u'].shape[0]
            for j in range(seed_group['u'].shape[0]):
                idxs.append(i*seed_group['u'].shape[0] + j)
        #print(self.data_list)
        idxs = [i for i in range(len(self.data_list))] #TODO 

        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        np.random.shuffle(idxs)
        self.idxs = idxs[:num_samples]

        # Split indices into
        #print(idxs)
        #raise
        if(self.split_style == 'equation'):
            train_idx = int(num_samples * (1 - test_ratio - val_ratio))
            val_idx = int(num_samples * (1-test_ratio))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:num_samples]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.idx)


        self.data = []
        self.grid = []
        self.time = []
        self.w0 = []
        self.temp_tokens = []
        self.available_idxs = []
        self.data_list = np.array(self.data_list)[self.idxs]
        #self.data_list = np.array([self.data_list[0]])
        #print(self.data_list)
        #self.data = torch.empty((40,200,64,64,201)).float()
        #self.data = torch.empty((len(self.data_list),200,64,64,201)).float()
        #print(vars(self.h5_file))
        #print(self.h5_file.filename)
        if('10s' in self.h5_file.filename):
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,201)).float()
        elif('30s' in self.h5_file.filename):
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,121)).float()
        elif('1s' in self.h5_file.filename):
            #print(len(self.data_list))
            self.data = torch.empty((len(self.data_list),self.samples_per_equation,64,64,201)).float()
            #raise
        for i in tqdm(range(len(self.data_list))):
            seed_group = self.h5_file[self.data_list[i]]
            #data = seed_group['u'][:]
            data = seed_group['u'][:][:,::reduced_resolution,::reduced_resolution,...]
            #print(data.shape)
            #raise

            # Get extra info
            base_tokens = seed_group['tokens'][:]
            x = seed_group['X'][:][::reduced_resolution,::reduced_resolution,np.newaxis]
            y = seed_group['Y'][:][::reduced_resolution,::reduced_resolution,np.newaxis]
            w0 = seed_group['a'][:][...,::reduced_resolution,::reduced_resolution,np.newaxis]

            # Add initial condition
            complete_data = np.concatenate((w0, data), axis=3)
            #self.data.append(torch.Tensor(complete_data).clone())
            #complete_data = complete_data[...,:101]
            #print(complete_data.shape)
            #raise
            self.data[i] = torch.Tensor(complete_data[:self.samples_per_equation])
            #print(complete_data.shape)

            # Add initial time
            time = list(seed_group['t'][:])
            time.insert(0, 0.0)
            self.time.append(time)

            # Get grid
            self.grid.append(np.dstack((x,y)))

            # Get tokens
            #print(base_tokens)
            # Correct for small issue.
            base_tokens[38] = 12
            #base_tokens = torch.cat((base_tokens[:33], torch.Tensor([16]), base_tokens[33:]))
            base_tokens = np.insert(base_tokens, 34, 16)
            #base_tokens[8] = 11
            #base_tokens[17] = 11
            #base_tokens[35] = 11
            #print(base_tokens)
            #raise
            #raise
            self.temp_tokens.append(base_tokens)
            #print("\nGOT SAMPLE {}\n".format(i))
            del complete_data

        # Arrange data
        print("ARRANGING DATA")
        self.data = torch.swapaxes(self.data, 2, 4)
        self.data = torch.swapaxes(self.data, 3, 4)
        print("\n\nDATA SHAPE:")
        print(self.data.shape)
        #print()
        #print()
        #raise

        # Get valid indices for returning data
        #print("Getting available idxs...")
        self.available_idxs = []
        #print(len(self.data_list))
        #raise
        if(self.train_style in ['next_step', 'arbitrary_step']):
            for i in tqdm(range(len(self.data_list))):
                if(self.train_style == 'next_step'):
                    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
                    if(self.split_style == 'equation'):
                        for j in range(1, self.samples_per_equation):
                            idxs = np.append(idxs, np.arange(0, self.data.shape[2])[self.initial_step:] + idxs[-1]+1)
                elif(self.train_style == 'arbitrary_step'):
                    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
                
                # Take into account that the first self.initial_step samples can't be used as target
                if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
                    idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
                            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
	    					self.available_idxs[-1] + 1
                self.available_idxs.extend(idxs)

        elif(self.train_style == 'fixed_future'): # Only need to keep track of total number of valid samples
            idxs = np.arange(0, self.data.shape[0]*self.data.shape[1])
            self.available_idxs = idxs

        # Flatten data to combine simulations
        self.data = self.data.flatten(start_dim=0, end_dim=1)

        # Grid to tensor
        self.grid = torch.Tensor(np.array(self.grid))

        # Add tokenized time to each equation for each simulation
        #print("Getting tokens...")
        self.tokens = []
        self.tokens = torch.empty(len(self.time), self.data.shape[1], 100)
        for idx, token in enumerate(self.temp_tokens):
            for jdx, time in enumerate(self.time[idx]):

                # Tokenize time
                slice_tokens = self._encode_tokens("&" + str(time))

                # Add tokenized time to equation
                full_tokens = copy.copy(list(token))
                full_tokens.extend(list(slice_tokens))

                # Pad tokens to all have same length
                full_tokens.extend([len(self.WORDS)]*(100 - len(full_tokens)))

                # Hold on to tokens
                self.tokens[idx][jdx] = torch.Tensor(full_tokens)

        # Time and tokens to tensors
        self.time = torch.Tensor(np.array(self.time))
        self.tokens = torch.Tensor(self.tokens)

        if(self.split_style == 'initial_condition'):
            #train_idx = int(len(self.available_idxs) * (1 - test_ratio - val_ratio))
            train_idx = int(self.data.shape[0] * (1 - test_ratio - val_ratio))
            val_idx = int(self.data.shape[0] * (1-test_ratio))
            #self.idxs = [i for i in range(len(self.available_idxs))]
            self.idxs = [i for i in range(self.data.shape[0])]
            np.random.shuffle(self.idxs)
            #print(train_idx, val_idx, num_samples, len(self.idxs))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
            self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}

        self.h5_file.close()
        print("DATA SHAPE: {}".format(self.data.shape))
        print("NUM AVAILABLE IDXS: {}".format(len(self.available_idxs)))
        print("NUM IDXS: {}".format(len(self.idxs)))
        print("{} good samples.".format(len(self.data)))
        print(self.split_style)
        print(self.train_style)

        # Create data tuples
        self.data_tuples = []
        dt = self.time[0][1] - self.time[0][0] # TODO Assumes single timestep
        if(self.split_style == 'initial_condition'):
            if(self.train_style == 'next_step'):
                #for idx in range(len(self.idxs)):
                for idx in self.idxs:
                    #print(self.idxs)
                    #print(self.data.shape)
                    #raise
                    for jdx in range(self.initial_step, self.data.shape[1]):
                    #for jdx in range(self.initial_step, 101):
                        #idx = self.idx_to_avail_map[self.idxs[idx]]

                        #print(self.data.shape)
                        sim_idx = self.available_idxs[idx]
                        sim_num = sim_idx // self.data.shape[1] # Get simulation number
                        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                        self.data_tuples.append((self.data[idx][jdx-self.initial_step:jdx],
                                self.data[idx][jdx][...,np.newaxis],
                                self.grid[idx//self.samples_per_equation],
                                self.tokens[idx//self.samples_per_equation][jdx], dt))
                                #self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
                        #self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                        #        self.data[sim_num][sim_time][...,np.newaxis],
                        #        self.grid[sim_num],
                        #        self.tokens[sim_num][sim_time],
                        #        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            elif(self.train_style == 'fixed_future'):
                #for idx in tqdm(range(self.data.shape[0])):
                for idx in tqdm(self.idxs):
                    sim_num = idx // self.data.shape[1] # Get simulation number
                    sim_time = idx % self.data.shape[1] # Get time from
                    #print(idx, sim_num, sim_time, self.data.shape)
                    self.data_tuples.append((
                        self.data[idx][:self.initial_step],
                        self.data[idx][self.sim_time].unsqueeze(-1),
                        self.grid[sim_num],
                        self.tokens[sim_num][self.sim_time],
                        self.time[sim_num][self.sim_time] - \
                                self.time[sim_num][self.sim_time-1]
                    ))

            del self.data
            del self.tokens
            del self.grid
            del self.time
            gc.collect()
            print("TOTAL SAMPLES: {}".format(len(self.data_tuples)))
            print("Done.")

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            if(self.split_style == 'equation'):
                print(len(self.available_idxs))
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.available_idxs)
            else:
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''
        if(self.split_style == 'initial_condition'):
            return self.data_tuples[idx]
            idx = self.idx_to_avail_map[self.idxs[idx]]

        sim_idx = self.available_idxs[idx]
        sim_num = sim_idx // self.data.shape[1] # Get simulation number
        sim_time = sim_idx % self.data.shape[1] # Get time from that simulation
        if(self.train_style == "next_step"):
            if(self.return_text):
                #print(sim_idx, sim_num, sim_time)
                return self.data[sim_num][sim_time-self.initial_step:sim_time], \
                        self.data[sim_num][sim_time][...,np.newaxis], \
                        self.grid[sim_num//2], \
                        self.tokens[sim_num//2][sim_time], \
                        self.time[sim_num//2][sim_time] - self.time[sim_num//2][sim_time-1]#, \
            else:
                return self.data[idx][...,:self.initial_step,:], \
                       self.data[idx][self.sim_time], \
                       self.grid[udx][self.sim_time]

        elif(self.train_style == 'fixed_future'):
            #print(self.time[0][:self.initial_step], self.time[0][self.sim_time])
            #raise
            if(self.return_text):
                return self.data[sim_num][:self.initial_step], \
                        self.data[sim_num][self.sim_time][...,np.newaxis], \
                        self.grid[sim_num], \
                        self.tokens[sim_num][self.sim_time], \
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]#, \
            else:
                return self.data[idx][:self.initial_step], \
                       self.data[idx][self.sim_time], \
                       self.grid[idx][self.sim_time]


class ElectricTransformerOperatorDataset2D(Dataset):
    def __init__(self, f, 
                 initial_step=10,
                 saved_folder='./data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 num_t=200,
                 num_x=200,
                 sim_time=-1,
                 split="train",
                 #test_ratio=0.2,
                 #val_ratio=0.2,
                 test_ratio=0.2,
                 val_ratio=0.2,
                 num_samples=None,
                 return_text=False,
                 train_style='fixed_future',
                 rollout_length=10,
                 split_style='equation',
                 seed=0
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(f.filename)
        self.return_text = return_text
        self.train_style = train_style
        self.rollout_length = rollout_length
        self.split_style = split_style
        
        # Extract list of seeds
        self.data_list = list(f.keys())

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.WORDS = ['(', ')', '[', ']', '+', '-', '*', '/', '=', 'Derivative', 'sin', 'cos', 't', 'u', 'x', 'w', 'y',
                 'pi', 'Delta', 'nabla', 'dot', "None", '0', '1', '2', '3', '4', '5', '6', '7', '8',
                 '9', '10^', 'E', 'e', ',', '.', '&', "Dirichlet", "Neumann"]
        self.word2id = {w: i for i, w in enumerate(self.WORDS)}
        self.id2word = {i: w for i, w in enumerate(self.WORDS)}

        self.num_t = num_t
        self.num_x = num_x

        self.h5_file = h5py.File(self.file_path, 'r')
        self.sim_time = sim_time

        sample_num = 0
        # Get all indices
        #TODO this shuffles by EQUATION, need to shuffle by SIMULATION?

        # Split indices into
        #print(idxs)
        #raise
        #TODO If using single BC combination, select at random for each run?
        print("\nSEED: {}".format(seed))
        np.random.seed(seed)
        if(self.split_style == 'equation'):
            np.random.shuffle(self.data_list)
            train_idx = int(num_samples * (1 - test_ratio - val_ratio))
            val_idx = int(num_samples * (1-test_ratio))
            if(split == "train"):
                self.data_list = self.data_list[:train_idx]
            elif(split == "val"):
                self.data_list = self.data_list[train_idx:val_idx]
            elif(split == "test"):
                self.data_list = self.data_list[val_idx:num_samples]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
        #print(self.data_list)
        #idxs = []
        #for i in tqdm(range(len(self.data_list))):
        #    seed_group = self.h5_file[self.data_list[i]]
        #    samples_per_sim = seed_group['b'].shape[0]
        #    for j in range(seed_group['b'].shape[0]):
        #        idxs.append(i*seed_group['b'].shape[0] + j)

        idxs = [i for i in range(len(self.data_list))] #TODO 
        np.random.shuffle(idxs)
        self.idxs = idxs[:num_samples]

        self.data = []
        self.grid = []
        self.time = []
        self.b = []
        self.temp_tokens = []
        self.available_idxs = []
        self.data_list = np.array(self.data_list)[self.idxs]
        #print(self.data_list)
        for i in tqdm(range(len(self.data_list))):
            seed_group = self.h5_file[self.data_list[i]]
            data = seed_group['v'][:]

            # Get extra info
            base_tokens = seed_group['tokens'][:]
            x = seed_group['X'][:][...,np.newaxis]
            y = seed_group['Y'][:][...,np.newaxis]
            b = seed_group['b'][:][...,np.newaxis]  #TODO Add this + time of 0 to data/tokens.

            # Add initial condition
            #complete_data = np.concatenate((w0, data), axis=3)
            self.data.append(data)
            self.b.append(b)
            #print(complete_data.shape)

            # Get grid
            self.grid.append(np.dstack((x,y)))

            # Get tokens
            self.temp_tokens.append(base_tokens)
            #print("\nGOT SAMPLE {}\n".format(i))

        # Arrange data
        self.data = torch.Tensor(np.array(self.data))

        # Get valid indices for returning data
        #print("Getting available idxs...")
        #self.available_idxs = []
        #for i in tqdm(range(len(self.data_list))):
            #if(self.train_style == 'next_step'):
            #    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
            #elif(self.train_style == 'arbitrary_step'):
            #    idxs = np.arange(0, self.data.shape[2])[self.initial_step:]
            
            # Take into account that the first self.initial_step samples can't be used as target
            #if(len(self.available_idxs) != 0): #TODO Make this robust to initial step
            #    idxs += self.available_idxs[-1] + 1 if(self.train_style == 'next_step') else \
            #            self.available_idxs[-1] + 1 + self.rollout_length if(self.train_style == 'rollout') else \
	#					self.available_idxs[-1] + 1
            #self.available_idxs.extend(idxs)


        # Grid to tensor
        self.grid = torch.Tensor(np.array(self.grid))

        # Add tokenized time to each equation for each simulation
        #print("Getting tokens...")
        self.tokens = []
        self.tokens = torch.empty(self.data.shape[0], 100)
        for idx, token in enumerate(self.temp_tokens):

            # Hold on to tokens
            full_tokens = copy.copy(list(token))
            full_tokens.extend([len(self.WORDS)]*(100 - len(full_tokens)))
            self.tokens[idx] = torch.Tensor(full_tokens)

        # Time and tokens to tensors
        #self.time = torch.Tensor(np.array(self.time))
        self.tokens = torch.Tensor(self.tokens)

        if(self.split_style == 'initial_condition'):
            train_idx = int(len(self.available_idxs) * (1 - test_ratio - val_ratio))
            val_idx = int(len(self.available_idxs) * (1-test_ratio))
            self.idxs = [i for i in range(len(self.available_idxs))]
            np.random.shuffle(self.idxs)
            #print(train_idx, val_idx, num_samples, len(self.idxs))
            if(split == "train"):
                self.idxs = self.idxs[:train_idx]
            elif(split == "val"):
                self.idxs = self.idxs[train_idx:val_idx]
            elif(split == "test"):
                self.idxs = self.idxs[val_idx:]
            else:
                raise ValueError("Select train, val, or test split. {} is invalid.".format(split))
            self.idx_to_avail_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}
            self.sample_to_idx_map = {i[0]: i[1] for i in zip(self.idxs, self.available_idxs)}

        self.h5_file.close()
        print("Number of samples: {}".format(len(self.data)))
        print("Done.")

        # Create data tuples?
        self.data_tuples = []
        if(self.split_style == 'initial_condition'):
            for idx in range(len(self.idxs)):
                idx = self.idx_to_avail_map[self.idxs[idx]]

                sim_idx = self.available_idxs[idx]
                sim_num = sim_idx // self.data.shape[1] # Get simulation number
                sim_time = sim_idx % self.data.shape[1] # Get time from that simulation

                self.data_tuples.append((self.data[sim_num][sim_time-self.initial_step:sim_time],
                        self.data[sim_num][sim_time][...,np.newaxis],
                        self.grid[sim_num],
                        self.tokens[sim_num][sim_time],
                        self.time[sim_num][sim_time] - self.time[sim_num][sim_time-1]))
            del self.data
            del self.tokens
            del self.grid
            del self.time
            gc.collect()
            #print(len(self.data_tuples))

    def _encode_tokens(self, all_tokens):
        encoded_tokens = []
        num_concat = 0
        for i in range(len(all_tokens)):
            try: # All the operators, bcs, regular symbols
                encoded_tokens.append(self.word2id[all_tokens[i]])
                if(all_tokens[i] == "&"): # 5 concatenations before we get to lists of sampled values
                    num_concat += 1
            except KeyError: # Numerical values
                if(isinstance(all_tokens[i], str)):
                    for v in all_tokens[i]:
                        print(i, all_tokens[i])
                        try:
                            encoded_tokens.append(self.word2id[v])
                        except KeyError:
                            print(all_tokens)
                            raise
                    if(num_concat >= 5): # We're in a list of sampled parameters
                        encoded_tokens.append(self.word2id[","])
                else:
                    raise KeyError("Unrecognized token: {}".format(all_tokens[i]))
    
        return encoded_tokens

    def __len__(self):
        if(self.train_style == 'fixed_future'):
            return len(self.data_list)
        elif(self.train_style == 'next_step'):
            if(self.split_style == 'equation'):
                return len(self.data)
            else:
                #return len(self.idxs)
                return len(self.data_tuples)
        elif(self.train_style == 'rollout'):
            return len(self.available_idxs)

    def __getitem__(self, idx):
        '''
        idx samples the file.
        Need to figure out a way to sample the snapshots within the file...
        '''

        if(self.return_text):
            return self.b[idx], \
                    self.data[idx][...,np.newaxis], \
                    self.grid[idx], \
                    self.tokens[idx], \
                    1.0
