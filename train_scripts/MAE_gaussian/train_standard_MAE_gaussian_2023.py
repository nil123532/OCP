#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: xinyu li
# Time: 05/21/23 21:58:01
# File: train_standard.py
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader

from schnet_nilesh import SchNetWrap as SchNet
from ocpmodels.datasets import LmdbDataset
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F

import torch.nn as nn
# Set seed
SEED = 2023
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

# Hyperparameters
train_dir = "MAE_gaussian_2023"
lr = 5e-4
energy_coef = 0.01
force_coef = 0.99
train_epochs = 100
num_samples = 50000

np.random.seed(0)
dataset = LmdbDataset({"src": "/storage/Dataset/ocp/henry/train/"}) # Preprossed oc20-200k dataset
idx = np.arange(200000)
np.random.shuffle(idx)
dataset = Subset(dataset, idx[:num_samples])  # We only use 50000 (num_samples) for traning and test as we do not have enough time

# Train, valid, test split with a ration of 80:10:10
train, valid, test = random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
    )

# Get train, valid, test and large test data loader
train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)

# Model
model = SchNet(1,1,2,
    use_pbc=False,
    regress_forces=True,
    otf_graph=False,
    hidden_channels=1024,
    num_filters=256,
    num_interactions=3,
    num_gaussians=200,
    cutoff=6.0,
    readout='add',
    addi_node_targets=1)

# Set optimizer and scheduler
train_writer = SummaryWriter(log_dir=train_dir)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

def combined_loss(y_true, y_pred, var):
    mae_loss = nn.L1Loss()(y_pred, y_true)
  
    gaussian_nll_loss = torch.nn.GaussianNLLLoss(full=False,eps=1e-06,reduction='sum')(y_pred, y_true, var)
    
    # Weighted combination of the two losses
    combined = 0.8 * mae_loss + 0.2 * gaussian_nll_loss
    
    return combined

if torch.cuda.is_available():
    model = model.cuda()
    model = model.float()
    #loss_fn.cuda()

# Evaluation function
def eval(model, device, loader):
    model.eval()
    e_true = []
    e_pred = []
    f_true = []
    f_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #batch = batch.float()
        #with torch.no_grad():
        gout, nout = model(batch)
        energy = gout[:, 0]
        forces = nout[:, :3]
        mask  = batch.fixed == 0
        
        e_true.append(batch.y.view(energy.shape).detach())
        e_pred.append(energy.detach())
        f_true.append((batch.force[mask]).detach())
        f_pred.append((forces[mask]).detach())

    e_true = torch.cat(e_true, dim = 0).cpu().numpy()
    e_pred = torch.cat(e_pred, dim = 0).cpu().numpy()
    f_true = torch.cat(f_true, dim = 0).cpu().numpy()
    f_pred = torch.cat(f_pred, dim = 0).cpu().numpy()
    
    return e_true, e_pred, f_true, f_pred

# Now train and test the model
losses = []
energy_losses = []
force_losses = []
energy_maes = []
force_maes = []
lowest_loss = 1.0e9
lowest_ve = 1.0e9
lowest_vf = 1.0e9
lowest_vmae = 1.0e9
for epoch in tqdm(range(train_epochs)):
    _losses = []
    _energy_losses = []
    _force_losses = []
    _energy_maes = []
    _force_maes = []
    model.train()
    for batch in tqdm(train_loader):
        batch = batch.to("cuda")
        #batch = batch.float()
        optimizer.zero_grad()
        # with autocast("cuda"):
        graph_out, node_out = model(batch)
        #print(graph_out.shape)
        energy = graph_out[:, 0]
        energy_var = graph_out[:, 1]
        energy_var = torch.nn.functional.softplus(energy_var)
        forces = node_out[:, :3]
        #force_var = node_out[:, [1]]
        force_var = node_out[:,[3]]
        force_var = torch.nn.functional.softplus(force_var)

        energy = energy.view(batch.y.size())
        energy_var = energy_var.view(batch.y.size())
        energy_loss = combined_loss(batch.y.float(), energy.float(), energy_var.float())
        mask  = batch.fixed == 0
        natoms = batch.natoms
        natoms = torch.repeat_interleave(natoms, natoms)[mask].view(-1, 1)
        # print("size: ", (batch.force[mask]/natoms).size(), (forces[mask]/natoms).size(), force_var[mask].repeat([1, 3]).size())
        force_loss = combined_loss((batch.force[mask]/natoms).float(), (forces[mask]/natoms).float(), (force_var[mask].repeat([1, 3])).float())
        #force_loss = torch.nn.functional.mse_loss(batch.force[mask]/natoms, forces[mask]/natoms)
        loss = energy_loss * energy_coef + force_loss * force_coef
        loss.backward()
        if float(loss.cpu().detach().numpy()) < lowest_loss:
            torch.save(model, Path(train_dir, 'model_best_MAEg2023.pth'))
            lowest_loss = loss
        optimizer.step()
        
        _losses.append(loss.cpu().detach().numpy())
        _energy_losses.append(energy_loss.cpu().detach().numpy())
        _force_losses.append(force_loss.cpu().detach().numpy())
    train_writer.add_scalar("loss", np.mean(_losses), epoch)
    train_writer.add_scalar("energy_loss", np.mean(_energy_losses), epoch)
    train_writer.add_scalar("force_loss", np.mean(_force_losses), epoch)    
    
    scheduler.step()
    losses.append(np.mean(_losses))
    if epoch % 1 == 0:
        print(f"training epoch: {epoch} loss: {np.mean(_losses)} ")
        ve_true, ve_pred, vf_true, vf_pred = eval(model, "cuda", val_loader)
        ve_mae = mean_absolute_error(ve_true, ve_pred)
        vf_mae = mean_absolute_error(vf_true, vf_pred)
        te_true, te_pred, tf_true, tf_pred = eval(model, "cuda", test_loader)
        te_mae = mean_absolute_error(te_true, te_pred)
        tf_mae = mean_absolute_error(tf_true, tf_pred)
        print(f"epoch: {epoch} val energy mae: {ve_mae} val force mae: {vf_mae} test energy mae: {te_mae} test force mae: {tf_mae} ")
        if ve_mae * energy_coef + vf_mae * force_coef < lowest_vmae:
            lowest_vmae = ve_mae * energy_coef + vf_mae * force_coef
            lowest_te_mae = te_mae
            lowest_tf_mae = tf_mae
#
print(f"Test at the best epoch: energy mae {lowest_te_mae} force mae: {lowest_tf_mae}")
print(f"Test at the last epoch: energy mae {te_mae} force mae: {tf_mae}")


        
            
