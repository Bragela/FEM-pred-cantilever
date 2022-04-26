from matplotlib import projections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import tqdm
import wandb
from NN import NeuralNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from sklearn import preprocessing


@torch.no_grad()
def no_grad_loop(data_loader, model, png_cnt, epoch=2, device="cuda", batch_size = 64):
    no_grad_loss_disp = 0
    no_grad_loss_mises = 0
    no_grad_loss = 0
    cnt = 0
    for i, (center_pts, vectors, forces, coords, FEM_disp, FEM_mises) in enumerate(data_loader):

        # transfer data to device
        center_pts = center_pts.to(device)
        vectors = vectors.to(device)
        forces = forces.to(device)
        FEM_disp = FEM_disp.to(device)
        FEM_mises = FEM_mises.to(device)     
        coords = coords.to(device)


        with autocast():
            disp_pred, mises_pred = model(center_pts,vectors,forces, coords)        
            loss_disp = F.l1_loss(disp_pred, FEM_disp)
            loss_mises = F.l1_loss(mises_pred, FEM_mises)   
            loss = loss_disp + loss_mises
    
        no_grad_loss_disp += loss_disp
        no_grad_loss_mises += loss_mises
        no_grad_loss += loss
        cnt += 1
        
        if i == len(data_loader) -1:

            case = 0
            fig = plt.figure(figsize=plt.figaspect(0.5))
            coords = coords[case].cpu().squeeze().detach().numpy()
            max, min = coords.max().item(), coords.min().item()
           
            x = coords[0]
            y = coords[1]
            z = coords[2]

            

            # FEM plot
            FEM_disp = FEM_disp[case].cpu().squeeze().detach().numpy()*20
            x_FEM = x + FEM_disp[:,0]
            y_FEM = y + FEM_disp[:,1]
            z_FEM = z + FEM_disp[:,2]

            stress_FEM = FEM_mises[case].cpu().squeeze().detach().numpy()
            max_a = stress_FEM.max().item()
            ax = fig.add_subplot(1,2,1, projection = '3d')
            a = ax.scatter(x_FEM, y_FEM, z_FEM, s = 20, c= stress_FEM, cmap = 'viridis', vmin= 0, vmax=max_a)
            fig.colorbar(a, pad=0.1, shrink=0.5, aspect=10)
            ax.set_title(f'FEM displacement and stresses.\n Max von Mises: {round(max_a,2)}')
            ax.set_xlim(min,max)
            ax.set_ylim(min,max)
            ax.set_zlim(min,max)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # pred plot
            disp_pred = disp_pred[case].cpu().squeeze().detach().numpy()*20
            x_pred = x + disp_pred[:,0]
            y_pred = y + disp_pred[:,1]
            z_pred = z + disp_pred[:,2]
            stress_pred = mises_pred[case].cpu().squeeze().detach().numpy()
            

            ax = fig.add_subplot(1,2,2, projection = '3d')
            max_b = stress_pred.max().item()
            b = ax.scatter(x_pred, y_pred, z_pred, s = 20, c= stress_pred, cmap = 'viridis', vmin= 0, vmax=max_a)
            fig.colorbar(b, pad=0.1, shrink=0.5, aspect=10)
            ax.set_title(f'Predicted displacement and stresses.\n Max von Mises: {round(max_b,2)}')
            ax.set_xlim(min,max)
            ax.set_ylim(min,max)
            ax.set_zlim(min,max)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            fig.tight_layout()
            if png_cnt != 100000000:
                plt.savefig(f'./GIFS/pngs/{png_cnt}.png')

            plt.savefig('train.png')
            wandb.log({"images": wandb.Image("train.png")}, commit=False)
            
            plt.close(fig)
            plt.close('all')
        
        
    return no_grad_loss_disp/cnt, no_grad_loss_mises/cnt, no_grad_loss/cnt


def train(model: NeuralNet, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=1e-3, device="cuda"):
    no_grad_loss_disp, no_grad_loss_mises, no_grad_loss = no_grad_loop(validation_loader, model, png_cnt=0, epoch=0, device="cuda", batch_size=batch_size)
    curr_lr =  learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=4)

    # training loop
    iter = 0
    png_cnt = 1
    training_losses = {
        "train": {},
        "valid": {}
    }
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        loader = tqdm.tqdm(train_loader)
        for center_pts, vectors, forces, coords, FEM_disp, FEM_mises in loader:  

            # transfer data to device
            center_pts = center_pts.to(device)
            vectors = vectors.to(device)
            forces = forces.to(device)
            FEM_disp = FEM_disp.to(device)
            FEM_mises = FEM_mises.to(device)
            coords = coords.to(device)
            
            # Forward pass
            with autocast():
                disp_pred, mises_pred = model(center_pts,vectors,forces, coords)
                loss_disp = F.l1_loss(disp_pred, FEM_disp)
                loss_mises = F.l1_loss(mises_pred, FEM_mises)
                loss = loss_disp + loss_mises
                
            loader.set_postfix(disp = loss_disp.item(), mises = loss_mises.item())
            
            # Backward and optimize
            optimizer.zero_grad()               # clear gradients
            scaler.scale(loss).backward()       # calculate gradients
            scaler.step(optimizer)
            scaler.update()    
            iter += 1  

            training_losses["train"][iter] = loss.item()
        
            if (iter+1) % 100 == 0:   

                # validation loop
                model = model.eval()
                valid_loss_disp, valid_loss_mises, valid_loss = no_grad_loop(validation_loader, model, png_cnt, epoch, device="cuda", batch_size=batch_size)
                png_cnt += 1
                scheduler.step(valid_loss)
                curr_lr =  optimizer.param_groups[0]["lr"]
                wandb.log({"valid loss": valid_loss.item(), "valid loss disp": valid_loss_disp.item(), "valid loss von Mises": valid_loss_mises.item(), "lr": curr_lr}, commit=False)
                model = model.train()
            wandb.log({"train loss": loss.item(), "train loss disp": loss_disp.item(), "train loss von Mises": loss_mises.item()})
    
    # test loop
    test_loss_disp, test_loss_mises, test_loss = no_grad_loop(test_loader, model, png_cnt=100000000, device="cuda", batch_size=batch_size)
    print(f'Testloss: tot={test_loss:.5f}, disp={test_loss_disp:.5f}, mises={test_loss_mises:.5f}')