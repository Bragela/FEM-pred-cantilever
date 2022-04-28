from tkinter import Y
from matplotlib.pyplot import axis
from sympy import Float
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import math
import matplotlib.pylab as plt
from sklearn import preprocessing



class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train", force_scaler=None, coords_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split
        self.force_scaler = force_scaler
        self.coords_scaler = coords_scaler

    def __len__(self):
        return len(self.data)
        #return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

        forces = []
        with open(f'{full_path}/Input.txt','r') as f:
            
            for i, line in enumerate(f):
                F = line.rstrip('\n')
                forces.append(float(F))

        if self.force_scaler != None:
            forces = self.force_scaler.transform(np.array(forces).reshape(-1,1))
            forces = torch.from_numpy(forces).float().squeeze(0)
        else:
            forces = torch.tensor(forces)

        FEM_stress = []
        FEM_disp = []
        coords = []

        with open(f'{full_path}/Stress_Box_1.txt','r') as f:
            for i, line in enumerate(f):
                ux, uy, uz, stress, x, y, z = line.rstrip('\n').split(',')
                FEM_stress.append(float(stress))
                coords.append([float(x), float(y), float(z)])
                FEM_disp.append([float(ux), float(uy), float(uz)])

        coords_original = torch.tensor(coords)

        if self.coords_scaler != None:
            coords = self.coords_scaler.transform(np.array(coords).reshape(-1,3))
            coords = torch.from_numpy(coords).float().squeeze(0)
        else:
            coords = torch.tensor(coords)

        FEM_stress = torch.tensor(FEM_stress)
        FEM_disp = torch.tensor(FEM_disp)
    
        return forces, coords, coords_original, FEM_stress, FEM_disp

def main():

    dataset = GridDataset(split='validation')
    forces, coords, coords_original, FEM_stress, FEM_disp = dataset[6]

    vec_trans = torch.mean(coords_original, axis=0)
    coords_original = coords_original - vec_trans
    rot_mat = torch.tensor([[math.cos(math.pi/2), -math.sin(math.pi/2),0], [math.sin(math.pi/2), math.cos(math.pi/2),0],[0, 0,1]])
    coords_original = coords_original @ rot_mat.T

    coords_original = coords_original.cpu().squeeze().detach().numpy()
    FEM_disp = FEM_disp.cpu().squeeze().detach().numpy()*50
    max, min = coords.max().item(), coords.min().item()

    FEM_x = coords_original[:,0] + FEM_disp[:,0]
    FEM_y = coords_original[:,1] + FEM_disp[:,1]
    FEM_z = coords_original[:,2] + FEM_disp[:,2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(FEM_x,FEM_y,FEM_z, s=50, c = FEM_stress, cmap='viridis')

    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()