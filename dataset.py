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


class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train"):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split

    def __len__(self):
        #return len(self.data)
        return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

        center_pts = []
        vector = []
        forces = []
        with open(f'{full_path}/Input.txt','r') as f:
            for i, line in enumerate(f):
                if i<4:
                    x, y, z = line.rstrip('\n').split(',')
                    center_pts.append([float(x), float(y), float(z)])
                elif 3 < i < 8:
                    x, y, z = line.rstrip('\n').split(',')
                    vector.append([float(x), float(y), float(z)])
                else:
                    F = line.rstrip('\n')
                    forces.append(float(F))
        
        center_pts = torch.tensor(center_pts)
        vector = torch.tensor(vector)
        forces = torch.tensor(forces)



        FEM_disp = []
        FEM_mises = []
        coords = []

        with open(f'{full_path}/Stress_Box_1.txt','r') as f:
            for i, line in enumerate(f):
                ux, uy, uz, mises, x, y, z = line.rstrip('\n').split(',')
                FEM_disp.append([float(ux), float(uy), float(uz)])
                FEM_mises.append(float(mises))
                coords.append([float(x), float(y), float(z)])

        FEM_disp = torch.tensor(FEM_disp)
        FEM_mises = torch.tensor(FEM_mises)
        coords = torch.tensor(coords)


        
        # translation
        #coords = coords.permute(1,0)
        vec_trans = torch.mean(coords, axis=0)
        coords = coords - vec_trans
        
    

        # rotation
        vec_unit = vector[0]/torch.linalg.norm(vector[0])
        x_axis = torch.tensor([1,0,0]).float()
        dot_prod = torch.dot(vec_unit, x_axis)
        phi = torch.arccos(dot_prod)
        rot_mat = torch.tensor([[math.cos(phi), -math.sin(phi), 0],[math.sin(phi), math.cos(phi), 0], [0,0,1]])

        FEM_disp = FEM_disp @ rot_mat.T
        coords = coords @ rot_mat.T


        # rotation
        # vec_unit = vector[0]/torch.linalg.norm(vector[0])
        # x_axis = torch.tensor([1,0,0]).float()
        # dot_prod = torch.dot(vec_unit, x_axis)
        # phi = torch.arccos(dot_prod)
        # rot_mat = torch.tensor([[math.cos(phi), -math.sin(phi), 0],[math.sin(phi), math.cos(phi), 0], [0,0,1]])

        # center_pts = center_pts @ rot_mat.T
        # vector = vector @ rot_mat.T
        # FEM_disp = FEM_disp @ rot_mat.T
        # coords = coords.permute(1,0)
        # coords = coords @ rot_mat.T
        # coords = coords.permute(1,0)

                
        return center_pts, vector, forces, coords, FEM_disp, FEM_mises


def main():
    dataset = GridDataset()
    center_pts, vector, forces, coords, FEM_disp, FEM_mises = dataset[0]

    coords = np.array(coords)
    FEM_disp = np.array(FEM_disp)*100
    max, min = coords.max().item(), coords.min().item()

    FEM_disp_pts = coords + FEM_disp

    print(FEM_disp_pts.shape)
    

    x, y, z = FEM_disp_pts[:,0],FEM_disp_pts[:,1],FEM_disp_pts[:,2]

    #print(x,y,z)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, s=50, c = FEM_mises, cmap='viridis')

    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()