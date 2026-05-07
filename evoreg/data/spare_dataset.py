import glob, random
from torch.utils.data import DataLoader, Dataset
import os 
from evoreg.data.utils import random_translation,random_scaling,partial_visibility_mask
import numpy as np 
import trimesh
import torch 

class SpareDataset(Dataset):
    """
    Dataset for Spare registration pairs.
    """
    
    def __init__(
        self,
        Spare_Dataset_Path: str,
        Indices: int,
        Train: bool = True
        

    ):
        """
        Instantiates the Faust Dataset from a folder path
        
        Args:
            Spare_Dataset_Path: SPARE_DATA/release_data/SPARE_Data_plys - the path where all the .plys are located
            
        """
        files = sorted(os.listdir(Spare_Dataset_Path))[:175]
        if Train:
            files = files[:Indices]
        else:
            files = files[Indices:]
        
        self.pairs = [(f"{Spare_Dataset_Path}/{indice}/source.ply",f"{Spare_Dataset_Path}/{indice}/target.ply") for indice in files]
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        source_path = self.pairs[idx][0]
        #print(source_path)
        target_path = self.pairs[idx][1]
        source = trimesh.load_mesh(source_path, process=False)
        target = trimesh.load_mesh(target_path, process=False)
        source_vertices = torch.from_numpy(np.asarray(source.vertices)).float()
        source_faces = torch.from_numpy(np.asarray(source.faces)).float()
        target_vertices = torch.from_numpy(np.asarray(target.vertices)).float()
        target_faces = torch.from_numpy(np.asarray(target.faces)).float()
        return {
            'source': source_vertices,
            'target': target_vertices,
            'source_faces':source_faces,
            'target_faces':target_faces,
            #'source_mesh':source,
            #'target_mesh':target
            
        }
