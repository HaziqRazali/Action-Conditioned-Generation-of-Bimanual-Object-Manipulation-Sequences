import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import time
import sys
sys.path.append("..")

from models.components import *
from models.utils import *

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                                
        # pose embedder
        self.pose_embedder = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
                       
        # pose rnn
        self.pose_encoder = nn.GRU(self.pose_rnn_encoder_units[0], self.pose_rnn_encoder_units[1], self.pose_rnn_encoder_units[2], batch_first=False)
        self.pose_encoder_hi = nn.ModuleList([make_mlp(self.pose_rnn_encoder_units[3:], self.pose_rnn_encoder_activations) for _ in range(self.pose_rnn_encoder_units[2])])
        
        # classifier
        if "kit_mocap" in self.data_loader:
            all_actions = self.main_actions + [x for x in self.fine_actions if x not in self.main_actions]
        elif "kit_rgbd" in self.data_loader:
            all_actions = self.fine_actions
        else:
            print("Unknown self.data_loader:", self.data_loader)
            sys.exit()
        self.lclf = nn.Linear(self.pose_rnn_encoder_units[1],len(all_actions))
        self.rclf = nn.Linear(self.pose_rnn_encoder_units[1],len(all_actions))
        
    def forward(self, data, mode):
                                        
        rhand_xyz = data["xyz"]                                                             # [batch, pose_padded_length, num_joints, 3]
        rhand_xyz = torch.reshape(rhand_xyz,[rhand_xyz.shape[0], rhand_xyz.shape[1], -1])   # [batch, pose_padded_length, num_joints* 3]
        
        rhand_xyz = self.pose_embedder(rhand_xyz)       # [batch, pose_padded_length, pose_embedder]
        rhand_xyz = torch.permute(rhand_xyz,[1,0,2])    # [pose_padded_length, batch, pose_embedder]
        
        h = None
        h_list = []
        for i in range(len(rhand_xyz)):
        
            rhand_xyz_i = rhand_xyz[i]                      # [batch, pose_embedder]
            rhand_xyz_i = torch.unsqueeze(rhand_xyz_i,0)    # [1, batch, pose_embedder]
        
            # compute h
            h = torch.stack([self.pose_encoder_hi[i](torch.squeeze(rhand_xyz_i)) for i in range(len(self.pose_encoder_hi))]) if h is None else h # [num_layers, batch, rnn_encoder_hidden]
        
            # forward
            _, h = self.pose_encoder(rhand_xyz_i, h) # [num_layers, batch, rnn_encoder_hidden]
            h_list.append(h[-1])
        h_list = torch.stack(h_list) # [pose_padded_length, batch, rnn_encoder_hidden]
        lactions = self.lclf(h_list) # [pose_padded_length, batch, len(all_actions)]
        ractions = self.rclf(h_list) # [pose_padded_length, batch, len(all_actions)]
        lactions = torch.permute(lactions,[1,0,2])
        ractions = torch.permute(ractions,[1,0,2])
        
        # reform output
        reformed_lactions, reformed_ractions = [], []
        for i in range(self.batch_size):
            
            # # # # # # # # # # # # # # # # # # # # # # # #
            # reform                                      #
            # - set values beyond pose_padded_length to 0 #
            # # # # # # # # # # # # # # # # # # # # # # # #
            
            inp_frame = 0
            key_frame = data["obj_xyz_unpadded_length"][i]
            
            reformed_lactions_i = reform_data(data["lhand_action_ohs"][i], lactions[i], inp_frame, key_frame)
            reformed_ractions_i = reform_data(data["rhand_action_ohs"][i], ractions[i], inp_frame, key_frame)
            reformed_lactions.append(reformed_lactions_i)
            reformed_ractions.append(reformed_ractions_i)
        reformed_lactions = torch.stack(reformed_lactions)
        reformed_ractions = torch.stack(reformed_ractions) 
        
        #print(reformed_lactions.shape, reformed_ractions.shape) # [32, 80, 15]
        return_data = {"lhand_action_ids":reformed_lactions, "rhand_action_ids":reformed_ractions}        
        return return_data
