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
                                
        # only allow even number of hand joints
        assert len(self.hand_xyz_dims) % 2  == 0
        
        # hand encoder
        if self.hand_encoder_type == "torch.mean":
            self.hand_encoder = eval(self.hand_encoder_type)
        elif self.hand_encoder_type == "make_mlp":
            self.hand_encoder = make_mlp(self.hand_encoder_units, self.hand_encoder_activations)
        else:
            print("Unknown hand_encoder_type:",self.hand_encoder_type)
                                
        # object encoder
        self.object_encoder = make_mlp(self.object_encoder_units, self.object_encoder_activations) if self.object_encoder_units is not None else nn.Identity()
        
        # edge convolution
        self.edge_conv = tgnn.EdgeConv(make_mlp(self.pose_encoder_units, self.pose_encoder_activations))
        
        # pose rnn
        self.pose_encoder = nn.GRU(self.pose_rnn_encoder_units[0], self.pose_rnn_encoder_units[1], self.pose_rnn_encoder_units[2], batch_first=False)
        self.pose_encoder_hi = nn.ModuleList([make_mlp(self.pose_rnn_encoder_units[3:], self.pose_rnn_encoder_activations) for _ in range(self.pose_rnn_encoder_units[2])])
                
        # classifier
        #all_actions = self.main_actions + [x for x in self.fine_actions if x not in self.main_actions]
        all_actions = self.clf_actions
        if self.use_bidirectional == 1:
            print("Using bidirectional")
            self.lclf = nn.Linear(self.pose_rnn_encoder_units[1]*2,len(all_actions))
            self.rclf = nn.Linear(self.pose_rnn_encoder_units[1]*2,len(all_actions))
        else:
            print("Not using bidirectional")
            self.lclf = nn.Linear(self.pose_rnn_encoder_units[1],len(all_actions))
            self.rclf = nn.Linear(self.pose_rnn_encoder_units[1],len(all_actions))
            
    def forward(self, data, mode):
    
        if "3d" in self.object_type:
            num_coords = 3
        elif "2d" in self.object_type:
            num_coords = 2
        else:
            print("Unknown self.object_type",self.object_type)
            sys.exit()    
            
        num_obj_wrist_classes = self.num_obj_wrist_classes
        obj_wrist_padded_length = self.obj_wrist_padded_length
                
        # # # # # # # # #
        #               #
        # Prepare Data  #
        #               #
        # # # # # # # # #

        # create the mega adjacency graph then convert it to the edge index list
        # - table not in
        # - 0 are padded objects not connected to anything
        # - hands and objects connected to each other except the zeros
        # - hands are the last 2 indices
        batch_adj = []
        for rhand_obj_ids in data["obj_ids"]:
        
            # form adjacency matrix for the current item in the batch
            adj = 1 - torch.eye(obj_wrist_padded_length) # [obj_wrist_padded_length, obj_wrist_padded_length] should be adj = 1 - torch.eye(obj_wrist_padded_length)
            for i,rhand_obj_id in enumerate(rhand_obj_ids):
                # if rhand_obj_id is zero, detach it from all
                if rhand_obj_id == 0:
                    adj = detach(adj, i)
            batch_adj.append(adj)
        edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())
                
        # embed obj_xyz
        obj_xyz = data["obj_xyz"]                                                                                   # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers, 3]
        obj_xyz = torch.reshape(obj_xyz,[self.batch_size,self.pose_padded_length,obj_wrist_padded_length-2,-1])     # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers* 3]
        obj_data = self.object_encoder(obj_xyz)                                                                     # [batch, pose_padded_length, obj_wrist_padded_length-2, object_encoder]
                
        # embed wrist
        wrist_xyz  = data["wrist_xyz"]                                                                                              # [batch, pose_padded_length,    hand_xyz_dims,   3]
        wrist_xyz  = wrist_xyz.view(self.batch_size, self.pose_padded_length, 2, int(len(self.hand_xyz_dims)/2), num_coords)        # [batch, pose_padded_length, 2, hand_xyz_dims/2, 3]
        wrist_xyz  = torch.reshape(wrist_xyz,(self.batch_size,self.pose_padded_length, 2, int(len(self.hand_xyz_dims)/2*num_coords))) # [batch, pose_padded_length, 2, hand_xyz_dims/2* 3]
        wrist_data = self.hand_encoder(wrist_xyz)                                                                                   # [batch, pose_padded_length, 2, hand_encoder]
                
        # concatenate obj pos and wrist pos
        all_data  = torch.cat((obj_data,wrist_data),dim=2)                          # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder]        
        all_ohs  = torch.cat((data["obj_ohs"],data["wrist_ohs"]),dim=1)             # [batch,                     obj_wrist_padded_length, num_obj_wrist_classes]
        all_ohs  = all_ohs[:,None].repeat(1,self.pose_padded_length,1,1)            # [batch, pose_padded_length, obj_wrist_padded_length, num_obj_wrist_classes]
        all_data = torch.cat((all_data,all_ohs),dim=-1)                             # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]        
        all_data_dim = all_data.shape[-1]
        
        all_data = torch.permute(all_data, [1,0,2,3])                                                                   # [pose_padded_length, batch, obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes]
        all_data = torch.reshape(all_data, [all_data.shape[0], self.batch_size*obj_wrist_padded_length, all_data_dim])  # [pose_padded_length, batch* obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes]
        all_ohs = all_ohs[:,0]                                                          # [batch, obj_wrist_padded_length, num_obj_wrist_classes]
        all_ohs = torch.reshape(all_ohs,[self.batch_size*obj_wrist_padded_length,-1])   # [batch* obj_wrist_padded_length, num_obj_wrist_classes]
        
        if 1:
            # embed reversed_obj_xyz
            reversed_obj_xyz = data["reversed_obj_xyz"]                                                                                   # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers, 3]
            reversed_obj_xyz = torch.reshape(reversed_obj_xyz,[self.batch_size,self.pose_padded_length,obj_wrist_padded_length-2,-1])     # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers* 3]
            reversed_obj_data = self.object_encoder(reversed_obj_xyz)                                                                     # [batch, pose_padded_length, obj_wrist_padded_length-2, object_encoder]
                    
            # embed wrist
            reversed_wrist_xyz  = data["reversed_wrist_xyz"]                                                                                              # [batch, pose_padded_length,    hand_xyz_dims,   3]
            reversed_wrist_xyz  = reversed_wrist_xyz.view(self.batch_size, self.pose_padded_length, 2, int(len(self.hand_xyz_dims)/2), num_coords)        # [batch, pose_padded_length, 2, hand_xyz_dims/2, 3]
            reversed_wrist_xyz  = torch.reshape(reversed_wrist_xyz,(self.batch_size,self.pose_padded_length,2,int(len(self.hand_xyz_dims)/2*num_coords))) # [batch, pose_padded_length, 2, hand_xyz_dims/2* 3]
            reversed_wrist_data = self.hand_encoder(reversed_wrist_xyz)                                                                                   # [batch, pose_padded_length, 2, hand_encoder]
                    
            # concatenate obj pos and wrist pos
            reversed_all_data  = torch.cat((reversed_obj_data,reversed_wrist_data),dim=2)                          # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder]        
            all_ohs  = torch.cat((data["obj_ohs"],data["wrist_ohs"]),dim=1)             # [batch,                     obj_wrist_padded_length, num_obj_wrist_classes]
            all_ohs  = all_ohs[:,None].repeat(1,self.pose_padded_length,1,1)            # [batch, pose_padded_length, obj_wrist_padded_length, num_obj_wrist_classes]
            reversed_all_data = torch.cat((reversed_all_data,all_ohs),dim=-1)                             # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]        
            all_data_dim = reversed_all_data.shape[-1]
            
            reversed_all_data = torch.permute(reversed_all_data, [1,0,2,3])                                                                   # [pose_padded_length, batch, obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes]
            reversed_all_data = torch.reshape(reversed_all_data, [reversed_all_data.shape[0], self.batch_size*obj_wrist_padded_length, all_data_dim])  # [pose_padded_length, batch* obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes]
            all_ohs = all_ohs[:,0]                                                          # [batch, obj_wrist_padded_length, num_obj_wrist_classes]
            all_ohs = torch.reshape(all_ohs,[self.batch_size*obj_wrist_padded_length,-1])   # [batch* obj_wrist_padded_length, num_obj_wrist_classes]
        
        h, reversed_h = None, None
        h_list, reversed_h_list = [], []
        for i in range(len(all_data)):
        
            data_i = all_data[i] # [batch* obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
        
            # edge conv
            data_i = self.edge_conv(data_i,edge_index) # [batch* obj_wrist_padded_length, edge_conv]
            
            # if using context
            if self.context:
                data_i = torch.cat([data_i,all_ohs],dim=-1) # [batch* obj_wrist_padded_length, edge_conv + num_obj_wrist_classes] 
            
            # compute h
            h = torch.stack([self.pose_encoder_hi[i](data_i) for i in range(len(self.pose_encoder_hi))]) if h is None else h # [num_layers, batch* obj_wrist_padded_length, rnn_encoder_hidden]
            
            # rnn             
            data_i = torch.unsqueeze(data_i,0)  # [1,          batch* obj_wrist_padded_length, edge_conv + num_obj_wrist_classes]
            _, h = self.pose_encoder(data_i,h)  # [num_layers, batch* obj_wrist_padded_length, rnn_encoder_hidden]        
            
            if self.use_bidirectional == 1:
                reversed_data_i = reversed_all_data[i] # [batch* obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
            
                # edge conv
                reversed_data_i = self.edge_conv(reversed_data_i,edge_index) # [batch* obj_wrist_padded_length, edge_conv]
                
                # if using context
                if self.context:
                    reversed_data_i = torch.cat([reversed_data_i,all_ohs],dim=-1) # [batch* obj_wrist_padded_length, edge_conv + num_obj_wrist_classes] 
                
                # compute h
                reversed_h = torch.stack([self.pose_encoder_hi[i](reversed_data_i) for i in range(len(self.pose_encoder_hi))]) if reversed_h is None else reversed_h # [num_layers, batch* obj_wrist_padded_length, rnn_encoder_hidden]
                
                # rnn             
                reversed_data_i = torch.unsqueeze(reversed_data_i,0)  # [1,          batch* obj_wrist_padded_length, edge_conv + num_obj_wrist_classes]
                _, reversed_h = self.pose_encoder(reversed_data_i,reversed_h)  # [num_layers, batch* obj_wrist_padded_length, rnn_encoder_hidden] 
                
                # collect
                reversed_h_list.append(reversed_h[-1])
            
            # collect
            h_list.append(h[-1])
                
        h_list = torch.stack(h_list)                        # [pose_padded_length, batch* obj_wrist_padded_length, rnn_encoder_hidden]
        if self.use_bidirectional == 1:
            reversed_h_list = torch.stack(reversed_h_list)  # [pose_padded_length, batch* obj_wrist_padded_length, rnn_encoder_hidden]
            h_list = torch.stack([h_list, reversed_h_list],dim=-1)
        
        h_list = torch.reshape(h_list,[self.pose_padded_length,self.batch_size,obj_wrist_padded_length,-1]) # [pose_padded_length, batch, obj_wrist_padded_length, rnn_encoder_hidden]
        h_list = F.tanh(h_list) if self.tanh_before_clf == 1 else h_list
        lactions = self.lclf(h_list[:,:,-2]) # [pose_padded_length, batch, len(all_actions)]
        ractions = self.rclf(h_list[:,:,-1]) # [pose_padded_length, batch, len(all_actions)]
        lactions = torch.permute(lactions,[1,0,2])
        ractions = torch.permute(ractions,[1,0,2])
                
        # collect features before classification
        lh_list = h_list[:,:,-2]                    # [pose_padded_length, batch, rnn_encoder_hidden]
        rh_list = h_list[:,:,-1]                    # [pose_padded_length, batch, rnn_encoder_hidden]        
        lh_list = torch.permute(lh_list,[1,0,2])    # [batch, pose_padded_length, rnn_encoder_hidden]
        rh_list = torch.permute(rh_list,[1,0,2])    # [batch, pose_padded_length, rnn_encoder_hidden]        
        lh_list = torch.stack([zero_pad(x,l) for x,l in zip(lh_list,data["xyz_unpadded_length"])])
        rh_list = torch.stack([zero_pad(x,l) for x,l in zip(rh_list,data["xyz_unpadded_length"])])
                
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
        if mode == "val":
            return_data = {**return_data, 
                            "lh_list":lh_list, "lh_list_unpadded_length":data["xyz_unpadded_length"],
                            "rh_list":rh_list, "rh_list_unpadded_length":data["xyz_unpadded_length"]}
        return return_data
