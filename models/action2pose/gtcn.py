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
    
class Encoder(nn.Module):
    def __init__(self, encoder_units, encoder_activations, args):
        super(Encoder, self).__init__()
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        
        # split filters
        self.num_convs = int(len(encoder_activations) / 2)
        assert self.num_convs % 2 == 0
        
        # edge convolution
        edge_convs = []
        for i in range(self.num_convs):        
            encoder_units_i = encoder_units[i*5:i*5+2]
            encoder_activations_i = encoder_activations[i*2:i*2+1]
            edge_convs.append(tgnn.EdgeConv(make_mlp(encoder_units_i, encoder_activations_i)))  
        self.edge_convs = nn.ModuleList(edge_convs)    
        
        # temporal convolution
        temp_convs = []
        for i in range(self.num_convs):
            inp_channels_i = encoder_units[2+i*5]
            out_channels_i = encoder_units[3+i*5]
            kernel_size_i  = encoder_units[4+i*5]
            temp_convs.append(nn.Sequential(nn.Conv1d(inp_channels_i,out_channels_i,kernel_size_i,padding=kernel_size_i//2),
                                            make_activation(encoder_activations[i*2+1]),
                                            nn.MaxPool1d(2)))
        self.temp_convs = nn.ModuleList(temp_convs)    
        
    def forward(self, **kwargs):
    
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        data = self.data              # [pose_padded_length, batch, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
        all_ohs = self.all_ohs        # [pose_padded_length, batch, obj_wrist_padded_length, num_obj_wrist_classes]
        edge_index = self.edge_index
                
        # fast but memory intensive graph op
        if self.gtcn == "fast":
            
            # length of data
            length = data.shape[0]
            
            # reshape for intial graph input
            data = torch.reshape(data, [data.shape[0], data.shape[1]* data.shape[2], data.shape[3]])                    # [pose_padded_length, batch* obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
            data = torch.permute(data,[1,0,2])                                                                          # [batch* obj_wrist_padded_length, pose_padded_length, object_encoder + num_obj_wrist_classes] # i permuted it this way because of how the edge indices were formed for the fast op
            data = torch.reshape(data,[-1,data.shape[-1]])                                                              # [batch* obj_wrist_padded_length* pose_padded_length, object_encoder + num_obj_wrist_classes]
            
            # reshape ohs
            all_ohs = torch.reshape(all_ohs, [all_ohs.shape[0], all_ohs.shape[1]* all_ohs.shape[2], all_ohs.shape[3]])  # [pose_padded_length, batch* obj_wrist_padded_length,    num_obj_wrist_classes]
            all_ohs = torch.permute(all_ohs,[1,0,2])                                                                    # [batch* obj_wrist_padded_length, pose_padded_length,    num_obj_wrist_classes]
            all_ohs = all_ohs[:,0:1]                                                                                    # [batch* obj_wrist_padded_length, 1,                     num_obj_wrist_classes]
            
            for i in range(self.num_convs):
                                
                # graph
                data = self.edge_convs[i](data,edge_index[i]) # [batch* obj_wrist_padded_length* length, f1]
                
                # concat ohs for tcn
                if self.context:
                    all_ohs_i = all_ohs.repeat(1,length,1)                          # [batch* obj_wrist_padded_length, length, num_obj_wrist_classes]
                    all_ohs_i = torch.reshape(all_ohs_i,[-1,all_ohs_i.shape[-1]])   # [batch* obj_wrist_padded_length* length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)                       # [batch* obj_wrist_padded_length* length, f1 + num_obj_wrist_classes]
                                
                # tcn
                data = torch.reshape(data,[-1,length,data.shape[-1]]) # [batch* obj_wrist_padded_length, length, f1]
                data = torch.permute(data,[0,2,1])                    # [batch* obj_wrist_padded_length, f1, length]
                data = self.temp_convs[i](data)                       # [batch* obj_wrist_padded_length, f2, length/2]
                length = data.shape[2]
                
                # reshape for graph
                data = torch.permute(data,[0,2,1])                    # [batch* obj_wrist_padded_length, length/2, f2]
                data = torch.reshape(data,[-1,data.shape[-1]])        # [batch* obj_wrist_padded_length* length/2, f2]
                
                # concat ohs for graph
                # do not concat if exiting
                if self.context and i != self.num_convs-1:
                    all_ohs_i = all_ohs.repeat(1,length,1)                          # [batch* obj_wrist_padded_length, length/2, num_obj_wrist_classes]
                    all_ohs_i = torch.reshape(all_ohs_i,[-1,all_ohs_i.shape[-1]])   # [batch* obj_wrist_padded_length* length/2, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)                       # [batch* obj_wrist_padded_length* length/2, f2 + num_obj_wrist_classes]
                
        # slow graph op
        elif self.gtcn == "slow":
        
            all_ohs = torch.reshape(all_ohs, [all_ohs.shape[0], all_ohs.shape[1]* all_ohs.shape[2], all_ohs.shape[3]])  # [pose_padded_length, batch* obj_wrist_padded_length, num_obj_wrist_classes]
            all_ohs = all_ohs[0:1]                                                                                      # [1,                  batch* obj_wrist_padded_length, num_obj_wrist_classes]
            data = torch.reshape(data, [data.shape[0], data.shape[1]* data.shape[2], data.shape[3]])                    # [pose_padded_length, batch* obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
            length = data.shape[0]
            for i in range(self.num_convs):
                
                # graph
                data = torch.stack([self.edge_convs[i](d,edge_index) for d in data]) # [length, batch* obj_wrist_padded_length, f1]
            
                # concat ohs for tcn
                if self.context:
                    all_ohs_i = all_ohs.repeat(length,1,1)      # [length, batch* obj_wrist_padded_length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)   # [length, batch* obj_wrist_padded_length, f1 + num_obj_wrist_classes]
            
                # tcn 
                data = torch.permute(data,[1,2,0]) # [batch* obj_wrist_padded_length, f1, length]
                data = self.temp_convs[i](data)    # [batch* obj_wrist_padded_length, f2, length/2]
                length = data.shape[-1]
                
                # reshape for graph
                data = torch.permute(data,[2,0,1]) # [length/2, batch* obj_wrist_padded_length, f2]
                
                # concat ohs for graph
                # do not concat if exiting
                if self.context and i != self.num_convs-1:
                    all_ohs_i = all_ohs.repeat(length,1,1)      # [length/2, batch* obj_wrist_padded_length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)   # [length/2, batch* obj_wrist_padded_length, f2 + num_obj_wrist_classes]
        else:
            print("Unknown self.gtcn",self.gtcn)
        return data

class Decoder(nn.Module):
    def __init__(self, decoder_units, decoder_activations, args):
        super(Decoder, self).__init__()
        
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        
        # split filters
        self.num_convs = int(len(decoder_activations) / 2)
        assert self.num_convs % 2 == 0
        
        # temporal convolution
        temp_convs = []
        for i in range(self.num_convs):
            inp_channels_i = decoder_units[0+i*5]
            out_channels_i = decoder_units[1+i*5]
            kernel_size_i  = decoder_units[2+i*5]
            temp_convs.append(nn.Sequential(nn.Conv1d(inp_channels_i,out_channels_i,kernel_size_i,padding=kernel_size_i//2),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=2)))
        self.temp_convs = nn.ModuleList(temp_convs)          
        
        # edge convolution
        edge_convs = []
        for i in range(self.num_convs):        
            decoder_units_i = decoder_units[i*3+i*2+3:i*3+i*2+5]
            decoder_activations_i = decoder_activations[i*2+1:i*2+2]
            edge_convs.append(tgnn.EdgeConv(make_mlp(decoder_units_i, decoder_activations_i)))  
        self.edge_convs = nn.ModuleList(edge_convs)  
  
    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)  
        
        # [length, batch* obj_wrist_padded_length, f1] if gtcn == "slow"
        # [batch* obj_wrist_padded_length* length, f1] if gtcn == "fast"  # shaped this way due to how i permuted it before the encoder, because of how the edge indices were formed for the fast op
        data = self.data               
        all_ohs = self.all_ohs        # [pose_padded_length, batch, obj_wrist_padded_length, num_obj_wrist_classes]
        edge_index = self.edge_index
                
        # fast but memory intensive graph op
        if self.gtcn == "fast":
            
            # reshape ohs
            all_ohs = torch.reshape(all_ohs, [all_ohs.shape[0], all_ohs.shape[1]* all_ohs.shape[2], all_ohs.shape[3]])  # [pose_padded_length, batch* obj_wrist_padded_length,    num_obj_wrist_classes]
            all_ohs = torch.permute(all_ohs,[1,0,2])                                                                    # [batch* obj_wrist_padded_length, pose_padded_length,    num_obj_wrist_classes]
            all_ohs = all_ohs[:,0:1]                                                                                    # [batch* obj_wrist_padded_length, 1,                     num_obj_wrist_classes]
            
            # reverse edge_index
            edge_index = edge_index[::-1]
            
            # reshape for initial input
            data = torch.reshape(data,[self.batch_size*(self.object_padded_length+2), -1, data.shape[-1]]) # [batch* obj_wrist_padded_length, length, f1]
            length = data.shape[1]
            
            for i in range(self.num_convs):
            
                # concat ohs for tcn
                if self.context:
                    all_ohs_i = all_ohs.repeat(1,length,1)      # [batch* obj_wrist_padded_length, length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)   # [batch* obj_wrist_padded_length, length, f1 + num_obj_wrist_classes]
                    
                # tcn
                data = torch.permute(data,[0,2,1]) # [batch* obj_wrist_padded_length, f1, length]
                data = self.temp_convs[i](data)    # [batch* obj_wrist_padded_length, f2, length*2]
                length = length*2
                    
                # graph
                data = torch.permute(data,[0,2,1])              # [batch* obj_wrist_padded_length, length*2, f2]
                data = torch.reshape(data,[-1,data.shape[-1]])  # [batch* obj_wrist_padded_length* length*2, f2]
                
                # concat ohs for graph
                if self.context:
                    all_ohs_i = all_ohs.repeat(1,length,1)                          # [batch* obj_wrist_padded_length, length*2, num_obj_wrist_classes]
                    all_ohs_i = torch.reshape(all_ohs_i,[-1,all_ohs_i.shape[-1]])   # [batch* obj_wrist_padded_length* length*2, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)                       # [batch* obj_wrist_padded_length* length*2, f2 + num_obj_wrist_classes]
                data = self.edge_convs[i](data,edge_index[i])                       # [batch* obj_wrist_padded_length* length*2, f1]
                
                # reshape for tcn
                data = torch.reshape(data,[self.batch_size*(self.object_padded_length+2),length,data.shape[-1]]) # [batch* obj_wrist_padded_length, length*2, f1]
        
            # reshape for output
            data = torch.permute(data,[1,0,2])
        
        # slow graph op
        elif self.gtcn == "slow":
        
            all_ohs = torch.reshape(all_ohs, [all_ohs.shape[0], all_ohs.shape[1]* all_ohs.shape[2], all_ohs.shape[3]])  # [pose_padded_length, batch* obj_wrist_padded_length, num_obj_wrist_classes]
            all_ohs = all_ohs[0:1]                                                                                      # [1,                  batch* obj_wrist_padded_length, num_obj_wrist_classes]
            length = data.shape[0]
            for i in range(self.num_convs):
                
                # concat ohs for tcn
                if self.context:
                    all_ohs_i = all_ohs.repeat(length,1,1)      # [length, batch* obj_wrist_padded_length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)   # [length, batch* obj_wrist_padded_length, f1 + num_obj_wrist_classes]
                                
                # tcn 
                data = torch.permute(data,[1,2,0]) # [batch* obj_wrist_padded_length, f1, length]
                data = self.temp_convs[i](data)    # [batch* obj_wrist_padded_length, f1, length*2]
                length = data.shape[-1]
                
                # concat ohs for graph
                data = torch.permute(data,[2,0,1])              # [length*2, batch* obj_wrist_padded_length, f1]
                if self.context:
                    all_ohs_i = all_ohs.repeat(length,1,1)      # [length*2, batch* obj_wrist_padded_length, num_obj_wrist_classes]
                    data = torch.cat([data,all_ohs_i],dim=-1)   # [length*2, batch* obj_wrist_padded_length, f1 + num_obj_wrist_classes]
            
                # graph
                data = torch.stack([self.edge_convs[i](d,edge_index) for d in data]) # [length*2, batch* obj_wrist_padded_length, f2]
        
        else:
            print("Unknown self.gtcn",self.gtcn)
        
        return data # [pose_padded_length, batch* obj_wrist_padded_length, f2]

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
        
        # # # # #
        # gtcn  #
        # # # # #
        
        # both encoder and decoder must have the same number of gcn-tcn blocks
        assert len(self.pose_encoder_activations) == len(self.pose_decoder_activations)
        # both encoder and decoder must have at least one of gcn-tcn block
        assert len(self.pose_encoder_activations) % 2 == 0
        
        # encoder
        self.encoder = Encoder(self.pose_encoder_units, self.pose_encoder_activations, args)
                                                      
        # decoder
        self.decoder = Decoder(self.pose_decoder_units, self.pose_decoder_activations, args)
        
        # classifier
        # - do not classify them together since the actions are asynchronous
        if "kit_mocap" in self.data_loader:
            all_actions = self.main_actions + [x for x in self.fine_actions if x not in self.main_actions]
        elif "kit_rgbd" in self.data_loader:
            all_actions = self.fine_actions
        else:
            print("Unknown self.data_loader:",self.data_loader)
            sys.exit()
        self.lclf = nn.Linear(self.pose_decoder_units[-1],len(all_actions))
        self.rclf = nn.Linear(self.pose_decoder_units[-1],len(all_actions))
                        
    def forward(self, data, mode):
                
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
                        
            # replicate adjacency pose_padded_length times
            # [batch_item1_t1, batch_item1_t2, ... batch_item2_t1, batch_item2_t2]
            # which is equivalent to [batch, pose_padded_length]
            if self.gtcn == "fast":
                for _ in range(self.pose_padded_length):
                    batch_adj.append(adj)
            elif self.gtcn == "slow":
                batch_adj.append(adj)
            else:
                print("Unknown self.gtcn:",self.gtcn)
                
        # create the edge index list
        if self.gtcn == "fast":
            batch_adj = torch.stack(batch_adj)
            batch_adj = [batch_adj[::1*2**k] for k in range(len(self.pose_encoder_activations)//2)]
            edge_index = [dense_to_sparse(batch_adj_i).to(device=torch.cuda.current_device()) for batch_adj_i in batch_adj]
        elif self.gtcn == "slow":
            edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())
        else:
            print("Unknown self.gtcn:",self.gtcn)
        #print("batch_adj",torch.stack(batch_adj).shape) # [4, 5, 5]
        #print("edge_index",edge_index.shape)            # [2, 80]
        #print(edge_index)
        """[[ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,
              4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,
              9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13,
             13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17,
             18, 18, 18, 18, 19, 19, 19, 19],
            [ 1,  2,  3,  4,  0,  2,  3,  4,  0,  1,  3,  4,  0,  1,  2,  4,  0,  1,
              2,  3,  6,  7,  8,  9,  5,  7,  8,  9,  5,  6,  8,  9,  5,  6,  7,  9,
              5,  6,  7,  8, 11, 12, 13, 14, 10, 12, 13, 14, 10, 11, 13, 14, 10, 11,
             12, 14, 10, 11, 12, 13, 16, 17, 18, 19, 15, 17, 18, 19, 15, 16, 18, 19,
             15, 16, 17, 19, 15, 16, 17, 18]]"""
        # [20, 128]
        
        
        # embed obj_xyz
        obj_xyz = data["obj_xyz"]                                                                                   # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers, 3]
        obj_xyz = torch.reshape(obj_xyz,[self.batch_size,self.pose_padded_length,obj_wrist_padded_length-2,-1])     # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers* 3]
        obj_data = self.object_encoder(obj_xyz)                                                                     # [batch, pose_padded_length, obj_wrist_padded_length-2, object_encoder]
                
        # embed wrist
        wrist_xyz  = data["wrist_xyz"]                                                                                       # [batch, pose_padded_length,    hand_xyz_dims,   3]
        wrist_xyz  = wrist_xyz.view(self.batch_size, self.pose_padded_length, 2, int(len(self.hand_xyz_dims)/2), 3)          # [batch, pose_padded_length, 2, hand_xyz_dims/2, 3]
        wrist_xyz  = torch.reshape(wrist_xyz,(self.batch_size,self.pose_padded_length,2, int(len(self.hand_xyz_dims)/2*3)))  # [batch, pose_padded_length, 2, hand_xyz_dims/2* 3]
        wrist_data = self.hand_encoder(wrist_xyz)                                                                            # [batch, pose_padded_length, 2, hand_encoder]
                
        # concatenate obj pos and wrist pos
        all_data  = torch.cat((obj_data,wrist_data),dim=2)                          # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder]        
        all_ohs  = torch.cat((data["obj_ohs"],data["wrist_ohs"]),dim=1) # [batch,                     obj_wrist_padded_length, num_obj_wrist_classes]
        all_ohs  = all_ohs[:,None].repeat(1,self.pose_padded_length,1,1)            # [batch, pose_padded_length, obj_wrist_padded_length, num_obj_wrist_classes]
        all_data = torch.cat((all_data,all_ohs),dim=-1)                             # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]        
        all_data_dim = all_data.shape[-1]
        
        all_data = torch.permute(all_data, [1,0,2,3])                                                                       # [pose_padded_length, batch, obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes] 
                        
        # forward
        all_ohs = torch.permute(all_ohs,[1,0,2,3]) # [pose_padded_length, batch, obj_wrist_padded_length, num_obj_wrist_classes]
        encoded = self.encoder(data=all_data, edge_index=edge_index, all_ohs=all_ohs)
        decoded = self.decoder(data=encoded,  edge_index=edge_index, all_ohs=all_ohs) # [pose_padded_length, batch* obj_wrist_padded_length, f]
        
        # classify left and right hand
        decoded = torch.reshape(decoded,[self.pose_padded_length,self.batch_size,obj_wrist_padded_length,self.pose_decoder_units[-1]]) # [pose_padded_length, batch, obj_wrist_padded_length, f]
        decoded = torch.permute(decoded,[1,0,2,3])
        lactions = self.lclf(decoded[:,:,-2])
        ractions = self.rclf(decoded[:,:,-1])
                
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
        
        return_data = {"lhand_action_ids":reformed_lactions, "rhand_action_ids":reformed_ractions}        
        return return_data
