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
    
def project_to_image(x, y, z, cx, cy, fx, fy):
        
    z = z * -1
    x =  (x * fx / z) + cx
    y = -(y * fy / z) + cy
    return x,y    
    
class EncoderGraphGRU(nn.Module):
    def __init__(self, encoder_units, encoder_activations, rnn_units, rnn_activations, mu_var_units, mu_var_activations):
        super(EncoderGraphGRU, self).__init__()
                            
        # mlp encoder
        #self.encoder     = make_mlp(encoder_units, encoder_activations)  
        self.encoder     = tgnn.EdgeConv(make_mlp(encoder_units, encoder_activations))  

        # rnn and hidden initializer
        self.num_layers  = rnn_units[2]
        self.rnn         = nn.GRU(rnn_units[0], rnn_units[1], rnn_units[2])
        self.rnn_hi      = nn.ModuleList([make_mlp(rnn_units[3:], rnn_activations) for _ in range(self.num_layers)])
        
        # mu var mlp
        self.mu          = make_mlp(mu_var_units, mu_var_activations)
        self.log_var     = make_mlp(mu_var_units, mu_var_activations)

        # norm
        self.norm = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        
    def forward(self, **kwargs):
    
        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
                                    
        # encode
        data = self.encoder(self.data, self.edge_index) # [batch* num_obj, pose_encoder]   [128, 64]
        data = torch.cat((data,self.t,self.a),dim=-1)   # [batch* num_obj, pose_encoder+1] [128, 64+1+a]
                
        # # # #
        # GRU #
        # # # #
        
        # compute hidden states
        h = torch.stack([self.rnn_hi[i](data) for i in range(self.num_layers)]) if self.h is None else self.h # [num_layers, batch* num_objects, rnn_encoder_hidden]
        
        # reshape input
        data = torch.unsqueeze(data,0) # [1, batch* num_obj, pose_encoder]
                
        # compute GRU
        _, h = self.rnn(data, h) # [num_layers, batch* num_obj, rnn_encoder_hidden]
        
        # compute mu and log_var 
        data = h[-1]                                        # [batch* num_obj, rnn_encoder_hidden]
        mu, log_var = self.mu(data), self.log_var(data)     # [batch* num_obj, pose_mu_var]
        
        # sample z
        std = torch.exp(0.5*log_var)                                # [batch* num_obj, pose_mu_var]
        eps = self.norm.sample([mu.shape[0], mu.shape[1]]).cuda()   # [batch* num_obj, pose_mu_var]
        z   = mu + eps * std                                        # [batch* num_obj, pose_mu_var]
        
        return {"h":h, "mu":mu, "log_var":log_var, "z":z}

class DecoderGraphGRU(nn.Module):
    def __init__(self, encoder_units, encoder_activations, rnn_units, rnn_activations, decoder_units, decoder_activations):
        super(DecoderGraphGRU, self).__init__()
                            
        # encoder
        self.encoder     = tgnn.EdgeConv(make_mlp(encoder_units, encoder_activations))   

        # rnn and hidden initializer
        self.num_layers  = rnn_units[2]
        self.rnn         = nn.GRU(rnn_units[0], rnn_units[1], rnn_units[2])
        self.rnn_hi      = nn.ModuleList([make_mlp(rnn_units[3:], rnn_activations) for _ in range(self.num_layers)])
        
        # mu var mlp
        self.decoder     = tgnn.EdgeConv(make_mlp(decoder_units, decoder_activations))  

    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
                
        # encode
        data = self.encoder(self.data, self.edge_index)                                                                         # [batch* num_obj, pose_encoder]                [128, 64]
        data = torch.cat((self.z,data,self.t,self.a),dim=-1) #if self.a is not None else torch.cat((self.z,data,self.t),dim=-1)  # [batch* num_obj, pose_encoder + noise + 1]    [128, 64 + 4 + 1]
        
        # # # #
        # GRU #
        # # # #
                
        # compute hidden states
        h = torch.stack([self.rnn_hi[i](data) for i in range(self.num_layers)]) if self.h is None else self.h # [num_layers, batch* num_obj, rnn_encoder_hidden]
        
        # reshape input
        data = torch.unsqueeze(data,0) # [1, batch* num_obj, pose_encoder + noise + 1]
        
        # compute GRU
        _, h = self.rnn(data, h) # [num_layers, batch* num_obj, rnn_encoder_hidden]
        
        # compute mu and log_var 
        data = h[-1]                                     # [batch* num_obj, rnn_encoder_hidden]
        out = self.decoder(data, self.edge_index)        # [batch* num_obj, 3]
        
        return {"h":h, "out":out}

class BodyDecoder(nn.Module):
    def __init__(self, body_decoder_type, body_decoder_units, body_decoder_activations, decoder_dim):
        super(BodyDecoder, self).__init__()
        
        self.body_decoder_type = body_decoder_type
        if self.body_decoder_type == "rnn":
            # gru
            self.num_layers = body_decoder_units[2]
            self.rnn        = nn.GRU(body_decoder_units[0], body_decoder_units[1], body_decoder_units[2])
            self.rnn_hi     = nn.ModuleList([make_mlp(body_decoder_units[3:], body_decoder_activations) for _ in range(self.num_layers)])
            
            # decoder
            self.decoder    = make_mlp([body_decoder_units[1], decoder_dim], ["none"])
            
        elif self.body_decoder_type == "make_mlp":
            self.decoder = make_mlp(body_decoder_units, body_decoder_activations)
        
        else:
            print("Unknown self.body_decoder_type:",self.body_decoder_type)
            sys.exit()
        
    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        data = self.data    # [batch, 159]

        if self.body_decoder_type == "rnn":
            # compute hidden states
            h = torch.stack([self.rnn_hi[i](data) for i in range(self.num_layers)]) if self.h is None else self.h # [num_layers, batch, rnn_encoder_hidden]
            
            # reshape input
            data = torch.unsqueeze(data,0) # [1, batch, 159]
            
            # compute GRU
            _, h = self.rnn(data, h)    # [num_layers, batch, rnn_encoder_hidden]
            
            # compute out
            out = self.decoder(h[-1])       # [32, 159 - len(hand_xyz_dims)]
            return {"h":h, "out":out}

        elif self.body_decoder_type == "make_mlp":
            data = torch.reshape(data,[data.shape[0],-1])
            #print(data.shape)
            out = self.decoder(data)
            #print(out.shape)
            return {"out":out}
            
        else:
            print("Unknown self.body_decoder_type:",self.body_decoder_type)
            sys.exit()

class FingerDecoder(nn.Module):
    def __init__(self, finger_decoder_type, finger_decoder_units, finger_decoder_activations, decoder_dim):
        super(FingerDecoder, self).__init__()
        
        self.finger_decoder_type = finger_decoder_type        
        if self.finger_decoder_type == "rnn":
        
            # gru
            # 1 rnn per finger
            self.num_layers = finger_decoder_units[2]
            self.lrnn        = nn.GRU(finger_decoder_units[0], finger_decoder_units[1], finger_decoder_units[2])
            self.lrnn_hi     = nn.ModuleList([make_mlp(finger_decoder_units[3:], finger_decoder_activations) for _ in range(self.num_layers)])
            self.rrnn        = nn.GRU(finger_decoder_units[0], finger_decoder_units[1], finger_decoder_units[2])
            self.rrnn_hi     = nn.ModuleList([make_mlp(finger_decoder_units[3:], finger_decoder_activations) for _ in range(self.num_layers)])
            
            # decoder
            # 1 decoder per fingers
            self.ldecoder    = make_mlp([finger_decoder_units[1], decoder_dim], ["none"])
            self.rdecoder    = make_mlp([finger_decoder_units[1], decoder_dim], ["none"])
        
        elif self.finger_decoder_type == "make_mlp":
            self.decoder    = make_mlp(finger_decoder_units, finger_decoder_activations)
        
        else:
            print("Unknown self.finger_decoder_type:",self.finger_decoder_type)
            sys.exit()
        
    def forward(self, **kwargs):

        # pass kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.finger_decoder_type == "rnn":
                
            data = self.data    # [batch, 2, 19]
            ldata = data[:,0,:] # [batch, 19]
            rdata = data[:,1,:] # [batch, 19]
            #data = torch.reshape(data, [data.shape[0], -1])    # [batch, 38]
                
            # compute hidden states
            lh = torch.stack([self.lrnn_hi[i](ldata) for i in range(self.num_layers)]) if self.lh is None else self.lh # [num_layers, batch, rnn_encoder_hidden]
            rh = torch.stack([self.rrnn_hi[i](rdata) for i in range(self.num_layers)]) if self.rh is None else self.rh # [num_layers, batch, rnn_encoder_hidden]
            
            # reshape input
            ldata = torch.unsqueeze(ldata,0) # [1, batch, 19]
            rdata = torch.unsqueeze(rdata,0) # [1, batch, 19]
                        
            # compute GRU
            _, lh = self.lrnn(ldata, lh)    # [num_layers, batch, rnn_encoder_hidden]
            _, rh = self.lrnn(rdata, rh)    # [num_layers, batch, rnn_encoder_hidden]
            
            # compute out
            lout = self.ldecoder(lh[-1]) # [32, 19]
            rout = self.rdecoder(rh[-1]) # [32, 19]
            return {"lh":lh, "rh":rh, "out":torch.stack([lout,rout],dim=1)}

        elif self.finger_decoder_type == "make_mlp":            
            data = self.data 
            out = self.decoder(data)
            return {"out":out}

        else:
            print("Unknown self.finger_decoder_type:",self.finger_decoder_type)
            sys.exit()

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                
        # only allow even number of hand joints
        assert len(self.hand_xyz_dims) % 2  == 0
        
        # hand encoder decoder
        if self.hand_encoder_type == "torch.mean":
            self.hand_encoder = eval(self.hand_encoder_type)
        elif self.hand_encoder_type == "make_mlp":
            self.hand_encoder = make_mlp(self.hand_encoder_units, self.hand_encoder_activations)
        else:
            print("Unknown hand_encoder_type:",self.hand_encoder_type)
        self.hand_decoder = make_mlp(self.hand_decoder_units, self.hand_decoder_activations) if self.hand_decoder_units is not None else nn.Identity()
                                 
        # object encoder decoder
        self.object_encoder = make_mlp(self.object_encoder_units, self.object_encoder_activations) if self.object_encoder_units is not None else nn.Identity()
        self.object_decoder = make_mlp(self.object_decoder_units, self.object_decoder_activations) if self.object_decoder_units is not None else nn.Identity()
        
        # prior
        self.prior_net = EncoderGraphGRU(self.pose_encoder_units,     self.pose_encoder_activations, 
                                         self.pose_rnn_encoder_units, self.pose_rnn_encoder_activations,
                                         self.pose_mu_var_units,      self.pose_mu_var_activations)
        
        # posterior
        self.posterior_net = EncoderGraphGRU(self.pose_encoder_units,     self.pose_encoder_activations, 
                                             self.pose_rnn_encoder_units, self.pose_rnn_encoder_activations,
                                             self.pose_mu_var_units,      self.pose_mu_var_activations)   
                                              
        # decoder
        self.decoder_net = DecoderGraphGRU(self.pose_encoder_units,     self.pose_encoder_activations,
                                           self.pose_rnn_decoder_units, self.pose_rnn_decoder_activations,
                                           self.pose_decoder_units,     self.pose_decoder_activations)
        
        # body decoder
        if "xyz" in self.loss_names:
        
            if "kit_mocap" in self.data_root:
                self.body_decoder = BodyDecoder(self.body_decoder_type, self.body_decoder_units, self.body_decoder_activations, 159 - 3*len(self.hand_xyz_dims))
        
            elif "kit_rgbd" in self.data_root:
                self.body_decoder = BodyDecoder(self.body_decoder_type, self.body_decoder_units, self.body_decoder_activations, 30)
                
            else:
                print("Unknown self.data_root for BodyDecoder:", self.data_root)
                sys.exit()
        
        # embeddings
        if self.position_embedder_type != "None":
            self.position_embedder = eval(self.position_embedder_type)
                
        if "finger" in self.loss_names:
        
            if "kit_mocap" in self.data_root:        
                # finger decoder
                self.finger_decoder = FingerDecoder(self.finger_decoder_type, self.finger_decoder_units, self.finger_decoder_activations, 19)
        
            elif "kit_rgbd" in self.data_root:
                self.finger_decoder = FingerDecoder(self.finger_decoder_type, self.finger_decoder_units, self.finger_decoder_activations, 42)
                
            else:
                print("Unknown self.data_root for FingerDecoder:", self.data_root)
                sys.exit()
        
            if self.use_edges_for_finger in ["average","attention","pool"]:
            
                if "kit_mocap" in self.data_root:
                    # [object label, action] attention
                    self.hand_encoder2   = make_mlp([15,32], ["none"])
                    self.object_encoder2 = make_mlp([12,32], ["none"]) if self.object_encoder_units is not None else nn.Identity()       
                    self.object_embedder = make_mlp([self.num_obj_wrist_classes,128],["relu"])
                    self.action_embedder = make_mlp([len(self.main_actions),128],["relu"])
                elif "kit_rgbd" in self.data_root:
                    # [object label, action] attention
                    self.hand_encoder2   = make_mlp([3,32], ["none"])
                    self.object_encoder2 = make_mlp([3,32], ["none"]) if self.object_encoder_units is not None else nn.Identity()       
                    self.object_embedder = make_mlp([17,128],["relu"])
                    self.action_embedder = make_mlp([8,128],["relu"])
                else:
                    print("Unknown self.data_root for FingerDecoder:", self.data_root)
                    sys.exit()
        
    def forward(self, data, mode):
                                
        num_obj_wrist_classes = self.num_obj_wrist_classes
        obj_wrist_padded_length = self.obj_wrist_padded_length
        
        if "3d" in self.object_type:
            num_obj_coords = 3
        elif "2d" in self.object_type:
            num_obj_coords = 2
        else:
            print("Unknown self.object_type",self.object_type)
            sys.exit()
                
        # teacher forcing, 0 uses ground truth, 1 uses predictions
        rand = torch.rand(self.batch_size)
        teacher_force = rand > self.teacher_force_ratio
        teacher_force = teacher_force.long()
        
        # # # # # # # # #
        #               #
        # Prepare Data  #
        #               #
        # # # # # # # # #
        
        """
        action = None
        if self.use_main_action:
            # main action labels 
            main_action_oh = data["rhand_main_action_oh"]*self.oh_scale                                 # [batch, 6]
            main_action_oh = torch.unsqueeze(main_action_oh,1).repeat(1,obj_wrist_padded_length,1)      # [batch, obj_wrist_padded_length, 6]
            main_action_oh = torch.reshape(main_action_oh,[self.batch_size*obj_wrist_padded_length,-1]) # [batch* obj_wrist_padded_length, 6]
            action = main_action_oh
        """

        main_action_oh = data["rhand_main_action_oh"]                                               # [batch, 6]
        main_action_oh = torch.unsqueeze(main_action_oh,1).repeat(1,obj_wrist_padded_length,1)      # [batch, obj_wrist_padded_length, 6]
        main_action_oh = torch.reshape(main_action_oh,[self.batch_size*obj_wrist_padded_length,-1]) # [batch* obj_wrist_padded_length, 6]
        action = main_action_oh
        action = action if self.use_main_action == 1 else torch.zeros(action.size(),dtype=action.dtype,device=action.get_device())
        #print("action:",action.shape,torch.sum(action))
        
        if self.use_semantic_variation:
            # semantic variation labels
            semantic_variation_oh = data["rhand_semantic_variation_oh"]                                                # [batch, max_semantic_variation]
            semantic_variation_oh = torch.unsqueeze(semantic_variation_oh,1).repeat(1,obj_wrist_padded_length,1)       # [batch, obj_wrist_padded_length, max_semantic_variation]
            semantic_variation_oh = torch.reshape(semantic_variation_oh,[self.batch_size*obj_wrist_padded_length,-1])  # [batch* obj_wrist_padded_length, 6]
            action = semantic_variation_oh
                
        # create mega adjacency graph then convert it to the edge index list
        # - table not in
        # - 0 are padded objects not connected to anything
        # - hands and objects connected to each other except the zeros
        # - hands are the last 2 indices
        batch_adj = []
        for rhand_obj_ids in data["obj_ids"]:
            adj = 1 - torch.eye(obj_wrist_padded_length) # [obj_wrist_padded_length, obj_wrist_padded_length] should be adj = 1 - torch.eye(obj_wrist_padded_length)
            for i,rhand_obj_id in enumerate(rhand_obj_ids):
                # if rhand_obj_id is zero, detach it from all
                if rhand_obj_id == 0:
                    adj = detach(adj, i)
            batch_adj.append(adj)
            #print(adj)
            #sys.exit()
        edge_index = dense_to_sparse(torch.stack(batch_adj)).to(device=torch.cuda.current_device())
                
        # embed obj_xyz
        obj_xyz = data["obj_xyz"]                                                                                   # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers, 3]
        obj_xyz = torch.reshape(obj_xyz,[self.batch_size,self.pose_padded_length,obj_wrist_padded_length-2,-1])     # [batch, pose_padded_length, obj_wrist_padded_length-2, num_markers* 3]
        obj_data = self.object_encoder(obj_xyz)                                                                     # [batch, pose_padded_length, obj_wrist_padded_length-2, object_encoder]
                
        # embed wrist
        wrist_xyz  = data["wrist_xyz"]                                                                                                  # [batch, pose_padded_length,    hand_xyz_dims,   3]
        wrist_xyz  = wrist_xyz.view(self.batch_size, self.pose_padded_length, 2, int(len(self.hand_xyz_dims)/2), num_obj_coords)            # [batch, pose_padded_length, 2,                  3]
        wrist_xyz  = torch.reshape(wrist_xyz,(self.batch_size,self.pose_padded_length,2,int(len(self.hand_xyz_dims)/2*num_obj_coords)))     # [batch, pose_padded_length, 2, hand_xyz_dims/2* 3]
        wrist_data = self.hand_encoder(wrist_xyz)                                                                                       # [batch, pose_padded_length, 2, hand_encoder]
                
        # concatenate obj pos and wrist pos
        all_data  = torch.cat((obj_data,wrist_data),dim=2)                          # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder]        
        all_ohs  = torch.cat((data["obj_ohs"],data["wrist_ohs"]),dim=1)             # [batch,                     obj_wrist_padded_length, num_obj_wrist_classes]
        all_ohs  = all_ohs[:,None].repeat(1,self.pose_padded_length,1,1)            # [batch, pose_padded_length, obj_wrist_padded_length, num_obj_wrist_classes]
        all_ohs  = all_ohs if self.use_object_label == 1 else torch.zeros(all_ohs.size(),dtype=all_ohs.dtype,device=all_ohs.get_device())
        #print("all_ohs:",all_ohs.shape,torch.sum(all_ohs))
        all_data = torch.cat((all_data,all_ohs),dim=-1)                             # [batch, pose_padded_length, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]        
        all_data_dim = all_data.shape[-1]
        
        # reshape obj data
        all_data = torch.permute(all_data, [1,0,2,3])                                       # [pose_padded_length, batch, obj_wrist_padded_length,  object_encoder + num_obj_wrist_classes]    
        posterior_all_data = all_data.reshape(self.pose_padded_length, -1, all_data_dim)    # [pose_padded_length, batch*(obj_wrist_padded_length), object_encoder + num_obj_wrist_classes]
        
        # positional embedding
        pos_emb = self.position_embedder(data["sequence_duration"], data["obj_xyz_unpadded_length"], self.pose_padded_length, self.position_embedder_units, self.reverse_position_embedder) # [batch, pose_padded_length, position_embedder]
        pos_emb = pos_emb.repeat(obj_wrist_padded_length,1,1) # [batch*(obj_wrist_padded_length), pose_padded_length, position_embedder]
        
        # outputs
        prior_mu, prior_log_var = [], []
        posterior_mu, posterior_log_var = [], []
        pred_obj_xyz, pred_wrist_xyz = [], []
        pred_obj_xyz_vel, pred_wrist_xyz_vel = [], []
        pred_rhand_xyz, pred_rhand_xyz_vel = [], []
        pred_rhand_finger = []
        
        # initialize hidden states
        prior_h, posterior_h = None, None
        decoder_h = None
        body_decoder_h = None
        finger_decoder_lh = None
        finger_decoder_rh = None
        for i in range(self.pose_padded_length-1):
        
            # # # # # # # # # # # # # # # # #
            #                               #
            # predict body and finger data  #
            #                               #
            # # # # # # # # # # # # # # # # #        
        
            if i == 0:
            
                # # # # # # # # # # # # #
                # hand and object data  #
                # # # # # # # # # # # # #
                if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
            
                    # encode object
                    pred_obj_xyz_t0 = data["obj_xyz"][:,i]                                                      # [batch, obj_wrist_padded_length-2, num_markers, 3]
                    obj_xyz_i  = torch.reshape(pred_obj_xyz_t0,[self.batch_size,obj_wrist_padded_length-2,-1])  # [batch, obj_wrist_padded_length-2, num_markers* 3]
                    obj_data_i = self.object_encoder(obj_xyz_i)                                                 # [batch, obj_wrist_padded_length-2, object_encoder]

                    # encode wrist
                    pred_wrist_xyz_t0 = data["wrist_xyz"][:,i]                                                                  # [batch,    hand_xyz_dims,   3]
                    wrist_xyz_i  = pred_wrist_xyz_t0.view(self.batch_size, 2, int(len(self.hand_xyz_dims)/2), num_obj_coords)       # [batch, 2, hand_xyz_dims/2, 3]
                    wrist_xyz_i  = torch.reshape(wrist_xyz_i,(self.batch_size,2,int(len(self.hand_xyz_dims)/2*num_obj_coords)))     # [batch, 2, hand_xyz_dims/2* 3]
                    wrist_data_i = self.hand_encoder(wrist_xyz_i)                                                               # [batch, 2, object_encoder]    
                    
                    # concat
                    all_data_i = torch.cat((obj_data_i,wrist_data_i),dim=1)     # [batch, obj_wrist_padded_length, object_encoder]
                    all_data_i = torch.cat((all_data_i,all_ohs[:,i]),dim=-1)    # [batch, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                        
                    # get prior input
                    prior_inp = all_data_i                                                                           # [batch, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                    prior_inp = torch.reshape(prior_inp, [self.batch_size*(obj_wrist_padded_length), all_data_dim])  # [batch* obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]                
                        
                # # # # # # # # # # # # #
                # body and finger data  #
                # # # # # # # # # # # # #
                
                if "xyz" in self.loss_names:
                    #pred_rhand_xyz_t0 = data["xyz"][:,i,[i for i in range(53) if i not in self.hand_xyz_dims],:]                   # [batch, remaining joints, 3]
                    pred_rhand_xyz_t0 = data["xyz"][:,i,[i for i in range(data["xyz"].shape[2]) if i not in self.hand_xyz_dims],:]  # [batch, remaining joints, 3]
                    
                if "finger" in self.loss_names:
                    pred_rhand_finger_t0 = data["finger"][:,i]                                                      # [batch, 2, 19]
            
            else:  
            
                # # # # # # # # # # # # #
                # hand and object data  #
                # # # # # # # # # # # # #
                if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
                
                    # encode object
                    pred_obj_xyz_t0 = pred_obj_xyz_t1
                    obj_xyz_i  = torch.reshape(pred_obj_xyz_t1,[self.batch_size,obj_wrist_padded_length-2,-1])  # [batch, obj_wrist_padded_length-2, num_markers* 3]
                    obj_data_i = self.object_encoder(obj_xyz_i)                                                 # [batch, obj_wrist_padded_length-2, object_encoder]
                    
                    # encode wrist
                    pred_wrist_xyz_t0 = pred_wrist_xyz_t1
                    wrist_xyz_i  = pred_wrist_xyz_t0.view(self.batch_size, 2, int(len(self.hand_xyz_dims)/2), num_obj_coords)   # [batch, 2, hand_xyz_dims/2, 3]
                    wrist_xyz_i  = torch.reshape(wrist_xyz_i,(self.batch_size,2,int(len(self.hand_xyz_dims)/2*num_obj_coords))) # [batch, 2, hand_xyz_dims/2* 3]
                    wrist_data_i = self.hand_encoder(wrist_xyz_i)                                                           # [batch, 2, object_encoder]
                    
                    # concat
                    all_data_i = torch.cat((obj_data_i,wrist_data_i),dim=1)     # [batch, obj_wrist_padded_length, object_encoder]
                    all_data_i = torch.cat((all_data_i,all_ohs[:,i]),dim=-1)    # [batch, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                    
                    # get true data
                    true_all_data_i = posterior_all_data[i].reshape(self.batch_size,obj_wrist_padded_length,all_data_dim)  # [batch, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]   
                    
                    # get prior input
                    prior_inp = torch.stack((true_all_data_i,all_data_i),dim=1)                             # [batch, 2, obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                    prior_inp = torch.stack([p[tf] for p,tf in zip(prior_inp,teacher_force)])               # [batch,    obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                    prior_inp = prior_inp.reshape(self.batch_size*(obj_wrist_padded_length), all_data_dim)  # [batch*    obj_wrist_padded_length, object_encoder + num_obj_wrist_classes]
                
                # # # # # # # # # # # # #
                # body and finger data  #
                # # # # # # # # # # # # #
                
                if "xyz" in self.loss_names:
                    pred_rhand_xyz_t0 = pred_rhand_xyz_t1
                    
                if "finger" in self.loss_names:
                    pred_rhand_finger_t0 = pred_rhand_finger_t1

            # # # # # # # # # # # # # # # # #
            #                               #
            # predict object and wrist data #
            #                               #
            # # # # # # # # # # # # # # # # #
                    
            if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
            
                # compute prior
                # =======================
                prior_inp = prior_inp                       # [batch*(obj_wrist_padded_length), 3]
                prior_out = self.prior_net(data=prior_inp, edge_index=edge_index, h=prior_h, t=pos_emb[:,i], a=action)
                prior_h   = prior_out["h"]
                
                # compute posterior
                # =======================
                posterior_inp = posterior_all_data[i+1]     # [batch*(obj_wrist_padded_length), 3]
                posterior_out = self.posterior_net(data=posterior_inp, edge_index=edge_index, h=posterior_h, t=pos_emb[:,i], a=action)
                posterior_h   = posterior_out["h"]
                            
                # decode
                # =======================
                # get prior and posterior z
                prior_z     = prior_out["z"].reshape(self.batch_size,obj_wrist_padded_length,self.pose_mu_var_units[-1])        # [batch,    obj_wrist_padded_length, -1]
                posterior_z = posterior_out["z"].reshape(self.batch_size,obj_wrist_padded_length,self.pose_mu_var_units[-1])    # [batch,    obj_wrist_padded_length, -1]
                # get z
                z = torch.stack((posterior_z,prior_z),dim=1)                                                                        # [batch, 2, obj_wrist_padded_length, -1]
                z = torch.stack([z_i[tf] for z_i,tf in zip(z,teacher_force)])                                                       # [batch,    obj_wrist_padded_length, -1]
                z = z.reshape(-1, self.pose_mu_var_units[-1])                                                                       # [batch*    obj_wrist_padded_length, -1]
                
                decoder_inp = prior_inp # obj_data[i] # [batch*(obj_wrist_padded_length), 3]
                decoder_out = self.decoder_net(z=z, data=decoder_inp, edge_index=edge_index, h=decoder_h, t=pos_emb[:,i], a=action)
                decoder_h   = decoder_out["h"]
                            
                # decode pred_obj_xyz_i and pred_wrist_xyz_i
                # =======================
                # reshape from [batch*(obj_wrist_padded_length), object_encoder] to 
                #              [batch, obj_wrist_padded_length,  object_encoder]
                out = decoder_out["out"]                                                    # [batch*(obj_wrist_padded_length),     pose_decoder]
                out = torch.reshape(out, [self.batch_size, obj_wrist_padded_length, -1])    # [batch, obj_wrist_padded_length,      pose_decoder]
                out = torch.cat((out,all_ohs[:,i]),dim=-1)                                  # [batch, obj_wrist_padded_length,      pose_decoder + num_obj_wrist_classes]            
                pred_obj_data_i      = out[:,:-2,:]                                         # [batch, obj_wrist_padded_length-2,    pose_decoder + num_obj_wrist_classes]
                pred_wrist_data_i    = out[:,-2:,:]                                         # [batch, 2,                            pose_decoder + num_obj_wrist_classes]
                
                # decode pred_obj_xyz_i and wrist
                pred_obj_xyz_t1   = self.object_decoder(pred_obj_data_i)                                                                            # [batch, obj_wrist_padded_length-2, num_markers* 3]
                pred_obj_xyz_t1   = torch.reshape(pred_obj_xyz_t1,[self.batch_size,obj_wrist_padded_length-2,-1,num_obj_coords])                        # [batch, obj_wrist_padded_length-2, num_markers, 3]   
                pred_wrist_xyz_t1 = self.hand_decoder(pred_wrist_data_i)                                                                            # [batch, 2, hand_xyz_dims/2* 3]
                pred_wrist_xyz_t1 = torch.reshape(pred_wrist_xyz_t1,(self.batch_size, 2, int(pred_wrist_xyz_t1.shape[-1]/num_obj_coords), num_obj_coords))  # [batch, 2, hand_xyz_dims/2, 3]
                pred_wrist_xyz_t1 = torch.reshape(pred_wrist_xyz_t1,(self.batch_size, -1, num_obj_coords))              
                
                # collect velocities
                # pred_obj_xyz_vel are the absolute positions if self.predict_object_velocity == 0
                pred_obj_xyz_vel.append(pred_obj_xyz_t1)
                pred_wrist_xyz_vel.append(pred_wrist_xyz_t1)
                
                if self.predict_object_velocity:
                    pred_obj_xyz_t1     = pred_obj_xyz_t0 + pred_obj_xyz_t1
                    pred_wrist_xyz_t1   = pred_wrist_xyz_t0 + pred_wrist_xyz_t1
                else:
                    pred_obj_xyz_t1     = pred_obj_xyz_t1
                    pred_wrist_xyz_t1   = pred_wrist_xyz_t1
                            
                # collect outputs
                pred_obj_xyz.append(pred_obj_xyz_t1)
                pred_wrist_xyz.append(pred_wrist_xyz_t1)
                prior_mu.append(prior_out["mu"])
                prior_log_var.append(prior_out["log_var"])
                posterior_mu.append(posterior_out["mu"])
                posterior_log_var.append(posterior_out["log_var"])
            
            # # # # # # # # #
            #               #
            # predict body  #
            #               #
            # # # # # # # # #
            
            if "xyz" in self.loss_names:
                        
                # project 3D wrist coordinates to 2D
                if "kit_rgbd" in self.data_root:
                    
                    # reverse scale prediction
                    if "wrist_xyz" in self.loss_names:
                        pred_wrist_xyz_t1_detached = pred_wrist_xyz_t1.detach().clone() / self.xyz_scale # [batch, hands=2, dim=3]
                        #print(pred_wrist_xyz_t1_detached.shape, data["table_center"].shape) # [batch, hands=2, dim=3], [batch, dim=3]
                    else:
                        pred_wrist_xyz_t1_detached = data["wrist_xyz"][:,i+1] / self.xyz_scale
                    
                    # denormalize
                    pred_wrist_xyz_t1_detached = pred_wrist_xyz_t1_detached + torch.unsqueeze(data["table_center"],dim=1)
                    pred_wrist_xyz_t1_detached = torch.matmul(data["rx"], torch.permute(pred_wrist_xyz_t1_detached,[0,2,1]))
                    pred_wrist_xyz_t1_detached = torch.permute(pred_wrist_xyz_t1_detached,[0,2,1])                              # [batch, hands=2, dim=3]
                    
                    # project to image
                    x = pred_wrist_xyz_t1_detached[:,:,0] # [batch, hands=2]
                    y = pred_wrist_xyz_t1_detached[:,:,1] # [batch, hands=2]
                    z = pred_wrist_xyz_t1_detached[:,:,2] # [batch, hands=2]
                    x, y = project_to_image(x, y, z, data["cx"], data["cy"], data["fx"], data["fy"]) # [batch, hands=2]
                    x /= torch.tensor([640,480]).to(device=torch.cuda.current_device())
                    y /= torch.tensor([640,480]).to(device=torch.cuda.current_device()) 
                    pred_wrist_xyz_t1_detached = torch.stack([x,y],dim=1)
                                        
                    """
                    # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    # sanity check to make sure the values are correct  #
                    # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    
                    print(data["wrist_xyz"].shape)      # [2, 150, 2, 3]     
                    print(data["table_center"].shape)   # [2, 3]
                    wrist = data["wrist_xyz"][:,0,:,:]  # [2, 2, 3]
                    wrist = wrist / self.xyz_scale
                    
                    # denormalize
                    wrist = wrist + torch.unsqueeze(data["table_center"],dim=1)
                    wrist = torch.matmul(data["rx"], torch.permute(wrist,[0,2,1]))
                    wrist = torch.permute(wrist,[0,2,1])                              # [batch, hands=2, dim=3]
                    
                    x = wrist[:,:,0]  # [batch=2, hands=2]
                    y = wrist[:,:,1]  # [batch=2, hands=2]
                    z = wrist[:,:,2]  # [batch=2, hands=2]
                    x, y = project_to_image(x, y, z, data["cx"], data["cy"], data["fx"], data["fy"]) # [2,2], [2,2], [2,2]
                    pred_wrist_xyz_t1_detached = torch.stack([x,y],dim=2) # [batch=2, hands=2, dim=2]
                    print(pred_wrist_xyz_t1_detached.shape)
                    for i in range(2):                        
                        print(data["sequence"][i])
                        print(pred_wrist_xyz_t1_detached[i,0])
                        print(pred_wrist_xyz_t1_detached[i,1])
                    sys.exit()
                    """
                    
                elif "kit_mocap" in self.data_root:
                    pred_wrist_xyz_t1_detached = pred_wrist_xyz_t1.detach().clone() if "wrist_xyz" in self.loss_names else data["wrist_xyz"][:,i+1]
                else:
                    print("Unknown self.data_root")
                    sys.exit()
                
                # concatenate partial body to wrist
                pred_rhand_xyz_t1 = torch.cat((pred_rhand_xyz_t0,pred_wrist_xyz_t1_detached),dim=1)     # [batch, 53, 3] or [batch, 17, 2]
                pred_rhand_xyz_t1 = torch.reshape(pred_rhand_xyz_t1, (self.batch_size, -1))             # [batch, 159]   or [batch, 34]
                                
                # decode body
                # =======================
                if self.body_decoder_type == "rnn":
                    body_decoder_out    = self.body_decoder(data=pred_rhand_xyz_t1, h=body_decoder_h)
                    pred_rhand_xyz_t1   = body_decoder_out["out"]
                    body_decoder_h      = body_decoder_out["h"]
                elif self.body_decoder_type == "make_mlp":
                    body_decoder_out    = self.body_decoder(data=pred_wrist_xyz_t1_detached)
                    pred_rhand_xyz_t1   = body_decoder_out["out"]
                else:
                    print("Unknown self.body_decoder_type:",self.body_decoder_type)
                    sys.exit()
                
                if "kit_mocap" in self.data_root:
                    pred_rhand_xyz_t1   = torch.reshape(pred_rhand_xyz_t1,[self.batch_size,-1,3]) # [batch, 53-len(hand_xyz_dims), 3]
                elif "kit_rgbd" in self.data_root:
                    pred_rhand_xyz_t1   = torch.reshape(pred_rhand_xyz_t1,[self.batch_size,-1,2]) # [batch, 53-len(hand_xyz_dims), 3]
                
                # collect velocities
                # pred_rhand_xyz_vel are the absolute positions if self.predict_body_velocity == 0
                pred_rhand_xyz_vel.append(pred_rhand_xyz_t1)
                
                if self.predict_body_velocity:
                    pred_rhand_xyz_t1 = pred_rhand_xyz_t0 + pred_rhand_xyz_t1
                else:
                    pred_rhand_xyz_t1 = pred_rhand_xyz_t1
                
                # collect output
                pred_rhand_xyz.append(pred_rhand_xyz_t1)

            # # # # # # # # # #
            #                 #
            # predict finger  #
            #                 #
            # # # # # # # # # #
                        
            if "finger" in self.loss_names:
            
                # decode finger
                # =======================
                
                if self.use_edges_for_finger in ["average","attention","pool"]:
                
                    # compute [action,object] probabilities
                    embedded_action = self.action_embedder(torch.unsqueeze(data["rhand_main_action_oh"],dim=1))      # [batch, 1,                  128]
                    embedded_object = self.object_embedder(data["obj_ohs"])   # [batch, num_padded_objects, 128]
                    
                    raw_lqk = torch.stack([torch.mm(ea,torch.t(eo)) for ea,eo in zip(embedded_action,embedded_object)])  # [batch, 1, num_padded_objects]
                    raw_lqk = raw_lqk[:,0,:]            # [batch, num_padded_objects]
                    lqk = F.softmax(raw_lqk,dim=1)      # [batch, num_padded_objects]
                    lqk = torch.unsqueeze(lqk,dim=-1)   # [batch, num_padded_objects, 1]
                    raw_rqk = torch.stack([torch.mm(ea,torch.t(eo)) for ea,eo in zip(embedded_action,embedded_object)])  # [batch, 1, num_padded_objects]
                    raw_rqk = raw_rqk[:,0,:]            # [batch, num_padded_objects]
                    rqk = F.softmax(raw_rqk,dim=1)      # [batch, num_padded_objects]
                    rqk = torch.unsqueeze(rqk,dim=-1)   # [batch, num_padded_objects, 1]
                                            
                    # embed wrist xyz
                    att_lrwrist_xyz = pred_wrist_xyz_t1.detach().clone() if "wrist_xyz" in self.loss_names else data["wrist_xyz"][:,i+1] # [batch, 10, 3] or [batch, 2, 3]
                    if "kit_mocap" in self.data_loader:
                        att_lrwrist_xyz = torch.reshape(att_lrwrist_xyz,[self.batch_size,2,5,3])    # [batch, 2, 5, 3]
                        att_lrwrist_xyz = torch.reshape(att_lrwrist_xyz,[self.batch_size,2,-1])     # [batch, 2, 15]
                    elif "kit_rgbd" in self.data_loader:
                        att_lrwrist_xyz = att_lrwrist_xyz                                           # [batch, 2, 3]
                    else:
                        print("Unknown self.data_loader:", self.data_loader)
                        sys.exit()
                    att_lrwrist_xyz = self.hand_encoder2(att_lrwrist_xyz)                           # [batch, 2, 32]
                    att_lrwrist_all = torch.cat([att_lrwrist_xyz,data["wrist_ohs"]],dim=-1)         # [batch, 2, 32 + kit_mocap=22] or [batch, 2, 32 + kit_rgbd=17]
                    att_lwrist_all = torch.unsqueeze(att_lrwrist_all[:,0],dim=1) # [batch, 1, 32 + kit_mocap=22] or [batch, 1, 32 + kit_rgbd=17]
                    att_rwrist_all = torch.unsqueeze(att_lrwrist_all[:,1],dim=1) # [batch, 1, 32 + kit_mocap=22] or [batch, 1, 32 + kit_rgbd=17]
                    
                    # embed object xyz
                    att_obj_xyz = pred_obj_xyz_t1.detach().clone() if "obj_xyz" in self.loss_names else data["obj_xyz"][:,i+1]  # [batch, 8, 4, 3] or [batch, 11, 1, 3] [batch, num_padded_objects
                    att_obj_xyz = torch.reshape(att_obj_xyz,[self.batch_size,self.object_padded_length,-1])                     # [batch, 8, 12] or [batch, num_padded_objects, 3]
                    att_obj_xyz = self.object_encoder2(att_obj_xyz)                                                             # [batch, 8, 32]
                    att_obj_all = torch.cat([att_obj_xyz,data["obj_ohs"]],dim=-1)                                               # [batch, 8, 32 + kit_mocap=22] or [batch, 8, 32 + kit_rgbd=17]
                    
                    # form edges
                    att_lwrist_obj_edges = att_lwrist_all - att_obj_all             # [batch, 8, 32 + 22] or [batch, n, 32 + 17]
                    att_lwrist_obj_edges = att_lwrist_obj_edges * lqk               # [batch, 8, 32 + 22] or [batch, n, 32 + 17]
                    att_rwrist_obj_edges = att_rwrist_all - att_obj_all             # [batch, 8, 32 + 22] or [batch, n, 32 + 17]
                    att_rwrist_obj_edges = att_rwrist_obj_edges * rqk               # [batch, 8, 32 + 22] or [batch, n, 32 + 17]
                    
                    # aggregate the edges
                    if self.use_edges_for_finger == "attention":
                        att_lwrist_obj_edge  = torch.sum(att_lwrist_obj_edges,dim=1)    # [batch, 32 + 22] or [batch, 32 + 17]
                        att_rwrist_obj_edge  = torch.sum(att_rwrist_obj_edges,dim=1)    # [batch, 32 + 22] or [batch, 32 + 17]
                    if self.use_edges_for_finger == "average":
                        att_lwrist_obj_edge  = torch.mean(att_lwrist_obj_edges,dim=1)    # [batch, 32 + 22] or [batch, 32 + 17]
                        att_rwrist_obj_edge  = torch.mean(att_rwrist_obj_edges,dim=1)    # [batch, 32 + 22] or [batch, 32 + 17]
                    if self.use_edges_for_finger == "pool":
                        att_lwrist_obj_edge = F.adaptive_max_pool1d(torch.permute(att_lwrist_obj_edges,[0,2,1]),1)
                        att_lwrist_obj_edge = torch.squeeze(torch.permute(att_lwrist_obj_edge,[0,2,1]))
                        att_rwrist_obj_edge = F.adaptive_max_pool1d(torch.permute(att_rwrist_obj_edges,[0,2,1]),1)
                        att_rwrist_obj_edge = torch.squeeze(torch.permute(att_rwrist_obj_edge,[0,2,1]))
                    
                    # concat
                    att_lrwrist_obj_edge = torch.stack([att_lwrist_obj_edge,att_rwrist_obj_edge],dim=1)     # [batch, 2, 32 + 22]  or [batch, 2, 32 + 17] 
                    # print(pred_rhand_finger_t0.shape) #[32, 2, 42]
                    pred_rhand_finger_t0 = torch.cat([pred_rhand_finger_t0,att_lrwrist_obj_edge],dim=-1)    # [batch, 2, 32 + 22 + 19] or kit_rgbd=[batch, 2, 49 + 42]
                    
                # rnn
                # input = finger at previous timestep
                # output = finger at next timestep
                if self.finger_decoder_type == "rnn":
                    finger_decoder_out    = self.finger_decoder(data=pred_rhand_finger_t0, lh=finger_decoder_lh, rh=finger_decoder_rh)
                    pred_rhand_finger_t1  = finger_decoder_out["out"]
                    finger_decoder_lh     = finger_decoder_out["lh"]
                    finger_decoder_rh     = finger_decoder_out["rh"]
                    pred_rhand_finger_t1  = torch.reshape(pred_rhand_finger_t1,[self.batch_size,2,-1]) # [batch, 2, 19] or [batch, 2, 42]
                    
                elif self.finger_decoder_type == "make_mlp":
                    # print(decoder_h.shape)                                                                                                            # [2, 192,   128] [num_layers, batch*num_objects,       hidden_units]    
                    finger_decoder_inp    = decoder_h[-1]                                                                                               #    [192,   128]             [batch*num_objects,       hidden_units]    
                    finger_decoder_inp    = torch.reshape(finger_decoder_inp,[self.batch_size,obj_wrist_padded_length,self.pose_rnn_decoder_units[1]])  #    [32, 6, 128]             [batch,num_objects,       hidden_units]    
                    finger_decoder_inp    = finger_decoder_inp[:,-2:,:]                                                                                 #    [32, 2, 128]             [batch,left/right wrists, hidden_units]
                    finger_decoder_out    = self.finger_decoder(data=finger_decoder_inp)
                    pred_rhand_finger_t1  = finger_decoder_out["out"]                                                                                   #    [32, 2, 19]              [batch,left/right wrists, 19]
                
                else:
                    print("Unknown self.finger_decoder_type:",self.finger_decoder_type)
                    sys.exit()
                
                if self.predict_finger_velocity:
                    pred_rhand_finger_t1 = pred_rhand_finger_t1 + pred_rhand_finger_t0
                else:
                    pred_rhand_finger_t1 = pred_rhand_finger_t1
                    
                # collect output
                pred_rhand_finger.append(pred_rhand_finger_t1)

        # # # # # # # # #
        #               #
        # collect data  #
        #               #
        # # # # # # # # #
        
        # collect object and wrist
        if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
            pred_obj_xyz       = torch.stack(pred_obj_xyz)                      # [pose_padded_length-1, batch, 3, num_markers,   3]
            pred_obj_xyz_vel   = torch.stack(pred_obj_xyz_vel)                  # [pose_padded_length-1, batch, 3, num_markers,   3]
            pred_wrist_xyz     = torch.stack(pred_wrist_xyz)                    # [pose_padded_length-1, batch,    hand_xyz_dims, 3]
            pred_wrist_xyz_vel = torch.stack(pred_wrist_xyz_vel)                # [pose_padded_length-1, batch,    hand_xyz_dims, 3]
            # collect object and wrist velocities
            pred_obj_xyz       = torch.permute(pred_obj_xyz,[1,0,2,3,4])        # [batch, pose_padded_length-1, 3, num_markers,   3]
            pred_obj_xyz_vel   = torch.permute(pred_obj_xyz_vel,[1,0,2,3,4])    # [batch, pose_padded_length-1, 3, num_markers,   3]
            pred_wrist_xyz     = torch.permute(pred_wrist_xyz,[1,0,2,3])        # [batch, pose_padded_length-1,    hand_xyz_dims, 3]
            pred_wrist_xyz_vel = torch.permute(pred_wrist_xyz_vel,[1,0,2,3])    # [batch, pose_padded_length-1,    hand_xyz_dims, 3]
        
            # collect distributions 
            prior_mu        = torch.stack(prior_mu)                 # [pose_padded_length-1, batch*(obj_wrist_padded_length), mu_var_units]
            prior_log_var   = torch.stack(prior_log_var)            # [pose_padded_length-1, batch*(obj_wrist_padded_length), mu_var_units]
            posterior_mu        = torch.stack(posterior_mu)         # [pose_padded_length-1, batch*(obj_wrist_padded_length), mu_var_units]
            posterior_log_var   = torch.stack(posterior_log_var)    # [pose_padded_length-1, batch*(obj_wrist_padded_length), mu_var_units]                
            # reshape distributions
            prior_mu        = torch.permute(torch.reshape(prior_mu,      [self.pose_padded_length-1, self.batch_size, obj_wrist_padded_length, self.pose_mu_var_units[-1]]), [1,0,2,3])            # [batch, pose_padded_length-1, obj_wrist_padded_length, 3]
            prior_log_var   = torch.permute(torch.reshape(prior_log_var, [self.pose_padded_length-1, self.batch_size, obj_wrist_padded_length, self.pose_mu_var_units[-1]]), [1,0,2,3])            # [batch, pose_padded_length-1, obj_wrist_padded_length, 3]
            posterior_mu        = torch.permute(torch.reshape(posterior_mu,      [self.pose_padded_length-1, self.batch_size, obj_wrist_padded_length, self.pose_mu_var_units[-1]]), [1,0,2,3])    # [batch, pose_padded_length-1, obj_wrist_padded_length, 3]
            posterior_log_var   = torch.permute(torch.reshape(posterior_log_var, [self.pose_padded_length-1, self.batch_size, obj_wrist_padded_length, self.pose_mu_var_units[-1]]), [1,0,2,3])    # [batch, pose_padded_length-1, obj_wrist_padded_length, 3]
        
        # collect body
        if "xyz" in self.loss_names:
        
            # position
            pred_rhand_xyz      = torch.stack(pred_rhand_xyz)                   # [pose_padded_length-1, batch, 53-len(hand_xyz_dims), 3]
            pred_rhand_xyz      = torch.permute(pred_rhand_xyz,[1,0,2,3])       # [batch, pose_padded_length-1, 53-len(hand_xyz_dims), 3]        
        
            # velocity
            pred_rhand_xyz_vel  = torch.stack(pred_rhand_xyz_vel)
            pred_rhand_xyz_vel  = torch.permute(pred_rhand_xyz_vel,[1,0,2,3])
        
            if "kit_mocap" in self.data_root:
                # ONLY FOR KIT_MOCAP DATASET
                # merge body with wrist
                # unfortunately needs to be done manually
                wrist         = pred_wrist_xyz.detach().clone() if "wrist_xyz" in self.loss_names else data["wrist_xyz"][:,1:] # do not take the first timestep
                full_pred_rhand_xyz = pred_rhand_xyz[:,:,0:13]
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, wrist[:,:,0:3]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, pred_rhand_xyz[:,:,13:22]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, wrist[:,:,3:5]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, pred_rhand_xyz[:,:,22:32]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, wrist[:,:,5:8]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, pred_rhand_xyz[:,:,32:41]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, wrist[:,:,8:10]),dim=2)
                full_pred_rhand_xyz = torch.cat((full_pred_rhand_xyz, pred_rhand_xyz[:,:,41:43]),dim=2)
                pred_rhand_xyz = full_pred_rhand_xyz
                
                wrist_vel   = pred_wrist_xyz_vel.detach().clone() if "wrist_xyz_vel" in self.loss_names else data["wrist_xyz_vel"][:,1:]
                full_pred_rhand_xyz_vel = pred_rhand_xyz_vel[:,:,0:13]
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, wrist_vel[:,:,0:3]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, pred_rhand_xyz_vel[:,:,13:22]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, wrist_vel[:,:,3:5]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, pred_rhand_xyz_vel[:,:,22:32]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, wrist_vel[:,:,5:8]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, pred_rhand_xyz_vel[:,:,32:41]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, wrist_vel[:,:,8:10]),dim=2)
                full_pred_rhand_xyz_vel = torch.cat((full_pred_rhand_xyz_vel, pred_rhand_xyz_vel[:,:,41:43]),dim=2)
                pred_rhand_xyz_vel = full_pred_rhand_xyz_vel
                
            elif "kit_rgbd" in self.data_root:
                pred_rhand_xyz = pred_rhand_xyz

        # collect finger
        if "finger" in self.loss_names:        
            pred_rhand_finger   = torch.stack(pred_rhand_finger)                # [pose_padded_length-1, batch, 2,                     19]
            pred_rhand_finger   = torch.permute(pred_rhand_finger,[1,0,2,3])    # [batch, pose_padded_length-1, 2,                     19]   

        # # # # # # # #
        #             #
        # reform data #
        #             #
        # # # # # # # #
        
        # reform output
        # replace inp and key frame onwards with the ground truth
        reformed_obj_xyz, reformed_wrist_xyz = [], []
        reformed_obj_xyz_vel, reformed_wrist_xyz_vel = [], []
        reformed_prior_mu, reformed_prior_log_var = [], []
        reformed_posterior_mu, reformed_posterior_log_var = [], []
                
        reformed_rhand_xyz, reformed_rhand_xyz_vel = [], []
        reformed_rhand_finger = []
        for i in range(self.batch_size):
            
            # # # # # # # # # # # # # # # # # # # # #
            # reform                                #
            # - ground truth at t=0                 #
            # - ground truth from key frame onwards #
            # # # # # # # # # # # # # # # # # # # # #
            
            inp_frame = 0
            key_frame = data["obj_xyz_unpadded_length"][i]
            
            if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
                # reform obj xyz
                reformed_obj_xyz_i = reform_data(data["obj_xyz"][i], pred_obj_xyz[i], inp_frame, key_frame)
                reformed_obj_xyz_i = reform_obj(data["obj_xyz"][i], reformed_obj_xyz_i, obj_wrist_padded_length-2 - torch.sum(data["obj_ids"][i] == 0))
                reformed_obj_xyz.append(reformed_obj_xyz_i)
                            
                # reform obj xyz vel
                reformed_obj_xyz_vel_i = reform_data(data["obj_xyz_vel"][i], pred_obj_xyz_vel[i], inp_frame, key_frame-1)
                reformed_obj_xyz_vel_i = reform_obj(data["obj_xyz_vel"][i], reformed_obj_xyz_vel_i, obj_wrist_padded_length-2 - torch.sum(data["obj_ids"][i] == 0))
                reformed_obj_xyz_vel.append(reformed_obj_xyz_vel_i)
                
                # reform wrist                    
                reformed_wrist_xyz_i = reform_data(data["wrist_xyz"][i], pred_wrist_xyz[i], inp_frame, key_frame)
                reformed_wrist_xyz.append(reformed_wrist_xyz_i)
                
                # reform wrist xyz vel                   
                reformed_wrist_xyz_vel_i = reform_data(data["wrist_xyz_vel"][i], pred_wrist_xyz_vel[i], inp_frame, key_frame)
                reformed_wrist_xyz_vel.append(reformed_wrist_xyz_vel_i)
            
                # zero distributions from the key_frame onwards
                reformed_prior_mu.append(zero_pad(prior_mu[i], key_frame-1))
                reformed_prior_log_var.append(zero_pad(prior_log_var[i], key_frame-1))
                reformed_posterior_mu.append(zero_pad(posterior_mu[i], key_frame-1))
                reformed_posterior_log_var.append(zero_pad(posterior_log_var[i], key_frame-1))
            
            if "xyz" in self.loss_names:
                # reform body
                reformed_rhand_xyz_i = reform_data(data["xyz"][i], pred_rhand_xyz[i], inp_frame, key_frame)
                reformed_rhand_xyz.append(reformed_rhand_xyz_i)
                
                reformed_rhand_xyz_vel_i = reform_data(data["xyz_vel"][i], pred_rhand_xyz_vel[i], inp_frame, key_frame)
                reformed_rhand_xyz_vel.append(reformed_rhand_xyz_vel_i)
            
            if "finger" in self.loss_names:
                # reform finger
                reformed_rhand_finger_i = reform_data(data["finger"][i], pred_rhand_finger[i], inp_frame, key_frame)
                reformed_rhand_finger.append(reformed_rhand_finger_i)
               
        # # # # # #
        # collect #
        # # # # # #
        
        return_data = {}
        
       # wrist and object motion
        if "obj_xyz" in self.loss_names and "wrist_xyz" in self.loss_names:
            reformed_obj_xyz        = torch.stack(reformed_obj_xyz)
            reformed_wrist_xyz      = torch.stack(reformed_wrist_xyz)
            reformed_obj_xyz_vel    = torch.stack(reformed_obj_xyz_vel)
            reformed_wrist_xyz_vel  = torch.stack(reformed_wrist_xyz_vel)        
            
            reformed_prior_mu           = torch.stack(reformed_prior_mu)
            reformed_prior_log_var      = torch.stack(reformed_prior_log_var)
            reformed_posterior_mu       = torch.stack(reformed_posterior_mu)
            reformed_posterior_log_var  = torch.stack(reformed_posterior_log_var)
            return_data = {"obj_xyz":reformed_obj_xyz,         "wrist_xyz":reformed_wrist_xyz,
                           "obj_xyz_vel":reformed_obj_xyz_vel, "wrist_xyz_vel":reformed_wrist_xyz_vel,
                           "obj_distribution_prior":{"mu":reformed_prior_mu, "log_var":reformed_prior_log_var},
                           "obj_distribution_posterior":{"mu":reformed_posterior_mu, "log_var":reformed_posterior_log_var}
                           }
        
        # xyz
        if "xyz" in self.loss_names:
            reformed_rhand_xyz      = torch.stack(reformed_rhand_xyz)
            reformed_rhand_xyz_vel  = torch.stack(reformed_rhand_xyz_vel)
            return_data = {**return_data, "xyz":reformed_rhand_xyz, "xyz_vel":reformed_rhand_xyz_vel}
        
        # finger
        if "finger" in self.loss_names:
            reformed_rhand_finger = torch.stack(reformed_rhand_finger)
            return_data = {**return_data, "finger":reformed_rhand_finger}
            
            if self.use_edges_for_finger in ["average","attention"]:
                return_data = {**return_data, "lqk":raw_lqk, "rqk":raw_rqk}
                                
        return return_data
