import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

class compute_loss(nn.Module):
    def __init__(self, args):
        super(compute_loss, self).__init__()
    
        for key, value in args.__dict__.items():
            setattr(self, key, value)

    # # # # # # # # #
    # forward pass  #
    # # # # # # # # #

    def forward(self, inp_data, out_data, loss_name, loss_function):
    
        # kl divergence
        if any(x == loss_function for x in ["nonvanilla_kl_loss","kd_mse"]):
            loss = eval("self."+loss_function)(out_data["_".join([loss_name,"prior"])], out_data["_".join([loss_name,"posterior"])])
        
        elif any(x == loss_function for x in ["kl_loss"]):
            loss = eval("self."+loss_function)(out_data["_".join([loss_name])])
            
        # l1, mse, soft_cross_entropy
        elif any(x == loss_function for x in ["l1","mse","binary_cross_entropy","cosine_distance"]):
            confidence_mask = 1 if loss_name+"_confidence" not in inp_data else inp_data[loss_name+"_confidence"]
            loss = eval("self."+loss_function)(out_data[loss_name], inp_data[loss_name], confidence_mask)
        
        # soft_cross_entropy
        elif any(x == loss_function for x in ["soft_cross_entropy"]):
            loss = eval("self."+loss_function)(out_data[loss_name], inp_data[loss_name], inp_data[loss_name+"_class_weights"])            
        
        # cross_entropy
        elif loss_function == "cross_entropy":
            loss = eval("self."+loss_function)(out_data[loss_name], inp_data[loss_name])
        
        # padded_cross_entropy
        elif loss_function == "padded_cross_entropy":
            #print(loss_name)
            #print(inp_data["sequence"])
            #print(inp_data["rhand_timesteps"].shape, inp_data["rhand_timesteps"][0])
            mask = inp_data[loss_name+"_mask"] if loss_name+"_mask" in inp_data else None
            loss = eval("self."+loss_function)(out_data[loss_name], inp_data[loss_name], inp_data[loss_name+"_unpadded_length"], mask)
        
        # tree mse
        elif loss_function == "indexed_mse":
            loss = eval("self."+loss_function)(out_data, inp_data, loss_name)
        
        # error
        else:
            print("Unknown loss function:", loss_function)
            sys.exit()
            
        # make sure the final loss is average over the batch size
        losses = torch.sum(loss) / loss.shape[0] if len(loss.shape) != 0 else loss        
        return losses

    # # # # # # # # # #
    # loss functions  #
    # # # # # # # # # #   
        
    # mse knowledge distillation loss
    ####################################################
    def kd_mse(self, data):
        
        true_data = data["true"]
        pred_data = data["pred"]
            
        loss = torch.nn.functional.mse_loss(true_data,pred_data,reduction="none")
        return torch.sum(loss / loss.shape[0])
    
    def cosine_distance(self, pred_data, true_data):
        
        #print(pred_data.shape) # [batch, len, n, 3, 2]
        #print(true_data.shape) # [batch, len, n, 3, 2]
        
        true_data = torch.permute(true_data,[0,1,2,4,3]) # [batch, len, n, 2, 3]
        pred_data = torch.permute(pred_data,[0,1,2,4,3]) # [batch, len, n, 2, 3]        
        loss = 1 - torch.nn.functional.cosine_similarity(true_data,pred_data,dim=-1)
        return torch.sum(loss / loss.shape[0])
    
    # mse loss
    ####################################################
    def mse(self, pred_data, true_data, confidence_mask):

        # data could be one of the following formats
        # [batch, seq len, num joints, data dim = 3]   for xyz joints 
        # [batch, seq len, num joints, data dim = 1]   for kin joints
        # sum over the batch, seq len, num joints, data dim
        # avg over the batch
                
        """
        # sanity check for graph or a2m_obj
        # - the loss should be zero if the object id = 0
        loss = (true_data - pred_data)**2 # [batch, 150, num_obj, num_markers=4, 3]
        #print(inp_data["rhand_obj_ids"])
        #print(inp_data["rhand_obj_xyz_unpadded_length"])
        print(loss.shape)
        print(torch.sum(true_data[0,89:]), torch.sum(pred_data[0,89:]))
        print(torch.sum(loss[0,:,0]))
        print(torch.sum(loss[0,:,1]))
        print(torch.sum(loss[0,:,2]))
        print(torch.sum(loss[0,:,3]))
        sys.exit()
        """
        
        """
        # sanity check for Action2Pose
        # - the loss at the t=0 and t=key_frame-1 onwards must be zero
        # - inp_data["lhand_kin_vel_unpadded_length"] = [19, 30, 11, ...]
        loss = (true_data - pred_data)**2 # [batch, padded_length, num_joints]
        inp_loss = torch.sum(loss[0,0]); print(inp_loss)
        key_loss = torch.sum(loss[0,19-1:]); print(key_loss)
        sys.exit()
        """
        
        """
        # sanity check for a2m or actor
        # - the loss from key_frame onwards must be zero
        # - inp_data["lhand_kin_vel_unpadded_length"] = [19, 30, 11, ...]
        loss = (true_data - pred_data)**2 # [batch, padded_length, num_joints]
        print(loss.shape)
        inp_loss = print(torch.sum(loss[0,0]))
        key_loss = print(torch.sum(loss[0,80:]))
        sys.exit()
        """
                
        return torch.sum(((true_data - pred_data)**2)*confidence_mask) / true_data.shape[0]
    
    # mse loss
    ####################################################
    def time_mse(self, pred_data, true_data):
    
        assert len(pred_data.shape) > 2    
        return torch.sum((true_data - pred_data)**2) / (true_data.shape[0] * true_data.shape[1])
    
    # soft cross entropy loss
    ####################################################
    def soft_cross_entropy(self, pred_data, true_data, weights):

        # pred_data.shape = [batch, num_classes]
        # true_data.shape = [batch, num_classes]
        # weights.shape   = [batch, num_classes]

        # pred_data values are logits i.e. raw (non-normalized) predictions
        log_probs = torch.nn.functional.log_softmax(pred_data, dim=1) # [batch, num_classes] 
        return -(weights * true_data * log_probs).sum() / pred_data.shape[0]

    # cross entropy
    ####################################################  
    def cross_entropy(self, out_data, inp_data):
        
        #print(out_data.shape) # [batch, length, num_classes]
        #print(inp_data.shape) # [batch, length]
                
        batch = out_data.shape[0]
        out_data = torch.reshape(out_data,[-1,out_data.shape[-1]])
        inp_data = torch.reshape(inp_data,[-1]).long()
        
        #print(out_data.shape) # [batch* length, num_classes]
        #print(inp_data.shape) # [batch* length]
        
        loss = torch.nn.functional.cross_entropy(out_data,inp_data,reduction="none") # [batch* length]
        loss = torch.reshape(loss,[batch,-1])                                        # [batch, length]
        
        #print(loss.shape)
        #print(loss[0,32:])
        #sys.exit()
        
        return loss.sum() / loss.shape[0]

    # padded cross entropy
    ####################################################  
    def padded_cross_entropy(self, out_data, inp_data, padded_length, mask):
        
        #print(out_data.shape) # [batch, length, num_classes]
        #print(inp_data.shape) # [batch, length]
        #print(padded_length)  # [batch]
        
        inp_data = inp_data.long()
        losses = []
        for i,length in enumerate(padded_length):    
            loss = torch.nn.functional.cross_entropy(out_data[i],inp_data[i],reduction="none") # [anticipate_length]
            loss[length:] *= 0
                    
            if mask is not None:
                loss *= mask[i]
                #if torch.sum(mask[i]) == 0:
                #    print(i)
                #    print(mask[i])
                #    sys.exit()
                #print(i)
                #print(mask[i])
                    
            if 0: #self.use_exponential_cross_entropy:
                weights = -1 * torch.arange(0,loss.shape[0],dtype=loss.dtype,device=loss.device) # [loss.shape[0]]
                weights = torch.exp(weights)
            else:
                weights = torch.ones(loss.shape,dtype=loss.dtype,device=loss.device)             # [loss.shape[0]]
            losses.append(torch.sum(weights * loss))

        losses = torch.stack(losses) # [batch]
        return torch.mean(losses)
    
    # non vanilla KL loss
    ####################################################  
    def nonvanilla_kl_loss(self, prior, posterior):

        mu1    = prior["mu"]
        sigma1 = prior["log_var"]
        mu2    = posterior["mu"]
        sigma2 = posterior["log_var"]     
        
        """
        # sanity check for action2motion
        # - the loss from key_frame onwards must be zero
        # - inp_data["lhand_kin_vel_unpadded_length"] = [19, 30, 11, ...]
        #mu1 = torch.sum(mu1[0,7:])
        #mu2 = torch.sum(mu2[0,7:])
        #sigma1 = torch.sum(sigma1[0,7:])
        #sigma2 = torch.sum(sigma2[0,7:])
        #print(mu1,sigma1)
        #print(mu2,sigma2)
        loss = 0.5*sigma2 - 0.5*sigma1 + (torch.exp(sigma1) + (mu1-mu2)**2)/(2*torch.exp(sigma2)) - 0.5
        print(torch.sum(loss[0,:]))
        #sys.exit()      
        """
        
        loss = 0.5*sigma2 - 0.5*sigma1 + (torch.exp(sigma1) + (mu1-mu2)**2)/(2*torch.exp(sigma2)) - 0.5
        return torch.sum(loss) / mu1.shape[0]

    # KL Divergence of mu and sigma
    ####################################################    
    def kl_loss(self, out_data):
    
        # inp_data not used
        # 0 - e^0 - 0^2 + 1 = 0        
        kl_loss = -0.5 * (out_data["log_var"] - out_data["log_var"].exp() - out_data["mu"].pow(2) +1) # [batch, num_units]
        kl_loss = torch.sum(kl_loss, dim=1) # [batch]
        return torch.sum(kl_loss) / kl_loss.shape[0]

    # # # # # # # # # # # # # #
    # special loss functions  #
    # # # # # # # # # # # # # #

    # computes the mse at indexed locations
    ####################################################
    def indexed_mse(self, out_data, inp_data, loss_name):
        
        pred_mid_pose = out_data[loss_name] # [batch, 2^self.tree_levels - 1, 21, 3]        
        true_mid_pose = inp_data[loss_name] # [batch, 500, 21, 3]
                
        pose_loss = []
        for i,timestep in enumerate(out_data[loss_name+"_timestep"]):
            pose_loss_i = pred_mid_pose[i] - true_mid_pose[i,timestep]          # [2^self.tree_levels - 1, 21, 3]
            pose_loss_i = torch.sqrt(torch.sum(pose_loss_i ** 2, dim=(-1,-2)))  # [2^self.tree_levels - 1]
            pose_loss_i = torch.mean(pose_loss_i)
            pose_loss.append(pose_loss_i)
        pose_loss = torch.stack(pose_loss)
        return pose_loss
        
