import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"),"tmp")
import re
import cv2
import sys
import time
import json
import torch
import socket
import logging
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

#from torch.nn.functional import cross_entropy, mse_loss, l1_loss

from glob import glob
from pathlib import Path
from collections import OrderedDict, Counter
from tensorboardX import SummaryWriter
from misc.misc import *
from misc.losses import *

torch.manual_seed(1337)

# Import for args
##################################################### 
parser = argparse.ArgumentParser()
parser.add_argument('--args', required=True, type=str)
parser.add_argument('--config_file', required=True, type=str)
args, unknown = parser.parse_known_args()
args_import = "from {} import *".format(args.args)
exec(args_import)
args = argparser(args)

# Imports for Architecture, Data Loader 
##################################################### 
architecture_import = "from {} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from {} import *".format(args.data_loader)
exec(data_loader_import)
            
# Prepare Data Loaders
##################################################### 
long_dtype, float_dtype = get_dtypes(args)
# load data
va_data = dataloader(args, "val")
# data loader
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

# Prepare Dump File
#####################################################
dump = [None]*int((len(va_data) - len(va_data)%args.batch_size))

# Prepare Network and Optimizers
##################################################### 
net = model(args)
print("Total # of parameters: ", count_parameters(net))
net.type(float_dtype)

# Load weights and initialize checkpoints
##################################################### 
print("Attempting to load from: " + os.path.join(args.weight_root,args.experiment_name))
if os.path.join(args.weight_root,args.experiment_name) is not None and os.path.isdir(os.path.join(args.weight_root,args.experiment_name)):
        
    # load the best epoch for each task
    for epoch_name,layer_names,task_name in zip(args.epoch_names,args.layer_names,args.task_names):
            
        # load best
        if epoch_name == -1:
            #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
            pt_files = os.listdir(os.path.join(args.weight_root,args.experiment_name))
            pt_files = sorted([x for x in pt_files if task_name in x])
            epoch_name = os.path.basename(pt_files[-1])
        
        # load closest
        if type(epoch_name) == type(1337):
            #pt_files = glob(os.path.join(args.weight_root,args.experiment_name,"*"))
            pt_files = os.listdir(os.path.join(args.weight_root,args.experiment_name))
            pt_files = sorted([x for x in pt_files if task_name in x])
            int_pt_files = [int(re.search('best_(.*?).pt', pt_file).group(1)) for pt_file in pt_files]
            epoch_name  = pt_files[int_pt_files.index(min(int_pt_files, key=lambda x:abs(x-epoch_name)))]
            epoch_name  = os.path.basename(epoch_name)
        
        print("Attempting to load from: " + os.path.join(args.weight_root,args.experiment_name,epoch_name), "into", layer_names)
    
        # load checkpoint dictionary
        checkpoint = torch.load(os.path.join(args.weight_root,args.experiment_name,epoch_name))
                
        # load weights
        model_state = checkpoint["model_state"]
        print("model_state.items()")
        for k,v in model_state.items():
            print(k)
        print()

        # not good because "encoder" will also grab "hand_encoder" or "object_encoder" etc
        #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}       

        # not good if the layer name is e.g. main.hand_encoder
        #layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name == k.split(".")[0]}  
        
        # better
        layer_dict = {}
        for k,v in model_state.items():
            for layer_name in layer_names:
                k_split = k.split(".")
                if any([layer_name == k for k in k_split]):
                    layer_dict[k] = v
                    break
        
        """layer_dict = {}
        for k,v in model_state.items():
            for layer_name in layer_names:
                k = k.split(".")[0]
                if layer_name == k:
                    layer_dict[k] = v
                    break
                print(k)
                sys.exit()"""        
        print("=======================")
        print("epoch_name", epoch_name)
        print("layer_dict.keys()")
        for key in layer_dict.keys():
            print(key)
        print()
        net.load_state_dict(layer_dict,strict=args.strict)
    
    print("Model Loaded")
    
else:
    print(os.path.join(args.weight_root,args.experiment_name) + " not found")
    sys.exit() 
   
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, args, mode):        
    # {'human_joints_t0':human_joints_t0, 'human_joints_t1':human_joints_t1, 'object_data':object_data, "key_object":key_object, "frame":frame, "key_frame":key_frame}
    
    assert mode == "val"

    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and type(v) != type([]) else inp_data[k]
        
    # Forward pass
    t1 = time.time()
    out_data = net(inp_data, mode=mode)
    t2 = time.time()
    print("Foward Pass Time: ", 1/(t2-t1), flush=True) 

    # move all to cpu numpy
    #losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
    
    return {"out_data":out_data}
    
# save results 
####################################################  
def save(out, inp, args):
        
    # handle conflicting names
    keys = set(inp.keys()) | set(out.keys())
    for key in keys:
        if key in inp and key in out:
            inp["_".join(["true",key])] = inp[key]
            out["_".join(["pred",key])] = out[key]
            del inp[key]
            del out[key]
    
    # merge dict
    data = {**inp, **out}
        
    # remove items i do not want
    data = {k:v for k,v in data.items() if all([x not in k for x in args.remove])}
        
    # json can only save list
    for k,v in data.items():
        data[k] = data[k].tolist() if isinstance(v, type(np.array([0]))) else data[k]
    #    print(type(v), type(data[k])) # either numpy array or list
    #sys.exit()
    
    # save each frame
    for i in range(len(data["sequence"])):
    
        # create folder
        foldername = os.path.join(args.result_root,args.result_name,data["sequence"][i]) # "/tmp/haziq/datasets/mogaze/humoro/results/" 
        path = Path(foldername)
        path.mkdir(parents=True,exist_ok=True)
                               
        # save filename
        #print(args.result_root, model, data["sequence"][i], str(int(data["inp_frame"][i])).zfill(10)+".json")
        filename = os.path.join(args.result_root,args.result_name,data["sequence"][i],str(int(data["inp_frame"][i])).zfill(10)+".json") # "/tmp/haziq/datasets/mogaze/humoro/results/"
                
        # create json for each frame 
        data_i = {k:v[i] if type(v) == type([]) else v for k,v in data.items()}
        #data_i = {k:v[i] if type(v) == type(np.array([0])) else v for k,v in data.items()}
        #for k,v in data_i.items():
        #    print(k, type(v))
        #print(data_i["sequence"])
        #sys.exit()

        # get all keys with unpadded_length
        variables_to_unpad = [k.replace("_unpadded_length","") for k in data_i.keys() if "unpadded_length" in k]
        for k in variables_to_unpad:
            
            # unpad true
            if any(["true_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["true_"+k] = data_i["true_"+k][:data_i[k+"_unpadded_length"]]

            # unpad pred
            if any(["pred_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["pred_"+k] = data_i["pred_"+k][:data_i[k+"_unpadded_length"]]
            
            # unpad
            if any([k == data_i_key for data_i_key in data_i.keys()]):
                data_i[k] = data_i[k][:data_i[k+"_unpadded_length"]]     
        
        # get all keys with unpadded_objects
        variables_to_unpad = [k.replace("_unpadded_objects","") for k in data_i.keys() if "unpadded_objects" in k]
        for k in variables_to_unpad:
        
            # unpad true
            if any(["true_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["true_"+k] = np.array(data_i["true_"+k])[:,:data_i[k+"_unpadded_objects"]].tolist()

            # unpad pred
            if any(["pred_"+k == data_i_key for data_i_key in data_i.keys()]):
                data_i["pred_"+k] = np.array(data_i["pred_"+k])[:,:data_i[k+"_unpadded_objects"]].tolist()
            
            # unpad
            if any([k == data_i_key for data_i_key in data_i.keys()]):
                data_i[k] = np.array(data_i[k])[:,:data_i[k+"_unpadded_objects"]].tolist()
        
        """
        if args.unpad is not None:
            for unpad in args.unpad:
                unpadded=0
                for k,v in data_i.items():
                    if k == unpad:
                        data_i[unpad] = data_i[unpad][:data_i[unpad+"_unpadded_length"]]
                        unpadded=1
                        #print(k, len(data_i[unpad]))
                    if k == "true_"+unpad:
                        data_i["true_"+unpad] = data_i["true_"+unpad][:data_i[unpad+"_unpadded_length"]]
                        unpadded=1
                        #print(k, len(data_i["true_"+unpad]))
                    if k == "pred_"+unpad:
                        data_i["pred_"+unpad] = data_i["pred_"+unpad][:data_i[unpad+"_unpadded_length"]]
                        unpadded=1
                        #print(k, len(data_i["pred_"+unpad]))
                if unpadded == 0:
                    print("Error unpadding", unpad)
                    print("=========================")
                    print("Available Keys")
                    for k,v in data_i.items():
                        print(k)
                    sys.exit()
        """
        
        """
        if args.unpad is not None:
            for unpad in args.unpad:
            
                try:
                    data_i[unpad] = data_i[unpad][:data_i[unpad+"_unpadded_length"]]
                    
                # handle this for e.g. when
                # unpad = true_mid_pose but its padded_length variable = mid_pose_unpadded_length
                except:
                    if "true" in unpad:
                        data_i[unpad] = data_i[unpad][:data_i[unpad.replace("true_","")+"_unpadded_length"]]
                    elif "pred" in unpad:
                        data_i[unpad] = data_i[unpad][:data_i[unpad.replace("pred_","")+"_unpadded_length"]]
                    else:
                        print("Unimplemented variable")
                        sys.exit()
        """  
        
        """
        for k,v in data_i.items():
            print(k, type(v))
        sys.exit()
        """
        
        #for k in data_i.keys():
        #    print(k)
        #sys.exit()
        
        #sys.exit()
        # write to json
        with open(filename, 'w') as f:
            json.dump(data_i, f)
        
# validation
####################################################  
with torch.no_grad():        
    net.eval()
    va_losses = {}
    for batch_idx, va_data in enumerate(va_loader):
    
        print("Validation batch ", batch_idx, " of ", len(va_loader))
        
        # forward pass
        va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=None,args=args,mode="val")
                
        # save results
        save(va_output["out_data"], va_data, args)
