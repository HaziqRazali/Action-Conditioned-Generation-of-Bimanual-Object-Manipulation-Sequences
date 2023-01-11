import os
import sys
import ast
import copy
import json
import time
import math
import torch 
import random
import numpy as np

from scipy import interpolate
from collections import Counter, OrderedDict
       
sys.path.append(os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation","dataloaders"))
from kit_rgbd_main_loader import *
from utils import *
import kit_rgbd_variables as var
           
class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
    
        """
        load data    
        """

        t1 = time.time() 
        data = main_loader(args, dtype)
        t2 = time.time() 
        print("Loading time: {}".format(t2-t1))
        
        # pass all to self
        for key, value in data.__dict__.items():
            setattr(self, key, value)
            
        """
        create indexer                    
        - this is where i discard sequences / subsequences
        - self.sequence_data remains unaffected
        """

        self.indexer = []
        for sequence_idx,sequence_data in enumerate(self.sequence_data):
            # indexer to index sequence_data
            indexer = [{"sequence_idx":sequence_idx, "main_action":sequence_data["metadata"]["main_action"]}]
            self.indexer.extend(indexer)
        self.data_len = len(self.indexer)
        
        print("Processing time: {}".format(t2-t1))
        print("Num {} samples: {}".format(self.dtype, self.data_len))
        
    def __len__(self):
        
        # X or 32
        return max(len(self.indexer),self.batch_size)
    
    def __getitem__(self, idx, is_distractor=0):

        # resample a random value if the sampled idx goes beyond data_len. This ensures that it does not matter how I augment the data
        if idx > self.data_len:
            idx = random.randint(0,self.__len__())
            
        # data
        indexer = self.indexer[idx]
        sequence_data = self.sequence_data[indexer["sequence_idx"]]
        
        # path to take
        subject = sequence_data["metadata"]["subject"]
        main_action  = sequence_data["metadata"]["main_action"]
        take    = sequence_data["metadata"]["take"]
        
        # sequence name
        sequence = sequence_data["metadata"]["filename"]
        sequence = sequence.split("/")
        sequence = "_".join([sequence[-3],sequence[-2],sequence[-1]])
                
        # get frames
        frame_data = self.get_frame_data(sequence_data)
        
        # # # # # # # # # # # # #
        # denormalization data  #  
        # # # # # # # # # # # # #
        
        table_center = sequence_data["table_center"]
        angle = sequence_data["angle"] 
        rx    = compute_rotation_matrix(angle * np.pi / 180, "x")
        
        # # # # # # # #
        # camera data #
        # # # # # # # #
        
        cx, cy = np.expand_dims(np.array(sequence_data["camera_data"]["cx"]),axis=0), np.expand_dims(np.array(sequence_data["camera_data"]["cy"]),axis=0)
        fx, fy = np.expand_dims(np.array(sequence_data["camera_data"]["fx"]),axis=0), np.expand_dims(np.array(sequence_data["camera_data"]["fy"]),axis=0)
        
        # # # # # # # # # # # # # #
        # get action data         #  
        # # # # # # # # # # # # # #
        
        lhand_action_data = self.get_action_data(sequence_data, frame_data, "lhand")
        rhand_action_data = self.get_action_data(sequence_data, frame_data, "rhand")
        hand_action_data = {}
        hand_action_data["main_action_id"]  = rhand_action_data["rhand_main_action_id"]
        hand_action_data["main_action_oh"]  = rhand_action_data["rhand_main_action_oh"]
                
        # # # # # # # # # # # # # #
        # get pose and hand data  #
        # # # # # # # # # # # # # #

        # get pose data
        pose_data = self.get_pose_data(sequence_data, frame_data)
        #print(pose_data["xyz"].shape) # [pose_padded_length, num_joints, xy] [150, 15, 2]
        
        # get hand data
        hand_data = self.get_hand_data(sequence_data, frame_data)
        #print(hand_data["finger"].shape) # [pose_padded_length, hands, joints*xy] [150, 2, 42]
                
        # # # # # # # # # # # # # #
        # get object data         #  
        # # # # # # # # # # # # # #
                
        # get object data
        obj_data = self.get_object_data(sequence_data, frame_data, "rhand") 
        
        # # # # # # # #
        # distractors #
        # # # # # # # #
        
        # add distractor
        if is_distractor == 0 and self.add_distractors == 1:
        
            # get the indices for the valid actions
            distractor_main_action = indexer["main_action"]
            actions_with_no_similar_objects = self.actions_with_no_similar_objects[distractor_main_action]
            idxs = [i for i,x in enumerate(self.indexer) if x["main_action"] in actions_with_no_similar_objects]
            
            #sequences_with_no_similar_objects = self.sequences_with_no_similar_objects[idx]       
            sequence_with_no_similar_objects_idx = random.choice(idxs)
            distractors = self.__getitem__(sequence_with_no_similar_objects_idx,is_distractor=1)
            obj_data    = self.merge_distractors(indexer["main_action"], sequence_with_no_similar_objects_idx, obj_data, distractors)
        
        # # # # # # # #
        # return data #
        # # # # # # # #
        
        # only return object data if distractor
        if is_distractor == 0:
            return_data = {# metadata
                           "sequence":sequence, "inp_frame":0,
                           "subject":subject, "main_action":main_action, "take":take,
                           
                           # denormalization data
                           "xyz_scale":self.xyz_scale,
                           "table_center":table_center, "angle":angle, "rx":rx,
                           
                           # camera data
                           "cx":cx, "cy":cy,
                           "fx":fx, "fy":fy,
                           
                           # scale
                           "xyz_scale":self.xyz_scale,
                                    
                           # frame data
                           **frame_data,
                                    
                           # action data
                           **lhand_action_data, **rhand_action_data,
                           **hand_action_data,
                           
                           # pose and hand data
                           **pose_data,
                           **hand_data,
                           
                           # object data
                           **obj_data}
                           
            #sys.exit()
            if self.add_distractors == 1:    
                return_data["distractor_sequence"] = self.sequence_data[self.indexer[sequence_with_no_similar_objects_idx]["sequence_data_index"]]["metadata"]["filename"]
                return_data["distractor_sequence_idx"] = sequence_with_no_similar_objects_idx
        
        else:
            return_data = {**obj_data}

        # do not process distractors
        if is_distractor == 0:
            return_data = self.process_data(return_data)
            return_data = self.flip_data(return_data)
        
            # convert all list of strings into a single string
            for k,v in return_data.items():
                if type(v) == type([]) and type(v[0]) == type("string"):
                    return_data[k] = str(v)
                    
            # convert all array type to float32
            for k,v in return_data.items():
                if type(v) == type(np.array(1)):
                    return_data[k] = return_data[k].astype(np.float32)
        """
        for k,v in return_data.items():
            if type(v) == type(np.array([1])):
                print(k, v.shape)
        print()
        """  
        
        """
        print(sequence_data["metadata"]["filename"])
        print(obj_data["obj_xyz_unpadded_length"], obj_data["obj_xyz"].shape)
        print(obj_data["wrist_xyz_unpadded_length"], obj_data["wrist_xyz"].shape)
        for i in range(obj_data["wrist_xyz_unpadded_length"]):
            print(i)
            print()
            print(obj_data["obj_xyz"][i])
            print()
            print(obj_data["wrist_xyz"][i])
            input()
        sys.exit()
        """
        return return_data
    
    def merge_distractors(self, main_action, idx, real_obj_data_clone, distractor_obj_data_clone):
        
        real_obj_data = dict(real_obj_data_clone)
        distractor_obj_data = dict(distractor_obj_data_clone)
        
        # merge_distractors
        # - (done) remove kitchen_sideboard (table) from distractor_obj_data
        # - (done) set distractor positions throughout time to its initial position
        # - (done) keep name of distractors (this is why the distractors cannot be objects relevant to the action)
        # - (done) unpad number objects before concatenating obj_ids, obj_ohs, obj_xyz, obj_xyz_vel, obj_pos, obj_rot
        # - randomize order
        # - (done) pad them, object_paded_length must satisfy the new total number of objects
                                
        # set distractor positions throughout time to its initial position
        distractor_obj_data["obj_xyz"]     = np.repeat(distractor_obj_data["obj_xyz"][0:1,:,:,:],repeats=self.pose_padded_length,axis=0)
        distractor_obj_data["obj_xyz_vel"] = np.repeat(distractor_obj_data["obj_xyz_vel"][0:1,:,:,:],repeats=self.pose_padded_length,axis=0)
         
        # merged_data dictionary
        merged_data = {}
        
        # # # # # # # # # # # # # # # # #
        # obj_names and obj_mocap_names #
        # # # # # # # # # # # # # # # # #

        merged_data["obj_names"] = real_obj_data["obj_names"] + distractor_obj_data["obj_names"]
        
        # # # # # #
        # obj_ids #
        # # # # # #
        
        # merge
        real_obj_ids        = real_obj_data["obj_ids"][:real_obj_data["obj_ids_unpadded_length"]]               # [n]
        distractor_obj_ids  = distractor_obj_data["obj_ids"][:distractor_obj_data["obj_ids_unpadded_length"]]   # [m]
        merged_obj_ids      = np.concatenate((real_obj_ids,distractor_obj_ids),axis=0)                          # [n+m]
        padded_merged_obj_ids = pad(merged_obj_ids,self.object_padded_length)                                   # [10]
        merged_obj_ids_unpadded_length = real_obj_data["obj_ids_unpadded_length"] + distractor_obj_data["obj_ids_unpadded_length"]
                
        # update dictionary
        merged_data["obj_ids"] = padded_merged_obj_ids
        merged_data["obj_ids_unpadded_length"] = merged_obj_ids_unpadded_length
        
        # does not apply for kit_rgbd
        """
        if len(set(merged_obj_ids)) != len(merged_obj_ids):
            print(main_action, idx, self.supplementary_data[idx]["main_action"])
            print("real_obj_ids:", [self.object_id_to_name[x] for x in real_obj_ids])
            print("distractor_obj_ids:", [self.object_id_to_name[x] for x in distractor_obj_ids])
            print("merged_obj_ids:", [self.object_id_to_name[x] for x in merged_obj_ids])
            sys.exit()
        """    
        #print("obj_ids")
        #print(real_obj_ids, distractor_obj_ids, merged_obj_ids, padded_merged_obj_ids)
        #print(merged_obj_ids_unpadded_length)
        #print()
        
        # # # # # #
        # obj_ohs #
        # # # # # #
        
        # merge
        real_obj_ohs        = real_obj_data["obj_ohs"][:real_obj_data["obj_ohs_unpadded_length"]]               # [n, self.num_obj_wrist_classes]
        distractor_obj_ohs  = distractor_obj_data["obj_ohs"][:distractor_obj_data["obj_ohs_unpadded_length"]]   # [m, self.num_obj_wrist_classes]
        merged_obj_ohs      = np.concatenate((real_obj_ohs,distractor_obj_ohs),axis=0)                          # [n+m, self.num_obj_wrist_classes]
        padded_merged_obj_ohs = pad(merged_obj_ohs,self.object_padded_length)     
        merged_obj_ohs_unpadded_length = real_obj_data["obj_ohs_unpadded_length"] + distractor_obj_data["obj_ohs_unpadded_length"]
        
        # update dictionary
        merged_data["obj_ohs"] = padded_merged_obj_ohs
        merged_data["obj_ohs_unpadded_length"] = merged_obj_ohs_unpadded_length
        
        #print("obj_ohs")
        #print(real_obj_ohs.shape, distractor_obj_ohs.shape, merged_obj_ohs.shape, padded_merged_obj_ohs.shape)
        #print(merged_obj_ohs_unpadded_length)
        #print()
        
        # # # # # #
        # obj_xyz #
        # # # # # #
        
        # merge
        real_obj_xyz        = real_obj_data["obj_xyz"][:,:real_obj_data["obj_xyz_unpadded_objects"]]
        distractor_obj_xyz  = distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]
        merged_obj_xyz      = np.concatenate((real_obj_xyz,distractor_obj_xyz),axis=1)
        padded_merged_obj_xyz = np.transpose(pad(np.transpose(merged_obj_xyz,[1,0,2,3]),self.object_padded_length),[1,0,2,3])
        merged_obj_xyz_unpadded_objects = real_obj_data["obj_xyz_unpadded_objects"] + distractor_obj_data["obj_xyz_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_xyz"] = padded_merged_obj_xyz
        merged_data["obj_xyz_unpadded_length"]  = real_obj_data["obj_xyz_unpadded_length"]
        merged_data["obj_xyz_unpadded_objects"] = merged_obj_xyz_unpadded_objects
        
        #print("obj_xyz")
        #print(distractor_obj_xyz.shape, real_obj_xyz.shape, merged_obj_xyz.shape)
        #print(merged_obj_xyz_unpadded_objects)
        #print()
        
        # # # # # # # #
        # obj_xyz_vel #
        # # # # # # # #
        
        # merge
        real_obj_xyz_vel        = real_obj_data["obj_xyz_vel"][:,:real_obj_data["obj_xyz_vel_unpadded_objects"]]
        distractor_obj_xyz_vel  = distractor_obj_data["obj_xyz_vel"][:,:distractor_obj_data["obj_xyz_vel_unpadded_objects"]]
        merged_obj_xyz_vel      = np.concatenate((real_obj_xyz_vel,distractor_obj_xyz_vel),axis=1)
        padded_merged_obj_xyz_vel = np.transpose(pad(np.transpose(merged_obj_xyz_vel,[1,0,2,3]),self.object_padded_length),[1,0,2,3])
        merged_obj_xyz_vel_unpadded_objects = real_obj_data["obj_xyz_vel_unpadded_objects"] + distractor_obj_data["obj_xyz_vel_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_xyz_vel"] = padded_merged_obj_xyz_vel
        merged_data["obj_xyz_vel_unpadded_length"]  = real_obj_data["obj_xyz_vel_unpadded_length"]
        merged_data["obj_xyz_vel_unpadded_objects"] = merged_obj_xyz_vel_unpadded_objects
        
        #print("obj_xyz_vel")
        #print(distractor_obj_xyz_vel.shape, real_obj_xyz_vel.shape, merged_obj_xyz_vel.shape)
        #print(merged_obj_xyz_vel_unpadded_objects)
        #print()
        
        # # # # # # # #
        # wrist data  #
        # # # # # # # #
        
        for k,v in real_obj_data.items():
            if "wrist" in k:
                merged_data[k] = real_obj_data[k]
        
        """# # # # # #
        # obj_pos #
        # # # # # #
        
        # merge
        real_obj_pos        = real_obj_data["obj_pos"][:,:real_obj_data["obj_pos_unpadded_objects"]]
        distractor_obj_pos  = distractor_obj_data["obj_pos"][:,:distractor_obj_data["obj_pos_unpadded_objects"]]
        merged_obj_pos      = np.concatenate((real_obj_pos,distractor_obj_pos),axis=1)
        padded_merged_obj_pos = np.transpose(pad(np.transpose(merged_obj_pos,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_obj_pos_unpadded_objects = real_obj_data["obj_pos_unpadded_objects"] + distractor_obj_data["obj_pos_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_pos"] = padded_merged_obj_pos
        merged_data["obj_pos_unpadded_length"]  = real_obj_data["obj_pos_unpadded_length"]
        merged_data["obj_pos_unpadded_objects"] = merged_obj_pos_unpadded_objects
        
        print("obj_pos")
        print(distractor_obj_pos.shape, real_obj_pos.shape, merged_obj_pos.shape)
        print(merged_obj_pos_unpadded_objects)
        print()"""
        
        """# # # # # #
        # obj_rot #
        # # # # # #
        
        # merge
        real_obj_rot        = real_obj_data["obj_rot"][:,:real_obj_data["obj_rot_unpadded_objects"]]
        distractor_obj_rot  = distractor_obj_data["obj_rot"][:,:distractor_obj_data["obj_rot_unpadded_objects"]]
        merged_obj_rot      = np.concatenate((real_obj_rot,distractor_obj_rot),axis=1)
        padded_merged_obj_rot = np.transpose(pad(np.transpose(merged_obj_rot,[1,0,2]),self.object_padded_length),[1,0,2])
        merged_obj_rot_unpadded_objects = real_obj_data["obj_rot_unpadded_objects"] + distractor_obj_data["obj_rot_unpadded_objects"]
        
        # update dictionary
        merged_data["obj_rot"] = padded_merged_obj_rot
        merged_data["obj_rot_unpadded_length"]  = real_obj_data["obj_rot_unpadded_length"]
        merged_data["obj_rot_unpadded_objects"] = merged_obj_rot_unpadded_objects
        
        print("obj_rot")
        print(distractor_obj_rot.shape, real_obj_rot.shape, merged_obj_rot.shape)
        print(merged_obj_rot.shape)"""
        
        """# # # # # # # # #
        # obj_table_rot #
        # # # # # # # # #
        
        merged_data["obj_table_pos"]                    = real_obj_data["obj_table_pos"]
        merged_data["obj_table_pos_unpadded_length"]    = real_obj_data["obj_table_pos_unpadded_length"]
        merged_data["obj_table_rot"]                    = real_obj_data["obj_table_rot"]
        merged_data["obj_table_rot_unpadded_length"]    = real_obj_data["obj_table_rot_unpadded_length"]"""
                
        return merged_data

    def flip_data(self, data):
        
        # get data
        obj_xyz,   obj_xyz_unpadded_length   = data["obj_xyz"],   data["obj_xyz_unpadded_length"]   # [pose_padded_length, num_padded_objects, num_mocap_markers=1, 3]
        wrist_xyz, wrist_xyz_unpadded_length = data["wrist_xyz"], data["wrist_xyz_unpadded_length"] # [pose_padded_length, total_wrist_joints=2, 3]
                
        # unpad
        reversed_obj_xyz    = obj_xyz[:obj_xyz_unpadded_length]
        reversed_wrist_xyz  = wrist_xyz[:wrist_xyz_unpadded_length]
        
        # flip
        reversed_obj_xyz    = reversed_obj_xyz[::-1]
        reversed_wrist_xyz  = reversed_wrist_xyz[::-1]
        
        # pad
        reversed_obj_xyz = pad(reversed_obj_xyz,self.pose_padded_length)
        reversed_wrist_xyz = pad(reversed_wrist_xyz,self.pose_padded_length)
        
        data["reversed_obj_xyz"] = reversed_obj_xyz
        data["reversed_wrist_xyz"] = reversed_wrist_xyz
        return data
    
    def process_data(self, data):
        """
        # compare the values of the noise being added to the dimensions of the object
        rhand_main_action = data["rhand_main_action"] # get action with bowl
        if rhand_main_action == "task_1_k_cooking":
        
            # get name and idx of cutting board
            obj_names = data["obj_names"] # get name of bowl
            obj_ids   = data["obj_ids"]   # get idx of bowl
            print(obj_names, obj_ids)
            if any(["bowl" in obj_name for obj_name in obj_names]):
            #if "bottle_1" in obj_names:
                cutting_board_small_idx = [i for i,obj_name in enumerate(obj_names) if "bowl" in obj_name][0] # obj_names.index("bottle_1")
                cutting_board_small = data["obj_bbox_xyz"][0,cutting_board_small_idx]   
                print(obj_names[cutting_board_small_idx])
                print(cutting_board_small.shape)
                cutting_board_small_min_x, cutting_board_small_max_x = np.min(cutting_board_small[:,0]), np.max(cutting_board_small[:,0])
                cutting_board_small_min_y, cutting_board_small_max_y = np.min(cutting_board_small[:,1]), np.max(cutting_board_small[:,1])
                print("max_x - min_x:",cutting_board_small_max_x - cutting_board_small_min_x)
                print("max_y - min_y:",cutting_board_small_max_y - cutting_board_small_min_y)
        """
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Simple addition of the noise. Either                                                                  #
        # 1) (done) sample noise one time then add it to all objects and wrists throughout the sequence         #
        # 2) (done) sample noise for each object and wrist for the entire sequence                              # 
        #    - the same noise must be added to the wrists and their respective interactees                      #
        # 3) (very tedious) sample noise for the wrists and their respective interactees and at each keyframe   #
        #                                                                                                       #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                
        assert self.noise_add_type == "all_objects" or self.noise_add_type == "each_object"        
        noise_probability = np.random.uniform(low=0.0,high=1.0,size=1)[0]        
        if self.noise_probability >= noise_probability: #if noise_probability >= self.noise_probability:
        
            # 1) sample noise one time then add it to all objects and wrists throughout the sequence
            if self.noise_add_type == "all_objects":
                noise = np.random.normal(0, self.sigma, size=[3]) * self.noise_scale
                noise[-1] = 0
                #if rhand_main_action == "task_1_k_cooking" and len([i for i,obj_name in enumerate(obj_names) if "bowl" in obj_name]) > 0:
                #print("noise", noise)
                #print()

                # # # # # # #
                # get data  #
                # # # # # # #
                
                # get object data
                obj_ids,   obj_ids_unpadded_length   = data["obj_ids"],   data["obj_ids_unpadded_length"]   # 
                obj_xyz,   obj_xyz_unpadded_length   = data["obj_xyz"],   data["obj_xyz_unpadded_length"]   # [pose_padded_length, num_padded_objects, num_mocap_markers=4, 3]
                wrist_xyz, wrist_xyz_unpadded_length = data["wrist_xyz"], data["wrist_xyz_unpadded_length"] # [pose_padded_length, total_wrist_joints=10, 3]
                
                # # # # # # #
                # add noise #
                # # # # # # #
                                               
                # add noise to object
                obj_xyz[:obj_xyz_unpadded_length,:obj_ids_unpadded_length] += noise
                wrist_xyz[:wrist_xyz_unpadded_length] += noise
                
                # # # # # # # # # # # # # 
                # recompute velocities  #
                # # # # # # # # # # # # #
                
                # recompute object velocities
                obj_xyz_vel = np.zeros(obj_xyz.shape)
                obj_xyz_vel[1:] = obj_xyz[1:] - obj_xyz[:-1]
                obj_xyz_vel[data["obj_xyz_unpadded_length"]] = obj_xyz_vel[0]                
                wrist_xyz_vel = np.zeros(wrist_xyz.shape)
                wrist_xyz_vel[1:] = wrist_xyz[1:] - wrist_xyz[:-1]
                wrist_xyz_vel[data["wrist_xyz_unpadded_length"]] = wrist_xyz_vel[0]
                
                # # # # # # # # # # #
                # update dictionary #
                # # # # # # # # # # #
                
                # update object dictionary
                data["obj_xyz"]       = obj_xyz
                data["obj_xyz_vel"]   = obj_xyz_vel
                data["wrist_xyz"]     = wrist_xyz
                data["wrist_xyz_vel"] = wrist_xyz_vel
            
            elif self.noise_add_type == "each_object":
            
                print("Unused!")
                sys.exit()
            
            else:
                print("Unknown noise_add_type:",self.noise_add_type)
                sys.exit()
                
        return data
    
    # # # # # # # # # # # # # # # #
    #                             #
    # Action Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get action data
    def get_action_data(self, sequence_data, frame_data, hand):
                                
        # # # # # # # #
        # main action #
        # # # # # # # #
        
        main_action = sequence_data["metadata"]["main_action"]
        main_action_id = np.array(self.main_action_to_id[main_action])
        main_action_oh = np.squeeze(one_hot(main_action_id, len(self.main_actions)))
                     
        # # # # # # # #
        # fine action #
        # # # # # # # #
        
        start_frame = frame_data["start_frame"]
        end_frame = frame_data["end_frame"]
        timesteps = np.arange(start_frame, end_frame, self.time_step_size).astype(int)
        
        # get fine action at timesteps
        fine_action_ids = sequence_data["segmentation"][hand][timesteps]                                # some entries may contain None 
        fine_action_ids_mask = np.array([0 if x is None else 1 for x in fine_action_ids]).astype(int)   # mask for None. 0 means ignore. 1 means do not ignore.
        fine_action_ids = np.array([0 if x is None else x for x in fine_action_ids]).astype(int)        # convert None to 0 for the one_hot function
        fine_action_ohs = one_hot(fine_action_ids, len(self.fine_actions))
                
        #if np.sum(fine_action_ids_mask) == 0:
        #    print(sequence_data["metadata"]["filename"])
        #    print(frame_data["timesteps"][:frame_data["timesteps_unpadded_length"]])
        #    print(fine_action_ids_mask)
        #    print(fine_action_ids)
        #    input()
                          
        # pad
        fine_action_ids_padded = pad(fine_action_ids, self.pose_padded_length)
        fine_action_ids_mask_padded = pad(fine_action_ids_mask, self.pose_padded_length)
        fine_action_ohs_padded = pad(fine_action_ohs, self.pose_padded_length)
                
        return_data = {# main action
                       "main_action":main_action, "main_action_id":main_action_id, "main_action_oh":main_action_oh,
                       
                       # fine actions
                       "action_ids":fine_action_ids_padded, "action_ids_unpadded_length":fine_action_ids.shape[0],
                       "action_ids_mask":fine_action_ids_mask_padded, "action_ids_mask_unpadded_length":fine_action_ids.shape[0],
                       "action_ohs":fine_action_ohs_padded, "action_ohs_unpadded_length":fine_action_ohs.shape[0]
                       }        
        return_data = {hand+"_"+k:v for k,v in return_data.items()}
        return return_data

    # # # # # # # # # # # # # # # #
    #                             #
    # Frame Processing Functions  #
    #                             #
    # # # # # # # # # # # # # # # #
    
    def get_frame_data(self, sequence_data):
        
        key = "main_3d_objects" # must always be main_3d_objects or all_3d_objects because we want the z dimension
        
        # get start stop frames
        table_pos = sequence_data["table_center"]
        table_z   = table_pos[2]
        
        # get hand bbox
        # note that timesteps[-1] may not != sequence_data["time"][-1]
        timesteps = sequence_data["time"]
        timesteps = np.arange(timesteps[0], timesteps[-1], 1)
        hand_bbox = sample_readings(sequence_data, category=key, items=[x for x in sequence_data[key].keys() if "Hand" in x], x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 2, 3]
        
        # compute distance from hand to table
        z = []
        for t in range(hand_bbox.shape[1]):
            l = np.mean(hand_bbox[0,t],axis=0)[2]
            r = np.mean(hand_bbox[1,t],axis=0)[2]
            hand_closest_to_table_z = max(l,r)
            z.append(hand_closest_to_table_z - table_z)
        z = np.array(z)
        
        # start and end frame
        start_frame_idx = np.argmax(z>-500)
        end_frame_idx   = len(z) - np.argmax(z[::-1]>-500)
        start_frame = sequence_data["time"][start_frame_idx]
        end_frame   = sequence_data["time"][end_frame_idx]
        timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        timesteps_padded = pad(timesteps,self.pose_padded_length)
        #print((end_frame - start_frame)/30, "+ \\")
        #print(timesteps)
        
        # compute the sequence duration
        sequence_duration = int(np.ceil((end_frame - start_frame)/self.time_step_size))
                                
        return_data = {"start_frame_idx":start_frame_idx, "end_frame_idx":end_frame_idx,
                       "start_frame":start_frame, "end_frame":end_frame,
                       "timesteps":timesteps_padded, "timesteps_unpadded_length":timesteps.shape[0],
                       "sequence_duration":sequence_duration}
        #return_data = {hand+"_"+k:v for k,v in return_data.items()}
        return return_data
    
    # # # # # # # # # # # # # # #
    #                           #
    # Pose Processing Functions #
    #                           #
    # # # # # # # # # # # # # # #
    
    def get_pose_data(self, sequence_data, frame_data):
    
        start_frame = frame_data["start_frame"]
        end_frame   = frame_data["end_frame"]
        timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        
        # pose xy data in image space
        pose_xy            = sample_readings(sequence_data, category="person", items=["pose_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [t, 15, 2]
        pose_xy_confidence = sample_readings(sequence_data, category="person", items=["pose_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [t, 15]
        pose_xy = pose_xy[0]
        pose_xy_confidence = pose_xy_confidence[0]
        pose_xy_confidence = np.repeat(pose_xy_confidence[:,:,np.newaxis],2,axis=2)
        
        # pad
        pose_xy_padded            = pad(pose_xy, self.pose_padded_length)
        pose_xy_confidence_padded = pad(pose_xy_confidence, self.pose_padded_length)
            
        # compute velocities
        pose_xy_vel_padded = np.zeros(pose_xy_padded.shape)
        pose_xy_vel_padded[1:] = pose_xy_padded[1:] - pose_xy_padded[:-1]
        pose_xy_vel_padded[pose_xy.shape[0]] = pose_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
                
        return_data = {                       
               "xyz":pose_xy_padded,                           "xyz_unpadded_length":pose_xy.shape[0],
               "xyz_vel":pose_xy_vel_padded,                   "xyz_vel_unpadded_length":pose_xy.shape[0],
               "xyz_confidence":pose_xy_confidence_padded,     "xyz_confidence_unpadded_length":pose_xy_confidence.shape[0],    
               "xyz_vel_confidence":pose_xy_confidence_padded, "xyz_vel_confidence_unpadded_length":pose_xy_confidence.shape[0]   
              }    
              
        return return_data
    
    def get_hand_data(self, sequence_data, frame_data):
    
        start_frame = frame_data["start_frame"]
        end_frame   = frame_data["end_frame"]
        timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        
        # hand xy data in image space
        lhand_xy            = sample_readings(sequence_data, category="person", items=["lhand_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [1, t, 21, 2]
        lhand_xy_confidence = sample_readings(sequence_data, category="person", items=["lhand_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [1, t, 21]
        rhand_xy            = sample_readings(sequence_data, category="person", items=["rhand_2d"], x_name="time", y_name="xy",         timesteps=timesteps, return_dict=False) # [1, t, 21, 2]
        rhand_xy_confidence = sample_readings(sequence_data, category="person", items=["rhand_2d"], x_name="time", y_name="confidence", timesteps=timesteps, return_dict=False) # [1, t, 21]
        
        lhand_xy = lhand_xy[0]                                                          # [t, num_joints=21, dim=2]
        lhand_xy_confidence = lhand_xy_confidence[0]                                    # [t, num_joints=21]
        lhand_xy_confidence = np.repeat(lhand_xy_confidence[:,:,np.newaxis],2,axis=2)   # [t, num_joints=21, dim=2]
        rhand_xy = rhand_xy[0]                                                          # [t, num_joints=21]
        rhand_xy_confidence = rhand_xy_confidence[0]                                    # [t, num_joints=21, dim=2]
        rhand_xy_confidence = np.repeat(rhand_xy_confidence[:,:,np.newaxis],2,axis=2)   # [t, num_joints=21, dim=2]
                              
        # pad
        lhand_xy_padded            = pad(lhand_xy, self.pose_padded_length)
        lhand_xy_confidence_padded = pad(lhand_xy_confidence, self.pose_padded_length)
        rhand_xy_padded            = pad(rhand_xy, self.pose_padded_length)
        rhand_xy_confidence_padded = pad(rhand_xy_confidence, self.pose_padded_length)
        
        # compute velocities
        lhand_xy_vel_padded = np.zeros(lhand_xy_padded.shape)
        lhand_xy_vel_padded[1:] = lhand_xy_padded[1:] - lhand_xy_padded[:-1]
        lhand_xy_vel_padded[lhand_xy.shape[0]] = lhand_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
        
        rhand_xy_vel_padded = np.zeros(rhand_xy_padded.shape)
        rhand_xy_vel_padded[1:] = rhand_xy_padded[1:] - rhand_xy_padded[:-1]
        rhand_xy_vel_padded[rhand_xy.shape[0]] = rhand_xy_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
        
        # follow kit_mocap format
        # [pose_padded_length, hands=2, joints=21, dim=2] -> [pose_padded_length, 2, 42]
        finger_padded = np.stack((lhand_xy_padded,rhand_xy_padded),axis=1)                              # [t, 2, num_joints=21, dim=2]
        finger_padded = np.reshape(finger_padded,[finger_padded.shape[0], finger_padded.shape[1], -1])  # [t, 2, num_joints=21* dim=2]
        finger_vel_padded = np.stack((lhand_xy_vel_padded,rhand_xy_vel_padded),axis=1)
        finger_vel_padded = np.reshape(finger_vel_padded,[finger_vel_padded.shape[0], finger_vel_padded.shape[1], -1])
        
        # [pose_padded_length, hands=2, joints=21] -> [pose_padded_length, 2, 42]
        finger_confidence_padded = np.stack((lhand_xy_confidence_padded,rhand_xy_confidence_padded),axis=1)                                                         # [t, 2, num_joints=21, dim=2]
        finger_confidence_padded = np.reshape(finger_confidence_padded,[finger_confidence_padded.shape[0], finger_confidence_padded.shape[1], -1])                  # [t, 2, num_joints=21* dim=2]
        finger_vel_confidence_padded = np.stack((lhand_xy_confidence_padded,rhand_xy_confidence_padded),axis=1)                                                     # [t, 2, num_joints=21, dim=2]
        finger_vel_confidence_padded = np.reshape(finger_vel_confidence_padded,[finger_vel_confidence_padded.shape[0], finger_vel_confidence_padded.shape[1], -1])  # [t, 2, num_joints=21* dim=2]
                        
        return_data = {
            "finger":finger_padded,                                 "finger_unpadded_length":lhand_xy.shape[0],
            "finger_vel":finger_vel_padded,                         "finger_vel_unpadded_length":lhand_xy.shape[0],
            "finger_confidence":finger_confidence_padded,           "finger_confidence_unpadded_length":lhand_xy_confidence.shape[0],
            "finger_vel_confidence":finger_vel_confidence_padded,   "finger_vel_confidence_unpadded_length":lhand_xy_confidence.shape[0]
            }
        
        """
        return_data = {
            "lhand_xy":lhand_xy_padded,                             "lhand_xy_unpadded_length":lhand_xy.shape[0],
            "lhand_xy_vel":lhand_xy_vel_padded,                     "lhand_xy_vel_unpadded_length":lhand_xy.shape[0],
            "lhand_xy_confidence":lhand_xy_confidence_padded,       "lhand_xy_confidence_unpadded_length":lhand_xy_confidence.shape[0],
            "lhand_xy_vel_confidence":lhand_xy_confidence_padded,   "lhand_xy_vel_confidence_unpadded_length":lhand_xy_confidence.shape[0],
            
            "rhand_xy":rhand_xy_padded,                             "rhand_xy_unpadded_length":rhand_xy.shape[0],
            "rhand_xy_vel":rhand_xy_vel_padded,                     "rhand_xy_vel_unpadded_length":rhand_xy.shape[0],
            "rhand_xy_confidence":rhand_xy_confidence_padded,       "rhand_xy_confidence_unpadded_length":rhand_xy_confidence.shape[0],
            "rhand_xy_vel_confidence":rhand_xy_confidence_padded,   "rhand_xy_vel_confidence_unpadded_length":rhand_xy_confidence.shape[0]
            }
        """
        return return_data
    
    # # # # # # # # # # # # # # # #
    #                             #
    # Object Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get object data
    # - get object meta
    # - object position and rotation
    def get_object_data(self, sequence_data, frame_data, hand):
                
        # object meta data
        object_meta = self.get_object_meta(sequence_data)
                
        start_frame = frame_data["start_frame"]
        end_frame   = frame_data["end_frame"]
        timesteps   = np.arange(start_frame, end_frame, self.time_step_size)
        
        # # # # # # # # # # # # # # # #
        # get object and distractors  # 
        # # # # # # # # # # # # # # # #
        
        key = self.object_type
        
        # object 3d centroid
        object_names = object_meta["obj_names"] #[x for x in sequence_data[key].keys() if "Hand" not in x] # [x for x in sequence_data["metadata"][key] if "Hand" not in x]
        object_bbox  = sample_readings(sequence_data, category=key, items=object_names, x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 2, 3] or [1, t, 2, 3]
        object_bbox  = np.expand_dims(object_bbox,0) if len(object_bbox.shape) == 3 else object_bbox # for wipe action when there is only one object that gets squeezed
        object_pos   = np.mean(object_bbox,axis=2,keepdims=True) # [n, t, 1, 3]
                                
        """if self.num_extra_distractors != -1:
            print(object_meta["obj_names"])
            print(object_meta["obj_ids"])
            print(object_meta["obj_ohs"])
            print(object_bbox.shape)
            print(object_pos.shape)
            sys.exit()"""            
        
        # # # # # # #
        # get hands #
        # # # # # # #
        
        # hand centroids
        hand_names = [x for x in sequence_data[key].keys() if "LeftHand" in x] + [x for x in sequence_data[key].keys() if "RightHand" in x]
        hand_bbox  = sample_readings(sequence_data, category=key, items=hand_names, x_name="time", y_name="bbox", timesteps=timesteps, return_dict=False) # [n, t, 3]
        hand_pos   = np.mean(hand_bbox,axis=2,keepdims=True) # [n, t, 1, 3]

        # # # # # #
        # process #
        # # # # # #
        
        # process hand and object
        table_pos   = sequence_data["table_center"]
        object_bbox = self.process_obj(object_bbox, table_pos=table_pos, pad_object=True,   prefix="obj_bbox")  # print(object_data["rhand_obj_xyz"].shape) # [t, n, 1, 3]
        object_data = self.process_obj(object_pos,  table_pos=table_pos, pad_object=True,   prefix="obj")       # print(object_data["rhand_obj_xyz"].shape) # [t, n, 1, 3]
        hand_data   = self.process_obj(hand_pos,    table_pos=table_pos, pad_object=False,  prefix="wrist")     # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2, 1, 3]
        hand_data["wrist_xyz"]     = np.squeeze(hand_data["wrist_xyz"])                                         # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2,    3]
        hand_data["wrist_xyz_vel"] = np.squeeze(hand_data["wrist_xyz_vel"])                                     # print(hand_data["rhand_wrist_xyz"].shape) # [t, 2,    3]
           
        return_data = {**object_meta, **object_data, **hand_data, **object_bbox}
        return return_data

    """
    def get_extra_distractors(self, task, num_extra_distractors):
        distractor_names = list(set(self.all_objects) - set(self.action_to_objects[task]))
        sampled_distractor_names = [random.choice(distractor_names) for _ in range(num_extra_distractors)]
        return sampled_distractor_names
    """
        
    # get object meta
    # - object names
    # - id
    # - one hot vectors
    def get_object_meta(self, sequence_data):
                
        # len(distractor_names) = 3
        # self.num_extra_distractors = 0
        # -(len(distractor_names) - self.num_extra_distractors) = -3
        
        # len(distractor_names) = 3
        # self.num_extra_distractors = 1
        # -(len(distractor_names) - self.num_extra_distractors) = -2
        
        # # # # # # # # # # # # # # # #
        # get object and distractors  # 
        # # # # # # # # # # # # # # # #
        
        key = self.object_type
        
        # object names and ids
        object_names        = [x for x in sequence_data[key].keys() if "Hand" not in x]
        
        # get distractors
        main_objects = self.metadata.loc[(self.metadata["subject"] == sequence_data["metadata"]["subject"]) & (self.metadata["task"] == sequence_data["metadata"]["main_action"]) & (self.metadata["take"] == sequence_data["metadata"]["take"])]["main_objects"]
        main_objects = main_objects.iloc[0]
        distractor_names = []
        for i,object_name in enumerate(object_names):
            if object_name.split("_")[0] not in main_objects.keys():
                distractor_names.append(object_name)
        
        # object_names without distractors
        object_names = [x for x in object_names if x not in distractor_names]
        
        # select distractors
        if self.num_extra_distractors != -1:
            distractor_names = distractor_names[:self.num_extra_distractors]
            object_names = object_names + distractor_names
        
        """# remove distractors
        if self.num_extra_distractors != -1:
            #print("object_names", object_names)
            #print("main_objects", main_objects.keys())
            #print("distractor_names before deletion", distractor_names)
            distractor_names = distractor_names[-(len(distractor_names) - self.num_extra_distractors):]
            #print("distractor_names after deleteion", distractor_names)
            object_names = [x for x in object_names if x not in distractor_names]
            #print("filtered object_names", object_names)
            #print()"""
            
            
        
        object_ids        = np.array([self.object_name_to_id[object_name.split("_")[0]] for object_name in object_names])
        object_ids_padded = pad(object_ids,self.object_padded_length).astype(int)
        object_ohs_padded = one_hot(object_ids_padded,self.num_obj_wrist_classes)
                
        # # # # # # #
        # get hands # 
        # # # # # # #
        
        # hand names and ids
        hand_names = [x for x in sequence_data[key].keys() if "LeftHand" in x] + [x for x in sequence_data[key].keys() if "RightHand" in x]
        hand_ids   = np.array([self.object_name_to_id[hand_name.split("_")[0]] for hand_name in hand_names])
        hand_ohs   = one_hot(hand_ids,self.num_obj_wrist_classes)
                
                       # object data
        return_data = {"obj_names":object_names,
                       "obj_ids":object_ids_padded, "obj_ids_unpadded_length":object_ids.shape[0], 
                       "obj_ohs":object_ohs_padded, "obj_ohs_unpadded_length":object_ids.shape[0],
                       
                       # hand data
                       "hand_names":hand_names, "wrist_ids":hand_ids, "wrist_ohs":hand_ohs}
        
        """
        if self.use_edges_for_finger in ["attention"]:
            # get the objects handled by the left and right hand
            main_action = sequence_data["metadata"]["main_action"]
            #print(sequence_data["metadata"]["filename"]) /home_nfs/haziq/datasets/kit_mocap/data-sorted-simpleGT-v3/Cut/files_motions_3021/Cut1_c_0_05cm_01.xml
            lqk = [1 if x in self.held_object[main_action]["left"] else 0 for x in object_names]
            rqk = [1 if x in self.held_object[main_action]["right"] else 0 for x in object_names]
            
            filename = sequence_data["metadata"]["filename"]
            if "Transfer" in filename and any([x in filename for x in ignore_board_sequence]):
                lqk = [1 if x in self.held_object[main_action]["left"] and x != "cutting_board_small" else 0 for x in object_names]
            if "Transfer" in filename and any([x in filename for x in ignore_bowl_sequence]):
                lqk = [1 if x in self.held_object[main_action]["left"] and x != "mixing_bowl_green" else 0 for x in object_names]

            if np.sum(lqk) != 1 or np.sum(rqk) != 1:
                print(sequence_data["metadata"]["filename"])
                print(object_names)
                print("lqk:",lqk)
                print("rqk:",rqk)
                sys.exit()
            lqk = lqk.index(1)
            rqk = rqk.index(1)
            return_data = {**return_data, "lqk":lqk, "rqk":rqk}
        """
                       
        return return_data
        
    # process all objects
    # - subtract by table_pos
    # - scale (do not scale table)
    # - add scaled noise
    # - pad time and number of objects
    def process_obj(self, pos, table_pos, pad_object, prefix):

        # pos = [n, t, 1, 3]
        # table_pos = [3]
                
        # subtract by unscaled table_pos at every timestep and maybe by the reference object
        pos = pos - table_pos if table_pos is not None and "3d" in self.object_type else pos     # [n, t, 1, 3]     
        
        # scale
        pos = pos * self.xyz_scale
        #print("----")
        #print(pos)
        #print("----")
                                
        # add noise
        #if self.add_noise:
        #    n, t, _ = pos.shape
        #    pos = np.stack([add_gaussian_noise(pos[i], 0, self.sigma, self.window_length, self.xyz_scale) for i in range(pos.shape[0])]) # [n, t, 3]
        
        """
        if self.add_noise:
            n, t, num_markers, _ = pos.shape
            pos = np.transpose(pos,[0,2,1,3])           # [n-1, num_markers, t, 3]
            pos = np.reshape(pos,[n*num_markers,t,3])   # [n-1* num_markers, t, 3]
            pos = np.stack([add_gaussian_noise(pos[i], 0, self.sigma, self.window_length, self.xyz_scale) for i in range(pos.shape[0])]) # [n-1* num_markers, t, 3]
            pos = np.reshape(pos,[n,num_markers,t,3])   # [n-1, num_markers, t, 3]
            pos = np.transpose(pos,[0,2,1,3])           # [n-1, t, num_markers, 3]
        """
        
        # pad time
        pos_padded = np.stack([pad(x, self.pose_padded_length) for x in pos])   # [n, padded t, num_markers, 3]
        table_pos_padded = pad(table_pos, self.pose_padded_length)              #    [padded t, num_markers, 3]
        
        # pad object
        if pad_object:
            pos_padded = pad(pos_padded, self.object_padded_length) # [padded n, padded t, 3]
        
        # transpose
        pos_padded = np.transpose(pos_padded, (1,0,2,3)) # [padded t, padded n, num_markers, 3]
        """if pos_padded.shape[2] == 2:
            print(prefix)
            print("ERROR")
            print(pos_padded.shape)
            sys.exit()"""
        
        pos_vel_padded = np.zeros(pos_padded.shape)
        pos_vel_padded[1:] = pos_padded[1:] - pos_padded[:-1]
        pos_vel_padded[pos.shape[1]] = pos_vel_padded[0] # <--- set the velocity at the final timestep to the zero vector
            
        return_data = {                       
                       prefix+"_xyz":pos_padded,             prefix+"_xyz_unpadded_length":pos.shape[1],             prefix+"_xyz_unpadded_objects":pos.shape[0],       # obj_pos, obj_pos_unpadded_length
                       prefix+"_xyz_vel":pos_vel_padded,     prefix+"_xyz_vel_unpadded_length":pos.shape[1],         prefix+"_xyz_vel_unpadded_objects":pos.shape[0],   
                       prefix+"_table_pos":table_pos_padded, prefix+"_table_pos_unpadded_length":table_pos.shape[0]                                                     # table pos is not at origin
                      }
        return return_data

# # # # # # # # # # # # #
#                       #
# processing functions  #
#                       #
# # # # # # # # # # # # #

def one_hot(labels, max_label=None):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    one_hot_labels[np.arange(labels.size),labels] = 1
    
    return one_hot_labels
        
"""
def one_hot(labels, max_label=None, return_mask=False):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    mask           = np.ones((labels.size)) if max_label is None else np.ones((labels.size))
    
    for i in range(one_hot_labels.shape[0]):
        if labels[i] != -1:
            one_hot_labels[i,labels[i]] = 1
        else:
            one_hot_labels[i,0] = 1
            mask[i] = 0
    
    # cannot handle None or custom labels
    #one_hot_labels[np.arange(labels.size),labels] = 1
    
    if return_mask == True:
        return one_hot_labels, mask
    else:
        return one_hot_labels
"""

# get the id of the reference object
# - will be none for approach action
def get_reference_object_name(object_names, action):
    return None

# pad data
def pad(data, pad_length, return_unpadded_length=0):

    #print(data.shape)
    # data must be [t, ...]
    
    unpadded_length = data.shape[0]
    if pad_length < unpadded_length:
        print("pad_length too short !")
        print("Pad Length = ", pad_length)
        print("Unpadded Sequence Length =", unpadded_length)
        sys.exit()

    new_shape = [pad_length] + list(data.shape[1:])
    new_shape = tuple(new_shape)
    data_padded = np.zeros(shape=new_shape)
    data_padded[:unpadded_length] = data    
    
    assert np.array_equal(data_padded[:unpadded_length],data)
    assert np.all(data_padded[unpadded_length:] == 0)    
    
    if return_unpadded_length:
        return unpadded_length, data_padded.astype(np.float32) 
    return data_padded.astype(np.float32)

# compute data velocities
def compute_velocities(data, discard_t0=1):
    
    data_vel = np.zeros(data.shape, dtype=data.dtype)
    
    if len(data.shape) == 2:
        
        # data must be [t, num_joints]
        data_vel[1:] = data[1:] - data[:-1]
        if discard_t0 == 1:
            data_vel = data_vel[1:]
        
    elif len(data.shape) == 3:     
    
        # data must be [t, num_joints]
        data_vel[:,1:] = data[:,1:] - data[:,:-1]
        if discard_t0 == 1:
            data_vel = data_vel[:,1:]
        
    else:
        sys.exit("Wrong number of dimensions")
    
    return data_vel

# # # # # # # # # # # #
#                     #
# sampling functions  #
#                     #
# # # # # # # # # # # #
     
# sample the readings at timestep by interpolating the left and right values
def sample_readings(data, category, items, x_name, y_name, timesteps, return_dict=True):

    sampled_readings = {}
    for item in items:
        
        time = data[category][item][x_name]    # [n]
        values = data[category][item][y_name]  # [n,3] or [n,m,3]
        
        yt_list = []
        for t in timesteps:
            
            # get interpolating index
            # https://stackoverflow.com/questions/36275459/find-the-closest-elements-above-and-below-a-given-number
                        
            if t < np.min(time):
                yt_list.append(values[0])
                
            elif t >= np.max(time):
                yt_list.append(values[-1])
                
            else:
                #print("time",time)
                #print("t",t)
                #print("time[time < t]",time[time < t])
                try:
                    # cannot <= and >= else I get the denominator of 0 when I do the interpolation resulting in infinity
                    i1 = np.where(time == time[time <= t].max())[0][0]
                    i2 = np.where(time == time[time > t].min())[0][0]                    
                except:
                    print(time)
                    print("filename=",data["metadata"]["filename"])
                    print("t=",t)
                    print("time.shape=", time.shape)
                    print("values.shape=", values.shape)
                    print("Error here")
                    print(np.where(time == time[time <= t].max())[0][0])
                    print(np.where(time == time[time > t].min())[0][0])
                    sys.exit()
        
                # time.shape    [t]
                # values.shape  [t, num_mocap / num_joint /, 3]        
                # y = joint values
                # x = timestep
                x = np.array([time[i1],time[i2]])               
                y = np.stack([values[i1],values[i2]],axis=-1)
                
                # interpolate 
                f = interpolate.interp1d(x, y)
                yt = f(t)
                yt_list.append(yt)
        sampled_readings[item] = np.array(yt_list)
    
    if return_dict:
        return sampled_readings
    return np.stack([v for k,v in sampled_readings.items()])
    #return np.squeeze(np.stack([v for k,v in sampled_readings.items()]))