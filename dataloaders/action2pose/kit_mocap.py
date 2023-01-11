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
from utils import *
from kit_mocap_main_loader import *
import kit_mocap_variables as var
    
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

        #t1 = time.time()       

        """
        extra variables specific to this project
        """
        
        self.held_object = var.held_object

        """
        update object data      
        """

        for i in range(len(self.sequence_data)):
            filename = self.sequence_data[i]["metadata"]["filename"]
        
            for hand in ["lhand","rhand"]:                
                action_list     = self.sequence_data[i]["segmentation"][hand]["action"]
                object_list     = self.sequence_data[i]["segmentation"][hand]["object"]
                object_id_list  = self.sequence_data[i]["segmentation"][hand]["object_id"]
                target_list     = self.sequence_data[i]["segmentation"][hand]["target"]
                target_id_list  = self.sequence_data[i]["segmentation"][hand]["target_id"]
                
                previous_object = None
                for t,(current_action,current_object,current_target) in enumerate(zip(action_list,object_list,target_list)):
                    
                    # current action is such that hand should not be holding onto anything
                    # - Idle, Approach, Retreat
                    if current_action in ["Idle","Approach","Retreat"]:
                    
                        object_list[t]      = None
                        object_id_list[t]   = -1
                        previous_object     = None
                        
                        # look for target if hand approaching an object
                        if current_action == "Approach" and current_target is not None:
                            previous_object = current_target
                        
                        # (only happens if the ground truth is erroneous for the current action but correct later on)
                        if current_object is not None:
                            previous_object = current_object                            
                        
                    # current action is such that hand should be holding onto something
                    # - Hold + Cucumber, Cut + Knife
                    elif current_action in self.main_actions + ["Move", "Hold", "Place"]:
                    
                        # if ground truth is erroneous and says that the hand is not holding anything
                        if current_object is None:
                            # get the object the hand was previously holding onto
                            if previous_object is not None:
                                object_list[t]      = previous_object
                                object_id_list[t]   = self.object_name_to_id[previous_object]
                            # assert if nothing
                            else:
                                print("Hand is supposed to be holding onto something but the ground truth says otherwise and previous_object = {}!".format(previous_object))
                                print(filename)
                                print("{} {}".format(hand, object_list))
                                sys.exit()
                        
                        # if hand holding onto something
                        # - no need to update the list
                        elif current_object is not None:
                            previous_object = current_object
                            
                        else:
                            print("Unknown case 1!")
                            print(filename)
                            print("{} {}".format(hand, object_list))
                            sys.exit()            
                
                # assert
                if len(list(set(object_id_list))) > 2:
                    print("{} {} has incorrect number of interactions!".format(filename, hand))
                    print(object_list)
                    sys.exit()
                    
                # update list
                self.sequence_data[i]["segmentation"][hand]["object"]           = object_list
                self.sequence_data[i]["segmentation"][hand]["object_id_list"]   = object_id_list
        
        """
        create indexer                    
        - this is where i discard sequences / subsequences
        - self.sequence_data remains unaffected
        """
        
        self.indexer = []
        for sequence_idx,sequence_data in enumerate(self.sequence_data):
        
            indexer           = []                                                    
            main_action       = sequence_data["metadata"]["main_action"]
            motion_folder_num = sequence_data["metadata"]["motion_folder_num"]
            
            # min and max time given the input and output lengths
            min_time = self.inp_length*self.time_step_size
            max_time = np.max(np.concatenate([sequence_data["segmentation"][k]["time"] for k,v in sequence_data["segmentation"].items()],axis=0)) - self.out_length*self.time_step_size
                                
            # the list of actions and activity durations
            lhand_time = sequence_data["segmentation"]["lhand"]["time"]
            rhand_time = sequence_data["segmentation"]["rhand"]["time"]
            lhand_action = sequence_data["segmentation"]["lhand"]["action"]
            rhand_action = sequence_data["segmentation"]["rhand"]["action"]
            
            # todo 
            # - (done) do not take last x second of motion for each segment
            # - take only the first frame if the entire segment is shorter than "out_length" so I don't end up discarding the entire segment
            inp_frame = min_time
            break_now = 0
            while inp_frame < max_time:
                
                inp_lhand_action, inp_rhand_action = None, None
                
                # inp_lhand_action label at inp_frame
                for t1,t2,a in zip(lhand_time[:,0],lhand_time[:,1],lhand_action):
                    # must satisfy the following conditions
                    # 1. inp_frame must be between [t1,t2]
                    # 2. inp_frame must be more than "out_length" second from the key_frame
                    # 3. segment must be at least "out_length long"
                    if (inp_frame >= t1 and inp_frame <= t2) and (t2 - inp_frame >= self.time_step_size * self.out_length) and (t2 - t1) >= self.time_step_size * self.out_length:
                        inp_lhand_action = a
                        break
                        
                # inp_rhand_action label at inp_frame
                for t1,t2,a in zip(rhand_time[:,0],rhand_time[:,1],rhand_action):
                    # must satisfy the following conditions
                    # 1. inp_frame must be between [t1,t2]
                    # 2. inp_frame must be more than "out_length" second from the key_frame
                    # 3. segment must be at least "out_length long"
                    if (inp_frame >= t1 and inp_frame <= t2) and (t2 - inp_frame >= self.time_step_size * self.out_length) and (t2 - t1) >= self.time_step_size * self.out_length:
                        inp_rhand_action = a
                        break
                                        
                # collect the sequence_data_id and inp_frame
                if inp_lhand_action is not None and inp_rhand_action is not None:
                    if any([inp_lhand_action == x or inp_rhand_action == x for x in self.main_actions]):
                        
                        if inp_lhand_action in self.main_actions:
                            main_action = inp_lhand_action
                        elif inp_rhand_action in self.main_actions:
                            main_action = inp_rhand_action
                        else:
                            print("Missing main action")
                            sys.exit()
                        indexer.append([sequence_idx, inp_frame, inp_lhand_action, inp_rhand_action, main_action, motion_folder_num])
                                            
                        # only 1 sample per sequence since it makes no difference
                        break_now = 1
                
                # increment inp_frame
                inp_frame += self.resolution
                if break_now:
                    break
                
            # convert to dict
            indexer = [{"sequence_data_index":x[0], "inp_frame":x[1], "inp_lhand_action":x[2], "inp_rhand_action":x[3], "main_action":x[4], "motion_folder_num":x[5]} for x in indexer]
            self.indexer.extend(indexer) 
        self.data_len = len(self.indexer)
        #t2 = time.time()
        
        print("Processing time: {}".format(t2-t1))
        print("Num {} samples: {}".format(self.dtype, self.data_len))
                
    def __len__(self):
        
        # 54 or 32
        return max(len(self.indexer),self.batch_size)
    
    def __getitem__(self, idx, is_distractor=0):

        #idx = idx % self.data_len
        # resample a random value if the sampled idx goes beyond data_len. This ensures that it does not matter how I augment the data
        if idx > self.data_len:
            idx = random.randint(0,self.__len__())
            
        # get the data
        indexer = self.indexer[idx]
        sequence_data = copy.deepcopy(self.sequence_data[indexer["sequence_data_index"]])
        
        # augment data duration
        if 0:
            sequence_data = self.augment_data_duration(sequence_data)
        
        # main action
        main_action = indexer["main_action"]
        
        # sequence filename
        filename = sequence_data["metadata"]["filename"]
        filename = filename.split("/")
        filename = "_".join([filename[-3],filename[-2],filename[-1]])
        filename = os.path.splitext(filename)[0]
        
        # # # # # # # # # # # # # #
        #                         #
        # get input and key frame #
        #                         #
        # # # # # # # # # # # # # #   
        
        # get frame data
        frame_data = self.get_frame_data(sequence_data, hand="rhand", prefix="")
        inp_frame = frame_data["inp_frame"]
        
        # # # # # # # # # # # # # #
        #                         #
        # get action data         #  
        #                         #
        # # # # # # # # # # # # # #
        
        lhand_action_data = self.get_action_data(sequence_data, frame_data, "lhand", "lhand")
        rhand_action_data = self.get_action_data(sequence_data, frame_data, "rhand", "rhand")
        
        # merge
        hand_action_data = {}
        hand_action_data["main_action_id"]  = rhand_action_data["rhand_main_action_id"]
        hand_action_data["main_action_oh"]  = rhand_action_data["rhand_main_action_oh"]
        hand_action_data["action_ids"]      = np.stack([lhand_action_data["lhand_action_ids"],rhand_action_data["rhand_action_ids"]],axis=0)
        #hand_action_data["handled_obj_ids"] = np.stack([lhand_action_data["lhand_handled_obj_ids"],rhand_action_data["rhand_handled_obj_ids"]],axis=0)
                
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #                                                             #
        # get object data                                             # 
        # - xyz, xyz_vel, pos, rot all do not contain the table data  #
        #                                                             #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #        

        # if self.normalize
        # - interactee also needs idle
        # - other objects
        #   - scale individually
        #   - scale using table min max

        obj_data = self.get_object_data(sequence_data, frame_data)
                
        # scale interactee
        # what about the other objects?
        
        # # # # # # # # # # # # # #
        #                         #
        # get human data          #  
        #                         #
        # # # # # # # # # # # # # #

        human_data = self.get_human_data(sequence_data, frame_data, obj_data)
        
        # # # # # # # #
        # distractors #
        # # # # # # # #
                
        # add distractor
        if is_distractor == 0 and self.add_distractors == 1:
        
            # get the indices for the valid actions
            actions_with_no_similar_objects = self.actions_with_no_similar_objects[main_action]
            idxs = [i for i,x in enumerate(self.indexer) if x["main_action"] in actions_with_no_similar_objects]
            
            #sequences_with_no_similar_objects = self.sequences_with_no_similar_objects[idx]       
            sequence_with_no_similar_objects_idx = random.choice(idxs)
            distractors = self.__getitem__(sequence_with_no_similar_objects_idx,is_distractor=1)
            obj_data    = self.merge_distractors(indexer["main_action"], sequence_with_no_similar_objects_idx, obj_data, distractors, debug=0)
            
            # add again
            if self.num_extra_distractors >= 8:
                # get the indices for the valid actions
                actions_with_no_similar_objects = self.actions_with_no_similar_objects[main_action]
                idxs = [i for i,x in enumerate(self.indexer) if x["main_action"] in actions_with_no_similar_objects]
                
                #sequences_with_no_similar_objects = self.sequences_with_no_similar_objects[idx]       
                sequence_with_no_similar_objects_idx = random.choice(idxs)
                distractors = self.__getitem__(sequence_with_no_similar_objects_idx,is_distractor=1)
                obj_data    = self.merge_distractors(indexer["main_action"], sequence_with_no_similar_objects_idx, obj_data, distractors, debug=0)
        
        # # # # # # # #
        # return data #
        # # # # # # # #
        
        # only return object data if distractor
        if is_distractor == 0:
            return_data = {# filename
                           "sequence":filename, "inp_frame":inp_frame, "main_action":main_action,
                           
                           # scale
                           "xyz_scale":self.xyz_scale, "kin_scale":self.kin_scale,
            
                           # frame data
                            **frame_data, 
                            
                           # action data
                            **hand_action_data, 
                            **lhand_action_data, **rhand_action_data,
                            
                           # object data
                            **obj_data, 
                           
                           # human data
                            **human_data}
            if self.add_distractors == 1:    
                return_data["distractor_sequence"] = self.sequence_data[self.indexer[sequence_with_no_similar_objects_idx]["sequence_data_index"]]["metadata"]["filename"]
                return_data["distractor_sequence_idx"] = sequence_with_no_similar_objects_idx
        
        else:
            return_data = {**obj_data}
        
        # do not process distractors
        if is_distractor == 0:
        
            # add noise to data
            return_data = self.process_data(return_data)
                        
            # flip data
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
                
        return return_data
        
    """        
    def normalize_data(self, data):
        
        # use limits to scale distractors
        # 
        
        print(data["sequence"])
        print(data["obj_xyz"].shape) # [t, num_objects, num_markers=4, 3]
        print(data["obj_ids"])
        print(data["lhand_interactee_id"], data["rhand_interactee_id"])
        
        limits = self.stats["limits"]
        
        # # # # # # # # # # # # #
        # scale distractor xyz  #
        # # # # # # # # # # # # #
        
        # get distractor idxs
        lhand_distractor_idxs = np.where(data["obj_ids"] != data["lhand_interactee_id"])[0]
        rhand_distractor_idxs = np.where(data["obj_ids"] != data["rhand_interactee_id"])[0]
        distractor_idxs = np.intersect1d(lhand_distractor_idxs, rhand_distractor_idxs)
        
        # scale distractor xyz
        for idx in distractor_idxs:
            data["obj_xyz"][:,idx,:,0] = normalize(data["obj_xyz"][:,idx,:,0],limits[0],limits[1])
            data["obj_xyz"][:,idx,:,1] = normalize(data["obj_xyz"][:,idx,:,1],limits[2],limits[3])
            data["obj_xyz"][:,idx,:,2] = normalize(data["obj_xyz"][:,idx,:,2],limits[4],limits[5])
        
        # # # # # # # # # # # # #
        # scale interactee xyz  #
        # # # # # # # # # # # # #
        
        # get interactee idxs
        lhand_interactee_idxs = np.where(data["obj_ids"] == data["lhand_interactee_id"])[0]
        rhand_interactee_idxs = np.where(data["obj_ids"] == data["rhand_interactee_id"])[0]
        rhand_interactee_idxs = np.array([x for x in rhand_interactee_idxs if x not in lhand_interactee_idxs])
        #interactee_idxs = np.union1d(lhand_interactee_idxs, rhand_interactee_idxs) # cannot use union but what if lhand == rhand. Simply subtract lhand from rhand
        
        print(lhand_interactee_idxs)
        print(rhand_interactee_idxs)
        
        # scale interactee xyz
        for idx in lhand_interactee_idxs:            
            lhand_action_ids = data["lhand_action_ids"][:data["lhand_action_ids_unpadded_length"]]
            for t,lhand_action_id in enumerate(lhand_action_ids):
                main_action = data["main_action"]
                action = self.id_to_action[lhand_action_id]
                print(action)
                print(self.stats.keys())
                print(data["main_action"]])
                print(self.stats[data["main_action"]])
                sys.exit()
                data["obj_xyz"][t,idx,:,0] = normalize(data["obj_xyz"][t,idx,:,0],limits[0],limits[1])
                data["obj_xyz"][t,idx,:,1] = normalize(data["obj_xyz"][t,idx,:,1],limits[2],limits[3])
                data["obj_xyz"][t,idx,:,2] = normalize(data["obj_xyz"][t,idx,:,2],limits[4],limits[5])
        for idx in rhand_interactee_idxs:
            rhand_action_ids = data["rhand_action_ids"][:data["rhand_action_ids_unpadded_length"]]
            for t,rhand_action_id in enumerate(rhand_action_ids):
                action = self.id_to_action[lhand_action_id]
                    
        
        
        print(lhand_interactee_idxs)
        
        print(self.stats["limits"])
        print(self.stats.keys())
                
        print(lhand_distractor_idxs)
        print(rhand_distractor_idxs)
        sys.exit()
    """    
    
    def augment_data_duration(self, data):
        
        # add the same amount as ACTOR
        
        # find which sequence it belongs to
        # e.g. Cut
        main_action = data["metadata"]["main_action"]
        
        # get the augmentation dictionary for it
        augmentation_dict_for_main_action = self.augmentation_dict[main_action]
        
        # get the specific sequence
        augmentation_dict_for_folder = augmentation_dict_for_main_action[str(data["metadata"]["motion_folder_num"])]
        
        # get the augmentation data
        segments_to_grab = augmentation_dict_for_folder[:-1]
        location_to_insert = augmentation_dict_for_folder[1]
        
        """
        - [root]
            - [root][body]
                - [root][body][time]             # [t]
                - [root][body][root_position]    # [t,3]
                - [root][body][root_rotation]    # [t,3]
            - [root][object1]
                - [root][object1][time]          # [t]
                - [root][object1][root_position] # [t,3]
                - [root][object1][root_rotation] # [t,3]
            - [root][object2]
                - [root][object1][time]          # [t]
                - [root][object1][root_position] # [t,3]
                - [root][object1][root_rotation] # [t,3]
        
        - [kinematics]
            - [kinematics][body]
                - [kinematics][body][joint_names]      # [44]
                - [kinematics][body][time]             # [t]
                - [kinematics][body][joint_values]     # [t, 44]
            - [kinematics][lhand]
                - [kinematics][lhand][joint_names]     # [19]
                - [kinematics][lhand][time]            # [t]
                - [kinematics][lhand][joint_values]    # [t, 19]
            - [kinematics][rhand]
                - [kinematics][rhand][joint_names]     # [19]
                - [kinematics][rhand][time]            # [t]
                - [kinematics][rhand][joint_values]    # [t, 19]
        
        - [mocap]
            - [mocap][body]
                - [mocap][body][mocap_names]          # [self.markers_in_use]
                - [mocap][body][time]                 # [t]
                - [mocap][body][mocap_values]         # [t, self.markers_in_use, 3] (instead of [self.markers_in_use, t, 3] since its easier to interpolate)
                
        - [segmentation]
            - [segmentation][lhand]
                - [segmentation][lhand][time]
                - [segmentation][lhand][action]
        """
        
        # get start time and end time of the segments to grab
        # add the extra duration to all [time] data
        
        # get start time and end time of the segments to grab
        print("rhand data")
        for t,a in zip(data["segmentation"]["rhand"]["time"],data["segmentation"]["rhand"]["action"]):
            print(t, a)
        print()
        print("rhand finer data")
        for t,a in zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"]):
            print(t, a)
        print()
        
        # get start time and end time of the main action 
        action_time = [t for t,a in zip(data["segmentation"]["rhand"]["time"],data["segmentation"]["rhand"]["action"]) if a == main_action]
        assert len(action_time) == 1
        action_time = action_time[0]
        
        # find the finer_actions whose timing lie within the action
        finer_start_end_time = [t for t,a in zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"]) if t[0] >= action_time[0] and t[1] <= action_time[1]]
        finer_action = [a for t,a in zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"]) if t[0] >= action_time[0] and t[1] <= action_time[1]]
        print("finer action")
        for t,a in zip(finer_start_end_time,finer_action):
            print(t, a)
        print()
        
        # get the augmentation
        augmentation_actions = [finer_action[i] for i in segments_to_grab]
        augmentation_start_end_time_list = [finer_start_end_time[i] for i in segments_to_grab]
        augmentation_start_end_time = [np.min(augmentation_start_end_time_list), np.max(augmentation_start_end_time_list)]
        augmentation_duration = augmentation_start_end_time[1] - augmentation_start_end_time[0]
        print("before repeat")
        print("augmentation_actions:",augmentation_actions)
        print("augmentation_start_end_time_list:",augmentation_start_end_time_list)
        print("augmentation_start_end_time:",augmentation_start_end_time)
        print("augmentation_duration:",augmentation_duration,"\n")
        
        # maybe repeat
        # - will not work because of how i select the augmentation_idxs
        repeat = 2
        repeated_augmentation_actions = augmentation_actions * repeat        
        repeated_augmentation_start_end_time_list = augmentation_start_end_time_list
        for i in range(repeat-1):
            repeated_augmentation_start_end_time_list = np.concatenate([repeated_augmentation_start_end_time_list, repeated_augmentation_start_end_time_list + augmentation_duration*(i+1)],axis=0)        
        repeated_augmentation_start_end_time = [np.min(repeated_augmentation_start_end_time_list), np.max(repeated_augmentation_start_end_time_list)]
        repeated_augmentation_duration = repeated_augmentation_start_end_time[1] - repeated_augmentation_start_end_time[0]
        print("after repeat")
        print("repeated_augmentation_actions:",repeated_augmentation_actions)
        print("repeated_augmentation_start_end_time_list:",repeated_augmentation_start_end_time_list)
        print("repeated_augmentation_start_end_time:",repeated_augmentation_start_end_time)
        print("repeated_augmentation_duration:",repeated_augmentation_duration,"\n")
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # - sample all the data whose time lie between augmentation_start_end_time  #
        # - insert the data to the sequence                                         #
        # - shift the time variable                                                 #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        for k,v in data["root"].items():
            
            # get the indices where the time is between augmentation_start_end_time
            augmentation_idxs   = np.where((data["root"][k]["time"] >= augmentation_start_end_time[0]) & (data["root"][k]["time"] < augmentation_start_end_time[1]))
            insert_idx          = np.where((data["root"][k]["time"] >= augmentation_start_end_time[0]) & (data["root"][k]["time"] < augmentation_start_end_time[1]))[0][-1]
            
            # get the values where the time is between augmentation_start_end_time
            time          = data["root"][k]["time"][augmentation_idxs]
            root_position = data["root"][k]["root_position"][augmentation_idxs]
            root_rotation = data["root"][k]["root_rotation"][augmentation_idxs]
            
            # repeat
            for i in range(repeat-1):
                time = np.concatenate([time, time + augmentation_duration*(i+1)],axis=0)
            root_position = np.array(list(root_position)*repeat)
            root_rotation = np.array(list(root_rotation)*repeat)
            
            # insert augmentation data
            data["root"][k]["root_position"] = np.concatenate([data["root"][k]["root_position"][:insert_idx], root_position, data["root"][k]["root_position"][insert_idx:]],axis=0)
            data["root"][k]["root_rotation"] = np.concatenate([data["root"][k]["root_rotation"][:insert_idx], root_rotation, data["root"][k]["root_rotation"][insert_idx:]],axis=0)
            
            # shift time data
            data["root"][k]["time"] = np.concatenate([data["root"][k]["time"][:insert_idx], time + augmentation_duration, repeated_augmentation_duration + data["root"][k]["time"][insert_idx:]],axis=0) # [t + augmentation_duration for t in data["root"][k]["time"][:index]]
        
        for k,v in data["joint"].items():
            
            # get the indices where the time is between augmentation_start_end_time
            augmentation_idxs   = np.where((data["joint"][k]["time"] >= augmentation_start_end_time[0]) & (data["joint"][k]["time"] < augmentation_start_end_time[1]))
            insert_idx          = np.where((data["joint"][k]["time"] >= augmentation_start_end_time[0]) & (data["joint"][k]["time"] < augmentation_start_end_time[1]))[0][-1]
            
            # get the values where the time is between augmentation_start_end_time
            time         = data["joint"][k]["time"][augmentation_idxs]
            joint_values = data["joint"][k]["joint_values"][augmentation_idxs]
            
            # repeat
            for i in range(repeat-1):
                time = np.concatenate([time, time + augmentation_duration*(i+1)],axis=0)
            joint_values = np.array(list(joint_values)*repeat)
                
            # insert augmentation data
            data["joint"][k]["joint_values"] = np.concatenate([data["joint"][k]["joint_values"][:insert_idx], joint_values, data["joint"][k]["joint_values"][insert_idx:]],axis=0)
                        
            # shift time data
            data["joint"][k]["time"] = np.concatenate([data["joint"][k]["time"][:insert_idx], time + augmentation_duration, repeated_augmentation_duration + data["joint"][k]["time"][insert_idx:]],axis=0)
        
        for k,v in data["mocap"].items():
            
            # get the indices where the time is between augmentation_start_end_time
            augmentation_idxs   = np.where((data["mocap"][k]["time"] >= augmentation_start_end_time[0]) & (data["mocap"][k]["time"] < augmentation_start_end_time[1]))
            insert_idx          = np.where((data["mocap"][k]["time"] >= augmentation_start_end_time[0]) & (data["mocap"][k]["time"] < augmentation_start_end_time[1]))[0][-1]
            
            # get the values where the time is between augmentation_start_end_time
            time         = data["mocap"][k]["time"][augmentation_idxs]
            mocap_values = data["mocap"][k]["mocap_values"][augmentation_idxs]
            
            # repeat
            for i in range(repeat-1):
                time = np.concatenate([time, time + augmentation_duration*(i+1)],axis=0)
            mocap_values = np.array(list(mocap_values)*repeat)
            
            # insert augmentation data
            data["mocap"][k]["mocap_values"] = np.concatenate([data["mocap"][k]["mocap_values"][:insert_idx], mocap_values, data["mocap"][k]["mocap_values"][insert_idx:]],axis=0)
            
            # shift time data
            data["mocap"][k]["time"] = np.concatenate([data["mocap"][k]["time"][:insert_idx], time + augmentation_duration, repeated_augmentation_duration + data["mocap"][k]["time"][insert_idx:]],axis=0)
        
        # update the segmentation for both action, time, finer_action, and finer_time
        # - lhand (both normal and finer data)
        #   - find the segment that encompasses the augmentation_start_end_time and extend its end time
        #   - extend subsequent start and end times
        # - rhand (normal data)
        #   - find the segment that encompasses the augmentation_start_end_time and extend its end time
        #   - extend subsequent start and end times
        # - rhand (finer data)
        #   - add the sampled segment at the insert_idx
        #   - extend subsequent start and end times by the augmentation_duration
        
        """
        print("==============================")
        print("lhand finer data before update")
        for t,a in zip(data["segmentation"]["lhand"]["finer_time"],data["segmentation"]["lhand"]["finer_action"]):
            print(t, a)
        print()
        lhand_idx_to_update = [i for i,(t,a) in enumerate(zip(data["segmentation"]["lhand"]["finer_time"],data["segmentation"]["lhand"]["finer_action"])) if t[0] <= augmentation_start_end_time[0] and t[1] >= augmentation_start_end_time[1]][0]
        print("lhand_idx_to_update:", data["segmentation"]["lhand"]["finer_time"][lhand_idx_to_update], data["segmentation"]["lhand"]["finer_action"][lhand_idx_to_update])
        # extend the segment
        # --- hold --- to ------- hold ------
        data["segmentation"]["lhand"]["finer_time"][lhand_idx_to_update][1] += repeated_augmentation_duration
        for i in range(lhand_idx_to_update+1,len(data["segmentation"]["lhand"]["finer_time"])):
            data["segmentation"]["lhand"]["finer_time"][i][0] += repeated_augmentation_duration
            data["segmentation"]["lhand"]["finer_time"][i][1] += repeated_augmentation_duration
        print("lhand finer data after update")
        for t,a in zip(data["segmentation"]["lhand"]["finer_time"],data["segmentation"]["lhand"]["finer_action"]):
            print(t, a)
        print("==============================")
        print()
        """
        
        print("==============================")
        print("lhand data before update")
        for t,a in zip(data["segmentation"]["lhand"]["time"],data["segmentation"]["lhand"]["action"]):
            print(t, a)
        print()
        lhand_idx_to_update = [i for i,(t,a) in enumerate(zip(data["segmentation"]["lhand"]["time"],data["segmentation"]["lhand"]["action"])) if t[0] <= augmentation_start_end_time[0] and t[1] >= augmentation_start_end_time[1]][0]
        print("lhand_idx_to_update:", data["segmentation"]["lhand"]["time"][lhand_idx_to_update], data["segmentation"]["lhand"]["action"][lhand_idx_to_update])
        data["segmentation"]["lhand"]["time"][lhand_idx_to_update][1] += repeated_augmentation_duration
        for i in range(lhand_idx_to_update+1,len(data["segmentation"]["lhand"]["time"])):
            data["segmentation"]["lhand"]["time"][i][0] += repeated_augmentation_duration
            data["segmentation"]["lhand"]["time"][i][1] += repeated_augmentation_duration
        print("lhand data after update")
        for t,a in zip(data["segmentation"]["lhand"]["time"],data["segmentation"]["lhand"]["action"]):
            print(t, a)
        print("==============================")
        print()
        
        print("==============================")
        print("rhand data before update")
        for t,a in zip(data["segmentation"]["rhand"]["time"],data["segmentation"]["rhand"]["action"]):
            print(t, a)
        print()        
        rhand_idx_to_update = [i for i,(t,a) in enumerate(zip(data["segmentation"]["rhand"]["time"],data["segmentation"]["rhand"]["action"])) if t[0] <= augmentation_start_end_time[0] and t[1] >= augmentation_start_end_time[1]][0]
        print("rhand_idx_to_update:", data["segmentation"]["rhand"]["time"][rhand_idx_to_update], data["segmentation"]["rhand"]["action"][rhand_idx_to_update])
        data["segmentation"]["rhand"]["time"][rhand_idx_to_update][1] += repeated_augmentation_duration
        for i in range(rhand_idx_to_update+1,len(data["segmentation"]["rhand"]["time"])):
            data["segmentation"]["rhand"]["time"][i][0] += repeated_augmentation_duration
            data["segmentation"]["rhand"]["time"][i][1] += repeated_augmentation_duration
        print("rhand data before update")
        for t,a in zip(data["segmentation"]["rhand"]["time"],data["segmentation"]["rhand"]["action"]):
            print(t, a)
        print()  
        print("==============================")
        
        """
        print("==============================")
        print("rhand finer data before update")
        for t,a in zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"]):
            print(t, a)
        print() 
        rhand_idx_to_update = [i for i,(t,a) in enumerate(zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"])) if t[1] == augmentation_start_end_time[1]][0]
        print("rhand_idx_to_update:", rhand_idx_to_update)
        # update segmentation data
        data["segmentation"]["rhand"]["finer_action"] = data["segmentation"]["rhand"]["finer_action"][:rhand_idx_to_update+1] + repeated_augmentation_actions + data["segmentation"]["rhand"]["finer_action"][rhand_idx_to_update+1:]
        
        # update time data
        time_immediately_before_augmentation = data["segmentation"]["rhand"]["finer_time"][:rhand_idx_to_update+1][-1][1]
        print("time_immediately_before_augmentation:",time_immediately_before_augmentation)
        augmentation_start_end_time = [t + augmentation_duration for t in augmentation_start_end_time_list]
        data["segmentation"]["rhand"]["finer_time"] = np.concatenate([data["segmentation"]["rhand"]["finer_time"][:rhand_idx_to_update+1], repeated_augmentation_start_end_time_list + augmentation_duration, data["segmentation"]["rhand"]["finer_time"][rhand_idx_to_update+1:] + repeated_augmentation_duration],axis=0)
        print("==============================")
        print("rhand finer data after update")        
        for t,a in zip(data["segmentation"]["rhand"]["finer_time"],data["segmentation"]["rhand"]["finer_action"]):
            print(t, a)
        print() 
        """
        return data
     
    def merge_distractors(self, main_action, idx, real_obj_data_clone, distractor_obj_data_clone, debug=0):
                
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
        distractor_obj_data["obj_pos"]     = np.repeat(distractor_obj_data["obj_pos"][0:1,:,:],repeats=self.pose_padded_length,axis=0)
        distractor_obj_data["obj_rot"]     = np.repeat(distractor_obj_data["obj_rot"][0:1,:,:],repeats=self.pose_padded_length,axis=0)
        
        # set values beyond pad to 0
        distractor_obj_data["obj_xyz"][real_obj_data["obj_xyz_unpadded_length"]:] = 0
        distractor_obj_data["obj_xyz_vel"][real_obj_data["obj_xyz_vel_unpadded_length"]:] = 0
        distractor_obj_data["obj_pos"][real_obj_data["obj_pos_unpadded_length"]:] = 0
        distractor_obj_data["obj_rot"][real_obj_data["obj_rot_unpadded_length"]:] = 0
        
        # limit number of extra distractors
        if self.num_extra_distractors != -1:
            """print("num_extra_distractors",self.num_extra_distractors)
            print(distractor_obj_data["obj_names"])
            print(ast.literal_eval(distractor_obj_data["obj_mocap_names"]))
            print(distractor_obj_data["obj_ids"], distractor_obj_data["obj_ids_unpadded_length"])
            print(distractor_obj_data["obj_ohs"], distractor_obj_data["obj_ohs_unpadded_length"])
            print(distractor_obj_data["obj_xyz"].shape, distractor_obj_data["obj_xyz_unpadded_objects"], np.sum(distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]), np.sum(distractor_obj_data["obj_xyz"][:,distractor_obj_data["obj_xyz_unpadded_objects"]:]))
            print(distractor_obj_data["obj_xyz_vel"].shape, distractor_obj_data["obj_xyz_vel_unpadded_objects"])
            print(distractor_obj_data["obj_pos"].shape, distractor_obj_data["obj_pos_unpadded_objects"])
            print(distractor_obj_data["obj_rot"].shape, distractor_obj_data["obj_rot_unpadded_objects"])"""
                        
            distractor_obj_data["obj_names"] = distractor_obj_data["obj_names"][:self.num_extra_distractors]
            distractor_obj_data["obj_mocap_names"] = ast.literal_eval(distractor_obj_data["obj_mocap_names"])
            distractor_obj_data["obj_mocap_names"] = distractor_obj_data["obj_mocap_names"][:self.num_extra_distractors]
            distractor_obj_data["obj_mocap_names"] = str(distractor_obj_data["obj_mocap_names"])
            distractor_obj_data["obj_ids"][self.num_extra_distractors:] = 0
            distractor_obj_data["obj_ohs"][self.num_extra_distractors:] = 0
            distractor_obj_data["obj_xyz"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_xyz_vel"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_pos"][:,self.num_extra_distractors:] = 0
            distractor_obj_data["obj_rot"][:,self.num_extra_distractors:] = 0
            
            # if the number of extra distractors is lesser than the number of objects currently in the distractor data
            if self.num_extra_distractors < distractor_obj_data["obj_ids_unpadded_length"]:
                distractor_obj_data["obj_ids_unpadded_length"] = self.num_extra_distractors
                distractor_obj_data["obj_ohs_unpadded_length"] = self.num_extra_distractors
                distractor_obj_data["obj_xyz_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_xyz_vel_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_pos_unpadded_objects"] = self.num_extra_distractors
                distractor_obj_data["obj_rot_unpadded_objects"] = self.num_extra_distractors
            
            """print("num_extra_distractors",self.num_extra_distractors)
            print(distractor_obj_data["obj_names"])
            print(ast.literal_eval(distractor_obj_data["obj_mocap_names"]))
            print(distractor_obj_data["obj_ids"], distractor_obj_data["obj_ids_unpadded_length"])
            print(distractor_obj_data["obj_ohs"], distractor_obj_data["obj_ohs_unpadded_length"])
            print(distractor_obj_data["obj_xyz"].shape, distractor_obj_data["obj_xyz_unpadded_objects"], np.sum(distractor_obj_data["obj_xyz"][:,:distractor_obj_data["obj_xyz_unpadded_objects"]]), np.sum(distractor_obj_data["obj_xyz"][:,distractor_obj_data["obj_xyz_unpadded_objects"]:]))
            print(distractor_obj_data["obj_xyz_vel"].shape, distractor_obj_data["obj_xyz_vel_unpadded_objects"])
            print(distractor_obj_data["obj_pos"].shape, distractor_obj_data["obj_pos_unpadded_objects"])
            print(distractor_obj_data["obj_rot"].shape, distractor_obj_data["obj_rot_unpadded_objects"])
            print()"""
        
        # merged_data dictionary
        merged_data = {}
        
        # # # # # # # # # # # # # # # # #
        # obj_names and obj_mocap_names #
        # # # # # # # # # # # # # # # # #

        merged_data["obj_names"] = real_obj_data["obj_names"] + distractor_obj_data["obj_names"]
        merged_data["obj_paths"] = str(ast.literal_eval(real_obj_data["obj_paths"]) + ast.literal_eval(distractor_obj_data["obj_paths"]))
        merged_data["obj_mocap_names"] = str(ast.literal_eval(real_obj_data["obj_mocap_names"]) + ast.literal_eval(distractor_obj_data["obj_mocap_names"])) 
        
        # # # # # #
        # obj_ids #
        # # # # # #
        
        # merge
        real_obj_ids        = real_obj_data["obj_ids"][:real_obj_data["obj_ids_unpadded_length"]]               # [n]
        distractor_obj_ids  = distractor_obj_data["obj_ids"][:distractor_obj_data["obj_ids_unpadded_length"]]   # [m]
        merged_obj_ids      = np.concatenate((real_obj_ids,distractor_obj_ids),axis=0)                               # [n+m]
        padded_merged_obj_ids = pad(merged_obj_ids,self.object_padded_length)                                   # [10]
        merged_obj_ids_unpadded_length = real_obj_data["obj_ids_unpadded_length"] + distractor_obj_data["obj_ids_unpadded_length"]
        
        # update dictionary
        merged_data["obj_ids"] = padded_merged_obj_ids
        merged_data["obj_ids_unpadded_length"] = merged_obj_ids_unpadded_length
        
        # to make sure the objects in the distractor sequence does not exist in the main sequence
        # does not apply when im adding twice or more number of distractors
        if len(set(merged_obj_ids)) != len(merged_obj_ids) and self.num_extra_distractors < 8:
            print(main_action, idx, self.indexer[idx]["main_action"])
            print("real_obj_ids:", [self.object_id_to_name[x] for x in real_obj_ids])
            print("distractor_obj_ids:", [self.object_id_to_name[x] for x in distractor_obj_ids])
            print("merged_obj_ids:", [self.object_id_to_name[x] for x in merged_obj_ids])
            sys.exit()
            
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
        
        # # # # # #
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
        
        #print("obj_pos")
        #print(distractor_obj_pos.shape, real_obj_pos.shape, merged_obj_pos.shape)
        #print(merged_obj_pos_unpadded_objects)
        #print()
        
        # # # # # #
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
        
        #print("obj_rot")
        #print(distractor_obj_rot.shape, real_obj_rot.shape, merged_obj_rot.shape)
        #print(merged_obj_rot.shape)
        
        # # # # # # # # #
        # obj_table_rot #
        # # # # # # # # #
        
        merged_data["obj_table_pos"]                    = real_obj_data["obj_table_pos"]
        merged_data["obj_table_pos_unpadded_length"]    = real_obj_data["obj_table_pos_unpadded_length"]
        merged_data["obj_table_rot"]                    = real_obj_data["obj_table_rot"]
        merged_data["obj_table_rot_unpadded_length"]    = real_obj_data["obj_table_rot_unpadded_length"]
        
        # # # # # #
        # lqk rqk #
        # # # # # #
        if "lqk" in real_obj_data.keys() and "rqk" in real_obj_data.keys():
            merged_data["lqk"]  = real_obj_data["lqk"]
            merged_data["rqk"]  = real_obj_data["rqk"]
        
        if debug == 1:
            print(merged_data["obj_xyz_unpadded_objects"])
        
        return merged_data
    
    def flip_data(self, data):
        
        # get data
        obj_xyz,   obj_xyz_unpadded_length   = data["obj_xyz"],   data["obj_xyz_unpadded_length"]   # [pose_padded_length, num_padded_objects, num_mocap_markers=4, 3]
        wrist_xyz, wrist_xyz_unpadded_length = data["wrist_xyz"], data["wrist_xyz_unpadded_length"] # [pose_padded_length, total_wrist_joints=10, 3]
                
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
    
        # compare the values of the noise being added to the dimensions of the object
        """
        rhand_main_action = data["rhand_main_action"] # get action with bowl
        if rhand_main_action == "Cut":
                
            # get name and idx of cutting board
            obj_names = data["obj_names"] # get name of bowl
            obj_ids   = data["obj_ids"]   # get idx of bowl
            print(obj_names, obj_ids)
            cutting_board_small_idx = obj_names.index("cucumber_attachment")
            cutting_board_small = data["obj_xyz"][0,cutting_board_small_idx]
            print(cutting_board_small_idx)            
            cutting_board_small_min_x, cutting_board_small_max_x = np.min(cutting_board_small[:,0]), np.max(cutting_board_small[:,0])
            cutting_board_small_min_y, cutting_board_small_max_y = np.min(cutting_board_small[:,1]), np.max(cutting_board_small[:,1])
            print("max_x - min_x:",cutting_board_small_max_x - cutting_board_small_min_x)
            print("max_y - max_y:",cutting_board_small_max_y - cutting_board_small_min_y)
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
                #print("noise",noise)
                #print()
                
                # # # # # # #
                # get data  #
                # # # # # # #
                
                # get human data
                pos,       pos_unpadded_length       = data["pos"],       data["pos_unpadded_length"]       # [pose_padded_length, 3]                         
                xyz,       xyz_unpadded_length       = data["xyz"],       data["xyz_unpadded_length"]       # [pose_padded_length, 53, 3]
                
                # get object data
                obj_ids,   obj_ids_unpadded_length   = data["obj_ids"],   data["obj_ids_unpadded_length"]   # 
                obj_pos,   obj_pos_unpadded_length   = data["obj_pos"],   data["obj_pos_unpadded_length"]   # [pose_padded_length, num_padded_objects, 3]
                obj_xyz,   obj_xyz_unpadded_length   = data["obj_xyz"],   data["obj_xyz_unpadded_length"]   # [pose_padded_length, num_padded_objects, num_mocap_markers=4, 3]
                wrist_xyz, wrist_xyz_unpadded_length = data["wrist_xyz"], data["wrist_xyz_unpadded_length"] # [pose_padded_length, total_wrist_joints=10, 3]
                
                # # # # # # #
                # add noise #
                # # # # # # #
                
                # add noise to human
                pos[:pos_unpadded_length] += noise
                xyz[:xyz_unpadded_length] += noise
                                               
                # add noise to object
                obj_pos[:obj_pos_unpadded_length,:obj_ids_unpadded_length] += noise
                obj_xyz[:obj_xyz_unpadded_length,:obj_ids_unpadded_length] += noise
                wrist_xyz[:wrist_xyz_unpadded_length] += noise
                
                # # # # # # # # # # # # # 
                # recompute velocities  #
                # # # # # # # # # # # # #
                
                # recompute human velocities                
                xyz_vel = np.zeros(xyz.shape)
                xyz_vel[1:] = xyz[1:] - xyz[:-1]
                xyz_vel[data["xyz_unpadded_length"]] = xyz_vel[0]   

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
                
                # update human dictionary
                data["pos"]           = pos
                data["xyz"]           = xyz
                data["xyz_vel"]       = xyz_vel
                
                # update object dictionary
                data["obj_pos"]       = obj_pos
                data["obj_xyz"]       = obj_xyz
                data["obj_xyz_vel"]   = obj_xyz_vel
                data["wrist_xyz"]     = wrist_xyz
                data["wrist_xyz_vel"] = wrist_xyz_vel
            
            elif self.noise_add_type == "each_object":
            
                obj_xyz, obj_xyz_unpadded_length = data["obj_xyz"], data["obj_xyz_unpadded_length"]
                wrist_xyz, wrist_xyz_unpadded_length = data["wrist_xyz"], data["wrist_xyz_unpadded_length"]
                #print(obj_ids, obj_ids_unpadded_length)
                #print(obj_xyz.shape, obj_xyz_unpadded_length)
                #print(wrist_xyz.shape, wrist_xyz_unpadded_length)            
                #print(np.sum(obj_xyz[obj_xyz_unpadded_length:]))
                            
                # add noise to object
                noise = np.random.normal(0, self.sigma, size=[obj_ids_unpadded_length,3]) * self.noise_scale
                noise[obj_ids_unpadded_length:] = 0
                noise[:,-1] = 0
                noise = noise[np.newaxis,:,np.newaxis,:]
                obj_xyz[:obj_xyz_unpadded_length,:obj_ids_unpadded_length] += noise
                                
                # add noise to wrist
                noise = np.random.normal(0, self.sigma, size=[2,3]) * self.noise_scale
                noise[:,-1] = 0
                noise = np.repeat(noise, 5, axis=0)
                noise = noise[np.newaxis,:,:]
                wrist_xyz[:wrist_xyz_unpadded_length] += noise
                
                # recompute velocities
                obj_xyz_vel = np.zeros(obj_xyz.shape)
                obj_xyz_vel[1:] = obj_xyz[1:] - obj_xyz[:-1]
                obj_xyz_vel[data["obj_xyz_unpadded_length"]] = obj_xyz_vel[0]
                wrist_xyz_vel = np.zeros(wrist_xyz.shape)
                wrist_xyz_vel[1:] = wrist_xyz[1:] - wrist_xyz[:-1]
                wrist_xyz_vel[data["wrist_xyz_unpadded_length"]] = wrist_xyz_vel[0]
                
                # update dictionary
                data["obj_xyz"]       = obj_xyz
                data["wrist_xyz"]     = wrist_xyz
                data["obj_xyz_vel"]   = obj_xyz_vel
                data["wrist_xyz_vel"] = wrist_xyz_vel
            
            else:
                print("Unknown noise_add_type:",self.noise_add_type)
                sys.exit()
        
        # # # # # # # # # # # # #
        # scale interactee xyz  #
        # # # # # # # # # # # # #
        
        if len(self.main_actions_scale) != 0:
        
            # get interactee idxs
            lhand_interactee_idxs = np.where(data["obj_ids"] == data["lhand_interactee_id"])[0]
            rhand_interactee_idxs = np.where(data["obj_ids"] == data["rhand_interactee_id"])[0]
            rhand_interactee_idxs = np.array([x for x in rhand_interactee_idxs if x not in lhand_interactee_idxs])
            assert len(lhand_interactee_idxs) < 2
            assert len(rhand_interactee_idxs) < 2
                    
            # get the timesteps where the main action is being performed
            # scale interactee xyz
            for idx in lhand_interactee_idxs:   
                        
                # get the timesteps where the main action is being performed
                T = [t for t,hand_action_id in enumerate(data["lhand_action_ids"][:data["lhand_action_ids_unpadded_length"]]) if self.id_to_action[hand_action_id] in self.main_actions]
                # left hand can be a zero if it is holding on to object
                if len(T) == 0:
                    #for t,hand_action_id in enumerate(data["lhand_action_ids"][:data["lhand_action_ids_unpadded_length"]]):
                    #    print(self.id_to_action[hand_action_id])
                    #print(data["sequence"])
                    #print(data["obj_ids"])
                    #print(data["obj_ids"][idx])
                    #print(self.object_id_to_name[data["obj_ids"][idx]])
                    #sys.exit()
                    lhand_interactee_idxs = []
                    continue
                main_action_id = list(set([data["lhand_action_ids"][t] for t in T]))
                if len(main_action_id) != 1:
                    print(data["sequence"])
                    print([self.id_to_action[x] for x in main_action_id])
                    sys.exit()
                main_action_id = main_action_id[0]
                main_action    = self.id_to_action[main_action_id]
                
                interactee_xyz = data["obj_xyz"][T,idx,:,:]                                 # [T,4,3]
                interactee_xyz_center    = np.mean(interactee_xyz,axis=1,keepdims=True)     # [T,1,3]
                interactee_xyz_center_t0 = interactee_xyz_center[0]                         # [  1,3]
                
                # scale the center and compute the difference from the new center to the old center
                scale = self.main_actions_scale[self.main_actions.index(main_action)]
                interactee_xyz_center = interactee_xyz_center - interactee_xyz_center_t0    # [T,1,3]
                interactee_xyz_scaled_center = interactee_xyz_center * scale                # [T,1,3]
                difference = interactee_xyz_scaled_center - interactee_xyz_center           # [T,1,3]
                            
                # add the difference to the mocap markers and the position
                data["obj_xyz"][T,idx,:,:]   = data["obj_xyz"][T,idx,:,:] + difference
                data["obj_pos"][T,idx,:]     = data["obj_pos"][T,idx,:] + np.squeeze(difference)
                data["lhand_interactee_xyz_center_t0"] = interactee_xyz_center_t0
                
            for idx in rhand_interactee_idxs:
                
                # get the timesteps where the main action is being performed
                T = [t for t,hand_action_id in enumerate(data["rhand_action_ids"][:data["rhand_action_ids_unpadded_length"]]) if self.id_to_action[hand_action_id] in self.main_actions]
                if len(T) == 0:
                    continue
                main_action_id = list(set([data["rhand_action_ids"][t] for t in T]))
                if len(main_action_id) != 1:
                    print(data["sequence"])
                    print([self.id_to_action[x] for x in main_action_id])
                    sys.exit()
                main_action_id = main_action_id[0]
                main_action    = self.id_to_action[main_action_id]
                
                interactee_xyz = data["obj_xyz"][T,idx,:,:]                                 # [T,4,3]
                interactee_xyz_center    = np.mean(interactee_xyz,axis=1,keepdims=True)     # [T,1,3]
                interactee_xyz_center_t0 = interactee_xyz_center[0]                         # [  1,3]
                
                # scale the center and compute the difference from the new center to the old center
                scale = self.main_actions_scale[self.main_actions.index(main_action)]
                interactee_xyz_center = interactee_xyz_center - interactee_xyz_center_t0    # [T,1,3]
                interactee_xyz_scaled_center = interactee_xyz_center * scale                # [T,1,3]
                difference = interactee_xyz_scaled_center - interactee_xyz_center           # [T,1,3]
                            
                # add the difference to the mocap markers and the position
                data["obj_xyz"][T,idx,:,:] = data["obj_xyz"][T,idx,:,:] + difference
                data["obj_pos"][T,idx,:]   = data["obj_pos"][T,idx,:] + np.squeeze(difference)
                data["rhand_interactee_xyz_center_t0"] = interactee_xyz_center_t0
            
            if len(lhand_interactee_idxs) == 0:
                data["lhand_interactee_xyz_center_t0"] = np.zeros([1,3],np.float32)
            if len(rhand_interactee_idxs) == 0:
                data["rhand_interactee_xyz_center_t0"] = np.zeros([1,3],np.float32)
                        
            # recompute object velocities
            obj_xyz, obj_xyz_unpadded_length = data["obj_xyz"], data["obj_xyz_unpadded_length"]
            obj_xyz_vel = np.zeros(obj_xyz.shape)
            obj_xyz_vel[1:] = obj_xyz[1:] - obj_xyz[:-1]
            obj_xyz_vel[data["obj_xyz_unpadded_length"]] = obj_xyz_vel[0]
            
            # update dictionary
            data["obj_xyz"]       = obj_xyz
            data["obj_xyz_vel"]   = obj_xyz_vel                   
        return data
    
    def split_sequence(self, sequence_data, frame_data, object_data, hand_human_data, hand):
        
        # for each action segment
        # extract data 
        
        action_segmentations = sequence_data["segmentation"][hand]["time"]
        actions     = sequence_data["segmentation"][hand]["action"]
        action_ids  = sequence_data["segmentation"][hand]["action_id"]
        frames      = frame_data[hand+"_frames"][:frame_data[hand+"_frames_unpadded_length"]]
        
        action_id_list = []
        hand_obj_xyz_padded_list, hand_obj_xyz_unpadded_length_list     = [], []
        hand_wrist_xyz_padded_list, hand_wrist_xyz_unpadded_length_list = [], []
        hand_xyz_padded_list, hand_xyz_unpadded_length_list             = [], []
        hand_finger_padded_list, hand_finger_unpadded_length_list       = [], []
        for i,(action,action_segmentation,action_id) in enumerate(zip(actions,action_segmentations,action_ids)):
                        
            # get the frames ids that lie in between the current segmentation
            t = np.where((frames >= action_segmentation[0]) & (frames < action_segmentation[1]))
            
            # get the data            
            hand_obj_xyz    = object_data[hand+"_obj_xyz"][t]
            hand_wrist_xyz  = hand_human_data[hand+"_wrist_xyz"][t]
            hand_xyz        = hand_human_data[hand+"_xyz"][t]
            hand_finger     = hand_human_data[hand+"_finger"][t]
                        
            # pad
            hand_obj_xyz_padded     = pad(hand_obj_xyz, self.segment_padded_length)
            hand_wrist_xyz_padded   = pad(hand_wrist_xyz, self.segment_padded_length)
            hand_xyz_padded         = pad(hand_xyz, self.segment_padded_length)
            hand_finger_padded      = pad(hand_finger, self.segment_padded_length)
            
            # append
            action_id_list.append(action_id)
            hand_obj_xyz_padded_list.append(hand_obj_xyz_padded)
            hand_wrist_xyz_padded_list.append(hand_wrist_xyz_padded)
            hand_xyz_padded_list.append(hand_xyz_padded)
            hand_finger_padded_list.append(hand_finger_padded)
            
            hand_obj_xyz_unpadded_length_list.append(hand_obj_xyz.shape[0])
            hand_wrist_xyz_unpadded_length_list.append(hand_wrist_xyz.shape[0])
            hand_xyz_unpadded_length_list.append(hand_xyz.shape[0])
            hand_finger_unpadded_length_list.append(hand_finger.shape[0])
                
        action_ids              = np.stack(action_id_list)
        hand_obj_xyz_padded     = np.stack(hand_obj_xyz_padded_list)
        hand_wrist_xyz_padded   = np.stack(hand_wrist_xyz_padded_list)
        hand_xyz_padded         = np.stack(hand_xyz_padded_list)
        hand_finger_padded      = np.stack(hand_finger_padded_list)
        
        hand_obj_xyz_unpadded_length    = np.stack(hand_obj_xyz_unpadded_length_list)
        hand_wrist_xyz_unpadded_length  = np.stack(hand_wrist_xyz_unpadded_length_list)
        hand_xyz_unpadded_length        = np.stack(hand_xyz_unpadded_length_list)
        hand_finger_unpadded_length     = np.stack(hand_finger_unpadded_length_list)
        
        return_data = {hand+"_segment_action_ids":action_ids,
                       hand+"_segment_obj_xyz":hand_obj_xyz_padded,     hand+"_segment_obj_xyz_unpadded_lengths":hand_obj_xyz_unpadded_length,
                       hand+"_segment_wrist_xyz":hand_wrist_xyz_padded, hand+"_segment_wrist_xyz_unpadded_lengths":hand_wrist_xyz_unpadded_length,
                       hand+"_segment_xyz":hand_xyz_padded,             hand+"_segment_xyz_unpadded_lengths":hand_xyz_unpadded_length,
                       hand+"_segment_finger":hand_finger_padded,       hand+"_segment_finger_unpadded_lengths":hand_finger_unpadded_length}
                
        return return_data
        
    # # # # # # # # # # # # # # # #
    #                             #
    # Frame Processing Functions  #
    #                             #
    # # # # # # # # # # # # # # # #

    # get frame data
    # - input frame, key frame, sequence duration
    def get_frame_data(self, sequence_data, hand, prefix):
        
        # time segmentations
        time_segmentations = sequence_data["segmentation"][hand]["time"]
               
        # for now, graph-v2 requires the frames where BOTH hands are holding onto the object
        # self.sample_from == "object_in_both_hands"
        # self.sample_till == "object_in_both_hands"
        
        # get the inp frame
        if self.sample_from == "first_frame":
            inp_frame_row = get_row(time_segmentations, inp_frame)
            inp_frame = time_segmentations[inp_frame_row][0]
        elif self.sample_from == "inp_frame":
            inp_frame = inp_frame
        elif self.sample_from == "full_sequence":
            inp_frame = 0    
        elif self.sample_from == "object_in_both_hands":
            lhand_segmentation_action = sequence_data["segmentation"]["lhand"]["action"]
            rhand_segmentation_action = sequence_data["segmentation"]["rhand"]["action"]
            # get the idx the moment the action is not "Idle","Approach"
            lhand_start_idx = np.min([i for i,segmentation in enumerate(lhand_segmentation_action) if segmentation not in ["Idle","Approach"]])
            rhand_start_idx = np.min([i for i,segmentation in enumerate(rhand_segmentation_action) if segmentation not in ["Idle","Approach"]])
            inp_frame = max(sequence_data["segmentation"]["lhand"]["time"][lhand_start_idx][0], sequence_data["segmentation"]["rhand"]["time"][rhand_start_idx][0])
            #print(sequence_data["segmentation"]["lhand"])
            #print(sequence_data["segmentation"]["rhand"])
            #print(lhand_start_idx, rhand_start_idx)
            #print(sequence_data["segmentation"]["lhand"]["time"][lhand_start_idx], sequence_data["segmentation"]["rhand"]["time"][rhand_start_idx])
            #print(inp_frame)
        else:
            print("Unknown self.sample_from=",self.sample_from)
            sys.exit()
        
        # get the key frame
        if self.sample_till == "full_sequence":
            key_frame = time_segmentations[-1][-1]
        elif self.sample_till == "object_in_both_hands":
            lhand_segmentation_action = sequence_data["segmentation"]["lhand"]["action"]
            rhand_segmentation_action = sequence_data["segmentation"]["rhand"]["action"]
            # get the idx the moment the action is not "Idle","Approach"
            lhand_end_idx = np.max([i for i,segmentation in enumerate(lhand_segmentation_action) if segmentation not in ["Idle","Approach","Retreat"]])
            rhand_end_idx = np.max([i for i,segmentation in enumerate(rhand_segmentation_action) if segmentation not in ["Idle","Approach","Retreat"]])
            key_frame = min(sequence_data["segmentation"]["lhand"]["time"][lhand_end_idx][1], sequence_data["segmentation"]["rhand"]["time"][rhand_end_idx][1])
            #print(sequence_data["segmentation"]["lhand"])
            #print(sequence_data["segmentation"]["rhand"])
            #print(lhand_end_idx, rhand_end_idx)
            #print(sequence_data["segmentation"]["lhand"]["time"][lhand_end_idx], sequence_data["segmentation"]["rhand"]["time"][rhand_end_idx])
            #print(key_frame)
        else:
            print("Unknown self.sample_till=",self.sample_till)
            sys.exit()
            
        # compute the sequence duration
        sequence_duration = int(np.ceil(key_frame - inp_frame)/self.time_step_size)
            
        # get every frame between [inp_frame, key_frame] but note that key_frame != frames[-1]
        frames = np.arange(inp_frame, key_frame, self.time_step_size)
        assert len(frames) > 1
        frames_unpadded_length, frames_padded = pad(frames, self.pose_padded_length, 1)
                       
        return_data = {# inp and key frame and full sequence duration
                       "inp_frame":inp_frame, "key_frame":key_frame, "sequence_duration":sequence_duration,
                        
                       # frames
                       "frames":frames_padded, "frames_unpadded_length":frames_unpadded_length}
        return_data = {prefix+"_"+k:v for k,v in return_data.items()} if len(prefix) > 0 else {k:v for k,v in return_data.items()}
        return return_data
        
    # # # # # # # # # # # # # # # #
    #                             #
    # Action Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get action data
    # - action_ids, handled_object_ids, semantic variation
    def get_action_data(self, sequence_data, frame_data, hand, prefix):

        # get the semantic variation
        #semantic_variation_id = np.array(self.get_semantic_variation(sequence_data))
        #semantic_variation_oh = np.squeeze(one_hot(semantic_variation_id,self.max_semantic_variation)) if semantic_variation_id != -1 else -1
        
        # time segmentations
        time_segmentations = sequence_data["segmentation"][hand]["time"]
        
        # get current fine grained action
        #action, action_id = sequence_data["segmentation"][hand]["action"][inp_frame_row], sequence_data["segmentation"][hand]["action_id"][inp_frame_row]
        
        """
        main_action = sequence_data["metadata"]["main_action"]
        main_action_id = np.array(sequence_data["metadata"]["main_action_id"])
        main_action_oh = np.squeeze(one_hot(main_action_id,len(self.main_actions))) if main_action_id != -1 else -1
        """
          
        # get main action
        if hand == "rhand":
            #main_action, main_action_id = self.get_main_action(sequence_data["segmentation"][hand])
            main_action, main_action_id = self.get_main_action(sequence_data["segmentation"])
            main_action_oh = np.squeeze(one_hot(main_action_id,len(self.main_actions)))
            # for debugging
            if len(main_action_oh.shape) == 0:
                main_action_oh = one_hot(main_action_id,len(self.main_actions))[0]
            #print(main_action_oh.shape, len(main_action_oh.shape), main_action_oh, type(main_action_oh))
            #sys.exit()
            #main_action_oh = np.squeeze(one_hot(main_action_id,self.len_all_actions))
        if hand == "lhand":
            main_action = "lhand"
            main_action_id = -1
            main_action_oh = -1
        
        # get unpadded frames
        frames = frame_data["frames"][:frame_data["frames_unpadded_length"]]
       
        # get fine grained actions at every timestep
        action_ids = sequence_data["segmentation"][hand]["action_id"] # [num_action_segments]
        #print(np.where((frames[0] >= time_segmentations[:,0]) & (frames[0] < time_segmentations[:,1])))                    #(array([0]),)
        #print(np.where((frames[0] >= time_segmentations[:,0]) & (frames[0] < time_segmentations[:,1]))[0])                 #[0]
        #print(action_ids[np.where((frames[0] >= time_segmentations[:,0]) & (frames[0] < time_segmentations[:,1]))[0]])     #[0]
        #print(action_ids[np.where((frames[0] >= time_segmentations[:,0]) & (frames[0] < time_segmentations[:,1]))[0][0]])  #0
        #action_ids = np.array([action_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))][0] for frame in frames])
        action_ids = np.array([action_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))[0][0]] for frame in frames])
        action_ids_padded = pad(action_ids, self.pose_padded_length).astype(int)
        action_ohs_padded = one_hot(action_ids_padded,len(self.all_actions))
        
        """
        # get objects being handled at every timestep
        handled_object_ids = sequence_data["segmentation"][hand]["object_id"]
        interactee_id = list(set([x for x in handled_object_ids if x != -1]))
        assert len(interactee_id) == 1
        interactee_id = interactee_id[0]
        #print(np.where((frames[0] >= time_segmentations[:,0]) & (frames[0] < time_segmentations[:,1]))[0][0]) #0
        handled_object_ids = np.array([handled_object_ids[np.where((frame >= time_segmentations[:,0]) & (frame < time_segmentations[:,1]))[0][0]] for frame in frames])
        handled_object_ids_padded = pad(handled_object_ids, self.pose_padded_length)
        """
        
        return_data = {# main action
                       "main_action":main_action, "main_action_id":main_action_id, "main_action_oh":main_action_oh,
                       
                       # semantic variation
                       #"semantic_variation_id":semantic_variation_id, "semantic_variation_oh":semantic_variation_oh,
                       
                       # action at every timestep
                       "action_ids":action_ids_padded, "action_ids_unpadded_length":len(frames),
                       "action_ohs":action_ohs_padded, "action_ohs_unpadded_length":len(frames)
                       
                       # handled object at every timestep
                       # VERY HELPFUL FOR WHEN I WANT TO DETERMINE WHETHER OR NOT THE HAND IS HOLDING ON TO AN OBJECT
                       #"handled_obj_ids":handled_object_ids_padded,
                       
                       # interactee_id
                       # for prototyping
                       #"interactee_id":interactee_id,                       
                       }        
        return_data = {prefix+"_"+k:v for k,v in return_data.items()} if len(prefix) > 0 else {k:v for k,v in return_data.items()}
        return return_data
    
    # get the smenatic variation
    # - -1 if it doenst exist
    def get_semantic_variation(self, data):
        main_action = data["metadata"]["main_action"]
        motion_folder_num = data["metadata"]["motion_folder_num"]
        
        # no semantic variation for the current action
        if main_action not in self.semantic_variation.keys():
            print(main_action)
            sys.exit()
            return -1
            
        for tuples in self.semantic_variation[main_action]:
            if motion_folder_num in tuples[0]:
                return tuples[1]
        
        print(main_action, motion_folder_num)
        sys.exit()
        return -1
    
    # get the main action
    # - action and action_id
    def get_main_action(self, segmentation):
    
        # check the right hand
        data = segmentation["rhand"]    
        for action, action_id in zip(data["action"],data["action_id"]):
            if any([action in main_action for main_action in self.main_actions]):
                return action, np.array(action_id)
                
        # check the left hand
        data = segmentation["lhand"]    
        for action, action_id in zip(data["action"],data["action_id"]):
            if any([action in main_action for main_action in self.main_actions]):
                return action, np.array(action_id)
        
        print("Error. Could not find action in get_main_action")
        sys.exit()

    # # # # # # # # # # # # # # # #
    #                             #
    # Object Processing Functions #
    #                             #
    # # # # # # # # # # # # # # # #
    
    # get object data
    # - get object meta
    # - object position and rotation
    def get_object_data(self, sequence_data, frame_data):
        
        # object meta data
        obj_meta = self.get_object_meta(sequence_data, frame_data)
                
        # unpad frames
        frames = frame_data["frames"]
        frames_unpadded_length = frame_data["frames_unpadded_length"]
        frames = frames[:frames_unpadded_length]
                
        # object position and rotation
        obj_xyz = sample_readings(sequence_data, category="mocap", items=obj_meta["obj_names"], x_name="time", y_name="mocap_values",  timesteps=frames, return_dict=False) # [n, t, num markers, 3]
        obj_pos = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False) # [n, t, 3]
        obj_rot = sample_readings(sequence_data, category="root",  items=obj_meta["obj_names"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False) # [n, t, 3]
        
        # get table data
        table_xyz = sample_readings(sequence_data, category="mocap", items=["kitchen_sideboard"], x_name="time", y_name="mocap_values",  timesteps=frames, return_dict=False) # [1, t, num_markers, 3]
        table_pos = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False) # [1, t, 3]
        table_rot = sample_readings(sequence_data, category="root",  items=["kitchen_sideboard"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False) # [1, t, 3]
        table_xyz = table_xyz[0] # [t, num_markers, 3]
        table_pos = table_pos[0] # [t, num_markers, 3]
        table_rot = table_rot[0] # [t, num_markers, 3]
        
        # process object
        obj_data = self.process_obj(obj_xyz, obj_pos, obj_rot, table_xyz=table_xyz, table_pos=table_pos, table_rot=table_rot, prefix="obj")
                
        return_data = {**obj_meta, **obj_data}
        return return_data

    # get object meta
    # - object names
    # - id
    # - one hot vectors
    def get_object_meta(self, sequence_data, frame_data):
        
        # object names
        obj_names = sequence_data["metadata"]["object_names"]
        
        # get the table index and remove the table from the object names
        table_idx = obj_names.index("kitchen_sideboard")
        obj_names.pop(table_idx)
        assert "kitchen_sideboard" not in obj_names
        
        # object xml paths
        obj_paths = sequence_data["metadata"]["object_paths"]
        obj_paths.pop(table_idx)
        obj_paths = str(obj_paths)
        
        # object mocap marker names
        obj_mocap_names = [sequence_data["mocap"][obj_name]["mocap_names"] for obj_name in obj_names]
        obj_mocap_names = str(obj_mocap_names)
                
        # object ids and one hots
        obj_ids          = np.array([self.object_name_to_id[obj_name] for obj_name in obj_names])    # ignore table
        obj_ids_padded   = pad(obj_ids,self.object_padded_length).astype(int)
        obj_ohs_padded   = one_hot(obj_ids_padded,self.num_obj_wrist_classes)
        
        return_data = {"obj_names":obj_names,    "obj_mocap_names":obj_mocap_names,         
                       "obj_paths":obj_paths,
                       "obj_ids":obj_ids_padded, "obj_ids_unpadded_length":obj_ids.shape[0],
                       "obj_ohs":obj_ohs_padded, "obj_ohs_unpadded_length":obj_ids.shape[0],
                       "table_idx":table_idx
                       }
        
        ignore_board_sequence = ["3035", "3036", "3037", "3038", "3047", "3048", "3049", "3050"]
        ignore_bowl_sequence  = ["3039", "3040", "3041", "3042", "3043", "3044", "3045", "3046", "3063", "3064", "3065", "3066", "3067", "3068", "3069", "3070"]
        
        if hasattr(self, "use_edges_for_finger"):
            if self.use_edges_for_finger in ["attention"]:
                # get the objects handled by the left and right hand
                main_action = sequence_data["metadata"]["main_action"]
                #print(sequence_data["metadata"]["filename"]) /home_nfs/haziq/datasets/kit_mocap/data-sorted-simpleGT-v3/Cut/files_motions_3021/Cut1_c_0_05cm_01.xml
                lqk = [1 if x in self.held_object[main_action]["left"] else 0 for x in obj_names]
                rqk = [1 if x in self.held_object[main_action]["right"] else 0 for x in obj_names]
                
                filename = sequence_data["metadata"]["filename"]
                if "Transfer" in filename and any([x in filename for x in ignore_board_sequence]):
                    lqk = [1 if x in self.held_object[main_action]["left"] and x != "cutting_board_small" else 0 for x in obj_names]
                if "Transfer" in filename and any([x in filename for x in ignore_bowl_sequence]):
                    lqk = [1 if x in self.held_object[main_action]["left"] and x != "mixing_bowl_green" else 0 for x in obj_names]

                if np.sum(lqk) != 1 or np.sum(rqk) != 1:
                    print(sequence_data["metadata"]["filename"])
                    print(obj_names)
                    print("lqk:",lqk)
                    print("rqk:",rqk)
                    sys.exit()
                lqk = lqk.index(1)
                rqk = rqk.index(1)
                return_data = {**return_data, "lqk":lqk, "rqk":rqk}
                       
        return return_data
        
    # process all objects
    # - subtract by table_pos and maybe by the reference object
    # - scale
    # - pad time and number of objects
    def process_obj(self, xyz, pos, rot, table_xyz, table_pos, table_rot, prefix):

        # xyz = [n, t, num_markers, 3]
        # pos = [n, t, 3]
        # rot = [n, t, 3]
        # table_pos = [t, 3]

        # remove table center from pos and rot
        #xyz = np.delete(xyz, table_id, axis=0) # [n-1, t, num_markers, 3]
        #pos = np.delete(pos, table_id, axis=0) # [n-1, t, 3]
        #rot = np.delete(rot, table_id, axis=0) # [n-1, t, 3]

        pos = np.expand_dims(pos,1) if len(pos.shape) == 2 else pos # [n-1, t, 3]
        rot = np.expand_dims(rot,1) if len(rot.shape) == 2 else rot # [n-1, t, 3]

        # convert rotation angles to rotation matrix
        #rot = np.stack([compute_rotation_matrix(obj_thetas) for obj_thetas in rot]) # [n, t, 3, 3]

        # subtract by unscaled table_pos at every timestep
        xyz         = xyz - np.expand_dims(table_pos,axis=1) if table_pos is not None else xyz                  # [n, t, num_markers=4, 3]
        pos         = pos - table_pos if table_pos is not None else pos                                         # [n, t, 3]
        table_xyz   = table_xyz - np.expand_dims(table_pos,axis=1) if table_pos is not None else table_xyz      # [t, num_markers=4, 3]
                
        """
        # transform data such that object and table are oriented wrt global axes
        if self.normalize == 1:
            # print(table_rot.shape)    # [107, 3]
            table_rot_t0 = table_rot[0] # [3]
            rx = compute_rotation_matrix(table_rot_t0[0],"x")
            ry = compute_rotation_matrix(table_rot_t0[1],"y")
            rz = compute_rotation_matrix(table_rot_t0[2],"z")
            r = rz @ ry @ rx
            
            # rotation matrix to view the object position wrt 
            # to the object's coordinate system
            r = r.T # [3x3]
            
            # transform object pos to view it wrt 
            # to the object's coordinate system
            xyz = r @ np.transpose(xyz,[3,0,1,2])
            xyz = np.transpose(xyz,[1,2,3,0])
            #pos_t = r @ np.expand_dims(pos_t,1)
            #transformed_pos.append(pos_t)
        """
        
        # scale
        xyz         = xyz * self.xyz_scale          # [n-1, t, num_markers, 3]
        pos         = pos * self.xyz_scale          # [n-1, t,              3]
        table_xyz   = table_xyz * self.xyz_scale    # [t,      num_markers, 3]
               
        # pad time
        xyz_padded = np.stack([pad(x, self.pose_padded_length) for x in xyz])   # [n, padded t, num_markers, 3]
        pos_padded = np.stack([pad(x, self.pose_padded_length) for x in pos])   # [n, padded t, 3]
        rot_padded = np.stack([pad(x, self.pose_padded_length) for x in rot])   # [n, padded t, 3]
        table_xyz_padded = pad(table_xyz, self.pose_padded_length)              # [   padded t, num_markers, 3]
        table_pos_padded = pad(table_pos, self.pose_padded_length)              #    [padded t, 3]
        table_rot_padded = pad(table_rot, self.pose_padded_length)              #    [padded t, 3]
                
        # pad objects
        xyz_padded = pad(xyz_padded, self.object_padded_length) # [padded n, padded t, num_markers, 3]
        pos_padded = pad(pos_padded, self.object_padded_length)
        rot_padded = pad(rot_padded, self.object_padded_length)
        
        # transpose
        xyz_padded = np.transpose(xyz_padded, (1,0,2,3))    # [padded t, padded n, num_markers, 3]
        pos_padded = np.transpose(pos_padded, (1,0,2))      # [padded t, padded n, 3]
        rot_padded = np.transpose(rot_padded, (1,0,2))      # [padded t, padded n, 3]
        #print(xyz_padded.shape)
        #print(xyz.shape)
        #print(np.sum(xyz_padded[93:]))
        #print(np.sum(xyz_padded[94:]))
        #sys.exit()
                                
        # object velocities
        xyz_vel_padded = np.zeros(xyz_padded.shape)
        xyz_vel_padded[1:] = xyz_padded[1:] - xyz_padded[:-1]
        xyz_vel_padded[xyz.shape[1]] = xyz_vel_padded[0]
        
        prefix = prefix+"_"
        return_data = {
                       prefix+"xyz":xyz_padded,                                    prefix+"xyz_unpadded_length":xyz.shape[1],     prefix+"xyz_unpadded_objects":xyz.shape[0],        # obj_xyz,     obj_xyz_unpadded_length
                       prefix+"xyz_vel":xyz_vel_padded,                            prefix+"xyz_vel_unpadded_length":xyz.shape[1], prefix+"xyz_vel_unpadded_objects":xyz.shape[0],    # obj_xyz_vel, obj_xyz_unpadded_length
                       
                       prefix+"pos":pos_padded,                                    prefix+"pos_unpadded_length":pos.shape[1], prefix+"pos_unpadded_objects":pos.shape[0],            # obj_pos, obj_pos_unpadded_length
                       prefix+"rot":rot_padded,                                    prefix+"rot_unpadded_length":rot.shape[1], prefix+"rot_unpadded_objects":rot.shape[0],            # obj_rot, obj_rot_unpadded_length
                       prefix+"table_xyz":table_xyz_padded,                        prefix+"table_xyz_unpadded_length":table_xyz.shape[0],
                       prefix+"table_pos":table_pos_padded,                        prefix+"table_pos_unpadded_length":table_pos.shape[0],                                             # table pos is not at origin
                       prefix+"table_rot":table_rot_padded,                        prefix+"table_rot_unpadded_length":table_rot.shape[0]
                      }
        return return_data

    # # # # # # # # # # # # # # # #
    #                             #
    # Human Processing Functions  #
    #                             #
    # # # # # # # # # # # # # # # #
    
    def get_human_data(self, sequence_data, frame_data, obj_data):
        
        # metadata
        subject_id          = sequence_data["metadata"]["subject_id"]
        subject_height      = np.float32(sequence_data["metadata"]["subject_height"])
        subject_mass        = np.float32(sequence_data["metadata"]["subject_mass"])
        subject_hand_length = np.float32(sequence_data["metadata"]["subject_hand_length"])
        kin_names       = sequence_data["joint"]["body"]["joint_names"]
        xyz_names       = sequence_data["mocap"]["body"]["mocap_names"]
        finger_names    = sequence_data["joint"]["lhand"]["joint_names"] + sequence_data["joint"]["rhand"]["joint_names"]

        # unpad frames
        frames = frame_data["frames"]
        frames_unpadded_length = frame_data["frames_unpadded_length"]
        frames = frames[:frames_unpadded_length]
        
        # get centers
        table_pos = obj_data["obj_table_pos"]
        table_pos = table_pos[:frames_unpadded_length]
        
        # get xyz and joint data
        # ========================================
        xyz = sample_readings(sequence_data, category="mocap", items=["body"], x_name="time", y_name="mocap_values", timesteps=frames, return_dict=False)
        kin = sample_readings(sequence_data, category="joint", items=["body"], x_name="time", y_name="joint_values", timesteps=frames, return_dict=False)
        xyz = xyz[0]
        kin = kin[0]
        
        # process xyz and joint data
        xyz = self.process_pose(xyz, table_pos=table_pos, scale=self.xyz_scale, pad_length=self.pose_padded_length, prefix="xyz")
        kin = self.process_pose(kin, table_pos=None,      scale=self.kin_scale, pad_length=self.pose_padded_length, prefix="kin")
        inp_xyz, inp_kin = xyz["xyz"][0], kin["kin"][0]
                
        # get global position and rotation
        # ========================================
        pos = sample_readings(sequence_data, category="root", items=["body"], x_name="time", y_name="root_position", timesteps=frames, return_dict=False)
        rot = sample_readings(sequence_data, category="root", items=["body"], x_name="time", y_name="root_rotation", timesteps=frames, return_dict=False)
        pos = pos[0]
        rot = rot[0]
        
        # process global position and rotation
        pos = self.process_pose(pos, table_pos=table_pos, scale=1, pad_length=self.pose_padded_length, prefix="pos")
        rot = self.process_pose(rot, table_pos=None,      scale=1, pad_length=self.pose_padded_length, prefix="rot")

        # get finger joint data
        # ========================================
        finger = sample_readings(sequence_data, category="joint", items=["lhand","rhand"], x_name="time", y_name="joint_values", timesteps=frames, return_dict=False) # [2, length, 19]
        finger = self.process_pose(np.transpose(finger,axes=[1,0,2]), table_pos=None, scale=1, pad_length=self.pose_padded_length, prefix="finger")

        # get hand data
        wrist_ids = np.array([i for i in range(self.num_obj_wrist_classes-2,self.num_obj_wrist_classes)])
        wrist_ohs = one_hot(wrist_ids,self.num_obj_wrist_classes)
        wrist_xyz = xyz["xyz"][:,self.hand_xyz_dims]
        
        #print(xyz["xyz"].shape) # (150, 53, 3)
        #print(wrist_xyz.shape)  # (150, 10, 3)
        
        # # # # # # # #
        # velocities  #
        # # # # # # # #
        
        # wrist velocity
        wrist_xyz_vel = np.zeros(wrist_xyz.shape)
        wrist_xyz_vel[1:] = wrist_xyz[1:] - wrist_xyz[:-1]
        wrist_xyz_vel[xyz["xyz_unpadded_length"]] = wrist_xyz_vel[0]
        """
        print(xyz[hand+"_xyz_unpadded_length"])
        print(np.sum(wrist_xyz[xyz[hand+"_xyz_unpadded_length"]:]))
        print(np.sum(wrist_xyz_vel[xyz[hand+"_xyz_unpadded_length"]:]))
        sys.exit()
        """     
    
        # body velocity
        xyz_vel = np.zeros(xyz["xyz"].shape)
        xyz_vel[1:] = xyz["xyz"][1:] - xyz["xyz"][:-1]
        xyz_vel[xyz["xyz_unpadded_length"]] = xyz_vel[0]
    
        # finger velocity
        finger_vel = np.zeros(finger["finger"].shape)
        finger_vel[1:] = finger["finger"][1:] - finger["finger"][:-1]
        finger_vel[finger["finger_unpadded_length"]] = finger_vel[0]

        # subject id and joint names
        return_data = {"subject_id":subject_id, "subject_height":subject_height, "subject_mass":subject_mass, "subject_hand_length":subject_hand_length,
                       "kin_names":kin_names, "xyz_names":xyz_names, "finger_names":finger_names,
        
                       # wrist data
                       "wrist_ohs":wrist_ohs,
                       "wrist_xyz":wrist_xyz,         "wrist_xyz_unpadded_length":xyz["xyz_unpadded_length"],
                       "wrist_xyz_vel":wrist_xyz_vel, "wrist_xyz_vel_unpadded_length":xyz["xyz_unpadded_length"],
                       
                       # finger data
                       **finger,
                       "finger_vel":finger_vel, "finger_vel_unpadded_length":finger["finger_unpadded_length"],
                       
                       # pose data
                       **xyz, **kin,
                       "xyz_vel":xyz_vel, "xyz_vel_unpadded_length":xyz["xyz_unpadded_length"],
                       **pos, **rot}

        """
                       # subject id and joint names
        return_data = {"subject_id":subject_id, "subject_height":subject_height, "subject_mass":subject_mass, "subject_hand_length":subject_hand_length,
                       prefix+"kin_names":kin_names, prefix+"xyz_names":xyz_names, prefix+"finger_names":finger_names,
        
                       # wrist data
                       prefix+"wrist_ohs":wrist_ohs,
                       prefix+"wrist_xyz":wrist_xyz,         prefix+"wrist_xyz_unpadded_length":xyz[prefix+"xyz_unpadded_length"],
                       prefix+"wrist_xyz_vel":wrist_xyz_vel, prefix+"wrist_xyz_vel_unpadded_length":xyz[prefix+"xyz_unpadded_length"],
                       
                       # finger data
                       **finger,
                       
                       # pose data
                       **xyz, **kin,
                       **pos, **rot}
        """        
        return return_data

    # standardize, lose and keep some dimensions before padding
    def process_pose(self, data, table_pos, scale, pad_length, prefix):
            
        # center
        if table_pos is not None:        
            # table_pos.shape = [t,3]
            # xyz.shape = [t,53,3]
            # pos.shape = [t,3]
            table_pos = np.expand_dims(table_pos,1) if len(data.shape) == 3 else table_pos
            data = data - table_pos
        
        # scale
        data = data * scale
            
        # not good to add so much noise throughout the motion
        if 0: #self.add_noise and "xyz" in prefix:
            #print(data.shape) # [t, num_markers, 3]
            t, num_markers, _ = data.shape
            data = np.transpose(data,[1,0,2]) # [num_markers, t, 3]
            data = np.stack([add_gaussian_noise(data[i], 0, self.sigma, self.window_length, self.xyz_scale) for i in range(data.shape[0])]) # [num_markers, t, 3]
            data = np.transpose(data,[1,0,2]) # [t, num_markers, 3]
        
        # pad 
        data_padded = pad(data, pad_length)
        return {prefix:data_padded, prefix+"_unpadded_length":data.shape[0]}

# # # # # # # # # # # # #
#                       #
# processing functions  #
#                       #
# # # # # # # # # # # # #

def normalize(data, a, b):
    return (data - a) / (b - a)

# get the id of the reference object
# - will be none for approach action
def get_reference_object_name(object_names, action):
    return None

# compute the rotation matrix given thetas in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1
    
    print(theta.shape)
    sys.exit()

    if axis == "x":
        r = np.array([[1, 0,              0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta),  np.cos(theta)]])
        #r[:,1,1] =  np.cos(theta)
        #r[:,1,2] = -np.sin(theta)
        #r[:,2,1] =  np.sin(theta)
        #r[:,2,2] =  np.cos(theta)
                     
    if axis == "y":
        r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
                      [ 0,             1,  0],
                      [-np.sin(theta), 0,  np.cos(theta)]])
        #r[:,0,0] =  np.cos(theta)
        #r[:,0,2] =  np.sin(theta)
        #r[:,2,0] = -np.sin(theta)
        #r[:,2,2] =  np.cos(theta)

    if axis == "z":
        r = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]])
        #r[:,0,0] =  np.cos(theta)
        #r[:,0,1] = -np.sin(theta)
        #r[:,1,0] =  np.sin(theta)
        #r[:,1,1] =  np.cos(theta)
    
    return r

"""
# compute the rotation matrix given thetas in radians and axes
def compute_rotation_matrix(thetas):

    #print(thetas.shape)                         # [t, 3]
    rx = np.zeros(shape=(thetas.shape[0],3,3))   # [t, 3, 3]
    ry = np.zeros(shape=(thetas.shape[0],3,3))   # [t, 3, 3]
    rz = np.zeros(shape=(thetas.shape[0],3,3))   # [t, 3, 3]
    r  = np.zeros(shape=(thetas.shape[0],3,3))   # [t, 3, 3]
    
    # form rx
    rx[:,0,0] =  1
    rx[:,1,1] =  np.cos(thetas[:,0])
    rx[:,1,2] = -np.sin(thetas[:,0])
    rx[:,2,1] =  np.sin(thetas[:,0])
    rx[:,2,2] =  np.cos(thetas[:,0])
    
    # form ry
    ry[:,1,1] =  1
    ry[:,0,0] =  np.cos(thetas[:,1])
    ry[:,0,2] =  np.sin(thetas[:,1])
    ry[:,2,0] = -np.sin(thetas[:,1])
    ry[:,2,2] =  np.cos(thetas[:,1])

    # form rz
    rz[:,2,2] =  1
    rz[:,0,0] =  np.cos(thetas[:,2])
    rz[:,0,1] = -np.sin(thetas[:,2])
    rz[:,1,0] =  np.sin(thetas[:,2])
    rz[:,1,1] =  np.cos(thetas[:,2])
        
    r = np.stack([rzt @ ryt @ rxt for rzt,ryt,rxt in zip(rz,ry,rx)])    
    return r
"""

# add scaled gaussian noise to data
def add_gaussian_noise(data, mu, sigma, window_length, scale):
    
    #print(data)
    
    #print(data.shape) # [t, dim]
    length = data.shape[0]   
    for i in range(0, length, window_length):
        # each window length will have the same noise added to it
        z = np.random.normal(mu, sigma, size=[1, data.shape[1]]) * scale
        data[i:i+window_length] += z
    #print(z)
    #sys.exit()
    return data

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

# get the first and final frame given the input frame
def get_row(segmentation_timesteps, inp_frame):

    row = np.where(np.logical_and(segmentation_timesteps[:,0] <= inp_frame, segmentation_timesteps[:,1] >= inp_frame))[0]
    assert row.shape[0] == 1
    return row[0]
     
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
                        
            if t <= np.min(time):
                yt_list.append(values[0])
                
            elif t >= np.max(time):
                yt_list.append(values[-1])
                
            else:
                #print.info("time",time)
                #print.info("t",t)
                #print.info("time[time < t]",time[time < t])
                try:
                    i1 = np.where(time == time[time <= t].max())[0][0]
                    i2 = np.where(time == time[time > t].min())[0][0]                    
                except:
                    print(time)
                    print("filename=",data["metadata"]["filename"])
                    print("t=",t)
                    print("time.shape=", time.shape)
                    print("values.shape=", values.shape)
                    print("Error here")
                    print(np.where(time == time[time < t].max())[0][0])
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