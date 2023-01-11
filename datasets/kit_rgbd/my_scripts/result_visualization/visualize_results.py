import os
import ast
import cv2
import sys
import json
import argparse

import numpy as np

from glob import glob
from pathlib import Path

sys.path.append("..")
from utils_data import *
from utils_processing import *

if __name__ == "__main__":
    
    # python visualize-results.py --result_folder "graph_v1_2022_08_08_visualization/easiest"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_rgbd"), type=str)
    parser.add_argument('--result_root', default=os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results"), type=str)
    parser.add_argument('--result_folder', default="action2pose/kit_rgbd/generation/combined", type=str)
    parser.add_argument('--draw_true', default=0, choices=[0,1], type=int)
    parser.add_argument('--save_video', default=0, type=int)
    args = parser.parse_args()
    
    sequence_folders = glob(os.path.join(args.result_root,args.result_folder,"*"))
    sequence_folders = natsorted(sequence_folders)
    for sequence_folder in sequence_folders:

        sequence_files = glob(os.path.join(sequence_folder,"*"))
        for sequence_file in sequence_files:
            
            print(sequence_file)
            
            # load result
            result = json.load(open(sequence_file,"r"))
            result = {k:np.array(v) if type(v) == type([]) else v for k,v in result.items()}
            
            # get sub-strings
            subject = result["subject"]
            action  = result["main_action"]
            take    = result["take"]
            
            print(subject, action, take)
            
            # path to rgb image
            rgb_files = read_chunks(os.path.join(args.root,"bimacs_rgbd_data",subject,action,take,"rgb"))
            timesteps = result["timesteps"].astype(int)
            rgb_files = [rgb_files[i] for i in timesteps]
            
            # denormalization data
            table_center = result["table_center"] # [3]
            angle = result["angle"]               # [1]
            rx    = compute_rotation_matrix(angle * np.pi / 180, "x")

            # # # # # # # # # # # #
            # load body pose data #
            # # # # # # # # # # # #
            
            if "pred_xyz" in result.keys(): 
                valid_joints = ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']
                true_body_xy = result["true_xyz"] # [t, num_joints=15, dim=2]
                true_body_xy = true_body_xy * np.array([640,480])
                true_body_xy = true_body_xy.astype(int)
                
                pred_body_xy = result["pred_xyz"] # [t, num_joints=15, dim=2]
                pred_body_xy = pred_body_xy * np.array([640,480])
                pred_body_xy = pred_body_xy.astype(int)
            
            else:
                valid_joints = ['LEar', 'LElbow', 'LEye', 'LHip', 'LShoulder', 'LWrist', 'MidHip', 'Neck', 'Nose', 'REar', 'RElbow', 'REye', 'RHip', 'RShoulder', 'RWrist']
                true_body_xy = result["xyz"] # [t, num_joints=15, dim=2]
                true_body_xy = true_body_xy * np.array([640,480])
                true_body_xy = true_body_xy.astype(int)
                
                pred_body_xy = result["xyz"] # [t, num_joints=15, dim=2]
                pred_body_xy = pred_body_xy * np.array([640,480])
                pred_body_xy = pred_body_xy.astype(int)                
            
            # # # # # # # # # # #
            # load finger data  #
            # # # # # # # # # # #
            
            if "pred_finger" in result.keys():
                true_finger_xy = result["true_finger"]                                              # [t, hands, total_num_joints*dim]
                true_finger_xy = np.reshape(true_finger_xy, [true_finger_xy.shape[0], 2, -1, 2])    # [t, hands, total_num_joints=21, dim=2]
                true_finger_xy = true_finger_xy * np.array([640,480])
                true_finger_xy = true_finger_xy.astype(int)
                
                pred_finger_xy = result["pred_finger"]                                              # [t, hands, total_num_joints*dim]
                pred_finger_xy = np.reshape(pred_finger_xy, [pred_finger_xy.shape[0], 2, -1, 2])    # [t, hands, total_num_joints=21, dim=2]
                pred_finger_xy = pred_finger_xy * np.array([640,480])
                pred_finger_xy = pred_finger_xy.astype(int)
            else:
                true_finger_xy = result["finger"]                                                   # [t, hands, total_num_joints*dim]
                true_finger_xy = np.reshape(true_finger_xy, [true_finger_xy.shape[0], 2, -1, 2])    # [t, hands, total_num_joints=21, dim=2]
                true_finger_xy = true_finger_xy * np.array([640,480])
                true_finger_xy = true_finger_xy.astype(int)
                
                pred_finger_xy = result["finger"]                                                   # [t, hands, total_num_joints*dim]
                pred_finger_xy = np.reshape(pred_finger_xy, [pred_finger_xy.shape[0], 2, -1, 2])    # [t, hands, total_num_joints=21, dim=2]
                pred_finger_xy = pred_finger_xy * np.array([640,480])
                pred_finger_xy = pred_finger_xy.astype(int)                
            
            # # # # # # # # # # #
            # load object data  #
            # # # # # # # # # # #
            
            # object names
            obj_names = ["lhand","rhand"] + ast.literal_eval(result["obj_names"])
                        
            if "pred_wrist_xyz" in result.keys():
                # true_wrist_obj_xyz
                true_wrist_xyz      = result["true_wrist_xyz"]        # [t, num_hands, 3]
                true_obj_xyz        = result["true_obj_xyz"][:,:,0,:] # [t, num_objs,  3] the third index is the number of markers which should be 1 for rgbd
                true_wrist_obj_xyz  = np.concatenate((true_wrist_xyz,true_obj_xyz),axis=1)  # [t, num_hands + num_objs, 3]
                true_wrist_obj_xyz  = true_wrist_obj_xyz / result["xyz_scale"]
                 
                # pred_wrist_obj_xyz
                pred_wrist_xyz      = result["pred_wrist_xyz"]        # [t, num_hands, 3]
                pred_obj_xyz        = result["pred_obj_xyz"][:,:,0,:] # [t, num_objs,  3]   
                pred_wrist_obj_xyz  = np.concatenate((pred_wrist_xyz,pred_obj_xyz),axis=1)  # [t, num_hands + num_objs, 3]
                pred_wrist_obj_xyz  = pred_wrist_obj_xyz / result["xyz_scale"]
            else:
                # true_wrist_obj_xyz
                true_wrist_xyz      = result["wrist_xyz"]        # [t, num_hands, 3]
                true_obj_xyz        = result["obj_xyz"][:,:,0,:] # [t, num_objs,  3]           
                true_wrist_obj_xyz  = np.concatenate((true_wrist_xyz,true_obj_xyz),axis=1)  # [t, num_hands + num_objs, 3]
                true_wrist_obj_xyz  = true_wrist_obj_xyz / result["xyz_scale"]
                 
                # pred_wrist_obj_xyz
                pred_wrist_xyz      = result["wrist_xyz"]        # [t, num_hands, 3]
                pred_obj_xyz        = result["obj_xyz"][:,:,0,:] # [t, num_objs,  3]   
                pred_wrist_obj_xyz  = np.concatenate((pred_wrist_xyz,pred_obj_xyz),axis=1)  # [t, num_hands + num_objs, 3]
                pred_wrist_obj_xyz  = pred_wrist_obj_xyz / result["xyz_scale"]                
            
            # if saving videos
            if args.save_video:
                output_filename = os.path.join(os.path.basename(subject),os.path.basename(action),os.path.basename(take))+".MP4"
                output_filename = os.path.join("video-results",output_filename.replace("/","_"))
                if args.save_video:
                    output_file = cv2.VideoWriter(
                        filename=output_filename,
                        # some installation of opencv may not support x264 (due to its license),
                        # you can try other format (e.g. MPEG)
                        fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                        fps=10,
                        frameSize=(640, 480),
                        isColor=True,
                    )   
            else:
                output_filename = os.path.join(os.path.basename(subject),os.path.basename(action),os.path.basename(take))
                output_filename = os.path.join("image-results",output_filename)            
                Path(output_filename).mkdir(parents=True, exist_ok=True)   
           
            body_xy = true_body_xy if args.draw_true == 1 else pred_body_xy
            finger_xy = true_finger_xy if args.draw_true == 1 else pred_finger_xy
            all_wrist_obj_xyz = true_wrist_obj_xyz if args.draw_true == 1 else pred_wrist_obj_xyz           
            for t,timestep in zip(range(all_wrist_obj_xyz.shape[0]),timesteps):
                                
                # read image
                rgb_file = rgb_files[t]
                rgb = cv2.imread(rgb_file)
                                
                # denormalize
                all_wrist_obj_xyz_t = all_wrist_obj_xyz[t]                # [num_hands + num_objs, 3]
                all_wrist_obj_xyz_t = all_wrist_obj_xyz_t + table_center  # [num_hands + num_objs, 3]
                all_wrist_obj_xyz_t = (rx @ all_wrist_obj_xyz_t.T).T      # [num_hands + num_objs, 3]
                                
                # # # # # # # # # # # # # #
                # project object to image #
                # # # # # # # # # # # # # #
                
                for xyz_t_n,obj_name in zip(all_wrist_obj_xyz_t,obj_names):
                
                    # # # # # # # # # # 
                    # project object  #
                    # # # # # # # # # #
                    
                    # obj name
                    instance_name = obj_name
                    obj_name = obj_name.split("_")[0]
                
                    # project to image
                    x,y = project_to_image(xyz_t_n[0], xyz_t_n[1], xyz_t_n[2], camera_data["cx"], camera_data["cy"], camera_data["fx"], camera_data["fy"])
                    #print(obj_name,x,y)
                    #input()
                    
                    # draw
                    cv2.circle(rgb, (x,y), 10, object_to_color[obj_name], 5)
                    cv2.putText(rgb, obj_name, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, object_to_color[obj_name], 2, cv2.LINE_AA)
                     
                # # # # # # # # # # # #
                # draw body on image  #
                # # # # # # # # # # # #
                
                body_xy_t = body_xy[t] # [num_joints=15, dim=2]
                for i,link_id in enumerate(link_ids):
                    
                    if joint_names[link_id[0]] in valid_joints and joint_names[link_id[1]] in valid_joints:
                        #print(joint_names[link_id[0]], joint_names[link_id[1]])
                        j1 = valid_joints.index(joint_names[link_id[0]])
                        j2 = valid_joints.index(joint_names[link_id[1]])
                        cv2.line(rgb, body_xy_t[j1], body_xy_t[j2], link_colors[i], thickness=5)
                
                # # # # # # # # # # # # #
                # draw finger on image  #
                # # # # # # # # # # # # #
                
                """finger_xy_t = finger_xy[t]                   # [2,21,2]
                for i,link_id in enumerate(finger_link_ids):
                    cv2.line(rgb, finger_xy_t[0,link_id[0]], finger_xy_t[0,link_id[1]], [255,0,0])
                    cv2.line(rgb, finger_xy_t[1,link_id[0]], finger_xy_t[1,link_id[1]], [255,0,0])"""
                
                save_folder = os.path.join("results", args.result_folder, subject, action, take)
                Path(save_folder).mkdir(parents=True, exist_ok=True)
                if args.save_video:
                    print("Saving", output_filename, timestep)
                    output_file.write(rgb)
                else:
                    print("Saving", os.path.join(save_folder,str(timestep).zfill(4)+".png"))
                    cv2.imwrite(os.path.join(save_folder,str(timestep).zfill(4)+".png"),rgb)
            