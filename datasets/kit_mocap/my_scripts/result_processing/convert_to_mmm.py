import os
import ast
import sys
import json
import argparse
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET

from pathlib import Path
from glob import glob
sys.path.append("..")
from utils_data import *
from utils_processing import *

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', default=os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation","results"), type=str) # "../../../../Keystate-Forecasting/results/"
    parser.add_argument('--result_name', required=True, type=str)
    args = parser.parse_args()
    args.result_root = os.path.expanduser(args.result_root)
        
    """
    xyz_stats = json.load(open(os.path.join("..","..","cached-data","xyz-stats.json"),"r"))
    kin_stats = json.load(open(os.path.join("..","..","cached-data","kin-stats.json"),"r"))
    xyz_stats = {k:np.array(v) for k,v in xyz_stats.items()}
    kin_stats = {k:np.array(v) for k,v in kin_stats.items()}
    """
                
    print(os.path.join(args.result_root,args.result_name,"*"))
    sequences = glob(os.path.join(args.result_root,args.result_name,"*"))
    print(sequences[-1])
    for sequence in sequences:    
        # print(sequence)
        
        files = glob(os.path.join(sequence,"*"))
        for file in files:
            # print(file)
                        
            # # # # # # #
            #           #
            # load data #
            #           #
            # # # # # # #
            
            data = json.load(open(file,"r"))
            #for k,v in data.items():
            #    print(k)
            
            for k,v in data.items():
                if "obj_names" not in k and "xyz_names" not in k and type(v) == type(list([])):
                    data[k] = np.array(v)
            
            # sequence name
            sequence = data["sequence"]
            print(sequence)
                          
            # time
            frames = data["frames"]
            print("frames:",frames, len(frames))
            
            # center
            table_pos = data["obj_table_pos"] if "obj_table_pos" in data.keys() else None
            table_rot = data["obj_table_rot"] if "obj_table_rot" in data.keys() else None
                        
            # scale
            xyz_scale, kin_scale = data["xyz_scale"], data["kin_scale"]
            
            # # # # # # # # # # # #
            #                     #
            # process object data #
            #                     #
            # # # # # # # # # # # #
                        
            # check whether i predicted xyz or kin 
            # ==================================== 
            has_pred_obj_xyz = any(["pred_obj_xyz" in k for k in data.keys()])
            has_pred_obj_pos = any(["pred_obj_pos" in k for k in data.keys()])
            has_pred_obj_rot = any(["pred_obj_rot" in k for k in data.keys()])
            print("has_pred_obj_xyz:",has_pred_obj_xyz)
            print("has_pred_obj_pos:",has_pred_obj_pos)
            print("has_pred_obj_rot:",has_pred_obj_rot)
            
            # object names
            obj_names = ast.literal_eval(data["obj_names"]) if "obj_names" in data.keys() else None   
            true_obj_names = ["true_"+obj_name for obj_name in obj_names]+["true_kitchen_sideboard"]
            pred_obj_names = ["pred_"+obj_name for obj_name in obj_names]
            
            # object mocap names
            obj_mocap_names = ast.literal_eval(data["obj_mocap_names"])
            
            # object path names
            obj_paths = ast.literal_eval(data["obj_paths"])
            true_obj_paths = obj_paths + ["/home/haziq/MMMTools/data/Model/Objects/kitchen_sideboard/kitchen_sideboard.xml"]
            pred_obj_paths = obj_paths
                        
            # =============== #
            # object position #
            # =============== #
            if has_pred_obj_pos:
                # true and pred pos
                true_obj_pos = np.transpose(data["true_obj_pos"], (1,0,2))                          # [n-1, len, 3] 
                pred_obj_pos = np.transpose(data["pred_obj_pos"], (1,0,2))                          # [n-1, len, 3] 
            else:           
                # true and pred pos
                true_obj_pos = np.transpose(data["obj_pos"], (1,0,2)) if "obj_pos" in data.keys() else None # [n-1, len, 3]             
                pred_obj_pos = None
            
            # =============== #
            # object xyz      #
            # =============== #
            if has_pred_obj_xyz:            
                # true and pred xyz 
                true_obj_xyz = np.transpose(data["true_obj_xyz"], (1,0,2,3))
                pred_obj_xyz = np.transpose(data["pred_obj_xyz"], (1,0,2,3))                # [n-1, len, num_markers, 3]  
            else:
                true_obj_xyz = np.transpose(data["obj_xyz"], (1,0,2,3))
                pred_obj_xyz = None
                        
            # =============== #
            # object rotation #
            # =============== #
            #table_idx = obj_names.index("kitchen_sideboard") 
            if has_pred_obj_rot:    
                # true and pred rot
                true_obj_rot = np.transpose(data["true_obj_rot"],(1,0,2))                               # (n-1, len, 3)
                pred_obj_rot = np.transpose(data["pred_obj_rot"],(1,0,2))                               # (n-1, len, 3)
                true_obj_rot = np.insert(true_obj_rot, len(true_obj_rot), np.expand_dims(table_rot,0), axis=0)
                pred_obj_rot = np.insert(pred_obj_rot, len(true_obj_rot), np.expand_dims(table_rot,0), axis=0)        
            else:            
                true_obj_rot = np.transpose(data["obj_rot"],(1,0,2))
                true_obj_rot = np.insert(true_obj_rot, len(true_obj_rot), np.expand_dims(table_rot,0), axis=0)
                pred_obj_rot = None
            
            # unprocess object 
            # ====================================  

            #table_idx = obj_names.index("kitchen_sideboard")            
            
            # unscale mocap before adding the table center
            true_obj_xyz = unprocess_obj(true_obj_xyz, table_pos, xyz_scale)
            pred_obj_xyz = unprocess_obj(pred_obj_xyz, table_pos, xyz_scale)
            
            # unscale centroid before adding the table center
            true_obj_pos = unprocess_obj(true_obj_pos, table_pos, xyz_scale) # [n-1, len, 3]
            pred_obj_pos = unprocess_obj(pred_obj_pos, table_pos, xyz_scale) # [n-1, len, 3]
            
            # insert table position
            true_obj_pos = np.insert(true_obj_pos, len(true_obj_pos), np.expand_dims(table_pos,0), axis=0)
            pred_obj_pos = None #np.insert(pred_obj_pos, table_idx, np.expand_dims(table_pos,0), axis=0)  if pred_obj_pos is not None else pred_obj_pos # [n,   len, 3] or None
             
            # # # # # # # # # # #
            #                   #
            # process pose data #
            #                   #
            # # # # # # # # # # #
                        
            # check whether i predicted xyz or kin 
            # ====================================  
            has_xyz_pred = any(["pred_xyz" in k for k in data.keys()])
            has_kin_pred = any(["pred_kin" in k for k in data.keys()])
            print("has_xyz_pred:",has_xyz_pred)
            print("has_kin_pred:",has_kin_pred)
            
            # subject id
            subject_id          = data["subject_id"]
            subject_height      = data["subject_height"]
            subject_mass        = data["subject_mass"]
            subject_hand_length = data["subject_hand_length"]
            xyz_names       = ast.literal_eval(data["xyz_names"])
            kin_names       = ast.literal_eval(data["kin_names"])
            finger_names    = ast.literal_eval(data["finger_names"])
            lfinger_names   = finger_names[:19]
            rfinger_names   = finger_names[19:]
            
            # position and rotation
            pos = data["pos"] # [len, 3]
            pos = unprocess_pose(pos, table_pos, 1)
            rot = data["rot"] # [len, 3]
                                                
            # we predicted xyz
            # ==================================== 
            if has_xyz_pred == 1:
                
                true_xyz = data["true_xyz"]
                true_xyz = unprocess_pose(true_xyz, table_pos, xyz_scale)
                
                pred_xyz = data["pred_xyz"]
                pred_xyz = unprocess_pose(pred_xyz, table_pos, xyz_scale)
            
            # we did not predict xyz
            # ====================================                     
            else:
                
                true_xyz = data["xyz"]
                true_xyz = unprocess_pose(true_xyz, table_pos, xyz_scale)     
                pred_xyz = None
                
            # we predicted kin
            # ==================================== 
            if has_kin_pred == 1:
                
                true_kin = data["true_kin"]
                true_kin = unprocess_pose(true_kin, None, kin_scale)
          
                pred_kin = data["pred_kin"]
                pred_kin = unprocess_pose(pred_kin, None, kin_scale)

            # we did not predict kin
            # ====================================   
            else:
            
                true_kin = data["kin"]
                true_kin = unprocess_pose(true_kin, None, kin_scale)
                pred_kin = None
  
            # check whether i predicted the fingers
            # ==================================== 
            has_finger_pred = any(["pred_finger" in k for k in data.keys()])
            print("has_finger_pred:",has_finger_pred)            
            if has_finger_pred == 1:
                true_finger = data["true_finger"]
                pred_finger = data["pred_finger"]
                true_lfinger, true_rfinger = true_finger[:,0], true_finger[:,1]
                pred_lfinger, pred_rfinger = pred_finger[:,0], pred_finger[:,1]
            else:
                true_finger = data["finger"]
                true_lfinger, true_rfinger = true_finger[:,0], true_finger[:,1]
                pred_lfinger, pred_rfinger = None, None
            
            # # # # # # # # # #
            #                 #
            # build xml file  #
            #                 #
            # # # # # # # # # #
                        
            m_encoding = 'UTF-8'
            
            # # # # # # # # # # # #
            # initialize root MMM #
            # # # # # # # # # # # #
            
            root = ET.Element("MMM")
            root.set("version", "2.0")
            root.set("name", sequence)
                                    
            # # # # # # # # # # # # # # # #
            # true and pred object nodes  #
            # # # # # # # # # # # # # # # #

            # try remove obj_pos and obj_rot
            # create true object node
            create_object_node(root, true_obj_names,        obj_mocap_names,  
                                     obj_xyz=true_obj_xyz,  obj_pos=true_obj_pos,   obj_rot=true_obj_rot, 
                                     frames=frames)
            
            # create pred object node
            print(pred_obj_names)
            print(obj_mocap_names)
            print(pred_obj_xyz.shape)
            create_object_node(root, pred_obj_names,        obj_mocap_names,
                                     obj_xyz=pred_obj_xyz,  obj_pos=None,           obj_rot=None, 
                                     frames=frames)

            # # # # # # # # # # # # # # #
            # true and pred human nodes #
            # # # # # # # # # # # # # # #
            
            # create true human node
            create_human_node(root, "true_"+subject_id, subject_height, subject_mass, subject_hand_length,
                                    pos_data=[pos, rot], 
                                    kin_data=[kin_names, true_kin],
                                    xyz_data=[xyz_names, true_xyz], 
                                    lfinger_data=[lfinger_names, true_lfinger], 
                                    rfinger_data=[rfinger_names, true_rfinger], 
                                    frames=frames)
             
            # create pred human node
            create_human_node(root, "pred_"+subject_id, subject_height, subject_mass, subject_hand_length, 
                                    pos_data=[pos, rot], 
                                    kin_data=[kin_names, pred_kin],
                                    xyz_data=[xyz_names, pred_xyz], 
                                    lfinger_data=[lfinger_names, pred_lfinger], 
                                    rfinger_data=[rfinger_names, pred_rfinger], 
                                    frames=frames)
                        
            # convert to string
            dom = xml.dom.minidom.parseString(ET.tostring(root))
            xml_string = dom.toprettyxml()
            part1, part2 = xml_string.split('?>')

            # create folder
            path = Path(os.path.dirname(file.replace(args.result_root,"./results/"))) # results\action2pose\agraph\oh\Cut_files_motions_3021_Cut1_c_0_05cm_01
            path.mkdir(parents=True, exist_ok=True)

            filename = file.replace(args.result_root,"results/").replace(".json",".xml")
            with open(filename, 'w') as xfile:
                xfile.write(part1 + 'encoding=\"{}\"?>\n'.format(m_encoding) + part2)
                xfile.close()

            # # # # # # # # # # #
            # create text file  #
            # # # # # # # # # # #
            
            #ubuntu_result_root = "/home/haziq/datasets/kit_mocap/my-scripts/result-processing/"
            ubuntu_result_root = os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation","datasets","kit_mocap","my_scripts","result_processing")
            
            # MMMMotionConverter
            MMMMotionConverter = "/home/haziq/MMMTools/build/bin/MMMMotionConverter"   
            
            # inputMotion and outputMotion
            inputMotion = outputMotion = os.path.join(ubuntu_result_root,filename)       
            
            print(filename.replace("xml","sh"))
            with open(filename.replace("xml","sh"),"wb") as f:
                
                # sanity check for the true pose, to make sure that the noise has been added
                if 0:
                
                    # motionName
                    motionName = "true_"+subject_id
                    
                    # converterConfigFile
                    #converterConfigFile = "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml"
                    converterConfigFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml")
                    
                    # outputModelFile
                    #outputModelFile = "/home/haziq/MMMTools/data/Model/Winter/mmm.xml"
                    outputModelFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/mmm.xml")
                    
                    # outputModelProcessorConfigFile
                    #outputModelProcessorConfigFile = os.path.join("/home/haziq/MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    outputModelProcessorConfigFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName {} --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputModelProcessorConfigFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputModelProcessorConfigFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())
                
                if has_xyz_pred == 1:
                
                    # motionName
                    motionName = "pred_"+subject_id
                    
                    # converterConfigFile
                    #converterConfigFile = "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml"
                    converterConfigFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml")
                    
                    # outputModelFile
                    outputModelFile = "/home/haziq/MMMTools/data/Model/Winter/mmm.xml"
                    outputModelFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/mmm.xml")
                    
                    # outputModelProcessorConfigFile
                    #outputModelProcessorConfigFile = os.path.join("/home/haziq/MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    outputModelProcessorConfigFile = os.path.join(os.path.expanduser("~"),"MMMTools/data/Model/Winter/config/",subject_id+".xml")
                    
                    # write
                    write = "{} --inputMotion \"{}\" --motionName {} --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputModelProcessorConfigFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 5 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputModelProcessorConfigFile, outputMotion)
                    #write = write.replace("\\","/")
                    f.write(write.encode())
            
                if has_pred_obj_xyz == 1:
                    print(pred_obj_names)
                    for pred_obj_name in pred_obj_names:
                        
                        # motionName
                        motionName = pred_obj_name
                        
                        # converterConfigFile
                        converterConfigFile = converterConfigFile_dict[pred_obj_name]
                        
                        # outputModelFile
                        outputModelFile = outputModelFile_dict[pred_obj_name]
                        
                        # outputModelProcessorConfigFile
                        #outputModelProcessorConfigFile = outputModelProcessorConfigFile_dict[pred_obj_name]
                        
                        # write
                        write = "{} --inputMotion \"{}\" --motionName {} --converterConfigFile \"{}\" --outputModelFile \"{}\" --outputMotion \"{}\" &&\\\nsleep 2 &&\\\n".format(MMMMotionConverter, inputMotion, motionName, converterConfigFile, outputModelFile, outputMotion)
                        #write = write.replace("\\","/")
                        f.write(write.encode())
                write = "echo \"{} DONE\"".format(filename)
                write = write.replace("xml","sh")#.replace("\\","/")
                f.write(write.encode())
            os.chmod(filename.replace("xml","sh"),0o755)
    
    Path(os.path.join("results",args.result_name)).mkdir(parents=True,exist_ok=True)  
    print()
    print("Creating convert.sh",os.path.join("results",args.result_name,"convert.sh"))
    print()
    # main script that runs all the conversion scripts
    with open(os.path.join("results",args.result_name,"convert.sh"),"wb") as f: # results\action2pose\agraph\oh
        
        files = glob(os.path.join("results",args.result_name,"**","*.sh"), recursive=True)
        files = [os.path.abspath(file) for file in files if "convert" not in file]
        #print(files[-1]) # results/action2pose/kit_mocap/graph_v1 2022_06_23/noise_scale=0_noise_add_type=each_object_vel=1_kl=1e-3/Cut_files_motions_3023_Cut3_c_30_05cm_03/0000000000.sh
        #print(os.path.abspath(files[-1]))
        #files = ["./"+os.path.basename(args.result_name)+file.replace(os.path.join("results",args.result_name),"") for file in files if "convert" not in file]
        #print(files[-1])
        #sys.exit()
        for file in files:
            print(file)
            write = file+"\n"
            write = write.replace(" ","\ ")
            #write = file.replace("\\","/")+"\n"
            f.write(write.encode())
    os.chmod(os.path.join("results",args.result_name,"convert.sh"),0o755)
    
def create_human_node(root, subject_id, subject_height, subject_mass, subject_hand_length, pos_data, kin_data, xyz_data, lfinger_data, rfinger_data, frames):

    # model pose
    pos = pos_data[0]
    rot = pos_data[1]
    
    # kin data
    kin_names = kin_data[0]
    kin = kin_data[1]
    
    # xyz data
    xyz_names = xyz_data[0]
    xyz = xyz_data[1]

    # # # # # # # # # # # # # # # # # # #
    # MMM                               #
    # - Motion name="1480"              # <-----
    # # # # # # # # # # # # # # # # # # #           
    
    motion = ET.SubElement(root, "Motion")
    motion.set("name",subject_id)
    motion.set("type","object")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    # 
    #   - Model path="mmm.xml"                # <-----
    #     - ModelProcessorConfig type="Winter # <-----
    #       - Height                          # <-----
    #       - Mass                            # <-----
    #       - HandLength                      # <-----
    # # # # # # # # # # # # # # # # # # # # # #   
        
    model = ET.SubElement(motion, "Model")
    model.set("path",outputModelFile_dict[subject_id])
    
    model_processor_config = ET.SubElement(model, "ModelProcessorConfig")
    model_processor_config.set("type","Winter")
    
    height = ET.SubElement(model_processor_config, "Height")
    height.text = str(subject_height)
    mass = ET.SubElement(model_processor_config, "Mass")
    mass.text = str(subject_mass)
    hand_length = ET.SubElement(model_processor_config, "HandLength")
    hand_length.text = str(subject_hand_length)
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             # <-----
    # # # # # # # # # # # # # # # # # # # # # #                
    
    sensors = ET.SubElement(motion, "Sensors")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
    
    sensor1 = ET.SubElement(sensors, "Sensor")
    sensor1.set("type","ModelPose")
    sensor1.set("version","1.0")           

    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           #
    #       - Configuration                   # <-----
    #       - Data                            # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
    
    configuration = ET.SubElement(sensor1, "Configuration")
    data = ET.SubElement(sensor1, "Data")
    
    # # # # # # # # # # # # # # # # # # # # # #
    # MMM                                     #
    # - Motion name="1480"                    #
    #   - Model path="mmm.xml"                #
    #   - Sensors                             #
    #     - Sensor type="ModelPose"           #
    #       - Configuration                   #
    #       - Data                            #
    #         - Measurement                   # <-----
    #           - RootPosition                # <-----
    #           - RootRotation                # <-----
    # # # # # # # # # # # # # # # # # # # # # # 
            
    # measurements
    for pos_t, rot_t, t in zip(pos, rot, frames):
        measurement = ET.SubElement(data, "Measurement")
        measurement.set("timestep",str(t))
        
        root_position = ET.SubElement(measurement, "RootPosition").text = " ".join([str(x) for x in pos_t])
        root_rotation = ET.SubElement(measurement, "RootRotation").text = " ".join([str(x) for x in rot_t])
    
    if kin is not None:
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","Kinematic")
        sensor1.set("version","1.0")           

        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration = ET.SubElement(sensor1, "Configuration")
        data = ET.SubElement(sensor1, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="Kinematic"           #
        #       - Configuration                   #
        #         - Joint name = "BLNx_joint"     # <-----
        #       - Data                            #
        # # # # # # # # # # # # # # # # # # # # # # 
        
        for kin_name in kin_names:
            joint = ET.SubElement(configuration, "Joint")
            joint.set("name",kin_name)
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #         - Joint name = "BLNx_joint"     #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - JointPosition               # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        # measurements
        for kin_t,t in zip(kin,frames):
            measurement = ET.SubElement(data, "Measurement")
            measurement.set("timestep",str(t))
            
            joint_position = ET.SubElement(measurement, "JointPosition").text = " ".join([str(x) for x in kin_t])
    
    if lfinger_data[1] is not None and rfinger_data[1] is not None:        
        for finger_data in [lfinger_data, rfinger_data]:
            
            finger_names = finger_data[0]
            finger = finger_data[1]
        
            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           # <-----
            # # # # # # # # # # # # # # # # # # # # # # 
            
            sensor1 = ET.SubElement(sensors, "Sensor")
            sensor1.set("type","Kinematic")
            sensor1.set("version","1.0")    

            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           #
            #       - Configuration                   # <-----
            #       - Data                            # <-----
            # # # # # # # # # # # # # # # # # # # # # #         
        
            configuration = ET.SubElement(sensor1, "Configuration")
            data = ET.SubElement(sensor1, "Data")
            
            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="1480"                    #
            #   - Model path="mmm.xml"                #
            #   - Sensors                             #
            #     - Sensor type="Kinematic"           #
            #       - Configuration                   #
            #         - Joint name = "BLNx_joint"     # <-----
            #       - Data                            #
            # # # # # # # # # # # # # # # # # # # # # # 
            
            for finger_name in finger_names :
                joint = ET.SubElement(configuration, "Joint")
                joint.set("name",finger_name)

            # # # # # # # # # # # # # # # # # # # # # #
            # MMM                                     #
            # - Motion name="kitchen_sideboard"       #
            #   - Model path="kitchen_sideboard.xml"  #
            #   - Sensors                             #
            #     - Sensor type="ModelPose"           #
            #       - Configuration                   #
            #         - Joint name = "BLNx_joint"     #
            #       - Data                            #
            #         - Measurement                   # <-----
            #           - JointPosition               # <-----
            # # # # # # # # # # # # # # # # # # # # # # 
            
            # measurements
            for finger_t,t in zip(finger,frames):
                measurement = ET.SubElement(data, "Measurement")
                measurement.set("timestep",str(t))
                
                joint_position = ET.SubElement(measurement, "JointPosition").text = " ".join([str(x) for x in finger_t])
    
    if xyz is not None:
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="MocapMarker"         # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","MoCapMarker")
        sensor1.set("version","1.0")           

        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="1480"                    #
        #   - Model path="mmm.xml"                #
        #   - Sensors                             #
        #     - Sensor type="MocapMarker"         #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration = ET.SubElement(sensor1, "Configuration")
        data = ET.SubElement(sensor1, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - MarkerPosition              # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
                
        # measurements
        for xyz_t,t in zip(xyz,frames):
            measurement = ET.SubElement(data, "Measurement")
            measurement.set("timestep",str(t))
            
            for name,value_t in zip(xyz_names,xyz_t):
                marker_position = ET.SubElement(measurement, "MarkerPosition")
                marker_position.set("name",name)
                marker_position.text = " ".join([str(x) for x in value_t])
            
def create_object_node(root, obj_names, obj_mocap_names, obj_xyz, obj_pos, obj_rot, frames):

    #if obj_names is None or obj_pos is None or obj_rot is None:
    #    return

    if obj_xyz is None and obj_pos is None and obj_rot is None:
        return

    # = = = = = = = = = = = = = = = = = #
    # object motion                     #
    # = = = = = = = = = = = = = = = = = #
    
    # # # # # # # # # # # # # # # # # # #
    # MMM                               #
    # - Motion name="kitchen_sideboard" # <-----
    # - Motion name="salad_fork"        # <-----
    # - Motion ...                      # <-----
    # # # # # # # # # # # # # # # # # # #
        
    for i,obj_name in enumerate(obj_names):
        motion = ET.SubElement(root, "Motion")
        motion.set("name",obj_name)
        motion.set("type","object")
        motion.set("synchronized","true")
    
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  # <-----
        # # # # # # # # # # # # # # # # # # # # # #
                
        model = ET.SubElement(motion, "Model")
        model.set("path",outputModelFile_dict[obj_name])
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             # <-----
        # # # # # # # # # # # # # # # # # # # # # #                
        
        sensors = ET.SubElement(motion, "Sensors")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        sensor1 = ET.SubElement(sensors, "Sensor")
        sensor1.set("type","ModelPose")
        sensor1.set("version","1.0")        
        sensor2 = ET.SubElement(sensors, "Sensor")
        sensor2.set("type","MoCapMarker")
        sensor2.set("version","1.0")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   # <-----
        #       - Data                            # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        configuration1 = ET.SubElement(sensor1, "Configuration")
        data1 = ET.SubElement(sensor1, "Data")
        configuration2 = ET.SubElement(sensor2, "Configuration")
        data2 = ET.SubElement(sensor2, "Data")
        
        # # # # # # # # # # # # # # # # # # # # # #
        # MMM                                     #
        # - Motion name="kitchen_sideboard"       #
        #   - Model path="kitchen_sideboard.xml"  #
        #   - Sensors                             #
        #     - Sensor type="ModelPose"           #
        #       - Configuration                   #
        #       - Data                            #
        #         - Measurement                   # <-----
        #           - RootPosition                # <-----
        #           - RootRotation                # <-----
        # # # # # # # # # # # # # # # # # # # # # # 
        
        #debug=3
        #print(object_names[debug])
        #for object_pos_t, object_rot_t, t in zip(object_pos[debug], object_rot[debug], frames):
        #    print(object_pos_t, object_rot_t, t)
        #    print(" ".join([str(x) for x in object_pos_t]))
        #    input()
        
        if obj_pos is not None and obj_rot is not None:
            # measurements
            for obj_pos_t, obj_rot_t, t in zip(obj_pos[i], obj_rot[i], frames):
                measurement1 = ET.SubElement(data1, "Measurement")
                measurement1.set("timestep",str(t))
                
                root_position = ET.SubElement(measurement1, "RootPosition").text = " ".join([str(x) for x in obj_pos_t])
                root_rotation = ET.SubElement(measurement1, "RootRotation").text = " ".join([str(x) for x in obj_rot_t])
                
        if obj_xyz is not None and i < obj_xyz.shape[0]:
            # measurements
            for obj_xyz_t,t in zip(obj_xyz[i],frames):
                measurement2 = ET.SubElement(data2, "Measurement")
                measurement2.set("timestep",str(t))
                                
                for j,name in enumerate(obj_mocap_names[i]):
                    
                    mocap_value = ET.SubElement(measurement2, "MarkerPosition")
                    mocap_value.text = " ".join([str(x) for x in obj_xyz_t[j]])
                    mocap_value.set("name",name)
                    
if __name__ == "__main__":

    main()