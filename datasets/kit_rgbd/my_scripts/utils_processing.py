import os
import re
import numpy as np
import pandas as pd

from glob import glob
from scipy import interpolate
from natsort import natsorted

def read_chunks(folder):
    
    all_folders = glob(os.path.join(folder,"*"))
    
    # metadata
    metadata = [x for x in all_folders if "metadata" in x][0]
    metadata = pd.read_csv(metadata, index_col=0)
    
    assert int(metadata.loc["frameWidth"]["value"]) == 640
    assert int(metadata.loc["frameHeight"]["value"]) == 480
    
    # actual chunk folders
    chunk_folders = [x for x in all_folders if "chunk" in x]
    chunk_folders = natsorted(chunk_folders)
    
    files = []
    for chunk_folder in chunk_folders:
        chunk_files = glob(os.path.join(chunk_folder,"*"))
        chunk_files = natsorted(chunk_files)
        files.extend(chunk_files)
    return files

def project_to_image(x,y,z,cx,cy,fx,fy,div=1):
    
    # cy = 240
    
    x/=div
    y/=div
    z/=div
        
    z *= -1
    x = (x * fx / z) + cx
    y = -(y * fy / z) + cy  
    
    return int(x),int(y)

# compute the rotation matrix given theta in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1

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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
 
def remove_duplicates(sequence_data, main_3d_objects):

    """
    sequence_data["3d"][instance_name] = {}
    sequence_data["3d"][instance_name]["time"] = []
    sequence_data["3d"][instance_name]["bbox"] = []   
    sequence_data["3d"][instance_name]["colour"] = x["colour"] 
    """
        
    for main_3d_object,count in main_3d_objects.items():
    
        # duplicates for the current main_3d_object
        # e.g. LeftHand_1, LeftHand_2, ...
        duplicates = {}
        duplicates["3d"] = {}
        duplicates["3d"] = {k:sequence_data["3d"][k] for k in sequence_data["3d"].keys() if main_3d_object in k}
                                
        # no duplicates
        if len(duplicates["3d"].keys()) == count:
            continue
            
        # contain duplicates
        elif len(duplicates["3d"].keys()) > count:
            #print("contain duplicates:")
            #print(sequence_data["3d"].keys())
            
            # find the top <count> keys with the longest lengths
            by_num_detections = sorted([(len(v["bbox"]), k) for k, v in duplicates["3d"].items()], reverse=True)
            to_keep = by_num_detections[:count]
            to_delete = by_num_detections[count:]
            
            # update the sequence_data
            for length,k in to_delete:
                del sequence_data["3d"][k] 
        
        # contain fewer than the needed count
        # except for woodenwedge
        elif len(duplicates["3d"].keys()) < count and len(duplicates["3d"].keys()) > 0 and "woodenwedge" in main_3d_object:
            continue
            
        # contain fewer than the needed count
        else:
            print("Error!")
            print(sequence_data["3d"].keys())
            sys.exit()
        
    return sequence_data
 
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
                    #print("filename=",data["metadata"]["filename"])
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
    return np.squeeze(np.stack([v for k,v in sampled_readings.items()]))