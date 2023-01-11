import numpy as np

def dict_arr_to_list(d):
    for k, v in d.items():
        if isinstance(v, dict):
            dict_arr_to_list(v)
        else:
            if type(v) == type(np.array(0)):
                v = v.tolist()
            d.update({k: v})
    return d

def dict_list_to_arr(d,skip=[]):
    for k, v in d.items():
        
        if any([k == x for x in skip]):
            continue    

        if isinstance(v, dict):
            dict_list_to_arr(v,skip=skip)

        else:
            if type(v) == type([]):
                v = np.array(v)
            d.update({k: v})

    return d
    
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

# convert integer labels to one hot vectors
def one_hot(labels, max_label=None):

    one_hot_labels = np.zeros((labels.size, labels.max()+1)) if max_label is None else np.zeros((labels.size, max_label))
    one_hot_labels[np.arange(labels.size),labels] = 1
    
    return one_hot_labels