import torch
from itertools import groupby

def masked_sum(data1, data2, idxs):

    # data1 [batch, inp_length, 53, 3]
    # data2 [batch, inp_length, 8,  3]
    # idxs  [batch, 8]

    data1_clone = torch.clone(data1)
    data2_clone = torch.clone(data2)
    batch_size = idxs.shape[0]
    for i in range(batch_size):
        
        # data1_clone[i,:,idxs_i] # [inp_length, 8, 3]
        # data2_clone[i,:]        # [inp_length, 8, 3]
        # idxs_i            # [8]
        
        idxs_i = idxs[i]
        data1_clone[i,:,idxs_i] = data1_clone[i,:,idxs_i] + data2_clone[i,:]
    return data1_clone

#def split_list(data, sep):
    
    #sep  = str(sep)
    #data = [str(x) for x in data]
    
#    return [list(group) for key, group in groupby(data, key=lambda x: x == sep) if not key]    

# scale center of data
def scale(obj, t0, scale):
    
    # Wrong to scale it down directly. Read kit_mocap   
    # Should instead subtract by t0 before scaling down
    #obj_center         = torch.mean(obj,dim=0)
    #scaled_obj_center  = obj_center / scale
    #difference         = scaled_obj_center - obj_center
    #obj                = obj - difference
        
    # should be right
    obj_center                  = torch.mean(obj,dim=0)
    obj_center_minus_t0         = obj_center - t0
    scaled_obj_center_minus_t0  = obj_center_minus_t0 / scale
    difference                  = obj_center_minus_t0 - scaled_obj_center_minus_t0
    obj                         = obj - difference
    return obj

def reform_data(inp_data, out_data, inp_frame, key_frame):
    
    inp  = inp_data[inp_frame:inp_frame+1].clone()
    pred = out_data[0:key_frame-1].clone()                    
    zero = inp_data[key_frame:].clone()  

    return torch.cat([inp,pred,zero])

def reform_obj(inp_data, out_data, key_frame):
    
    inp  = inp_data[:,key_frame:].clone()
    pred = out_data[:,:key_frame].clone()
    return torch.cat([pred,inp],dim=1)

def detach(adj, idx):
    
    i = torch.Tensor([i for i in range(adj.shape[0])]).long()
    j = torch.Tensor([idx]).long()    
    x, y = torch.meshgrid(i, j, indexing='xy')
    adj[x,y] = 0 
    x, y = torch.meshgrid(j, i, indexing='xy')
    adj[x,y] = 0
    
    return adj

#def reverse_data(data, un

def zero_pad(data_i, key_frame):

    pred_data_i = data_i[:key_frame].clone()
    zero_data_i = torch.zeros(size=data_i[key_frame:].shape).to(device=torch.cuda.current_device())
    pred_data_i = torch.cat([pred_data_i,zero_data_i])
    
    return pred_data_i

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0)

class return_1:
    def __init__(self, *args):
        return         
    
    def __call__(self, data):
        return torch.tensor(1, dtype=data.dtype, device=data.device)
    
    #def forward(self, data):
    #    return torch.tensor(1, dtype=data.type, device=data.device)

# compute the rotation matrix given thetas in radians and axes
def compute_rotation_matrix(theta, axis):

    assert axis == "x" or axis == "y" or axis == "z"
    
    # form n 3x3 identity arrays
    #n = theta.shape[0] if type(theta) == type(np.array(1)) else 1
    #r = np.zeros((n,3,3),dtype=np.float32)
    #r[:,0,0] = 1
    #r[:,1,1] = 1
    #r[:,2,2] = 1
        
    r = torch.zeros(size=[theta.shape[0],3,3], dtype=theta.dtype, device=theta.device) # [len(theta), 3, 3]

    if axis == "x":
        #r = np.array([[1, 0,              0],
        #              [0, np.cos(theta), -np.sin(theta)],
        #              [0, np.sin(theta),  np.cos(theta)]])
        r[:,0,0] =  1
        r[:,1,1] =  torch.cos(theta)
        r[:,1,2] = -torch.sin(theta)
        r[:,2,1] =  torch.sin(theta)
        r[:,2,2] =  torch.cos(theta)
                     
    if axis == "y":
        #r = np.array([[ np.cos(theta), 0,  np.sin(theta)],
        #              [ 0,             1,  0],
        #              [-np.sin(theta), 0,  np.cos(theta)]])
        r[:,1,1] =  1
        r[:,0,0] =  torch.cos(theta)
        r[:,0,2] =  torch.sin(theta)
        r[:,2,0] = -torch.sin(theta)
        r[:,2,2] =  torch.cos(theta)

    if axis == "z":
        #r = np.array([[np.cos(theta), -np.sin(theta), 0],
        #              [np.sin(theta),  np.cos(theta), 0],
        #              [0,              0,             1]])
        r[:,2,2] =  1
        r[:,0,0] =  torch.cos(theta)
        r[:,0,1] = -torch.sin(theta)
        r[:,1,0] =  torch.sin(theta)
        r[:,1,1] =  torch.cos(theta)
    
    return r