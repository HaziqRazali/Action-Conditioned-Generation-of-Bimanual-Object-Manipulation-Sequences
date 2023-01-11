import numpy as np

# # # # # # # # # #
# label to color  #
# # # # # # # # # #

object_to_color = {
"lhand": [255, 0, 0],
"rhand": [0, 0, 255],
"bottle": [207, 255, 0],
"whisk": [255, 143, 0],
"bowl": [255, 0, 175],
"cup": [47, 255, 0],
"knife": [255, 0, 95],
"banana": [0, 255, 31],
"screwdriver": [255, 0, 15],
"sponge": [0, 255, 191],
"cuttingboard": [255, 63, 0],
"cereals": [0, 255, 111],
"woodenwedge": [0, 239, 255],
"saw": [0, 159, 255],
"hammer": [255, 223, 0],
"harddrive": [0, 79, 255]}

# # # # # # #
# body data #
# # # # # # #

joint_names = ['LAnkle', 'LBigToe', 'LEar', 'LElbow', 'LEye', 'LHeel', 'LHip', 'LKnee', 'LShoulder', 'LSmallToe', 'LWrist', 
               'MidHip', 'Neck', 'Nose', 
               'RAnkle', 'RBigToe', 'REar', 'RElbow', 'REye', 'RHeel', 'RHip', 'RKnee', 'RShoulder', 'RSmallToe', 'RWrist']
link_names = [['LBigToe','LHeel'],['LHeel','LAnkle'],['LAnkle','LKnee'],['LKnee','LHip'],['LHip','MidHip'],['MidHip','Neck'],['Neck','LShoulder'],['LShoulder','LElbow'],['LElbow','LWrist'],
              ['RBigToe','RHeel'],['RHeel','RAnkle'],['RAnkle','RKnee'],['RKnee','RHip'],['RHip','MidHip'],['MidHip','Neck'],['Neck','RShoulder'],['RShoulder','RElbow'],['RElbow','RWrist'],
              ['Neck','Nose'],['Nose','LEye'],['LEye','LEar'],
              ['Neck','Nose'],['Nose','REye'],['REye','REar']]
link_ids = [[joint_names.index(x[0]),joint_names.index(x[1])] for x in link_names]
print(link_ids)

# # # # # # # # 
# finger data #
# # # # # # # #

finger_joint_names = ['LHand_0', 'LHand_1', 'LHand_10', 'LHand_11', 'LHand_12', 'LHand_13', 'LHand_14', 'LHand_15', 'LHand_16', 'LHand_17', 'LHand_18', 'LHand_19', 'LHand_2', 'LHand_20', 'LHand_3', 'LHand_4', 'LHand_5', 'LHand_6', 'LHand_7', 'LHand_8', 'LHand_9']
finger_link_names = [['LHand_0','LHand_1'],['LHand_1','LHand_2'],['LHand_2','LHand_3'],['LHand_3','LHand_4'],
                     ['LHand_0','LHand_5'],['LHand_5','LHand_6'],['LHand_6','LHand_7'],['LHand_7','LHand_8'],
                     ['LHand_0','LHand_9'],['LHand_9','LHand_10'],['LHand_10','LHand_11'],['LHand_11','LHand_12'],
                     ['LHand_0','LHand_13'],['LHand_13','LHand_14'],['LHand_14','LHand_15'],['LHand_15','LHand_16'],
                     ['LHand_0','LHand_17'],['LHand_17','LHand_18'],['LHand_18','LHand_19'],['LHand_19','LHand_20']]
finger_link_ids = [[finger_joint_names.index(x[0]),finger_joint_names.index(x[1])] for x in finger_link_names]

#finger_link_ids = [[0,1],[1,2],[2,3],[3,4],
#                   [0,5],[5,6],[6,7],[7,8],
#                   [0,9],[9,10],[10,11],[11,12],
#                   [0,13],[13,14],[14,15],[15,16],
#                   [0,17],[17,18],[18,19],[19,20]]

# misses 
# toe, heel, ankle
red = [255,0,0]

# spinal column 
red = [255,0,0]

# left arm
dark_green = [0,100,0]
green = [0,255,0]
light_green = [144,238,144]

# left eye
dark_purple = [139,0,139]
purple = [128,0,128]

# right eye
pink = [255,192,203]
dark_pink = [231,84,128]

# right arm
orange = [255,165,0]
dark_orange = [255,140,0]
yellow = [255,255,0]

# left leg
blue = [0,0,255]
light_blue = [173,216,230]

# right leg
green = [0,255,0]
light_green = [144,238,144]
              
link_colors = [red, # ['LBigToe','LHeel'],
               red, # ['LHeel','LAnkle'],
               red, # ['LAnkle','LKnee'],
               blue, # ['LKnee','LHip'],
               light_blue, # ['LHip','MidHip'],
               red, # ['MidHip','Neck'],
               dark_green, # ['Neck','LShoulder'],
               green, # ['LShoulder','LElbow'],
               light_green, # ['LElbow','LWrist']
               
               red, # ['RBigToe','RHeel'],
               red, # ['RHeel','RAnkle'],
               red, # ['RAnkle','RKnee'],
               green, # ['RKnee','RHip'],
               light_green, # ['RHip','MidHip'],
               red, # ['MidHip','Neck'],
               dark_orange, # ['Neck','RShoulder'],
               orange, # ['RShoulder','RElbow'],
               yellow, # ['RElbow','RWrist']]

               red,    # ['Neck','Nose'],
               dark_purple, # ['Nose','LEye'],
               purple, # ['LEye','LEar'],
               
               red,    # ['Neck','Nose'],
               dark_pink,   # ['Nose','REye'],
               pink]   # ['REye','REar']
               
# # # # # # # #
# camera data #
# # # # # # # #

# PrimeSense Carmine 1.09 data
camera_data = {"fov_x":54, "fov_y":45, "min_dist":300, "fx":628.0353617616482, "fy":579.4112549695428, "cx":320, "cy":240}
"""def fov_to_focal_length(fov, absolute):

    fov = float(fov)
    absolute = float(absolute)

    fov_rad = fov * np.pi / 180
    return absolute / (2 * np.tan(fov_rad / 2))"""
    
def fov_to_focal_length(fov, dimension):

    fov = float(fov)
    dimension = float(dimension)

    fov_rad = fov * np.pi / 180
    return dimension / (2 * np.tan(fov_rad / 2))
    
# # # # # # # # # # # # # # # # # #
# sequences with missing data #
# # # # # # # # # # # # # # # # # #

missing_data = []