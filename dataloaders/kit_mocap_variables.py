import os
import json
import numpy as np


"""

train test split
- description of each variant in /datasets/kit_mocap/

"""

train_test_split = {}
train_test_split["data-sorted-simpleGT-v2"] = {}
train_test_split["data-sorted-simpleGT-v3"] = {}
train_test_split["default"] = {}

# data-sorted-simpleGT-v2
train_test_split["data-sorted-simpleGT-v2"]["train"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                                        "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                                        "Mix":["3121","3122","3123","3124"],
                                                        "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                                        "Peel":["3109","3110","3111","3112"],
                                                        "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                                        "RollOut":["3113","3114","3117","3118"],
                                                        "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3002","3003","3004","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3018","3019","3020","3057","3058","3059","3060","3061","3062"],
                                                        "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                                        "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                                        "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                                        }
train_test_split["data-sorted-simpleGT-v2"]["val"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                                      "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                                      "Mix":["3121","3122","3123","3124"],
                                                      "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                                      "Peel":["3109","3110","3111","3112"],
                                                      "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                                      "RollOut":["3113","3114","3117","3118"],
                                                      "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3002","3003","3004","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3018","3019","3020","3057","3058","3059","3060","3061","3062"],
                                                      "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                                      "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                                      "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                                      }

# data-sorted-simpleGT-v3
train_test_split["data-sorted-simpleGT-v3"]["train"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                                        "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                                        "Mix":["3121","3122","3123","3124"],
                                                        "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                                        "Peel":["3109","3110","3111","3112"],
                                                        "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                                        "RollOut":["3113","3114","3117","3118"],
                                                        "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3057","3058","3059","3060","3061","3062"],
                                                        "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                                        "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                                        "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                                        }
train_test_split["data-sorted-simpleGT-v3"]["val"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                                      "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                                      "Mix":["3121","3122","3123","3124"],
                                                      "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                                      "Peel":["3109","3110","3111","3112"],
                                                      "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                                      "RollOut":["3113","3114","3117","3118"],
                                                      "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3002","3003","3004","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3018","3019","3020","3057","3058","3059","3060","3061","3062"],
                                                      "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                                      "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                                      "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                                      }

# default
train_test_split["default"]["train"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                        "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                        "Mix":["3121","3122","3123","3124"],
                                        "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                        "Peel":["3109","3110","3111","3112"],
                                        "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                        "RollOut":["3113","3114","3117","3118"],
                                        "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3057","3058","3059","3060","3061","3062"],
                                        "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                        "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                        "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                        }
train_test_split["default"]["val"] = {"Close":["3072","3074","3076","3078","3079","3080","3082","3084","3086","3088","3089","3090"],
                                      "Cut":["3021","3022","3023","3024","3025","3028","3029"],
                                      "Mix":["3121","3122","3123","3124"],
                                      "Open":["3071","3073","3075","3077","3081","3083","3085","3087"],
                                      "Peel":["3109","3110","3111","3112"],
                                      "Pour":["2967","2968","2969","2970","2971","2972","2973","2974","2975","2976","2977","2978","2979","2980","2981","2982","2983","2984","2985","2986","2987","2988","3051","3052","3053","3054","3055","3056"],
                                      "RollOut":["3113","3114","3117","3118"],
                                      "Scoop":["2989","2990","2991","2992","2993","2994","2995","2996","2997","2998","2999","3000","3001","3002","3003","3004","3005","3006","3007","3008","3009","3010","3011","3012","3013","3014","3015","3016","3017","3018","3019","3020","3057","3058","3059","3060","3061","3062"],
                                      "Stir":["2941","2942","2943","2944","2945","2946","2947","2948","2949","2950","2953","2954","2955","2956","2957","2958","2959","2960","2961","2962","2963","2966"],
                                      "Transfer":["3035","3036","3037","3038","3039","3040","3041","3042","3043","3044","3045","3046","3047","3048","3049","3050","3063","3064","3065","3066","3067","3068","3069","3070"],
                                      "Wipe":["3091","3092","3093","3094","3095","3096","3097","3098","3099","3100","3101","3102","3103","3104","3105","3106","3107","3108"]
                                      }
                                      
"""

mocap and joint marker names
xyz id = 3x, 3x+1, 3x+2

"""

# mocap names
mocap_names = ["C7", "CLAV", "L3", # final x = 2
               "LAEL", "LANK", "LAOL", "LASI", "LBAK", "LBHD", "LFHD", "LFRA", "LHEE", "LHIP", "LHPS", "LHTS", "LIFD", "LKNE", "LMT1", "LMT5", "LPSI", "LSHO", "LTHI", "LTIP", "LTOE", "LUPA", "LWPS", "LWTS", # final x = 26
               "RAEL", "RANK", "RAOL", "RASI", "RBAK", "RBHD", "RFHD", "RFRA", "RHEE", "RHIP", "RHPS", "RHTS", "RIFD", "RKNE", "RMT1", "RMT5", "RPSI", "RSHO", "RTHI", "RTIP", "RTOE", "RUPA", "RWPS", "RWTS", # final x = 50
               "STRN","T10"]       # final x = 52   
l_arm_mocap_names = ["LAEL", "LAOL", "LFRA", "LHPS", "LHTS", "LIFD", "LWPS", "LWTS"]
r_arm_mocap_names = ["RAEL", "RAOL", "RFRA", "RHPS", "RHTS", "RIFD", "RWPS", "RWTS"]
l_arm_mocap_idxs  = [mocap_names.index(l_arm_mocap_name) for l_arm_mocap_name in l_arm_mocap_names]
r_arm_mocap_idxs  = [mocap_names.index(r_arm_mocap_name) for r_arm_mocap_name in r_arm_mocap_names]

# joint names
joint_names = ["BLNx_joint", "BLNy_joint", "BLNz_joint", # lower neck        0,1,2
              "BPx_joint", "BPy_joint", "BPz_joint",     # pelvis            3,4,5
              "BTx_joint", "BTy_joint", "BTz_joint",     # thorax            6,7,8
              "BUNx_joint", "BUNy_joint", "BUNz_joint",  # upper neck        9,10,11
               
              "LAx_joint", "LAy_joint", "LAz_joint",     # left ankle        12,13,14
              "LEx_joint", "LEz_joint",                  # left elbow        15,16,
              "LHx_joint", "LHy_joint", "LHz_joint",     # left hip          17,18,19
              "LKx_joint",                               # left knee         20
              "LSx_joint", "LSy_joint", "LSz_joint",     # left shoulder     21,22,23
              "LWx_joint", "LWy_joint",                  # left wrist        24,25
              "LFx_joint",                               # left foot         26
              "LMrot_joint",                             # left metatarsal   27          (between foot and ankle)
              
              "RAx_joint", "RAy_joint", "RAz_joint",     # right ankle       28,29,30
              "REx_joint", "REz_joint",                  # right elbow       31,32
              "RHx_joint", "RHy_joint", "RHz_joint",     # right hip         33,34,35
              "RKx_joint",                               # right knee        36
              "RSx_joint", "RSy_joint", "RSz_joint",     # right shoulder    37,38,39
              "RWx_joint", "RWy_joint",                  # right wrist       40,41,
              "RFx_joint",                               # right foot        42,
              "RMrot_joint"]                             # right metatarsal  43          (between foot and ankle)
joint_idx_to_name = {i:joint_name for i,joint_name in enumerate(joint_names)}

# extended joint names with the root and column segment
root_name   = ["ROOTx_joint", "ROOTy_joint", "ROOTz_joint"]
CS_name     = ["CSx_joint",   "CSy_joint",   "CSz_joint"]
extended_joint_names     = joint_names + root_name + CS_name
extended_joint_idx_to_name = {i:joint_name for i,joint_name in enumerate(extended_joint_names)}

# left and right arm joint names and idxs
l_arm_joint_names = ["LSx_joint", "LSy_joint", "LSz_joint","LEx_joint", "LEz_joint","LWx_joint", "LWy_joint"]
r_arm_joint_names = ["RSx_joint", "RSy_joint", "RSz_joint","REx_joint", "REz_joint","RWx_joint", "RWy_joint"]
l_arm_joint_idxs  = [extended_joint_names.index(l_arm_joint_name) for l_arm_joint_name in l_arm_joint_names]
r_arm_joint_idxs  = [extended_joint_names.index(r_arm_joint_name) for r_arm_joint_name in r_arm_joint_names]

# the axis of rotation for each joint
extended_joint_axis = {}
for joint_name in extended_joint_names:
    # each joint_name must be hardcoded to use the x,y,z for only the axis of rotation
    if joint_name.count("x") > 1 or joint_name.count("y") > 1 or joint_name.count("z") > 1:
        print("Error in joint_name.count()")
        sys.exit()
    # each joint_name must be hardcoded to use the x,y,z for the axis of rotation
    if joint_name.count("x") == 0 and joint_name.count("y") == 0 and joint_name.count("z") == 0:
        if joint_name.count("rot") == 1:
            extended_joint_axis[joint_name] = "y"
        else:
            print("Error in joint_name.count()")
            sys.exit()
    if joint_name.count("x") == 1:
        extended_joint_axis[joint_name] = "x"
    if joint_name.count("y") == 1:
        extended_joint_axis[joint_name] = "y"
    if joint_name.count("z") == 1:
        extended_joint_axis[joint_name] = "z"

# the axis of rotation for each joint idx
extended_joint_idx_axis = {extended_joint_names.index(k):v for k,v in extended_joint_axis.items()}

"""
link order
- link order as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
- order x,y,z
"""

# complete link orderto generate the skeleton
link_order   = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
               # spinal column
               # 1) BP
               ["ROOTz_joint","BPx_joint"], ["BPx_joint","BPy_joint"], ["BPy_joint","BPz_joint"],
               # 2) BT
               ["BPz_joint","BTx_joint"],   ["BTx_joint","BTy_joint"], ["BTy_joint","BTz_joint"],
               # 3) CS
               ["BTz_joint","CSx_joint"],   ["CSx_joint","CSy_joint"], ["CSy_joint","CSz_joint"],
               # 3) BLN
               ["CSz_joint","BLNx_joint"],  ["BLNx_joint","BLNy_joint"], ["BLNy_joint","BLNz_joint"],
               # 3) BUN
               ["BLNz_joint","BUNx_joint"],  ["BUNx_joint","BUNy_joint"], ["BUNy_joint","BUNz_joint"],
               
               # top left half
               # 1) LS
               ["CSz_joint","LSx_joint"],   ["LSx_joint","LSy_joint"], ["LSy_joint","LSz_joint"],
               # 2) LE
               ["LSz_joint","LEx_joint"],   ["LEx_joint","LEz_joint"],
               # 3) LW
               ["LEz_joint","LWx_joint"],   ["LWx_joint","LWy_joint"],
                
               # top right half
               # 1) RS
               ["CSz_joint","RSx_joint"],   ["RSx_joint","RSy_joint"], ["RSy_joint","RSz_joint"],
               # 2) RE
               ["RSz_joint","REx_joint"],   ["REx_joint","REz_joint"],
               # 3) RW
               ["REz_joint","RWx_joint"],   ["RWx_joint","RWy_joint"],
 
               # bottom left half
               # 1) LH
               ["ROOTz_joint","LHx_joint"], ["LHx_joint","LHy_joint"], ["LHy_joint","LHz_joint"],
               # 2) LK
               ["LHz_joint","LKx_joint"],
               # 3) LA
               ["LKx_joint","LAx_joint"],   ["LAx_joint","LAy_joint"], ["LAy_joint","LAz_joint"],

               # bottom right half
               # 1) RH
               ["ROOTz_joint","RHx_joint"], ["RHx_joint","RHy_joint"], ["RHy_joint","RHz_joint"],
               # 2) RK
               ["RHz_joint","RKx_joint"],
               # 3) RA
               ["RKx_joint","RAx_joint"],   ["RAx_joint","RAy_joint"], ["RAy_joint","RAz_joint"]]

# link order from root to left wrist
link_order_to_left_wrist  = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
                            # 1) LS
                            ["CSz_joint","LSx_joint"],   ["LSx_joint","LSy_joint"], ["LSy_joint","LSz_joint"],
                            # 2) LE
                            ["LSz_joint","LEx_joint"],   ["LEx_joint","LEz_joint"],
                            # 3) LW
                            ["LEz_joint","LWx_joint"],   ["LWx_joint","LWy_joint"]]
# link order from root to right wrist
link_order_to_right_wrist = [["ROOTx_joint","ROOTy_joint"],["ROOTy_joint","ROOTz_joint"],
                            # 1) RS
                            ["CSz_joint","RSx_joint"],   ["RSx_joint","RSy_joint"], ["RSy_joint","RSz_joint"],
                            # 2) RE
                            ["RSz_joint","REx_joint"],   ["REx_joint","REz_joint"],
                            # 3) RW
                            ["REz_joint","RWx_joint"],   ["RWx_joint","RWy_joint"]]

# idx version
link_idx_order_to_left_wrist = []
for link in link_order_to_left_wrist:    
    parent_name = link[0]
    child_name  = link[1]    
    parent_idx = extended_joint_names.index(parent_name)
    child_idx  = extended_joint_names.index(child_name)
    link_idx_order_to_left_wrist.append([parent_idx,child_idx])
link_idx_order_to_right_wrist = []
for link in link_order_to_right_wrist:    
    parent_name = link[0]
    child_name  = link[1]    
    parent_idx = extended_joint_names.index(parent_name)
    child_idx  = extended_joint_names.index(child_name)
    link_idx_order_to_right_wrist.append([parent_idx,child_idx])

"""
parent_dict
- key = child
- value = parent
- order x,y,z
"""

               # ROOT
parent_dict = {"ROOTx_joint":"ROOTx_joint", "ROOTy_joint":"ROOTx_joint", "ROOTz_joint":"ROOTy_joint", 

               # spinal column
               # 1) BP
               "BPx_joint":"ROOTz_joint", "BPy_joint":"BPx_joint", "BPz_joint":"BPy_joint",
               # 2) BT
               "BTx_joint":"BPz_joint",   "BTy_joint":"BTx_joint", "BTz_joint":"BTy_joint",
               # 3) CS
               "CSx_joint":"BTz_joint",   "CSy_joint":"CSx_joint", "CSz_joint":"CSy_joint",
               # 3) BLN
               "BLNx_joint":"CSz_joint",   "BLNy_joint":"BLNx_joint", "BLNz_joint":"BLNy_joint",
               # 3) BUN
               "BUNx_joint":"BLNz_joint",   "BUNy_joint":"BUNx_joint", "BUNz_joint":"BUNy_joint",
               
               # top left half
               # 1) LS
               "LSx_joint":"CSz_joint",   "LSy_joint":"LSx_joint", "LSz_joint":"LSy_joint",
               # 2) LE
               "LEx_joint":"LSz_joint",   "LEz_joint":"LEx_joint", 
               # 3) LW
               "LWx_joint":"LEz_joint",   "LWy_joint":"LWx_joint",
               
               # top right half
               # 1) RS
               "RSx_joint":"CSz_joint", "RSy_joint":"RSx_joint", "RSz_joint":"RSy_joint",
               # 2) RE
               "REx_joint":"RSz_joint", "REz_joint":"REx_joint", "RWx_joint":"REz_joint",
               # 3) RW
               "RWy_joint":"RWx_joint", 
               
               # bottom left half
               # 1) LH
               "LHx_joint":"ROOTz_joint", "LHy_joint":"LHx_joint", "LHz_joint":"LHy_joint",
               # 2) LK
               "LKx_joint":"LHz_joint",
               # 3) LA
               "LAx_joint":"LKx_joint", "LAy_joint":"LAx_joint", "LAz_joint":"LAy_joint",
               
               # bottom right half
               # 1) RH
               "RHx_joint":"ROOTz_joint", "RHy_joint":"RHx_joint", "RHz_joint":"RHy_joint",
               # 2) RK
               "RKx_joint":"RHz_joint",
               # 3) RA
               "RAx_joint":"RKx_joint", "RAy_joint":"RAx_joint", "RAz_joint":"RAy_joint"}
extended_parent_idx_dict = {extended_joint_names.index(k):extended_joint_names.index(v) for k,v in parent_dict.items()}

"""
link direction
- link direction as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
"""

link_direction_dict = {}

# root
link_direction_dict[("ROOTx_joint","ROOTy_joint")] = [0,0,0]
link_direction_dict[("ROOTy_joint","ROOTz_joint")] = [0,0,0]

# spinal column
# 1) ROOT to BP
link_direction_dict[("ROOTz_joint","BPx_joint")] = [0,0,1]
link_direction_dict[("BPx_joint","BPy_joint")]   = [0,0,0]
link_direction_dict[("BPy_joint","BPz_joint")]   = [0,0,0]
# 1) BP to BT
link_direction_dict[("BPz_joint","BTx_joint")]   = [0,0,1]
link_direction_dict[("BTx_joint","BTy_joint")]   = [0,0,0]
link_direction_dict[("BTy_joint","BTz_joint")]   = [0,0,0]
# 1) BT to CS
link_direction_dict[("BTz_joint","CSx_joint")]   = [0,0,1]
link_direction_dict[("CSx_joint","CSy_joint")]   = [0,0,0]
link_direction_dict[("CSy_joint","CSz_joint")]   = [0,0,0]
# 1) CS to BLN
link_direction_dict[("CSz_joint","BLNx_joint")]  = [0,0,1]
link_direction_dict[("BLNx_joint","BLNy_joint")] = [0,0,0]
link_direction_dict[("BLNy_joint","BLNz_joint")] = [0,0,0]
# 1) BLN to BUN
link_direction_dict[("BLNz_joint","BUNx_joint")] = [0,0,1]
link_direction_dict[("BUNx_joint","BUNy_joint")] = [0,0,0]
link_direction_dict[("BUNy_joint","BUNz_joint")] = [0,0,0]

# top left half
# 1) CS to LS
link_direction_dict[("CSz_joint","LSx_joint")]   = [-1,0,0]
link_direction_dict[("LSx_joint","LSy_joint")]   = [0,0,0]
link_direction_dict[("LSy_joint","LSz_joint")]   = [0,0,0]
# 1) LS to LE
link_direction_dict[("LSz_joint","LEx_joint")]   = [0,0,-1]
link_direction_dict[("LEx_joint","LEz_joint")]   = [0,0,0]
# 1) LE to LW
link_direction_dict[("LEz_joint","LWx_joint")]   = [0,0,-1]
link_direction_dict[("LWx_joint","LWy_joint")]   = [0,0,0]

# top right half
# 1) CS to RS
link_direction_dict[("CSz_joint","RSx_joint")]   = [1,0,0]
link_direction_dict[("RSx_joint","RSy_joint")]   = [0,0,0]
link_direction_dict[("RSy_joint","RSz_joint")]   = [0,0,0]
# 1) RS to RE
link_direction_dict[("RSz_joint","REx_joint")]   = [0,0,-1]
link_direction_dict[("REx_joint","REz_joint")]   = [0,0,0]
# 1) RE to RW
link_direction_dict[("REz_joint","RWx_joint")]   = [0,0,-1]
link_direction_dict[("RWx_joint","RWy_joint")]   = [0,0,0]

# bottom left half
# 1) ROOT to LH
link_direction_dict[("ROOTz_joint","LHx_joint")] = [-1,0,0]
link_direction_dict[("LHx_joint","LHy_joint")]   = [0,0,0]
link_direction_dict[("LHy_joint","LHz_joint")]   = [0,0,0]
# 1) LH to LK
link_direction_dict[("LHz_joint","LKx_joint")]   = [0,0,-1]
# 1) LK to LA
link_direction_dict[("LKx_joint","LAx_joint")]   = [0,0,-1]
link_direction_dict[("LAx_joint","LAy_joint")]   = [0,0,0]
link_direction_dict[("LAy_joint","LAz_joint")]   = [0,0,0]

# bottom right half
# 1) ROOT to LH
link_direction_dict[("ROOTz_joint","RHx_joint")] = [1,0,0]
link_direction_dict[("RHx_joint","RHy_joint")]   = [0,0,0]
link_direction_dict[("RHy_joint","RHz_joint")]   = [0,0,0]
# 1) LH to LK
link_direction_dict[("RHz_joint","RKx_joint")]   = [0,0,-1]
# 1) LK to LA
link_direction_dict[("RKx_joint","RAx_joint")]   = [0,0,-1]
link_direction_dict[("RAx_joint","RAy_joint")]   = [0,0,0]
link_direction_dict[("RAy_joint","RAz_joint")]   = [0,0,0]

for k,v in link_direction_dict.items():
    link_direction_dict[k] = np.array(v)
    
# idx version
link_idx_direction_dict = {(extended_joint_names.index(k[0]),extended_joint_names.index(k[1])):v for k,v in link_direction_dict.items()}
    
"""
link length
- link length as described in
  https://ieeexplore.ieee.org/document/7506114 
  Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion
"""

link_length_dict = {}

# root
link_length_dict[("ROOTx_joint","ROOTy_joint")] = 0
link_length_dict[("ROOTy_joint","ROOTz_joint")] = 0

# spinal column
# 1) ROOT to BP
link_length_dict[("ROOTz_joint","BPx_joint")] = 40
link_length_dict[("BPx_joint","BPy_joint")]   = 0
link_length_dict[("BPy_joint","BPz_joint")]   = 0
# 1) BP to BT
link_length_dict[("BPz_joint","BTx_joint")]   = 60
link_length_dict[("BTx_joint","BTy_joint")]   = 0
link_length_dict[("BTy_joint","BTz_joint")]   = 0
# 1) BT to CS
link_length_dict[("BTz_joint","CSx_joint")]   = 188
link_length_dict[("CSx_joint","CSy_joint")]   = 0
link_length_dict[("CSy_joint","CSz_joint")]   = 0
# 1) CS to BLN
link_length_dict[("CSz_joint","BLNx_joint")]  = 22
link_length_dict[("BLNx_joint","BLNy_joint")] = 0
link_length_dict[("BLNy_joint","BLNz_joint")] = 0
# 1) BLN to BUN
link_length_dict[("BLNz_joint","BUNx_joint")] = 30
link_length_dict[("BUNx_joint","BUNy_joint")] = 0
link_length_dict[("BUNy_joint","BUNz_joint")] = 0

# top left half
# 1) CS to LS
link_length_dict[("CSz_joint","LSx_joint")]   = 110
link_length_dict[("LSx_joint","LSy_joint")]   = 0
link_length_dict[("LSy_joint","LSz_joint")]   = 0
# 1) LS to LE
link_length_dict[("LSz_joint","LEx_joint")]   = 188
link_length_dict[("LEx_joint","LEz_joint")]   = 0
# 1) LE to LW
link_length_dict[("LEz_joint","LWx_joint")]   = 145
link_length_dict[("LWx_joint","LWy_joint")]   = 0

# top right half
# 1) CS to RS
link_length_dict[("CSz_joint","RSx_joint")]   = 110
link_length_dict[("RSx_joint","RSy_joint")]   = 0
link_length_dict[("RSy_joint","RSz_joint")]   = 0
# 1) RS to RE
link_length_dict[("RSz_joint","REx_joint")]   = 188
link_length_dict[("REx_joint","REz_joint")]   = 0
# 1) RE to RW
link_length_dict[("REz_joint","RWx_joint")]   = 145
link_length_dict[("RWx_joint","RWy_joint")]   = 0

# bottom left half
# 1) ROOT to LH
link_length_dict[("ROOTz_joint","LHx_joint")] = 52
link_length_dict[("LHx_joint","LHy_joint")]   = 0
link_length_dict[("LHy_joint","LHz_joint")]   = 0
# 1) LH to LK
link_length_dict[("LHz_joint","LKx_joint")]   = 245
# 1) LK to LA
link_length_dict[("LKx_joint","LAx_joint")]   = 246
link_length_dict[("LAx_joint","LAy_joint")]   = 0
link_length_dict[("LAy_joint","LAz_joint")]   = 0

# bottom right half
# 1) ROOT to LH
link_length_dict[("ROOTz_joint","RHx_joint")] = 52
link_length_dict[("RHx_joint","RHy_joint")]   = 0
link_length_dict[("RHy_joint","RHz_joint")]   = 0
# 1) LH to LK
link_length_dict[("RHz_joint","RKx_joint")]   = 245
# 1) LK to LA
link_length_dict[("RKx_joint","RAx_joint")]   = 246
link_length_dict[("RAx_joint","RAy_joint")]   = 0
link_length_dict[("RAy_joint","RAz_joint")]   = 0

# idx version
link_idx_length_dict = {(extended_joint_names.index(k[0]),extended_joint_names.index(k[1])):v for k,v in link_length_dict.items()}

"""

object mocap names
- as defined in the xml files

"""

object_mocap_names = {"apple_juice":["aj_01","aj_02","aj_03","aj_04"],
                      "apple_juice_lid":["aj_lid1","aj_lid2","aj_lid3","aj_lid4"],
                      "broom":["broom1","broom2","broom3","broom4","broom5"],
                      "cucumber_attachment":["ca_01","ca_02","ca_03","ca_04"],
                      "cup_large":["cl_01","cl_02","cl_03","cl_04"],
                      "cup_small":["sc_01","sc_02","sc_03","sc_04"],
                      "cutting_board_small":["cbs_01","cbs_02","cbs_03","cbs_04"],
                      "draining_rack":["draining_rack_01","draining_rack_02","draining_rack_03","draining_rack_04"],
                      "egg_whisk":["ew_01","ew_02","ew_03","ew_04"],
                      "kitchen_sideboard":["ks_01","ks_02","ks_03","ks_04"],
                      "knife_black":["knife_black_01","knife_black_02","knife_black_03","knife_black_04"],
                      "ladle":["ladle_01","ladle_02","ladle_03","ladle_04"],
                      "milk_small":["milk_small_01","milk_small_02","milk_small_03","milk_small_04"],
                      "milk_small_lid":["ms_lid1","ms_lid2","ms_lid3","ms_lid4"],
                      "mixing_bowl_green":["mbg_01","mbg_02","mbg_03","mbg_04"],
                      "mixing_bowl_small":["mbs_01","mbs_02","mbs_03","mbs_04"],
                      "peeler":["peeler1","peeler2","peeler3","peeler4"],
                      "plate_dish":["plate_01","plate_02","plate_03","plate_04"],
                      "rolling_pin":["rolling_pin1","rolling_pin2","rolling_pin3","rolling_pin4"],
                      "salad_fork":["salad_fork1","salad_fork2","salad_fork3","salad_fork4"],
                      "salad_spoon":["salad_spoon1","salad_spoon2","salad_spoon3","salad_spoon4"],
                      "sponge_small":["sponge_dry_01","sponge_dry_02","sponge_dry_03","sponge_dry_04"],
                      "tablespoon":["tablespoon_01","tablespoon_02","tablespoon_03","tablespoon_04"],
                      }

"""
object mocap markers
- coordinates or object mocap markers after de-rotating and de-centering wrt its root_position
- load from the same json file no matter what subset since the mocap markers are computed wrt to each sequence
"""

object_mocap_markers = None
object_mocap_marker_path = os.path.join(os.path.expanduser("~"),"Action-Conditioned-Generation-of-Bimanual-Object-Manipulation","datasets","kit_mocap","cached-data","obj-marker-pos.json")
if os.path.isfile(object_mocap_marker_path):
    with open(object_mocap_marker_path,"r") as fp:
        object_mocap_markers = json.load(fp)
    for filename,_ in object_mocap_markers.items():
        for k,v in object_mocap_markers[filename].items():
            object_mocap_markers[filename][k]["xyz"] = np.array(object_mocap_markers[filename][k]["xyz"])
            object_mocap_markers[filename][k]["var"] = np.array(object_mocap_markers[filename][k]["var"])

"""

object names
- names of all relevant objects
- this variable prevents the code from loading the human pose or kinect cameras into the object dictionary

"""

all_objects =  ["apple_juice","apple_juice_lid", #0,1
                 "broom", #15
                 "cucumber","cucumber_attachment","cup_large","cup_small","cutting_board_small", #2,2,3,3,4
                 "draining_rack", #5
                 "egg_whisk", #6
                 "knife_black", #7
                 "ladle", #16
                 "milk_small","milk_small_lid","mixing_bowl_green","mixing_bowl_small", #0,1,8,8
                 "plate_dish","peeler", #9,10
                 "rolling_pin", #11
                 "salad_fork","salad_spoon","sponge_small", #12, 13, 14
                 "tablespoon",
                # extras
                 "kitchen_sideboard"
                ]
                
"""
action to objects
- the list of objects present in each action
- helps create the one-hot label as I need to know the total objects present in order to create the one-hot mapping
- this can be also be rephrased as folder to objects
- although it is not a good variable to be honest (not really)
"""

# action -> objects
action_to_objects = {}
action_to_objects["Close"]     = ["apple_juice","apple_juice_lid","milk_small","milk_small_lid"]
action_to_objects["Cut"]       = ["cucumber_attachment","cutting_board_small","knife_black"]
action_to_objects["Mix"]       = ["mixing_bowl_green","salad_fork","salad_spoon"]
action_to_objects["Open"]      = ["apple_juice","apple_juice_lid","milk_small","milk_small_lid"]
action_to_objects["Peel"]      = ["cucumber_attachment","cutting_board_small","peeler","mixing_bowl_green"]
action_to_objects["Pour"]      = ["apple_juice","milk_small","cup_small","cup_large"]
action_to_objects["RollOut"]   = ["rolling_pin"]
action_to_objects["Scoop"]     = ["mixing_bowl_green","cup_small","cup_large","ladle","tablespoon","plate_dish","salad_fork","salad_spoon"]
action_to_objects["Stir"]      = ["cup_large","tablespoon","egg_whisk","mixing_bowl_green","mixing_bowl_small"]
action_to_objects["Transfer"]  = ["cutting_board_small","knife_black","mixing_bowl_green"]
action_to_objects["Wipe"]      = ["draining_rack","plate_dish","sponge_small","mixing_bowl_green","cutting_board_small"]

"""
the object the left and right hand holds onto for each action (folder)
"""

held_object = {}
held_object["Close"]    = {"left":["apple_juice_lid","milk_small_lid"], "right":["apple_juice","milk_small"]}
held_object["Cut"]      = {"left":["knife_black"], "right":["cucumber_attachment"]}
held_object["Mix"]      = {"left":["salad_fork"], "right":["salad_spoon"]}
held_object["Open"]     = {"left":["apple_juice_lid","milk_small_lid"], "right":["apple_juice","milk_small"]}
held_object["Peel"]     = {"left":["cucumber_attachment"], "right":["peeler"]}
held_object["Pour"]     = {"left":["cup_small","cup_large"], "right":["apple_juice","milk_small"]}
held_object["RollOut"]  = {"left":["rolling_pin"], "right":["rolling_pin"]}
held_object["Scoop"]    = {"left":["salad_fork","cup_small","cup_large"], "right":["ladle","tablespoon","salad_spoon"]}
held_object["Stir"]     = {"left":["cup_large","mixing_bowl_green","mixing_bowl_small"], "right":["tablespoon","egg_whisk"]}
held_object["Transfer"] = {"left":["cutting_board_small","mixing_bowl_green"], "right":"knife_black"}
held_object["Wipe"]     = {"left":["plate_dish","mixing_bowl_green","cutting_board_small"], "right":"sponge_small"}