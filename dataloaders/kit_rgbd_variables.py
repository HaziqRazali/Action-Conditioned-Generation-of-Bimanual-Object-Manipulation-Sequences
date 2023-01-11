import os
import json
import numpy as np

# sequences to skip
skip = [
# todo: find out why i skip this
# - must be due to bad ground truths
["subject_1","task_6_w_hard_drive","take_0"],
# temporary skip while i manually fix get-2d-track-data.py
["subject_6","task_4_k_wiping","take_8"],

# bad track data
["subject_1","task_1_k_cooking","take_6"],
["subject_1","task_1_k_cooking","take_9"],
["subject_1","task_3_k_pouring","take_3"],
["subject_1","task_3_k_pouring","take_5"],
["subject_1","task_3_k_pouring","take_8"],
["subject_1","task_3_k_pouring","take_9"],
["subject_1","task_7_w_free_hard_drive","take_0"],

["subject_2","task_2_cooking_with_bowls","take_0"],
["subject_2","task_3_k_pouring","take_2"],
["subject_2","task_3_k_pouring","take_4"],
["subject_2","task_4_k_wiping","take_1"],
["subject_2","task_4_k_wiping","take_6"],
["subject_2","task_6_w_hard_drive","take_2"],
["subject_2","task_6_w_hard_drive","take_5"],
["subject_2","task_8_w_hammering","take_0"],
["subject_2","task_8_w_hammering","take_1"],

["subject_3","task_1_k_cooking","take_1"],
["subject_3","task_1_k_cooking","take_2"],
["subject_3","task_1_k_cooking","take_3"],
["subject_3","task_1_k_cooking","take_4"],
["subject_3","task_1_k_cooking","take_5"],
["subject_3","task_1_k_cooking","take_8"],                
["subject_3","task_2_k_cooking_with_bowls","take_3"],
["subject_3","task_2_k_cooking_with_bowls","take_4"],
["subject_3","task_2_k_cooking_with_bowls","take_5"],
["subject_3","task_2_k_cooking_with_bowls","take_6"],                
["subject_3","task_3_k_pouring","take_0"],
["subject_3","task_3_k_pouring","take_3"],
["subject_3","task_3_k_pouring","take_6"],
["subject_3","task_3_k_pouring","take_8"],
["subject_3","task_3_k_pouring","take_9"],                
["subject_3","task_6_w_hard_drive","take_9"],                
["subject_3","task_7_free_hard_drive","take_0"],
["subject_3","task_7_free_hard_drive","take_3"],
["subject_3","task_7_free_hard_drive","take_4"],
["subject_3","task_7_free_hard_drive","take_6"],
["subject_3","task_7_free_hard_drive","take_7"],
["subject_3","task_7_free_hard_drive","take_9"],                
["subject_3","task_8_w_hammering","take_5"],
["subject_3","task_8_w_hammering","take_6"],

["subject_4","task_1_k_cooking","take_2"],
["subject_4","task_1_k_cooking","take_7"],
["subject_4","task_1_k_cooking","take_8"],                
["subject_4","task_2_k_cooking_with_bowls","take_3"],                
["subject_4","task_3_k_pouring","take_2"],
["subject_4","task_3_k_pouring","take_5"],
["subject_4","task_3_k_pouring","take_6"],
["subject_4","task_3_k_pouring","take_8"],
["subject_4","task_3_k_pouring","take_9"],                
["subject_4","task_6_w_hard_drive","take_4"],                
["subject_4","task_7_w_free_hard_drive","take_2"],
["subject_4","task_7_w_free_hard_drive","take_5"],
["subject_4","task_7_w_free_hard_drive","take_7"],
["subject_4","task_8_w_hammering","take_3"],

["subject_5","task_3_k_pouring","take_4"],
["subject_5","task_3_k_pouring","take_5"],
["subject_5","task_3_k_pouring","take_7"],
["subject_5","task_3_k_pouring","take_9"],
["subject_5","task_6_w_hard_drive","take_0"],
["subject_5","task_6_w_hard_drive","take_4"],
["subject_5","task_6_w_hard_drive","take_7"],
["subject_5","task_7_w_free_hard_drive","take_1"],
["subject_5","task_7_w_free_hard_drive","take_3"],

["subject_6","task_3_k_pouring","take_3"],
["subject_6","task_3_k_pouring","take_6"],
["subject_6","task_6_w_hard_drive","take_2"],
["subject_6","task_6_w_hard_drive","take_8"],
["subject_6","task_7_w_free_hard_drive","take_8"],                
["subject_6","task_8_w_hammering","take_4"],
["subject_6","task_8_w_hammering","take_5"]                
]

# list of all objects
all_objects = ["bottle","whisk","bowl","cup","knife","banana","screwdriver","sponge","cuttingboard","cereals","woodenwedge","saw","hammer","harddrive","cuttingboard"]

# action -> objects
action_to_objects = {}
action_to_objects["task_1_k_cooking"]              = ["LeftHand","RightHand","bottle","bowl","whisk"]
action_to_objects["task_2_k_cooking_with_bowls"]   = ["LeftHand","RightHand","bowl","whisk"]
action_to_objects["task_3_k_pouring"]              = ["LeftHand","RightHand","bottle","cup"]
action_to_objects["task_4_k_wiping"]               = ["LeftHand","RightHand","sponge","bottle","bowl","whisk","cup","banana","knife","cuttingboard","cereals"]
action_to_objects["task_5_k_cereals"]              = []
action_to_objects["task_6_w_hard_drive"]           = ["LeftHand","RightHand","harddrive","screwdriver"]
action_to_objects["task_7_w_free_hard_drive"]      = ["LeftHand","RightHand","harddrive","screwdriver"]
action_to_objects["task_8_w_hammering"]            = ["LeftHand","RightHand","hammer","woodenwedge"]
action_to_objects["task_9_w_sawing"]               = ["LeftHand","RightHand","saw","woodenwedge"]