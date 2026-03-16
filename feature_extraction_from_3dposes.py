#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary:
This script processes 3D pose estimation data from mouse recordings to extract behavioral features for stroke analysis.

Workflow:
1. Define mouse groups (stroke and healthy), including mouse IDs and longitudinal recording timestamps.
2. Iterate through all recordings in the dataset folders and match them to the corresponding
   mouse ID and timestamp.
3. Load necessary data for each recording:
   - 3D keypoint predictions from DANNCE
   - Center of mass (COM) trajectories
   - Precomputed cylinder center locations
4. Apply a selected semi-traditional behavioral feature extraction method
   (e.g., rearing metrics, airtime, COM velocity, keypoint velocity).
5. Compute behavioral metrics for each recording based on the selected method.
6. Aggregate results across all mice and timepoints.
7. Save the extracted features into a CSV file for downstream analysis
   (e.g., longitudinal stroke recovery assessment or statistical comparison between groups).

The script is designed to support multiple feature extraction methods, controlled by
the `feature_function` option.
"""


import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import pickle
from tqdm import tqdm
from feature_function import *


######################################################
# options:
# 'HighMidRear', 'AirTime', 'ComVelocity', 'AutoLabel','keypoints_velocity','relative_keypoints_velocity', 'high_mid_rear_BothForepaws', 'only_moving_keypoints_velocity_raw'
semi_traditional_method = 'only_moving_keypoints_velocity_raw' 

# set up save folder and csv file name
save_base_folder_path = '/stroke_analysis/'
save_path = 'only_moving_keypoints_velocity_raw.csv'

pd_frame = pd.DataFrame(columns=['group','time_stamp','mouse_name','result_name','result'])
######################################################




mouse_group_information = {}
mouse_group_names = ['stroke','healthy']
mouse_group_ids = [['c1_m1','c1_m2','c1_m3','c1_m4','c1_m5'],
                   ['c2_m6','c2_m7','c2_m8','c2_m9','c2_m10']]
mouse_group_timestamps = [['12022021_d-1','12072021_d4','12102021_d7','12172021_w2','12232021_d20','12312021_w4','20220114_w6','20220128_w8','20220211_w10','20220225_w12','20220328_w16'],
                          ['12022021_d-1','12072021_d4','12102021_d7','12172021_w2','12232021_d20','12312021_w4','20220114_w6','20220128_w8','20220211_w10','20220225_w12','20220328_w16']]

for group_count, mouse_group in enumerate(mouse_group_names):
    mouse_group_information[mouse_group] = {}
    mouse_group_information[mouse_group]['mouse_ids'] = mouse_group_ids[group_count]
    mouse_group_information[mouse_group]['timestamps'] = mouse_group_timestamps[group_count]

# cylinder center data
cylinder_center_dict_path = '/hpc/group/tdunn/segura-behavior/sihan_find_cylinder_by_SAM/final_final_cylinder_center_214.pkl'
with open(cylinder_center_dict_path,'rb') as f:
    cylinder_center_dict = pickle.load(f)

pred_mat = 'smoothed_prediction_twd'

# cylinder radius
radius = 47.5

recording_count = 0
index_count = 0

for group_name in tqdm(mouse_group_information):
    
    print('start proccessing group ' + group_name + ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    temp_group_information = mouse_group_information[group_name]
    temp_group_timestamps = temp_group_information['timestamps']
    temp_group_mouseids = temp_group_information['mouse_ids']
    
    if group_name == 'stroke':
        except_recordings = []
        base_path = '/stroke_behavior'
        camera_parameters_name = '_label3d_dannce.mat'
    elif group_name == 'healthy':
        except_recordings = []
        base_path = '/healthy_behavior'
        camera_parameters_name = '_label3d_dannce.mat'
    elif group_name == 'sham':
        except_recordings = []
        base_path = '/healthy_behavior'
        camera_parameters_name = '_smallcylinder_label3d_dannce.mat'
    elif group_name == 'treated':
        except_recordings = [] 
        base_path = '/healthy_behavior'
        camera_parameters_name = '_smallcylinder_label3d_dannce.mat'


    filelist = os.listdir(base_path)
    # create pd data frame here

    for time in temp_group_timestamps:
        for mouse_name in temp_group_mouseids:
            for i in range(len(filelist)):
                if mouse_name in filelist[i] and time in filelist[i] and 'cylinder' in filelist[i] and filelist[i] not in except_recordings:
                    folder_name = filelist[i]
                    print(folder_name)
                    circle_center = cylinder_center_dict[folder_name]['circle_center']
                    temp_pred_path = base_path + '/' + filelist[i] + '/' + pred_mat       
                    temp_com_path = base_path + '/' + filelist[i] + '/DANNCE/predict_results/twd/com3d_used.mat'

                    if semi_traditional_method == 'HighMidRear':
                        temp_mouse_result_list, result_name_list = high_mid_rear(temp_pred_path,circle_center,radius)
                    elif semi_traditional_method == 'AirTime':
                        temp_mouse_result_list, result_name_list = air_time(temp_pred_path)
                    elif semi_traditional_method == 'ComVelocity':
                        temp_mouse_result_list, result_name_list = com_velocity(temp_com_path)
                    elif semi_traditional_method == 'AutoLabel':
                        temp_mouse_result_list, result_name_list = auto_label(temp_pred_path,circle_center,radius)
                    elif semi_traditional_method == 'keypoints_velocity':
                        temp_mouse_result_list, result_name_list = keypoints_velocity(temp_pred_path,filtered_threshold=10, gap=25, fix_outlier=False)
                    elif semi_traditional_method == 'relative_keypoints_velocity':
                        temp_mouse_result_list, result_name_list = relative_keypoints_velocity(temp_com_path,temp_pred_path,filtered_threshold=10, gap=25,fix_outlier=False)
                    elif semi_traditional_method == 'high_mid_rear_BothForepaws':
                        temp_mouse_result_list, result_name_list = high_mid_rear_BothForepaws(temp_pred_path,circle_center,radius,rearing_thresholds=[5,20,-5])
                    elif semi_traditional_method == 'only_moving_keypoints_velocity_raw':
                        temp_mouse_result_list, result_name_list = only_moving_keypoints_velocity_raw(temp_com_path,temp_pred_path,if_relative=True)




                    print('already finished recrding: ' + str(recording_count))
                    recording_count += 1

        

pd_frame.to_csv(save_base_folder_path+save_path, index=False)


