"""
This file contains behavioral feature extraction functions for 3D mouse
pose data obtained from DANNCE predictions and COM trajectories. The functions are designed
to quantify movement patterns in the cylinder test and related behavioral recordings for
group comparison, longitudinal analysis, and downstream statistical modeling.

Main functionality:
1. Preprocessing
   - Detect and correct outlier keypoint trajectories using z-score thresholding and
     linear interpolation (`detect_and_fix_outliers`).

2. Cylinder-contact / rearing features
   - Compute forepaw distance to the cylinder wall (`cal_paws_dis`,
     `cal_paws_dis_ForAutoLabel`).
   - Quantify high rearing and mid rearing behavior for left/right forepaws separately
     (`high_mid_rear`) or both forepaws combined (`high_mid_rear_BothForepaws`).

3. Limb airtime features
   - Measure the proportion of time each forepaw or hindpaw is elevated above a landing
     threshold (`air_time`).

4. Center-of-mass (COM) kinematics
   - Compute average COM speed and acceleration in 3D, in the XY plane, and along the Z axis
     after smoothing (`com_velocity`).

5. Keypoint-based motion features
   - Compute absolute keypoint velocity in the global coordinate frame
     (`keypoints_velocity`).
   - Compute relative keypoint velocity after subtracting either a body reference point
     or the COM trajectory (`relative_keypoints_velocity`).
   - Compute keypoint velocity only during frames when the animal is actively moving,
     determined by COM velocity thresholding (`only_moving_keypoints_velocity_raw`).

This file provides a collection of interpretable, hand-crafted behavioral metrics from
3D pose sequences, capturing rearing, limb support, whole-body locomotion, and local
body-part motion. These features can be used to study motor impairments, recovery, or
group differences in stroke and healthy mouse datasets.
"""

import scipy.io as sio
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import copy

def detect_and_fix_outliers(data, z_threshold=2):
    """
    Detect and fix outliers in a 3D time series using Z-score and linear interpolation.

    Args:
        data (np.array): Input array of shape (n_frames, 3), where each row represents a frame
                         and columns represent x, y, z axes.
        z_threshold (float): Z-score threshold to detect outliers (default is 3).

    Returns:
        np.array: Array with outliers detected and fixed by linear interpolation.
    """
    n_frames, n_axes = data.shape
    data_fixed = np.copy(data)
    
    for axis in range(n_axes):  # Iterate over each axis (x, y, z)
        # Calculate the Z-score for each point along the current axis
        mean_axis = np.mean(data[:, axis])
        std_axis = np.std(data[:, axis])
        z_scores = (data[:, axis] - mean_axis) / std_axis

        # Detect outliers (where the absolute Z-score is greater than the threshold)
        outliers = np.abs(z_scores) > z_threshold

        # Get indices of non-outliers to use for interpolation
        valid_indices = np.where(~outliers)[0]
        outlier_indices = np.where(outliers)[0]

        # Linear interpolation
        if len(valid_indices) > 1:  # Make sure there are enough points to interpolate
            interp_func = interp1d(valid_indices, data[valid_indices, axis], kind='linear', fill_value='extrapolate')
            data_fixed[outlier_indices, axis] = interp_func(outlier_indices)
    
    return data_fixed




def cal_paws_dis(radius, paws_prediction, circle_center):
    
    left_paws = radius - ((paws_prediction[:,0,0] - circle_center[0]) ** 2 + (paws_prediction[:,1,0] - circle_center[1]) ** 2) ** 0.5
    right_paws = radius - ((paws_prediction[:,0,1] - circle_center[0]) ** 2 + (paws_prediction[:,1,1] - circle_center[1]) ** 2) ** 0.5    
    
    left_count = left_paws < 15
    right_count = right_paws < 15
    
    return left_paws, right_paws, left_count, right_count



def cal_paws_dis_ForAutoLabel(radius, paws_prediction, circle_center, zz_low, zz_high):
    
    left_paws = radius - ((paws_prediction[:,0,0] - circle_center[0]) ** 2 + (paws_prediction[:,1,0] - circle_center[1]) ** 2) ** 0.5
    right_paws = radius - ((paws_prediction[:,0,1] - circle_center[0]) ** 2 + (paws_prediction[:,1,1] - circle_center[1]) ** 2) ** 0.5    
    
    prediction_length = np.shape(paws_prediction)[0]
    z_filter = np.zeros((prediction_length,), dtype=bool)
    
    for i in range(prediction_length):
        if paws_prediction[i,2,0] > zz_low and paws_prediction[i,2,1] > zz_low and paws_prediction[i,2,0] < zz_high and paws_prediction[i,2,1] < zz_high:
            z_filter[i] = True
    
    return left_paws, right_paws, z_filter


# high rear and mid rear
def high_mid_rear(pred_path,circle_center,radius,rearing_thresholds=[15,20,-5]): # thresholds is touching, high enough and landing
    '''
    INPUT:
        pred_path
        circle_center [x,y], xy are coordinates for circle
        radius is the radius of the cylinder, unit mm
        rearing_thresholds [x,y,z], x is threshold for touching, y is threshold for high enough, z is threshold for landing
        
    OUTPUT:
        result_list = [left_high, left_mid, right_high, right_mid]
        result_names_list contains all names in the result
    '''

    result_names_list = ['left_high', 'left_mid', 'right_high', 'right_mid']

    hands_prediction = sio.loadmat(pred_path)['pred'][:,:,[8,12]] # shape should be 30000*3*2; 8 is left and 12 is right
    left_distance_all, right_distance_all, left_close, right_close = cal_paws_dis(radius, hands_prediction, circle_center)

    sides = ['left','right']
    results_list = [] # order should be left_high, left_mid, right_high, right_mid
    for side in sides:
        if side == 'left':
            distance_data = left_distance_all
            heights_data = hands_prediction[:,2,0]
        elif side == 'right':
            distance_data = right_distance_all
            heights_data = hands_prediction[:,2,1]         

        temp_data = np.squeeze(np.dstack((distance_data, heights_data))) # shape is (30000,2)
        # print(temp_data.shape)
        high_rare = 0
        mid_rare = 0
        # threshold0
        for i in range(len(temp_data)):
            if temp_data[i,0] <= rearing_thresholds[0] and temp_data[i,1] >= rearing_thresholds[1]:
                high_rare = high_rare + 1
            elif temp_data[i,0] > rearing_thresholds[0] and temp_data[i,1] >= rearing_thresholds[2]:
                mid_rare = mid_rare + 1

        high_rate, mid_rate = high_rare/30000, mid_rare/30000   
        results_list.append(high_rate)
        results_list.append(mid_rate)


    return results_list, result_names_list

# high rear and mid rear for both
def high_mid_rear_BothForepaws(pred_path,circle_center,radius,rearing_thresholds=[15,20,-5]): # thresholds is touching, high enough and landing
    '''
    INPUT:
        pred_path
        circle_center [x,y], xy are coordinates for circle
        radius is the radius of the cylinder, unit mm
        rearing_thresholds [x,y,z], x is threshold for touching, y is threshold for high enough, z is threshold for landing
        
    OUTPUT:
        result_list = [left_high, left_mid, right_high, right_mid]
        result_names_list contains all names in the result
    '''

    result_names_list = ['high_rear', 'mid_rear']

    hands_prediction = sio.loadmat(pred_path)['pred'][:,:,[8,12]] # shape should be 30000*3*2; 8 is left and 12 is right
    left_distance_all, right_distance_all, left_close, right_close = cal_paws_dis(radius, hands_prediction, circle_center)

    results_list = [] # order should be high_rear, mid_rear
    distance_data = np.concatenate((left_distance_all,right_distance_all)) # shape is (60000,)
    heights_data = np.concatenate((hands_prediction[:,2,0],hands_prediction[:,2,1])) # shape is (60000,)
    temp_data = np.squeeze(np.dstack((distance_data, heights_data))) # shape is (60000,2)
    # print(temp_data.shape)
    high_rare = 0
    mid_rare = 0
    # threshold0
    for i in range(len(temp_data)):
        if temp_data[i,0] <= rearing_thresholds[0] and temp_data[i,1] >= rearing_thresholds[1]:
            high_rare = high_rare + 1
        elif temp_data[i,0] > rearing_thresholds[0] and temp_data[i,1] >= rearing_thresholds[2]:
            mid_rare = mid_rare + 1

    high_rate, mid_rate = high_rare/60000, mid_rare/60000   
    results_list.append(high_rate)
    results_list.append(mid_rate)


    return results_list, result_names_list

# hanging in the air time for each limb
def air_time(pred_path,landing_threshold = -8):
    '''
    INPUT:
        pred_path
        landing_threshold is threshold for landing, unit mm
        
    OUTPUT:
        result_list = [left forepaw, right forepaw, left hindpaw, right hindpaw] # ration that specific limb in the air
        result_names_list contains all names in the result
    '''
    limb_list = [8,12,16,19] # left forepaw, right forepaw, left hindpaw, right hindpaw
    results_list = []
    result_names_list = ['left_forepaw', 'right_forepaw', 'left_hindpaw', 'right_hindpaw']
    for limb_id in limb_list:
        limb_prediction = np.squeeze(sio.loadmat(pred_path)['pred'][:,:,limb_id]) # shape should be 30000*3
        heights_data = np.squeeze(limb_prediction[:,2]) # shape should be 30000*1
        in_the_air = 0
        for i in range(len(heights_data)):
            if heights_data[i] > landing_threshold:
                in_the_air += 1
        airing_ration = in_the_air / len(heights_data)
        results_list.append(airing_ration)
    return results_list, result_names_list



# com velocity and acceleration
def com_velocity(com_path, fps = 100):
    results_list = []
    result_names_list = ['average_speed','xy_plane_speed','z_speed',
    'average_acceleration','xy_plane_acceleration','z_acceleration']
    com_prediction = sio.loadmat(com_path)['com']
    # smoothing
    for i in range(3):
        com_prediction[:,i] = signal.savgol_filter(np.squeeze(com_prediction[:,i]), window_length=17, polyorder=3, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
        com_prediction[:,i] = signal.medfilt(np.squeeze(com_prediction[:,i]), kernel_size=11)

    speed_list, xy_speed_list, z_speed_list = [], [], []
    acceleration_list, xy_a_list, z_a_list = [], [], []
    for j in range(len(com_prediction)-1):
        distance = np.sqrt((com_prediction[j,0] - com_prediction[j+1,0]) ** 2 + (com_prediction[j,1] - com_prediction[j+1,1]) ** 2 + (com_prediction[j,2] - com_prediction[j+1,2]) ** 2)
        speed_list.append(distance/(1/fps)) # unit mm/s
        xy_distance = np.sqrt((com_prediction[j,0] - com_prediction[j+1,0]) ** 2 + (com_prediction[j,1] - com_prediction[j+1,1]) ** 2)
        xy_speed_list.append(xy_distance/(1/fps)) # unit mm/s
        z_distance = np.sqrt((com_prediction[j,2] - com_prediction[j+1,2]) ** 2)
        z_speed_list.append(z_distance/(1/fps)) # unit mm/s
    
    for k in range(len(speed_list)-1):
        acceleration_list.append(abs(speed_list[k]-speed_list[k+1])/(1/fps))
        xy_a_list.append(abs(xy_speed_list[k]-xy_speed_list[k+1])/(1/fps))
        z_a_list.append(abs(z_speed_list[k]-z_speed_list[k+1])/(1/fps))
    
    results_list = [np.mean(speed_list),np.mean(xy_speed_list),np.mean(z_speed_list),np.mean(acceleration_list),np.mean(xy_a_list),np.mean(z_a_list)]

    return results_list, result_names_list



# all keypoints speed
def keypoints_velocity(pred_path, filtered_threshold=0, fps = 100, gap = 5, fix_outlier=False):
    results_list = []
    result_names_list = ['EarL','EarR','Snout','SpineF','SpineM','TailB',
    'ForepawL','WristL','ElbowL','ShoulderL',
    'ForepawR','WristR','ElbowR','ShoulderR',
    'HindpawL','AnkleL','KneeL','HindpawR','AnkleR','KneeR']
    dannce_prediction = sio.loadmat(pred_path)['pred'] # shape should be 30000*3*22

    results_list = []
    for keypoint_id in range(22):
        if keypoint_id == 6 or keypoint_id == 7:
            continue
        temp_speed_list = []
        temp_keypoints_array = np.squeeze(dannce_prediction[:,:,keypoint_id]) # shape should be 30000*3
        if fix_outlier:
            temp_keypoints_array = detect_and_fix_outliers(temp_keypoints_array)
        for frame in range(len(dannce_prediction)-gap):
            gapped_distance = np.sqrt((temp_keypoints_array[frame,0] - temp_keypoints_array[frame+gap,0]) ** 2 \
            + (temp_keypoints_array[frame,1] - temp_keypoints_array[frame+gap,1]) ** 2 \
            + (temp_keypoints_array[frame,2] - temp_keypoints_array[frame+gap,2]) ** 2)

            temp_velocity = gapped_distance/(gap/fps) # unit mm/s
            if temp_velocity >= filtered_threshold:
                temp_speed_list.append(temp_velocity) # unit mm/s

        results_list.append(np.mean(temp_speed_list))

    return results_list, result_names_list


def relative_keypoints_velocity(com_path, pred_path, use_spineM = True, filtered_threshold=0,fps = 100, gap = 5, fix_outlier=False):
    results_list = []
    result_names_list = ['EarL','EarR','Snout','SpineF','SpineM','TailB',
    'ForepawL','WristL','ElbowL','ShoulderL',
    'ForepawR','WristR','ElbowR','ShoulderR',
    'HindpawL','AnkleL','KneeL','HindpawR','AnkleR','KneeR']
    dannce_prediction = sio.loadmat(pred_path)['pred'] # shape should be 30000*3*22


    com_prediction = sio.loadmat(com_path)['com'] # shape should be 30000*3
    # smoothing
    for i in range(3):
        com_prediction[:,i] = signal.savgol_filter(np.squeeze(com_prediction[:,i]), window_length=17, polyorder=3, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
        com_prediction[:,i] = signal.medfilt(np.squeeze(com_prediction[:,i]), kernel_size=5)

    # breakpoint()

    if use_spineM:
        spineM_array = copy.deepcopy(dannce_prediction[:,:,5]) # try tailB here, spineM should be 4
        for i in range(22):
            dannce_prediction[:,:,i] = dannce_prediction[:,:,i] - spineM_array
    else:
        for i in range(22):
            dannce_prediction[:,:,i] = dannce_prediction[:,:,i] - com_prediction

        

    results_list = []
    for keypoint_id in range(22):
        if keypoint_id == 6 or keypoint_id == 7:
            continue
        temp_speed_list = []
        temp_keypoints_array = np.squeeze(dannce_prediction[:,:,keypoint_id]) # shape should be 30000*3
        if fix_outlier:
            temp_keypoints_array = detect_and_fix_outliers(temp_keypoints_array)
        for frame in range(len(dannce_prediction)-gap):
            # breakpoint()
            gapped_distance = np.sqrt((temp_keypoints_array[frame,0] - temp_keypoints_array[frame+gap,0]) ** 2 \
            + (temp_keypoints_array[frame,1] - temp_keypoints_array[frame+gap,1]) ** 2 \
            + (temp_keypoints_array[frame,2] - temp_keypoints_array[frame+gap,2]) ** 2)

            temp_velocity = gapped_distance/(gap/fps) # unit mm/s
            if temp_velocity >= filtered_threshold:
                temp_speed_list.append(temp_velocity) # unit mm/s

        # breakpoint()
        results_list.append(np.mean(temp_speed_list))

    return results_list, result_names_list  



def only_moving_keypoints_velocity_raw(com_path, pred_path, filtered_threshold=10,fps = 100, gap = 25, fix_outlier=False, if_relative=False):
    results_list = []
    result_names_list = ['EarL','EarR','Snout','SpineF','SpineM','TailB',
    'ForepawL','WristL','ElbowL','ShoulderL',
    'ForepawR','WristR','ElbowR','ShoulderR',
    'HindpawL','AnkleL','KneeL','HindpawR','AnkleR','KneeR']
    dannce_prediction = sio.loadmat(pred_path)['pred'] # shape should be 30000*3*22


    com_prediction = sio.loadmat(com_path)['com'] # shape should be 30000*3
    # smoothing
    for i in range(3):
        com_prediction[:,i] = signal.savgol_filter(np.squeeze(com_prediction[:,i]), window_length=17, polyorder=3, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
        com_prediction[:,i] = signal.medfilt(np.squeeze(com_prediction[:,i]), kernel_size=5)


    # use tail B as the relative point
    if if_relative:
        spineM_array = copy.deepcopy(dannce_prediction[:,:,5]) # try tailB here, spineM should be 4
        for i in range(22):
            dannce_prediction[:,:,i] = dannce_prediction[:,:,i] - spineM_array



    results_list = []
    for keypoint_id in range(22):
        if keypoint_id == 6 or keypoint_id == 7:
            continue
        temp_speed_list = []
        temp_keypoints_array = np.squeeze(dannce_prediction[:,:,keypoint_id]) # shape should be 30000*3
        if fix_outlier:
            temp_keypoints_array = detect_and_fix_outliers(temp_keypoints_array)
        for frame in range(len(dannce_prediction)-gap):

            com_gapped_distance = np.sqrt((com_prediction[frame,0] - com_prediction[frame+gap,0]) ** 2 \
            + (com_prediction[frame,1] - com_prediction[frame+gap,1]) ** 2 \
            + (com_prediction[frame,2] - com_prediction[frame+gap,2]) ** 2)

            temp_com_velocity = com_gapped_distance/(gap/fps)

            if temp_com_velocity >= filtered_threshold:

                gapped_distance = np.sqrt((temp_keypoints_array[frame,0] - temp_keypoints_array[frame+gap,0]) ** 2 \
                + (temp_keypoints_array[frame,1] - temp_keypoints_array[frame+gap,1]) ** 2 \
                + (temp_keypoints_array[frame,2] - temp_keypoints_array[frame+gap,2]) ** 2)

                temp_velocity = gapped_distance/(gap/fps) # unit mm/s
                temp_speed_list.append(temp_velocity) # unit mm/s

        # breakpoint()
        results_list.append(np.mean(temp_speed_list))

    return results_list, result_names_list  

