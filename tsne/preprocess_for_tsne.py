"""
Author: Sophie
"""
import os
import numpy as np
import scipy.io as sio
import pandas as pd

"""
This script can be used only when the predictions you are using are all under experiment_root_folder, \
otherwise you will need to modify this script to run through your predictions and format them into \
desired mat format.
If your folder follow the naming rules and are all one level under experiment_root_folder, \
you do not need to have ClusterDirectory filled out in the metadata.csv
"""

if __name__ == '__main__':
    # the folder thatcontains all the videos and predictions
    base_path = "/hpc/group/seguralab/experiment_root_folder/"
    metadata = "data/metadata.csv"
    formatted_save_path = "data/formatted"
    os.makedirs(formatted_save_path, exist_ok=True) # create save_path

    df = pd.read_csv(metadata)
    bad_pred = [] # Put the folders you want to ignore here.
    for k, row in df.iterrows():
        # You will need to change this if you have a different naming rule 
        project_folder = \
            str(row["Date"]) + "_" + row["Timepoint"] + "_" + row["AnimalID"][:2] + \
                "_" + row["AnimalID"][2:] + "_cylinder" # folder for one recording
        if project_folder in bad_pred:
            continue
        prediction_path = base_path + project_folder + "/smoothed_prediction_twd5_medfilt5.mat"

        try:
            # You will need to change this if your recording is not 30000 frames
            pred = sio.loadmat(prediction_path)['pred'][:30000]
        except FileNotFoundError as error:
            print(error)
            breakpoint()

        # output
        print(len(pred))
        print("pred shape", pred.shape)
        sampleID = np.array(list(range(0, pred.shape[0])))
        # Note: this will change the skeleton to 18 key points later(ignoring shoulders). If you (only to whom from Segura lab) need help to change this, please contact sophie.shi@duke.edu in tdunn lab
        struct = {'EarL': pred[:,:,0],
                'EarR': pred[:,:,1],
                'Snout': pred[:,:,2],
                'SpineF': pred[:,:,3],
                'SpineM': pred[:,:,4],
                'Tail_base_': pred[:,:,5],
                'Tail_mid_': pred[:,:,6],
                'Tail_end_': pred[:,:,7],
                'Forepaw_L': pred[:,:,8],
                'Wrist_L': pred[:,:,9],
                'Forelimb_L': pred[:,:,10],
                'ShouderL': pred[:,:,11],
                'Forepaw_R': pred[:,:,12],
                'Wrist_R': pred[:,:,13],
                'ForeLimb_R': pred[:,:,14],
                'ShouderR': pred[:,:,15],
                'Hindpaw_L': pred[:,:,16],
                'Ankel_L': pred[:,:,17],
                'Hindlimb_L': pred[:,:,18],
                'Hindpaw_R': pred[:,:,19],
                'Ankel_R': pred[:,:,20],
                'Hindlimb_R': pred[:,:,21],
                'sampleID': sampleID
                }

        # You can change the save path. This is saved in one folder for easier access
        savefile = f"{formatted_save_path}/{project_folder}_5min_dappy_format.mat"
        mdict = {'predictions':struct,
                }
        # hdf5storage.savemat(savefile, mdict, appendmat=True, format='7.3')
        sio.savemat(savefile, mdict, format='5')
        print(savefile)
        df.at[k, "ClusterDirectory"] = savefile

    print("data formatted")
    df.to_csv(metadata, index=False) 
