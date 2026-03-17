from neuroposelib import read
from neuroposelib import vis
import numpy as np
import time
from IPython.display import Video
from pathlib import Path
import matplotlib.pyplot as plt

from neuroposelib import preprocess
from neuroposelib import write
from neuroposelib import features

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

########## Setting ##########
analysis_key = "stroke_exp" # this is just the fole name of your config yaml file
#############################
config = read.config("./configs/" + analysis_key + ".yaml")


connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)

# Make out_path
Path(config["out_path"]).mkdir(parents=True, exist_ok=True)

# Merge prediction
pose, ids, meta, meta_by_frame = read.pose_from_meta(
    path=config["meta_path"], connectivity=connectivity
)

write.pose_h5(pose,ids, config['data_path'] + 'pose_merged.h5')

# Read pose_merged.h if already saved
# pose, ids = read.pose_h5(config["data_path"] + "pose_merged.h5")

print("Pose shape (# frames x # keypoints x 3 coordinates): ")
print(pose.shape)

# meta, meta_by_frame = read.meta(path=config['meta_path'], id = ids)
print(meta)
print("\n" + str(ids))



pose_aligned = preprocess.align_floor_by_id(pose=pose, ids=ids, foot_id=12, head_id=0)
write.pose_h5(pose_aligned, ids, config["data_path"] + "pose_aligned.h5")

# Align the videos

############# Optional visulization ###############
# vis.pose.arena3D(
#     pose_aligned,
#     connectivity,
#     frames=[1000, 500000],
#     N_FRAMES=150,
#     dpi=100,
#     VID_NAME="aligned.mp4",
#     SAVE_ROOT=config["out_path"],
# )
####################################################


# Provide the mid-spine and the mid-spine -> front-spine indices.
pose = preprocess.rotate_spine(preprocess.center_spine(pose_aligned, keypt_idx=4), keypt_idx=[4, 3])

def shift_spine_to_xz_plane_relative(pose, keypt_idx=[4, 3]):
    """
    Shifts each frame in the pose sequence by the same vertical distance required to bring the spineF 
    of the middle frame onto the xz plane. This preserves the relative vertical positions of all frames.
    """
    num_frames, num_joints, _ = pose.shape
    middle_frame_idx = num_frames // 2  # Index of the middle frame

    # Determine the vertical shift needed to bring the spineF of the middle frame to the xz plane
    shift_y = -pose[middle_frame_idx, keypt_idx[1], 1]

    # Apply this shift to the y-coordinate of every joint in every frame
    pose_shifted = np.copy(pose)
    pose_shifted[:, :, 1] += shift_y

    return pose_shifted

#### This is optional shift, used for better visualization in grid
pose_visualize = shift_spine_to_xz_plane_relative(pose_aligned, keypt_idx=[4, 3])


############# Optional visulization for debug ###############
# vis.pose.arena3D(
#     pose,
#     connectivity,
#     frames=[50000],
#     N_FRAMES=150,
#     dpi=100,
#     VID_NAME="centered.mp4",
#     SAVE_ROOT=config["out_path"],
# )

# vis.pose.arena3D(
#     pose_visualize,
#     connectivity,
#     frames=[50000],
#     N_FRAMES=150,
#     dpi=100,
#     VID_NAME="shifted.mp4",
#     SAVE_ROOT=config["out_path"],
# )
# ####################################################

# Video(config["out_path"] + "vis_centered.mp4", width=600, height=600)


# Calculating joint angles
angles, angle_labels = features.get_angles(pose, connectivity.angles)
# deal with nan
nans, x= nan_helper(angles)
angles[nans]= np.interp(x(nans), x(~nans), angles[~nans])
print("angle", angles.shape)
print(np.unique(np.where(np.isnan(angles))[0]))
print(np.unique(np.where(np.isnan(angles))[1]))
# print(angle_labels[9])
# print(angle_labels[20])
print("nan", np.isnan(np.sum(angles)))
print("angles", angles)
print("inf", np.isinf(np.sum(angles)))

# Reshape pose to get egocentric pose features
ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)

t = time.time()

pc_feats, pc_labels = features.pca(
    np.concatenate((ego_pose, angles), axis=1),  labels + angle_labels, categories=["ang", "ego_euc"], n_pcs=5, method="fbpca"
)

print("PCA time: " + str(time.time() - t))

del ego_pose, labels
del angles, angle_labels


wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids, f_s=100, freq=np.linspace(1, 25, 25), w0=5
)


# PCA on wavelet features
pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    categories=["ang", "ego_euc"],
    n_pcs=5,
    method="fbpca",
)

del wlet_feats, wlet_labels
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
del pc_wlet, pc_wlet_labels

# Optionally save full PC features to file
# write.features_h5(
#     pc_feats, pc_labels, path="".join([config["out_path"], "pca_feats.h5"])
# )


from neuroposelib import DataStruct as ds

data_obj = ds.DataStruct(
    pose=pose,
    id=ids,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)

data_obj.features = pc_feats

# When using high framerate data, downsampling may be necessary in order to 
# discover granular structure in embedding, we used 10 by default in Segura lab setting, set in config
data_obj = data_obj[:: config["downsample"], :]


from neuroposelib.embed import Embed

embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

from neuroposelib.embed import Watershed
# Watershed clustering
data_obj.ws = Watershed(
    sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

print("Writing Data Object to pickle")
data_obj.write_pickle(''.join([config['out_path'],'/']))

# Plot density
vis.plot.density(
    data_obj.ws.density,
    data_obj.ws.borders,
    filepath=config["out_path"] + "/density.png",
    show=True,
)

# of each id 
vis.plot.density_cat(
    data=data_obj,
    column="id",
    watershed=data_obj.ws,
    filepath=config["out_path"] + "/density_id.png",
    show=True,
)


vis.plot.density_cat(
    data=data_obj,
    column="Condition",
    watershed=data_obj.ws,
    filepath=config["out_path"] + "/density_condition.png",
    show=True,
)

vis.plot.density_cat(
    data=data_obj,
    column="condition_timepoints",
    watershed=data_obj.ws,
    filepath=config["out_path"] + "/density_condition_timepoints.png",
    show=True,
)


# vis.plot.density_cat(
#     data=data_obj,
#     column="Condition",
#     watershed=data_obj.ws,
#     filepath=config["out_path"] + "/density_condition_vmax.png",
#     show=True,
#     vmax=0.35,
# )


vis.pose.sample_grid3D(
    # pose-pose.mean(axis=-2, keepdims=True),
    pose_visualize,
    connectivity=connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=16,
    centered=True,
    N_FRAMES=150,
    fps=90,
    dpi=100,
    watershed=data_obj.ws,
    embed_vals=None,
    VID_NAME = "cluster",
    filepath=config["out_path"],
)