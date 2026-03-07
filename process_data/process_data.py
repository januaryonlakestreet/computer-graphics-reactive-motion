from glob import glob
from fairmotion.data import bvh
import numpy as np
import torch
from fairmotion.ops.conversions import R2Q
from tqdm import tqdm
import os

def foot_detect(positions, thres):
    fid_l = [7, 10]
    fid_r = [8, 11]
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.1, 0.05])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
    return np.concatenate((np.zeros((1, 2)),feet_l)), np.concatenate((np.zeros((1, 2)),feet_r))


def interaction_contact_labels(positions_a,positions_b,thres):
    fid_l = [7, 10]
    fid_r = [8, 11]
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.1, 0.05])

    #positions_a = positions_a.positions(local=False)
    #positions_b = positions_b.positions(local=False)

    thres = 0.00001



    return torch.from_numpy((positions_a-positions_b < thres).astype(np.float32))
def process_character(loaded_bvh):
    motion = loaded_bvh

    # 1. Access the underlying Skeleton object
    skeleton = motion.skel

    # 2. Extract the joints list to map names to indices
    joints = skeleton.joints
    joint_name_to_index = {joint.name: i for i, joint in enumerate(joints)}

    # 3. Iterate through the joints to find parent-child connections
    connections = []
    for child_joint in joints:
        parent_joint = child_joint.parent_joint

        # The root joint has no parent, so its parent is None
        if parent_joint is not None:
            parent_name = parent_joint.name
            child_name = child_joint.name

            parent_idx = joint_name_to_index[parent_name]
            child_idx = joint_name_to_index[child_name]

            connections.append([parent_idx, child_idx])






    data = np.zeros((1, 267))

    positions = motion.positions(local=False)  # (frames, joints, 3)
    fl, fr = foot_detect(positions,0.1)
    new_displacements = []
    # calculate rx, ry, ra in global space
    root_linear_velocity = positions[1:, 0] - positions[:-1, 0]
    root_linear_velocity = np.concatenate(
        (np.expand_dims(np.zeros((root_linear_velocity.shape[1],)), axis=0), root_linear_velocity))
    rx = root_linear_velocity[:, 0]
    ry = root_linear_velocity[:, 2]

    v_linear = rx + ry
    r = np.sqrt(np.power(positions[:, 0, 0], 2) + np.power(positions[:, 0, 2], 2))
    r += 0.00001
    ra = v_linear / r

    # translate to the character's root space
    matrices = []
    for pose in motion.poses:
        matrix = pose.get_transform(0, local=False)
        matrices.append(matrix)

    transform_matrices = np.stack(matrices, axis=0)
    transform_matrices_inv = np.linalg.inv(transform_matrices)
    global_positions = np.swapaxes(positions, 1, 2)  # (frames, 3, joints)
    global_positions = np.insert(global_positions, 3, np.ones((1, global_positions.shape[2])),
                                 axis=1)  # (frames, 4, joints)

    root_local_positions = transform_matrices_inv @ global_positions
    #global_positions#
    root_local_positions = np.delete(root_local_positions, -1, axis=1)
    positions = root_local_positions.swapaxes(1, 2)

    velocities = positions[1:] - positions[:-1]
    orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, 22, 6)  # PFNN
    rotations = motion.rotations(local=False)


    velocities = np.concatenate((np.expand_dims(np.zeros((velocities.shape[1], velocities.shape[2])), axis=0),
                                 velocities))  # (frame, num of joints, 3)

    new_data = np.concatenate((rx.reshape(-1, 1), ry.reshape(-1, 1), ra.reshape(-1, 1)), axis=1)
    new_data = np.concatenate((new_data, positions.reshape(positions.shape[0], -1)), axis=1)
    new_data = np.concatenate((new_data, velocities.reshape(velocities.shape[0], -1)), axis=1)
    new_data = np.concatenate((new_data, orientations.reshape(orientations.shape[0], -1)), axis=1)


    root_location = torch.from_numpy(positions[:,0,:]).expand(positions.shape[0], -1)

    return_dict = \
        {
            "character_displacement_vector" : np.concatenate((rx.reshape(-1, 1), ry.reshape(-1, 1), ra.reshape(-1, 1)), axis=1),
            "joint_positions": motion.positions(local=False).reshape(positions.shape[0], -1), # motion.positions(local=False)
            "joint_velocities" :  velocities.reshape(velocities.shape[0], -1),
            "joint_rotations_quat": R2Q(rotations).reshape(orientations.shape[0],-1),
            "foot_contact_labels" :torch.cat((torch.from_numpy(fl),torch.from_numpy(fr)),dim=1)
        }

    return return_dict




def calculate_stats(data):
    mean = np.mean(data, axis=0)
    std = np.std(data,axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return mean, std, min_val, max_val

if __name__ == '__main__':

    all_motions = glob("../interaction_dataset/*.bvh")
    all_ready_processed = []
    done = 0

    stats_a = {"mean": [], "std": [], "min": [], "max": []}
    stats_b = {"mean": [], "std": [], "min": [], "max": []}

    for motion in tqdm(all_motions, desc="Processing Files"):
        done += 1
        motion_id = motion.split("_")[1].split("\\")[1]

        if motion_id in all_ready_processed:
            continue
        all_ready_processed.append(motion_id)

        # Define paths
        bvh_path_1 = f"../interaction_dataset/{motion_id}_1.bvh"
        bvh_path_2 = f"../interaction_dataset/{motion_id}_2.bvh"

        try:
            loaded_bvh_1 = bvh.load(bvh_path_1)
            loaded_bvh_2 = bvh.load(bvh_path_2)

            processed_bvh_1 = process_character(loaded_bvh_1)
            processed_bvh_2 = process_character(loaded_bvh_2)

            interaction_contact_label = interaction_contact_labels(
                processed_bvh_1["joint_positions"], processed_bvh_2["joint_positions"], 0.1
            )
            processed_bvh_1.update({"interaction_contact_label": interaction_contact_label})

            interaction_contact_label = interaction_contact_labels(
                processed_bvh_2["joint_positions"], processed_bvh_1["joint_positions"], 0.1
            )
            processed_bvh_2.update({"interaction_contact_label": interaction_contact_label})

            side_a = np.concatenate(list(processed_bvh_1.values()), axis=1)
            side_b = np.concatenate(list(processed_bvh_2.values()), axis=1)

            # Calculate statistics for side_a
            mean_a, std_a, min_a, max_a = calculate_stats(side_a)
            stats_a["mean"].append(mean_a)
            stats_a["std"].append(std_a)
            stats_a["min"].append(min_a)
            stats_a["max"].append(max_a)

            # Calculate statistics for side_b
            mean_b, std_b, min_b, max_b = calculate_stats(side_b)
            stats_b["mean"].append(mean_b)
            stats_b["std"].append(std_b)
            stats_b["min"].append(min_b)
            stats_b["max"].append(max_b)

            # Save processed files
            bvh_save_path_1 = f"../interaction_dataset/processed/{motion_id}_1.npy"
            bvh_save_path_2 = f"../interaction_dataset/processed/{motion_id}_2.npy"
            os.makedirs(os.path.dirname(bvh_save_path_1), exist_ok=True)

            with open(bvh_save_path_1, 'wb') as f:
                np.save(f, side_a)
            with open(bvh_save_path_2, 'wb') as f:
                np.save(f, side_b)

        except FileNotFoundError:
            continue

    # Save overall statistics

    stats_a["mean"] = np.mean(stats_a["mean"], axis=0)
    stats_a["std"] = np.mean(stats_a["std"], axis=0)
    stats_a["min"] = np.min(stats_a["min"], axis=0)
    stats_a["max"] = np.max(stats_a["max"], axis=0)


    stats_b["mean"] = np.mean(stats_b["mean"], axis=0)
    stats_b["std"] = np.mean(stats_b["std"], axis=0)
    stats_b["min"] = np.min(stats_b["min"], axis=0)
    stats_b["max"] = np.max(stats_b["max"], axis=0)


    np.save("../interaction_dataset/processed/stats/stats_a.npy", stats_a)
    np.save("../interaction_dataset/processed/stats/stats_b.npy", stats_b)