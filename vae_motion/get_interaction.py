import numpy as np
import torch
import glob
import random
import os
class Get_Interaction:
    def __init__(self,args):
        self.args = args
        self.future_length = args.future_length
        self.history_length = args.history_length

    def create_motion_primitives(self, motion_sequence, history_length=None, future_length=None):

        motion_primitives = []
        total_frames = motion_sequence.shape[0]

        # Iterate through the sequence to create primitives
        start_index = 0
        while start_index + history_length + future_length <= total_frames:
            history = motion_sequence[start_index:start_index + history_length]
            future = motion_sequence[start_index + history_length:start_index + history_length + future_length]
            motion_primitives.append((history, future))
            start_index += future_length

        # Handle the case where there might be a remaining partial sequence
        if total_frames - start_index > history_length:
            history = motion_sequence[start_index:start_index + history_length]
            future = motion_sequence[start_index + history_length:]
            # motion_primitives.append((history, future))

        return motion_primitives

    def Get_Interaction(self,id=False):
        frame_count = 0

        while frame_count < self.args.mini_batch_size:
            dir = os.path.dirname(os.path.dirname(os.getcwd()))+"\\computer & graphics two char motion\\interaction_dataset\\processed\\"
            motion_choice = random.choice(glob.glob(dir + "*.npy")[:1])
            motion_id = motion_choice.split("\\")[-1].split("_")[0]



            raw_data_1 = np.load(dir+str(motion_id)+"_1.npy")
            raw_data_2 = np.load(dir+str(motion_id)+"_2.npy")
            mocap_data_1 = torch.from_numpy(raw_data_1).float().to(self.args.device)
            mocap_data_2 = torch.from_numpy(raw_data_2).float().to(self.args.device)

            loaded_stats_a = np.load(dir+"stats\\stats_a.npy",allow_pickle=True).item()
            loaded_stats_b = np.load(dir+"stats\\stats_b.npy", allow_pickle=True).item()

            avg_a = torch.from_numpy(loaded_stats_a['mean']).float().to(self.args.device)
            std_a = torch.from_numpy(loaded_stats_a['std']).float().to(self.args.device)

            avg_b = torch.from_numpy(loaded_stats_b['mean']).float().to(self.args.device)
            std_b = torch.from_numpy(loaded_stats_b['std']).float().to(self.args.device)

            std_a[std_a == 0] = 1.0
            std_b[std_b == 0] = 1.0


            interaction_contact_label_a = mocap_data_1[:,-70:] # 66 i labels + 4 fc labels
            interaction_contact_label_b = mocap_data_2[:,-70:]

            mocap_data_1 = (mocap_data_1 - avg_a) / std_a
            mocap_data_2 = (mocap_data_2 - avg_b) / std_b

            mocap_data_1[:,-70:] = interaction_contact_label_a
            mocap_data_2[:,-70:] = interaction_contact_label_b

            frame_count += mocap_data_1.shape[0]


        return {"side_a":mocap_data_1,"side_b":mocap_data_2}