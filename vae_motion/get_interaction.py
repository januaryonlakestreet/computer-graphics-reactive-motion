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

    def collect_data(self,data):
        transl = torch.from_numpy(data["trans"])
        orient = torch.from_numpy(data["root_orient"])
        pose = torch.from_numpy(data["pose_body"])
        return torch.cat((transl,orient,pose),dim=1).to(self.args.device)



    def denormalize_a(self, data):
        dir = os.path.dirname(os.path.dirname(os.getcwd())) + "\\computer & graphics two char motion\\interaction_dataset_files\\"
        loaded_stats = torch.load(dir + "\\stats.pt")

        loaded_stats_a = loaded_stats["a"]
        avg_a = torch.cat((
            loaded_stats_a["transl"]["mean"],
            loaded_stats_a["orient"]["mean"],
            loaded_stats_a["pose"]["mean"]
        )).float().to(self.args.device)

        std_a = torch.cat((
            loaded_stats_a["transl"]["std"],
            loaded_stats_a["orient"]["std"],
            loaded_stats_a["pose"]["std"]
        )).float().to(self.args.device)

        std_a[std_a == 0] = 1.0

        denormed_mocap_data = data * std_a + avg_a
        return denormed_mocap_data

    def denormalize_b(self, data):
        dir = os.path.dirname(os.path.dirname(os.getcwd())) + "\\computer & graphics two char motion\\interaction_dataset_files\\"
        loaded_stats = torch.load(dir + "\\stats.pt")

        loaded_stats_b = loaded_stats["b"]
        avg_b = torch.cat((
            loaded_stats_b["transl"]["mean"],
            loaded_stats_b["orient"]["mean"],
            loaded_stats_b["pose"]["mean"]
        )).float().to(self.args.device)

        std_b = torch.cat((
            loaded_stats_b["transl"]["std"],
            loaded_stats_b["orient"]["std"],
            loaded_stats_b["pose"]["std"]
        )).float().to(self.args.device)

        std_b[std_b == 0] = 1.0

        denormed_mocap_data = data * std_b + avg_b
        return denormed_mocap_data


    def Get_Interaction(self,id=False):
        frame_count = 0

        while frame_count < self.args.mini_batch_size:
            dir = os.path.dirname(os.path.dirname(os.getcwd()))+"\\computer & graphics two char motion\\interaction_dataset_files\\"
            motion_choice = random.choice(glob.glob(dir + "*.pkl")[:1])
            motion_id = motion_choice.split("\\")[-1].split("_")[0].split(".")[0]

            loaded_data = np.load(dir+str(motion_id)+".pkl",allow_pickle=True)

            raw_data_1 = loaded_data["person1"]
            raw_data_2 = loaded_data["person2"]

            mocap_data_1 = self.collect_data(raw_data_1)
            mocap_data_2 = self.collect_data(raw_data_2)

            loaded_stats = torch.load(dir+"\\stats.pt")



            loaded_stats_a = loaded_stats["a"]
            loaded_stats_b = loaded_stats["b"]

            avg_a = torch.cat((loaded_stats_a["transl"]["mean"],loaded_stats_a["orient"]["mean"],
                               loaded_stats_a["pose"]["mean"])).float().to(self.args.device)
            std_a = torch.cat((loaded_stats_a["transl"]["std"],loaded_stats_a["orient"]["std"],
                               loaded_stats_a["pose"]["std"])).float().to(self.args.device)

            avg_b = torch.cat((loaded_stats_b["transl"]["mean"], loaded_stats_b["orient"]["mean"],
                               loaded_stats_b["pose"]["mean"])).float().to(self.args.device)
            std_b = torch.cat((loaded_stats_b["transl"]["std"], loaded_stats_b["orient"]["std"],
                               loaded_stats_b["pose"]["std"])).float().to(self.args.device)

            std_a[std_a == 0] = 1.0
            std_b[std_b == 0] = 1.0




            normed_mocap_data_1 = (mocap_data_1 - avg_a) / std_a
            normed_mocap_data_2 = (mocap_data_2 - avg_b) / std_b



            frame_count += mocap_data_1.shape[0]


        return {"side_a":normed_mocap_data_1,"side_b":normed_mocap_data_2}