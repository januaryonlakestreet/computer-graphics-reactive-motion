import os
from glob import glob
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frame_size = 138
text_dim = 384
pose_dim = 69
latent_dim = 256
def find_text_files_with_keyword(root_dir, keyword):

    results = []
    pattern = os.path.join(root_dir, "**", "*.txt")
    for path in glob(root_dir, recursive=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
                if keyword in data:
                    results.append(int(path.split(".txt")[0].split("\\")[4]))
        except OSError:
            pass
    return results

def breakup_tensor(tensor):

    transl = tensor[:,:3]
    global_orient = tensor[:,3:6]
    body_pose = tensor[:,6:]

    return {"transl":transl,
            "global_orient":global_orient,
            "body_pose":body_pose
            }

def breakup_tensor_with_time(tensor):

    transl = tensor[:,:,:3]
    global_orient = tensor[:,:,3:6]
    body_pose = tensor[:,:,6:]

    return {"transl":transl,
            "global_orient":global_orient,
            "body_pose":body_pose
            }


def collect_dict(sequence):
    data = torch.cat([torch.from_numpy(sequence["trans"]),
                      torch.from_numpy(sequence["root_orient"]),
                      torch.from_numpy(sequence["pose_body"])], dim=1)

    return data.to(DEVICE)

def collect_dict_tensor(sequence):
    data = torch.cat([sequence["transl"],
                      sequence["global_orient"],
                      sequence["body_pose"]], dim=1)

    return data.to(DEVICE)
def load_normalization(path="stats.pt"):
    return torch.load(path)

def apply_normalization_a_only(tensor_a):
    normalization_data = load_normalization("stats.pt")

    tensor_a[:, :, :3] = (tensor_a[:, :, :3] - normalization_data["a"]["transl"]["mean"].cuda()) / \
                         normalization_data["a"]["transl"]["std"].cuda()
    tensor_a[:, :, 3:6] = (tensor_a[:, :, 3:6] - normalization_data["a"]["orient"]["mean"].cuda()) / \
                          normalization_data["a"]["orient"]["std"].cuda()
    tensor_a[:, :, 6:] = (tensor_a[:, :, 6:] - normalization_data["a"]["pose"]["mean"].cuda()) / \
                         normalization_data["a"]["pose"]["std"].cuda()

    return tensor_a

def apply_normalization_b_only(tensor_b):
    normalization_data = load_normalization("stats.pt")

    tensor_b[:, :, :3] = (tensor_b[:, :, :3] - normalization_data["b"]["transl"]["mean"].cuda()) / \
                         normalization_data["b"]["transl"]["std"].cuda()
    tensor_b[:, :, 3:6] = (tensor_b[:, :, 3:6] - normalization_data["b"]["orient"]["mean"].cuda()) / \
                          normalization_data["b"]["orient"]["std"].cuda()
    tensor_b[:, :, 6:] = (tensor_b[:, :, 6:] - normalization_data["b"]["pose"]["mean"].cuda()) / \
                         normalization_data["b"]["pose"]["std"].cuda()

    return tensor_b

def apply_normalization(tensor_a,tensor_b):
    normalization_data = load_normalization("stats.pt")

    tensor_a[:, :, :3] = (tensor_a[:, :, :3] - normalization_data["a"]["transl"]["mean"].cuda()) / \
                       normalization_data["a"]["transl"]["std"].cuda()
    tensor_a[:, :, 3:6] = (tensor_a[:, :, 3:6] - normalization_data["a"]["orient"]["mean"].cuda()) / \
                        normalization_data["a"]["orient"]["std"].cuda()
    tensor_a[:, :, 6:] = (tensor_a[:, :, 6:] - normalization_data["a"]["pose"]["mean"].cuda()) / \
                       normalization_data["a"]["pose"]["std"].cuda()

    tensor_b[:, :, :3] = (tensor_b[:, :, :3] - normalization_data["b"]["transl"]["mean"].cuda()) / \
                       normalization_data["b"]["transl"]["std"].cuda()
    tensor_b[:, :, 3:6] = (tensor_b[:, :, 3:6] - normalization_data["b"]["orient"]["mean"].cuda()) / \
                        normalization_data["b"]["orient"]["std"].cuda()
    tensor_b[:, :, 6:] = (tensor_b[:, :, 6:] - normalization_data["b"]["pose"]["mean"].cuda()) / \
                       normalization_data["b"]["pose"]["std"].cuda()

    return tensor_a,tensor_b

def apply_denormalization(tensor_a,tensor_b):
    normalization_data = load_normalization("stats.pt")
    tensor_a[:, :3] = (tensor_a[:, :3] * normalization_data["a"]["transl"]["std"].to(DEVICE)) + \
                       normalization_data["a"]["transl"]["mean"].to(DEVICE)
    tensor_a[:, 3:6] = (tensor_a[:, 3:6] * normalization_data["a"]["orient"]["std"].to(DEVICE)) + \
                        normalization_data["a"]["orient"]["mean"].to(DEVICE)
    tensor_a[:, 6:] = (tensor_a[:, 6:] * normalization_data["a"]["pose"]["std"].to(DEVICE)) + \
                       normalization_data["a"]["pose"]["mean"].to(DEVICE)

    tensor_b[:, :3] = (tensor_b[:, :3] * normalization_data["b"]["transl"]["std"].to(DEVICE)) + \
                       normalization_data["b"]["transl"]["mean"].to(DEVICE)
    tensor_b[:, 3:6] = (tensor_b[:, 3:6] * normalization_data["b"]["orient"]["std"].to(DEVICE)) + \
                        normalization_data["b"]["orient"]["mean"].to(DEVICE)
    tensor_b[:, 6:] = (tensor_b[:, 6:] * normalization_data["b"]["pose"]["std"].to(DEVICE)) + \
                       normalization_data["b"]["pose"]["mean"].to(DEVICE)

    return tensor_a, tensor_b

def apply_denormalization_a_only(tensor_a):
    normalization_data = load_normalization("stats.pt")
    tensor_a[:, :3] = (tensor_a[:, :3] * normalization_data["a"]["transl"]["std"].to(DEVICE)) + \
                       normalization_data["a"]["transl"]["mean"].to(DEVICE)
    tensor_a[:, 3:6] = (tensor_a[:, 3:6] * normalization_data["a"]["orient"]["std"].to(DEVICE)) + \
                        normalization_data["a"]["orient"]["mean"].to(DEVICE)
    tensor_a[:, 6:] = (tensor_a[:, 6:] * normalization_data["a"]["pose"]["std"].to(DEVICE)) + \
                       normalization_data["a"]["pose"]["mean"].to(DEVICE)

    return tensor_a


def apply_denormalization_b_only(tensor_b):
    normalization_data = load_normalization("stats.pt")
    tensor_b[:, :3] = (tensor_b[:, :3] * normalization_data["b"]["transl"]["std"].to(DEVICE)) + \
                      normalization_data["b"]["transl"]["mean"].to(DEVICE)
    tensor_b[:, 3:6] = (tensor_b[:, 3:6] * normalization_data["b"]["orient"]["std"].to(DEVICE)) + \
                       normalization_data["b"]["orient"]["mean"].to(DEVICE)
    tensor_b[:, 6:] = (tensor_b[:, 6:] * normalization_data["b"]["pose"]["std"].to(DEVICE)) + \
                      normalization_data["b"]["pose"]["mean"].to(DEVICE)

    return tensor_b

def try_load_saved(model,path):
    try:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt, strict=False)
        return model
    except:
        return model


