import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d as p3d


def save_screenshot(data_a=None,data_b=None,file_name="3d_scatter_plot.png",joints=22,connect_joints=True):
    #assumption: the data_a is ether 293 or 66 in width if 293 we need to select the joints
    #it is also already denormed

    if data_a is None and data_b is None:
        return

    # Step 1: Setup
    np.random.seed(0)


    chains = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # Chain 1
        [0, 5], [5, 6], [6, 7], [7, 8],  # Chain 2 (linked to root 0)
        [0, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],  # Chain 3 (linked to 0)
        [0, 18], [18, 19], [19, 20], [20, 21]  # Chain 4 (linked to 0)  right arm

    ]



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=45, azim=-1.5,roll=90)  # Experiment with these values
    ax.view_init(elev=90, azim=0, roll=180)
    # Step 2: Plot points

    if data_a is not None:
        points_a = data_a.detach().cpu()

        if points_a.shape[-1] == 586:
            points_a = points_a[:,:,:293]
        if points_a.shape[-1] == 293:
            points_a = points_a[0,:, 3:69]

        points_a = points_a.reshape(-1, joints, 3).squeeze().squeeze()

        ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], c='b', s=50)
        if connect_joints:
            for chain in chains:
                chain_pts = points_a[chain]
                ax.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2], marker='o', color="blue")

    if data_b is not None:
        points_b = data_b.detach().cpu()
        if len(points_b.shape) == 2:
            points_b = points_b.unsqueeze(dim=1)
        if points_b.shape[-1] == 586:
            points_b = points_b[:, :, 293:]
        if points_b.shape[-1] == 293:
            points_b = points_b[0, :, 3:69]

        points_b = points_b.reshape(-1, 22, 3).squeeze().squeeze()

        ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], c='r', s=50)
        for chain in chains:
            chain_pts = points_b[chain]
            ax.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2], marker='o',color="red")


    plt.savefig(file_name, dpi=300, bbox_inches='tight')  # Save with high resolution
    plt.close(fig)
    elev = ax.elev
    azim = ax.azim
    #roll = ax.get_roll()  # Added get_roll()

    #print(f"Roll: {roll} degrees")


    def on_motion(event):
        if event.inaxes == ax:
            rotation_matrix = ax.get_proj()
            rotation_matrix = rotation_matrix[:3, :3]
            # Calculate elevation and azimuth from the rotation matrix.
            # This calculation is more accurate during interactive rotation.
            elev = np.degrees(np.arcsin(-rotation_matrix[2, 1]))
            azim = np.degrees(np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1]))

            print(f"Elevation: {elev} degrees")
            print(f"Azimuth: {azim} degrees")



    # Connect the 'motion_notify_event' to the on_move function

    #fig.canvas.mpl_connect('motion_notify_event', on_motion)

    #plt.show()  # Close the figure to free memory



def save_rollout_as_animation(rollout_a, rollout_b, output_path="motion_animation.gif", fps=30, skeleton_connections=None):
    """
    Saves a motion rollout as a 3D animation using Matplotlib, showing two characters.

    Args:
        rollout_a (np.ndarray): The motion rollout data for character A. Expected shape:
                               (num_frames, num_joints, 3), where 3 represents x, y, z coordinates.
        rollout_b (np.ndarray): The motion rollout data for character B. Expected shape:
                               (num_frames, num_joints, 3), where 3 represents x, y, z coordinates.
        output_path (str): The path to save the animation file (e.g., "motion.gif", "motion.mp4").
        fps (int): Frames per second for the animation.
        skeleton_connections (list of tuples, optional): A list of tuples, where each tuple
                                                       contains the indices of two connected joints.
                                                       If None, a default human skeleton structure is assumed.
    """
    num_frames, num_joints, _ = rollout_a.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=-1.5, roll=-0)
    # Default human skeleton connections (you might need to adjust this)


    if skeleton_connections is None:

        skeleton_connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Chain 1
            [0, 5], [5, 6], [6, 7], [7, 8],  # Chain 2 (linked to root 0)
            [0, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
            # Chain 3 (linked to 0)
            [0, 18], [18, 19], [19, 20], [20, 21]  # Chain 4 (linked to 0)  right arm

        ]

    # Create lines for both characters
    lines_a = [ax.plot([], [], [], 'b-')[0] for _ in range(len(skeleton_connections))]
    points_a = ax.plot([], [], [], 'ro')[0]

    lines_b = [ax.plot([], [], [], 'g-')[0] for _ in range(len(skeleton_connections))] #different color for B
    points_b = ax.plot([], [], [], 'mo')[0] #different color for B

    def update(frame_num):
        #ax.clear()
        ax.view_init(elev=90, azim=90,roll=180)  # Experiment with these values

        frame_data_a = rollout_a[frame_num]
        x_a = frame_data_a[:, 0]
        y_a = frame_data_a[:, 1]
        z_a = frame_data_a[:, 2]

        points_a.set_data(x_a, y_a)
        points_a.set_3d_properties(z_a)

        for i, (joint1_idx, joint2_idx) in enumerate(skeleton_connections):
            x_data = [frame_data_a[joint1_idx, 0], frame_data_a[joint2_idx, 0]]
            y_data = [frame_data_a[joint1_idx, 1], frame_data_a[joint2_idx, 1]]
            z_data = [frame_data_a[joint1_idx, 2], frame_data_a[joint2_idx, 2]]
            lines_a[i].set_data(x_data, y_data)
            lines_a[i].set_3d_properties(z_data)

        frame_data_b = rollout_b[frame_num]
        x_b = frame_data_b[:, 0]
        y_b = frame_data_b[:, 1]
        z_b = frame_data_b[:, 2]

        points_b.set_data(x_b, y_b)
        points_b.set_3d_properties(z_b)
      
        for i, (joint1_idx, joint2_idx) in enumerate(skeleton_connections):
            x_data = [frame_data_b[joint1_idx, 0], frame_data_b[joint2_idx, 0]]
            y_data = [frame_data_b[joint1_idx, 1], frame_data_b[joint2_idx, 1]]
            z_data = [frame_data_b[joint1_idx, 2], frame_data_b[joint2_idx, 2]]
            lines_b[i].set_data(x_data, y_data)
            lines_b[i].set_3d_properties(z_data)

        # Set axis labels and limits (you might need to adjust these based on your data)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        max_val = np.max(np.abs(np.concatenate((rollout_a, rollout_b)))) #Find max of both rollouts.
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_title(f"Frame: {frame_num}")

        return lines_a + [points_a] + lines_b + [points_b]

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=True)


    try:
        ani.save(output_path, writer='pillow', fps=fps) # Using pillow writer for GIF
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Consider installing ffmpeg for saving as MP4: 'pip install ffmpeg-python'")
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps) # Try saving as MP4 if ffmpeg is available
            print(f"Animation saved to {output_path} using ffmpeg.")
        except Exception as e2:
            print(f"Error saving with ffmpeg: {e2}")

    plt.close(fig)


def mean_l2di_(reaction, reaction_gt):
    x = np.mean(np.sqrt(np.sum((reaction - reaction_gt)**2, -1)))
    return x





def generate_rollout(vae,start_seq,args):
    tmp = start_seq(args)

    import numpy as np


    sequence = tmp.Get_Interaction()
    side_a = sequence["side_a"].unsqueeze(dim=1).unsqueeze(dim=1)
    side_b = sequence["side_b"].unsqueeze(dim=1).unsqueeze(dim=1)
    frames = side_a.shape[0]
    side_a = side_a.squeeze()
    side_b = side_b.squeeze()
    current = side_a[5].unsqueeze(dim=0).unsqueeze(dim=0)
    mh =   side_a[:5].unsqueeze(dim=0)


    generated_frames = []

    for frame in range(frames):
        nf, _ = vae(current,mh)
        generated_frames.append(nf)
        current = side_a[frame].unsqueeze(dim=0).unsqueeze(dim=0)
        mh = side_a[frame-5:frame].unsqueeze(dim=0)
        th = side_b[frame - 5:frame].unsqueeze(dim=0)

    real = torch.cat((sequence["side_a"],sequence["side_b"]),dim=1)
    fake = torch.cat((sequence["side_a"],torch.stack(generated_frames).squeeze()),dim=1)



    denormed_real = vae.denormalize(real.unsqueeze(dim=1)).squeeze()
    denormed_fake = vae.denormalize(fake.unsqueeze(dim=1)).squeeze()




    side_b_start = side_a.shape[1]  # where side_b starts after concat
    denormed_a = denormed_fake[:,:side_b_start]
    denormed_side_b = denormed_real[:, :side_b_start]
    denormed_generated_side_b = denormed_fake[:, side_b_start:]

    frames_real = np.array(denormed_side_b[:,3:69].reshape(-1, 22, 3).detach().cpu())
    frames_fake = np.array(denormed_generated_side_b[:,3:69].reshape(-1, 22, 3).detach().cpu())

    frame_real_not_generated = np.array(denormed_side_b[:,3:69].reshape(-1, 22, 3).detach().cpu())


    save_rollout_as_animation(frame_real_not_generated,frames_fake)

    print("mpjpe",mean_l2di_( np.array(denormed_side_b.detach().cpu()), np.array(denormed_generated_side_b.detach().cpu())))
    return denormed_a,denormed_generated_side_b





def generate_rollout_with_diffusion(vae,diffusion,interaction_dataset):
    from settings.settings import args

    start_frame = interaction_dataset.GetSingleRandom().unsqueeze(dim=0).unsqueeze(dim=0)
    num_generated_samples = 1
    generated_samples = diffusion.sample(shape=(num_generated_samples, 256),pose=start_frame.squeeze())

    mh = []

    mh.append(vae.motion_decoder(generated_samples, start_frame))

    for _ in range(400):
        pose = vae.motion_decoder(generated_samples,torch.concatenate(mh,dim=1))
        mh.append(pose)
        generated_samples = diffusion.sample(shape=(num_generated_samples, 256), pose=start_frame.squeeze())

    data = vae.denormalize(torch.concatenate((torch.concatenate(mh,dim=1).squeeze(),torch.concatenate(mh,dim=1).squeeze()),dim=1).unsqueeze(1)).squeeze()[:,:293]
    data = np.array(data[:,3:69].reshape(-1,22,3).detach().cpu())


    save_rollout_as_animation(data,data)





