import numpy as np 
import argparse
import os

def load_data_from_folder(folder_path, filepath, img_h, img_w):
    ee_position = np.empty((0, 6), dtype=np.float64)
    ee_orientation = np.empty((0, 8), dtype=np.float64)
    image = np.empty((0, img_h, img_w, 3), dtype=np.uint8)
    ego_image = np.empty((0, img_h, img_w, 3), dtype=np.uint8)
    action = np.empty((0, 12), dtype=np.float64)
    done = np.array([], dtype=bool)
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            
            # Load data from the .npz file
            loaded_data = np.load(file_path)

            print("Loading {}".format(file_path))
            
            # Assuming the data is stored in a specific variable name, e.g., 'data'
            ee_position = np.append(ee_position, loaded_data["ee_position"], axis=0)
            ee_orientation = np.append(ee_orientation, loaded_data["ee_orientation"], axis=0)
            image = np.append(image, loaded_data["image"], axis=0)
            ego_image = np.append(ego_image, loaded_data["ego_image"], axis=0)
            action = np.append(action, loaded_data["action"], axis=0)
            done_add = np.full(loaded_data["ee_position"].shape[0], False, dtype=bool)
            done_add[-1] = True
            done = np.append(done, done_add)

    # zeros_pos_ori = np.zeros((ee_position.shape[0], 3))
    # zeros_col = np.zeros((ee_position.shape[0], 1))
    zeros = np.zeros(ee_position.shape[0])
    # policy_type = np.full((ee_position.shape[0],1), 254)
    
    
    # Concatenate all the data together along the specified axis (0 for row-wise concatenation)
    np.savez(
        filepath, 
        ee_position=ee_position,
        ee_orientation=ee_orientation,
        image=image,
        ego_image=ego_image,
        action=action,
        done=done,
        # rest of params necessary to not cause errors
        # robot1_eef_pos=zeros_pos_ori,
        # robot1_eef_rot=zeros_pos_ori,
        rollout_timestep = zeros,
        # reward=zeros_col,
        # policy_type=policy_type,
        # policy_name=zeros_col,
        # policy_switch=zeros_col,
        # ee_orientation_eul=zeros_col,
        # q=zeros_col,
        # qdot=zeros_col,
        # gripper_width=zeros_col,
        # gripper_pos=zeros_col,
        # gripper_open=zeros_col
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    parser.add_argument("--filepath", type=str, default="data.npz")
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    args = parser.parse_args()

    load_data_from_folder(args.directory, args.filepath, args.h, args.w)