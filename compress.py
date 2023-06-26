import numpy as np 
import argparse
import os
import pickle
import cv2

def get_corrected_img_path(path):
    desired_level = 2

    for _ in range(desired_level):
        path, tail = os.path.split(path)

def load_data_from_folder(folder_path, filepath, img_h, img_w):
    ee_position = np.empty((0, 3), dtype=np.float64)
    # ee_orientation = np.empty((0, 8), dtype=np.float64)
    image = np.empty((0, img_h, img_w, 3), dtype=np.uint8)
    # ego_image = np.empty((0, img_h, img_w, 3), dtype=np.uint8)
    action = np.empty((0, 3), dtype=np.float64)
    done = np.array([], dtype=bool)
    
    # robosuite data
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".npz"):
    #         file_path = os.path.join(folder_path, filename)
            
    #         # Load data from the .npz file
    #         loaded_data = np.load(file_path)

    #         print("Loading {}".format(file_path))
            
    #         # Assuming the data is stored in a specific variable name, e.g., 'data'
    #         ee_position = np.append(ee_position, loaded_data["ee_position"], axis=0)
    #         ee_orientation = np.append(ee_orientation, loaded_data["ee_orientation"], axis=0)
    #         image = np.append(image, loaded_data["image"], axis=0)
    #         ego_image = np.append(ego_image, loaded_data["ego_image"], axis=0)
    #         action = np.append(action, loaded_data["action"], axis=0)
    #         done_add = np.full(loaded_data["ee_position"].shape[0], False, dtype=bool)
    #         done_add[-1] = True
    #         done = np.append(done, done_add)

    # real data
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)

                print("Loading {}".format(file_path))

                start_idx = 2
                for obj in loaded_data[2:]: # ignore first two objects (not a part of demo)
                    if np.any(obj["action_lipa"]): # find actual start of demo
                        break
                    start_idx += 1

                for obj in loaded_data[start_idx:]: 
                    ee_position = np.append(ee_position, np.array([obj["obs_lipa"][:3]]), axis=0)
                    action = np.append(action, np.array([obj["action_lipa"][:3]]), axis=0)
                    overhead_img = cv2.imread(obj["overhead_img"])
                    overhead_img = cv2.resize(overhead_img, (np.uint16(overhead_img.shape[1]/2), np.uint16(overhead_img.shape[0]/2)))
                    image = np.append(image, np.array([overhead_img]), axis=0)
                done_add = np.full(len(loaded_data)-start_idx, False, dtype=bool)
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
        # ee_orientation=ee_orientation,
        image=image,
        # ego_image=ego_image,
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
    parser.add_argument("--h", type=int, default=240)
    parser.add_argument("--w", type=int, default=424)
    args = parser.parse_args()

    load_data_from_folder(args.directory, args.filepath, args.h, args.w)