import numpy as np

if __name__ == "__main__":
    data = np.load("peg_insertion_data/demo_10.npz", allow_pickle=True)
    print(data.files)
    print(data["ee_position"].shape)
    print(data["image"].shape)
    print(data["ego_image"].shape)
    print(data["action"].shape)
    # data = np.load("peg_insertion.npz", allow_pickle=True)
    # print(data["action"])
    # prev = np.zeros(12)
    # for act in data["action"]:
    #     print(np.abs(act-prev))
    #     prev = act
    # print(data["robot1_eef_pos"].shape)
    # print(data["robot1_eef_rot"].shape)
    # print(data["rollout_timestep"].shape)
    # print(data["reward"].shape)
    # print(data["policy_type"].shape)
    # print(data["policy_name"].shape)
    # print(data["policy_switch"].shape)
    # print(data["ee_orientation_eul"].shape)
    # print(data["q"].shape)
    # print(data["qdot"].shape)
    # print(data["gripper_width"].shape)
    # print(data["gripper_pos"].shape)
    # print(data["gripper_open"].shape)
