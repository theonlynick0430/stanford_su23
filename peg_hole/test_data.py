import numpy as np

if __name__ == "__main__":
    data = np.load("peg_hole.npz", allow_pickle=True)
    print(data.files)
    print(data["ee_position"].shape)
    print(data["ee_orientation"].shape)
    print(data["image"].shape)
    print(data["ego_image"].shape)
    print(data["action"].shape)