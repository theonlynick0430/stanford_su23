import pickle
import cv2
import numpy as np

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_pickle_fields(data):
    for obj in data:
        for field, value in obj.items():
            print(f"{field}: {value}")

if __name__ == "__main__":

    # # Provide the path to your Pickle file
    # pickle_file_path = 'role_assignment/box_pack/dua/demo_0.pkl'

    # # Load the Pickle file
    # loaded_data = load_pickle_file(pickle_file_path)

    # # Print the fields of the loaded data
    # print_pickle_fields(loaded_data)

    # img_path = 'role_assignment/box_pack/lipa/images_0/overhead_image_0000.jpg'
    # overhead_img = cv2.imread(img_path)
    # print(overhead_img.shape)
    # print((overhead_img.shape[1]/2, overhead_img.shape[0]/2))
    # overhead_img = cv2.resize(overhead_img, (np.uint16(overhead_img.shape[1]/2), np.uint16(overhead_img.shape[0]/2)))
    # print(overhead_img.shape)

    loaded_data = np.load("peg_hole/test/demo_0.npz")
    print(loaded_data["image"].shape)
