import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import os
from tqdm import tqdm

def plot_images_low_res(xyz_array, folder_path, image_size):

    h, w = image_size
    n = xyz_array.shape[0]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n):
        img_path = os.path.join(folder_path, f'human_{i+1}.png')
        img = np.flip(plt.imread(img_path), axis=0)

        x_center, z_center, y_value = xyz_array[i]
        h_img, w_img = img.shape[:2]

        x, z = np.meshgrid(np.arange(w_img), np.arange(h_img))
        x = x - w_img // 2 + x_center
        z = z - h_img // 2 + z_center
        y = np.full_like(x, 1 - y_value)

        ax.plot_surface(x, y, z, facecolors=img)

    ax.set_xlim(0, w)
    ax.set_zlim(0, h)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-85)

    plt.show()


def plot_images_high_res(xyz_array, folder_path, image_size):

    h, w = image_size
    n = xyz_array.shape[0]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n):
        img_path = os.path.join(folder_path, f'human_{i+1}.png')
        img = np.flip(plt.imread(img_path), axis=0)

        x_center, z_center, y_value = xyz_array[i]
        h_img, w_img = img.shape[:2]

        x, z = np.meshgrid(np.arange(w_img), np.arange(h_img))
        x = x - w_img // 2 + x_center
        z = z - h_img // 2 + z_center
        y = np.full_like(x, 1 - y_value)

        ax.plot_surface(x, y, z, facecolors=img, rcount=img.shape[0], ccount=img.shape[1])

    ax.set_xlim(0, w)
    ax.set_zlim(0, h)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-85)

    plt.show()


def result_high_res_save(xyz_array, folder_path, image_size):
    h, w = image_size
    n = xyz_array.shape[0]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    print("Rendering 3D images...")
    for i in tqdm(range(n), desc="Processing images", unit="image"):
        img_path = os.path.join(folder_path, f'human_{i+1}.png')
        img = np.flip(plt.imread(img_path), axis=0)

        x_center, z_center, y_value = xyz_array[i]
        h_img, w_img = img.shape[:2]

        x, z = np.meshgrid(np.arange(w_img), np.arange(h_img))
        x = x - w_img // 2 + x_center
        z = z - h_img // 2 + z_center
        y = np.full_like(x, 1 - y_value)

        ax.plot_surface(x, y, z, facecolors=img, rcount=h_img, ccount=w_img)

    ax.set_xlim(0, w)
    ax.set_zlim(0, h)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-85)

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    save_folder = os.path.join(script_dir, 'DisplayResult')  
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, 'high_res_display.png')

    print("Saving high resolution image...")
    with tqdm(total=1, desc="Saving image", bar_format="{l_bar}{bar}| {remaining}") as pbar:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pbar.update(1)

    plt.close()
    print(f"Image saved to {save_path}")