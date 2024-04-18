import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from itertools import product

import skimage as ski
from skimage import io, transform, exposure

from segmentation import load_image
from data_inbalance import get_image_paths


def convert_to_irgb(image):
     image[:, :, 0] = image[:, :, 3]
     image = remove_last_channel(image)
     return image 

def remove_last_channel(image):
    image = image[:, :, :3]
    return image

def create_compare_figure(image1, image1_title,  image2, image2_title,  output_file=None):
    '''
    Plot tow images side by side
    '''
    fig, axes = plt.subplots(1, 2, figsize=(6, 12), constrained_layout=True)
    axes[0].imshow(image1)
    axes[0].set_title(image1_title, fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title(image2_title, fontsize=10)
    axes[1].axis('off')
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=1000)
    else:
        plt.show()

def create_solo_figure(image, title, output_file=None):
    '''
    Plot a single image with a title
    '''
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title, fontsize=12)
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=1000)
    else:
        plt.show()


# I used the old script because i cant find the package for crop function in augmentation.py  
def rotate_image(image, angle):
    return transform.rotate(image, angle, resize=False)

def generate_all_angles(image, image_name, flip_image=False):
    '''
    Generate rotated images.
    '''
    angles = [90, 180, 270]
    images_list = [image]
    names_list = [image_name.split('.')[0] + '_0.tif']

    for i in angles:
      images_list.append(rotate_image(image, i))
      names_list.append(image_name.split('.')[0] + '_' + str(i) + '.tif')

    if flip_image:
      for i in range(4):
        images_list.append(np.fliplr(images_list[i]))
        names_list.append(names_list[i].split('.')[0] + '_f.tif')
    return images_list, names_list