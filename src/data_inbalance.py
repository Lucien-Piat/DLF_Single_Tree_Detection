import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import os
from segmentation import load_image

'''
Script used compute and plot the data imbalance
'''

def get_image_paths(directory):
    '''
    Get list of files in directory.
    '''
    try:
        files = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            files.extend(filenames)
        return files
    except FileNotFoundError:
        print("Directory not found.")
        return []
    
def count_pixel(image): 
    ones = np.sum(image)
    zeros = image.size - ones
    return ones, zeros

def count_all(directory):
    images_path = get_image_paths(directory)
    total_ones = total_zeros = 0
    for i in images_path:
        curr_image = load_image(directory, i)
        ones, zeros = count_pixel(curr_image)
        total_ones += ones
        total_zeros += zeros
    return total_ones, total_zeros

def plot_percentage(total_ones, total_zeros, output_file=None):
    plt.bar(np.array(["Tree", "Not tree "]),np.array([total_ones, total_zeros]))
    plt.ylabel("Pixel (n)")
    if output_file:
        plt.savefig(output_file, dpi=1000)
    else:
        plt.show()

if __name__ == "__main__":
    total_ones, total_zeros = count_all("data\\bronze\\masks")
    plot_percentage(total_ones, total_zeros, output_file="assets\\computations\\class_imbalance")
    


