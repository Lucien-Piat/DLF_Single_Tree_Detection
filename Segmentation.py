import os
import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
from skimage import io
from skimage.morphology import erosion, square, opening, dilation, label

def load_image(input_folder, image_path):
    '''
    Load image from path.
    '''
    try:
        image = io.imread(os.path.join(input_folder, image_path))
        return np.array(image)
    except FileNotFoundError:
        print("Image not found")
        return None
    
def plot_data(image1, image2):
    '''
    Plot tow images side by side
    '''
    fig, axes = plt.subplots(1, 2, figsize=(6, 12), constrained_layout=True)
    axes[0].imshow(image1)
    axes[0].set_title('Original', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title('Modified', fontsize=10)
    axes[1].axis('off')
    plt.show()
    
# I applied this script on full size images, if you use it on scaled down ones,
# You might want to decrease the erosion_factor or small objects will disappear 
    
def label_image(binary_image, erosion_factor=10): 

    # Refine the segmentation by applying erosion and opening operations
    for i in range(erosion_factor):

        # Open the binary image to remove small objects
        binary_image = opening(binary_image, ski.morphology.square(3))
        
        # Erode the binary image to shrink object boundaries
        binary_image = erosion(binary_image, square(3))

    # Label connected components in the refined binary image
    labeled_image = label(binary_image)

    # Once labeled, revert changes
    for i in range(erosion_factor):
        labeled_image = opening(labeled_image, ski.morphology.square(3))
        labeled_image = dilation(labeled_image, square(3))
    return labeled_image # Fully labeled original image

if __name__ == "__main__":
    image1 = load_image('raw_data','mask_1.tif')
    image2 = label_image(image1)
    plot_data(image1, image2)