"""
@authors: cjep, kmah, feso, aith, nozo
"""


import matplotlib.pyplot as plt
import os

# Import our own file that has the feature extraction functions
from extract_features import asymmetry, color, dg


# ---Image processing function---

def load(image_dir, mask_dir):

    """
    Function to load images and masks

    Inputs:
     • Directory for images
     • Directory for masks

    Output:
     • List of images
     • List of masks
     • List of filenames/image IDs
    """

    # Initialize lists to store images, masks and filenames
    images = []
    masks = []
    filenames = []

    # Iterate over image files
    for im_filename in os.listdir(image_dir):

        if im_filename.endswith('.png'):  
            
            # Load image
            image_path = os.path.join(image_dir, im_filename)

            image = plt.imread(image_path)
            images.append(image)

            # Load corresponding mask
            mask_filename = os.path.splitext(im_filename)[0] + '_mask.png' 
            mask_path = os.path.join(mask_dir, mask_filename)

            mask = plt.imread(mask_path)
            masks.append(mask)

            filename = im_filename.rstrip('.png')
            filenames.append(filename)

    return images, masks, filenames



# Directory paths for images, masks and output
image_dir = input("Please enter image directory: ")
mask_dir = input("Please enter mask directory: ")
output_dir = input("Please enter output directory to store features.csv: ")


# Load images and masks
images, masks, filenames = load(image_dir, mask_dir)


# Path to the output CSV file
output_path = os.path.join(output_dir, "features.csv")


# Writing CSV file with the results
with open(output_path, "w") as outfile:

    outfile.write(f"image_id,A,C,DG\n")

    for index, (im, mask) in enumerate(zip(images, masks)):
        
        im_id = filenames[index]

        # Call functions
        a_score = asymmetry(im, mask)
        c_score = color(im, mask)
        dg_score = dg(im, mask)

        outfile.write(f"{im_id},{a_score},{c_score},{dg_score}\n")

print('A CSV file has been created with the results!')
