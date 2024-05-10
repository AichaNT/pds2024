# ---Plotting DG---

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

# Load image
im_path = "data/all_img_mask/img/PAT_1453_1566_310.png"
mask_path = "data/all_img_mask/mask/PAT_1453_1566_310_mask.png"
im = plt.imread(im_path)
mask = plt.imread(mask_path)

# ---DG code---

# Gaussian blur to filter out interfering features
im = filters.gaussian(im, sigma=5, channel_axis=-1)
    
# Extracting the color layers
red, green, blue = [im[:,:, c] for c in range(3)]

# The means for each color (with the mask - only where the lesion is)
mean_red = np.mean(red[mask.astype(bool)])
mean_green = np.mean(green[mask.astype(bool)])
mean_blue = np.mean(blue[mask.astype(bool)])

# Average intensity within the lesion mask
average_intensity = np.mean((mean_red, mean_green, mean_blue))

# Intensity of each pixel
intensity = (red + green + blue) / 3

# Threshold for comparison
threshold = 0.85  

# Binary mask where intensity is lower than the average intensity
dg_mask = (intensity < average_intensity * threshold)

# Apply the lesion mask
dg_mask = dg_mask & mask.astype(bool)

# Percentage of lesion with irregular dots/globules
dg_percentage = np.sum(dg_mask) / np.sum(mask)
    
# Deciding whether the feature is present
if dg_percentage >= 0.04:
    result = 1
else:
    result = 0

# ---Plotting---

# Creating filename for plot title
filename = im_path.split("/")[-1].split(".")[0]

# Size of plots
plot_width = 6
plot_height = 6

# Figure and axes for plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(plot_width, plot_height))

# Plotting filtered image and dg_mask
ax0, ax1 = axes

ax0.imshow(im)
ax0.set_title(filename)

ax1.imshow(dg_mask, cmap='gray')
ax1.set_title(f'Percentage: {round(dg_percentage, 3)} \n Result: {result}')

# Adjusting layout
plt.tight_layout()
plt.show()
