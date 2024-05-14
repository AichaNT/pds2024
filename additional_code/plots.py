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


# ---Plotting rotation---
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.transform import resize

# Load an example image and mask
im_path = "data/all_img_mask/img/PAT_1453_1566_310.png"
mask_path = "data/all_img_mask/mask/PAT_1453_1566_310_mask.png"

# Read image and mask
image = io.imread(im_path)
mask = io.imread(mask_path)

# ---Asymmetry code---
def prep_mask(m):
    m = resize(m, (m.shape[0] // 4, m.shape[1] // 4), anti_aliasing=False)

    mask_bounds = np.argwhere(m == 1)
    top = np.min(mask_bounds[:, 0])
    bottom = np.max(mask_bounds[:, 0])
    left = np.min(mask_bounds[:, 1])
    right = np.max(mask_bounds[:, 1])

    #setting crop parameters
    a = 10
    b = 10

    #cropping mask to standard crop parameters
    cropped_m = m[top - a : bottom + 10, left - b: right + 10]

    #if the cropped image cannot be halved equally
    m_shape = cropped_m.shape
    if m_shape[1] % 2 != 0:
        b -= 1

    if m_shape[0] % 2 != 0:
        a -= 1
    
    cropped_m = m[top-a : bottom + 10, left - b: right + 10] #re-cropping to make even halving possible

    return cropped_m

def prep_im(im, m):
    '''preps and crops an image for asymmetry testing'''
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)

    m = resize(m, (m.shape[0] // 4, m.shape[1] // 4), anti_aliasing=False)

    image_bounds = np.argwhere(m == 1)
    top = np.min(image_bounds[:, 0])
    bottom = np.max(image_bounds[:, 0])
    left = np.min(image_bounds[:, 1])
    right = np.max(image_bounds[:, 1])

    #setting crop parameters
    a = 10
    b = 10

    #cropping images to standard crop parameters
    cropped_im = im[top - a : bottom + 10, left - b : right + 10]
    cropped_m = m[top - a : bottom + 10, left - b: right + 10]

    #if the cropped image cannot be halved equally
    im_shape = cropped_im.shape
    if im_shape[1] % 2 != 0:
        b -= 1

    if im_shape[0] % 2 != 0:
        a -= 1
        
    #re-crop with one pixel less to ensure an equal halving, relevant for overlap calculation later
    cropped_im = im[top - a : bottom + 10, left - b : right + 10]
    cropped_m = m[top - a : bottom + 10, left - b: right + 10]

    return cropped_im, cropped_m

def axis_split(png):
    '''takes a prepped image or mask and splits it in two over the x-axis and the y-axis, yielding 4 images'''
    mid_v = png.shape[0] // 2 # Horizontal middle
    mid_h = png.shape[1] // 2 # Vertical middle
    
    # Cropping image to get vertical and horizontal splits
    top = png[:mid_v,:]
    bottom = png[mid_v:, :]
    left = png[:, :mid_h]
    right = png[:, mid_h:]

    return top, bottom, left, right

def shape(mask):
    '''takes a prepped mask and returns the shape asymmetry score'''
    top, bottom, left, right = axis_split(mask) #splitting the image in four parts
    btm = np.flip(bottom, axis = 0) #flipping the bottom onto the top
    r = np.flip(right, axis = 1) #flipping the right onto the left

    x_overlap = np.sum((top==1) & (btm==1)) #finding overlap between top and bottom

    x_total = np.sum(top) + np.sum(btm) - x_overlap #finding total area when overlapped (subtracting overlap because it is counted twice otherwise)

    x = x_overlap/x_total #finding percentage overlapping out of total area

    #repeating for left and right
    y_overlap = np.sum((left==1) & (r==1))

    y_total = np.sum(left) + np.sum(r) - y_overlap

    y = y_overlap/y_total

    #transforming into asymmetry score
    xs = 0 #default score
    ys = 0 #default score
    t = 0.75 #threshhold for asymmetry (symmetry must be above 90%)

    if x < t:
        xs = 1 #asymmetry score over x-axis is 1

    if y < t:
        ys = 1 #asymmetry score over y-axis is 1

    return x, xs, y, ys #returns overlap-percentages and asymmetry scores

def asymmetry(im, mask):
    rotated_mask = mask
    best_score = 0
    best_rotation = 0
    rotation = 0

    for _ in range(36):
        cropped_im, cropped_mask = prep_im(im, rotated_mask)
        h, x, v, y = shape(cropped_mask)
        overlap_score = h + v
        if overlap_score > best_score:
            best_score = overlap_score
            best_rotation = rotation
        rotation += 10
        rotated_mask = transform.rotate(mask, rotation)

    mask = transform.rotate(mask, best_rotation)
    im_rotated = transform.rotate(im, best_rotation, mode="constant", cval=0)
    final_im, final_mask = prep_im(im_rotated, mask)
    

    return im_rotated, rotated_mask, best_rotation


# Calling the asymmetry function to access im_rotated and best_rotation
im_rotated, rotated_mask, best_rotation = asymmetry(image, mask)

# Plotting original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

# Plotting best rotation
plt.subplot(1, 2, 2)
plt.imshow(im_rotated) 
plt.title(f'Rotated Image (Best Rotation: {best_rotation} degrees)')
plt.show()


