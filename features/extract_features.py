"""
Created on ...

@authors: 
"""
import numpy as np
import matplotlib.pyplot as plt

# Import packages for image processing
from skimage import morphology, filters, transform
from skimage.segmentation import slic
from skimage.transform import resize

# --Asymmetry--

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
    '''preps and crops an image for asymmetry testing
    im_id : the complete path and id of the image
    mask_id : the complete path and id of the mask
    '''

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
    mid_v = png.shape[0] // 2 #horizontal middle
    mid_h = png.shape[1] // 2 #vertical middle
    
    #cropping image to get vertical and horizontal splits
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
    t = 0.85 #threshhold for asymmetry (symmetry must be above 85%)

    if x < t:
        xs = 1 #asymmetry score over x-axis is 1

    if y < t:
        ys = 1 #asymmetry score over y-axis is 1

    return x, xs, y, ys #returns overlap-percentages and asymmetry scores

def avg(im, m):
    filtered = filters.gaussian(im, sigma=5, channel_axis=-1)
    red, green, blue = [filtered[:,:,c] for c in range(3)]
    mean_red = np.mean(red[m.astype(bool)])
    mean_green = np.mean(green[m.astype(bool)])
    mean_blue = np.mean(blue[m.astype(bool)])

    avg_color = np.mean((mean_red, mean_green, mean_blue))

    return avg_color

def get_color(im, mask):
   '''takes a prepped image and mask
   returns an rgb-score for each quadrant of the masked part of the image
   to be used for the color asymmetry score'''

   top, bottom, left, right = axis_split(im) #splitting the image on the x- and y-axis
   tm, bm, lm, rm = axis_split(mask) #splitting the mask on the x- and y-axis

   avg_top = avg(top, tm)
   avg_bottom = avg(bottom, bm)
   avg_left = avg(left, lm)
   avg_right = avg(right, rm)

   return avg_top, avg_bottom, avg_left, avg_right

def a_color(im, mask):
    '''
    takes a prepped image and mask and runs the get_color function on the image and mask
    returns an asymmetry score equal to either 0, 1 and 2 for no asymmetry or asymmetry on one or both of the x- and y-axis
    '''
    top, bottom, left, right = get_color(im, mask) #getting rgb-values
    x_axis = abs(top - bottom) #finding the absolute difference between the top and bottom color
    y_axis = abs(left - right) #finding the absolute difference between the left and right color

    xc_a = 0 #setting x-axis asymmetry score to 0 by default
    yc_a = 0 #setting y-axis asymmetry score to 0 by default

    if x_axis > 0.05: #if the difference exceeds 0.5
        xc_a = 1 #x-axis asymmetry score is 1
    
    if y_axis > 0.05: #if the difference exceeds 0.5
        yc_a = 1 #y-axis asymmetry score is 1

    return xc_a, yc_a #return axes' asymmetry score

def final_score(im, mask):
    '''
    takes a prepped image and mask
    combines the color and shape asymmetry scores
    by comparing the axes' scores
    gives each axis a score of 0 or at most 1
    and adds these for the final score, which is either 0, 1 or 2
    '''
    xc, yc = a_color(im, mask) #getting color asymmetry score
    x, xs, y, ys = shape(mask) #getting shape asymmetry score

    x_score = 0 #default score = 0
    y_score = 0 #default score = 0

    if xc == 1 or xs == 1: #if either shape or color asymmetry is 1
        x_score = 1 #score is 1
    
    if yc == 1 or ys == 1: #if either shape or color asymmetry is 1
        y_score = 1 #score is 1

    total = x_score + y_score #calculating total asymmetry score

    return total #returning total score

def asymmetry(im, mask):
    '''
    takes an image and mask
    rotates the mask until the orientation with the best symmetry is found
    rotates the image, gets the color symmetry score
    and combines these
    output is final asymmetry score where both color and shape is accounted for
    '''

    rotated_mask = mask
    best_score = 0 #starting at worst score - zero percent overlap on any axis
    best_rotation = 0 #starting rotation
    rotation = 0 #starting rotation

    #rotating mask-image to find best rotation
    for _ in range(36):
        cropped_mask = prep_mask(rotated_mask)
        h, x, v, y = shape(cropped_mask) #get asymmetry scores for this rotation
        overlap_score = h+v #symmetry difference score, the sum of the area-differences over the x and y axis
        if overlap_score > best_score: #if the current score is better than the best score
            best_score = overlap_score #set current difference as new best score
            best_rotation = rotation #set the current rotation as the best rotation
        rotation += 10 #increase rotation by 10
        rotated_mask = transform.rotate(mask, rotation) #rotate mask image to new rotation
    
    mask = transform.rotate(mask, best_rotation) #rotating mask-image to best rotation
    im = transform.rotate(im, best_rotation, mode = "constant", cval = 0) #rotating image to best rotation and filling blank spaces with zeros
    final_im, final_mask = prep_im(im, mask)
    score = final_score(final_im, final_mask)
    
    return score

# --Color--
def color(im, mask):

    """
    """

    # Segment the image using skimage SLIC (Simple Linear Iterative Clustering)
    segments = slic(im, n_segments=12, compactness=10, sigma=1, mask=mask)

    # Initialize a dictionary to store average RGB values for each segment
    # key = color average, value = list of segments with this color average
    segment_avg_rgb = {}

    # Iterate over each unique segment ID
    for segment_id in np.unique(segments):

        # Create a mask for the current segment
        segment_mask = segments == segment_id 

        # Extract pixels belonging to the current segment
        segment_pixels = im[segment_mask] 

        # Calculate the average RGB value for the segment
        avg_rgb = tuple(np.round(np.mean(segment_pixels, axis=0)))  # Use tuple for hashing

        # Group segments with the same color by storing them under the average RGB value
        if avg_rgb in segment_avg_rgb:
            segment_avg_rgb[avg_rgb].append(segment_id)
        else:
            segment_avg_rgb[avg_rgb] = [segment_id]

        # Calculate the number of unique colors in the image by counting the keys in the dictionary
        num_colors = len(segment_avg_rgb) 

    return num_colors*0.5

# --Irregular dots/Globules--
def dg(im, mask):

    """
    """

    # Gaussian blur
    im_blur = filters.gaussian(im, sigma=5, channel_axis=-1)

    # Extracting the color layers
    red, green, blue = [im_blur[:,:, c] for c in range(3)]

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

    # Binary mask where intensity is lower than the average intensity and color is brownish
    dg_mask = (intensity < average_intensity * threshold)

    # Apply the lesion mask
    dg_mask = dg_mask & mask.astype(bool)

    # Percentage of lesion with irregular dots/globules
    dg_percentage = np.sum(dg_mask) / np.sum(mask)
    
    # Deciding whether the feature is present
    if dg_percentage > 0.04:
        score = 1
    else:
        score = 0

    return score



