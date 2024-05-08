"""
@authors: cjep, kmah, feso, aith, nozo
"""


import numpy as np
import matplotlib.pyplot as plt

# Import packages for image processing
from skimage import morphology, filters, transform
from skimage.segmentation import slic
from skimage.transform import resize



# ---Help functions---

def prep_mask(m):
    '''
    Preps and crops a mask

    Parameters:
    • m: numpy array, the input mask
    
    Returns:
    • cropped_m: numpy array, the preprocessed and cropped mask
    '''

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
    '''
    Preps and crops an image for asymmetry testing
    
    Parameters:
    • im: numpy array, the input image
    • m: numpy array, the corresponding input mask

    Returns:
    • cropped_im: numpy array, the preprocessed and cropped image
    • cropped_m: numpy array, the preprocessed and cropped mask
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
    '''
    Takes a prepped image or mask and splits it in two over the x-axis and the y-axis, yielding 4 images

    Parameters:
    • png: numpy array, the input image or mask

    Returns:
    • four numpy arrays representing the top, bottom, left, and right parts of the input
    '''

    mid_v = png.shape[0] // 2 #horizontal middle
    mid_h = png.shape[1] // 2 #vertical middle
    
    #cropping image to get vertical and horizontal splits
    top = png[:mid_v,:]
    bottom = png[mid_v:, :]
    left = png[:, :mid_h]
    right = png[:, mid_h:]

    return top, bottom, left, right


def shape(mask):
    '''
    Takes a prepped mask and returns the shape asymmetry score

    Parameters:
    • mask: numpy array, the preprocessed mask
    '''

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
    '''
    Returns the average color of the masked region in the prepped image

    Parameters:
    • im: numpy array, the preprocessed image
    • m: numpy array, the preprocessed mask
    '''

    filtered = filters.gaussian(im, sigma=5, channel_axis=-1)
    red, green, blue = [filtered[:,:,c] for c in range(3)]
    mean_red = np.mean(red[m.astype(bool)])
    mean_green = np.mean(green[m.astype(bool)])
    mean_blue = np.mean(blue[m.astype(bool)])

    avg_color = np.mean((mean_red, mean_green, mean_blue))

    return avg_color


def get_color(im, mask):
   '''
   Returns an rgb-score for each quadrant of the masked part of the image,
   to be used for the color asymmetry score

   Parameters:
   • im: numpy array, the preprocessed image
   • mask: numpy array, the preprocessed mask
   '''

   top, bottom, left, right = axis_split(im) #splitting the image on the x- and y-axis
   tm, bm, lm, rm = axis_split(mask) #splitting the mask on the x- and y-axis

   avg_top = avg(top, tm)
   avg_bottom = avg(bottom, bm)
   avg_left = avg(left, lm)
   avg_right = avg(right, rm)

   return avg_top, avg_bottom, avg_left, avg_right


def a_color(im, mask):
    '''
    Calculates the color asymmetry score for a prepped image and mask

    Parameters:
    • im: numpy array, the preprocessed image
    • mask: numpy array, the preprocessed mask
    
    Returns:
    • The asymmetry score equal to either 0, 1 and 2 for no asymmetry or asymmetry on one or both of the x- and y-axis
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
    Combines the color and shape asymmetry scores by comparing the axes' scores
    
    Parameters:
    • im: numpy array, the preprocessed image
    • mask: numpy array, the preprocessed mask
    
    Returns:
    • Final score, which is either 0, 1 or 2
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



# ---Asymmetry---

def asymmetry(im, mask):
    '''
    Takes an image and mask.
    Rotates the input mask until the orientation with the best symmetry is found
    Rotates the input image, gets the color symmetry score, and combines these
    
    Parameters:
    • im: numpy array, the input image
    • mask: numpy array, the input mask corresponding to the image

    Returns:
    • final asymmetry score where both color and shape is accounted for (0, 1, or 2)
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



# ---Color---

def color(im, mask):

    """
    Computes the number of unique colors in a lesion image

    Parameters:
    • im: numpy array, the input image
    • mask: numpy array, the corresponding input mask

    Returns:
    • The number of unique colors in the masked region of the image
    """

    # Segment the image using skimage SLIC
    segments = slic(im, n_segments=12, compactness=10, sigma=1, mask=mask)

    # Initialize a dictionary to store average RGB values for each segment
    # key = color average, value = list of segments with this color average
    segment_avg_rgb = {}

    # Iterate over unique segment IDs
    for segment_id in np.unique(segments):

        # Create a mask for the current segment
        segment_mask = segments == segment_id 

        # Extract pixels from the current segment
        segment_pixels = im[segment_mask] 

        # Calculate the average RGB value for the segment
        avg_rgb = tuple(np.round(np.mean(segment_pixels, axis=0)))  # Use tuple for hashing

        # Group segments with the same color by storing them under the average RGB value
        if avg_rgb in segment_avg_rgb:
            segment_avg_rgb[avg_rgb].append(segment_id)
        else:
            segment_avg_rgb[avg_rgb] = [segment_id]

        # Calculate the number of unique colors in the image by counting the keys
        num_colors = len(segment_avg_rgb) 

    return num_colors*0.5 # Multiplying number of colors with 0.5 to normalize results



# ---Irregular dots/Globules---

def dg(im, mask):

    """
    Detects irregular dots/globules within the lesion area

    Parameters:
    • im: numpy array, the input image
    • mask: numpy array, the corresponding input mask 

    Returns:
    • A score of either 0 or 1, indicating the absence or presence of the feature
    """

    # Gaussian blur to filter out interfering features - such as hair
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

    # Binary mask where intensity is lower than the average intensity
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



