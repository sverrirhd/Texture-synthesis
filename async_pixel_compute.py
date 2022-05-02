import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Pool

def rolling_window(a, window):
    sh,sw = a.shape
    wh,ww = window
    
    shape = ((sh-wh+1), (sw-ww+1), wh, ww)
    strides = (a.strides[0], a.strides[1], a.strides[0], a.strides[1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def filter_patches(patches):
    wh,ww = patches.shape[-2:] 
    patches = patches.reshape((-1, wh*ww))
    
    # Throw away patches with 0 values (i.e. only patch row indexes where there are no 0s in the pathc)
    patches = patches[np.argwhere(np.all(patches != 0,axis=1))[:,0]].reshape(-1, wh, ww)
    return patches

def get_boundary_patches(mask,kernel_shape):
    
    sh,sw = mask.shape
    wh,ww = kernel_shape 
    # coordinates of center pixel inside each patch
    p_row = int(kernel_shape[0]//2)
    p_col = int(kernel_shape[1]//2)
    
    mask_patches = rolling_window(mask,kernel_shape)
    
    # Find all patches of mask where center pixel is missing
    p_miss = np.argwhere(mask_patches[:,:,p_row,p_col] == 0)
    
    # Check how many zeros in each patch
    zero_counts = np.sum(mask_patches==0,(-1,-2))
    
    # Get the minimum number of missing neighbours where center is also missing
    min_val = np.min([zero_counts[row,col] for row,col in p_miss])
    
    # Get all patch coordinates from p_miss with minimum number of missing neighbours
    indexes = np.argwhere((zero_counts == min_val)*(mask_patches[:,:,p_row,p_col] == 0))
    return indexes

def create_initial_mask(image):
    mask = np.zeros(image.shape)
    mask[np.where(image != 0)] = 1 
    return mask

def image_incomplete(mask):
    return np.sum(mask == 0) > 0

def gaussian(winsize=3, sigma=3):
    t = sigma**2
    x = np.arange(-(winsize//2),winsize//2+1,1)
    filt = 1.0/np.sqrt(t*2.0*np.pi)*np.exp(-(x**2.0)/(2.0*t))
    return filt

def normalized_SSD(POI,patches,kernel_shape):
    '''
    Computes the normalized sum of squared differences between 
    the patch of interest (POI)
    and all other patches given   
    '''
    # width of kernel
    wh,ww = kernel_shape
    POI_shape = POI.shape
    feat_vecs = patches.reshape(-1,POI_shape[0])
    # mask for missing values
    POI_mask = (POI > 0)*1
    feat_vecs_mask = POI_mask * feat_vecs
    
    
    # Scale the sigma to the size of the kernel
    sigma = (wh / 6)
    
    SD = (feat_vecs_mask - POI)**2
    SSD = np.sum(SD,axis=-1)
    
    # Normalize the SSD by the maximum possible contribution (based on missing pixels in POI mask)
    g2 = (gaussian(wh,sigma) * gaussian(wh,sigma).reshape(-1,1))
    total_SSD = np.sum(POI_mask * np.ravel(g2))
    
    normalized_ssd = SSD / total_SSD
    
    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=0.3):
    min_ssd = np.min(normalized_ssd)
    min_threshold = min_ssd * (1. + error_threshold)
    indices = np.argwhere(normalized_ssd <= min_threshold)
    return indices

def softmax(x):
    s = np.exp(x - np.max(x,axis=0))  # shift values
    s = s / s.sum(axis=0)
    return s

def select_pixel_index(normalized_ssd, indices):
    N = len(indices)
    
    # increased weight for closest matches
    weights = softmax(normalized_ssd[indices])
    # Select a random pixel index from the index list.
    selection = np.random.choice(np.arange(N), size=1, p=weights[:,0])
    selected_index = indices[selection]
    
    return selected_index

def get_pixel_value(index,POI_vec,filtered_patches,kernel_shape):
    
    p_row = int(kernel_shape[0]//2)
    p_col = int(kernel_shape[1]//2)
    selected_patch = filtered_patches[index].squeeze()
    pixel_value = selected_patch[p_row,p_col]
    return pixel_value
    
def get_results(response):
    try:
        outcome = response.get(timeout=10)
        return outcome
    except TimeoutError:
        return None

def async_compute_pixel(point_of_interest,patches_all,patches_filtered,kernel_shape,epsilon):
    irow,icol = point_of_interest
    
    # Make feature vector
    feat_vec = np.ravel(patches_all[irow,icol])

    # Compute 
    n_ssd = normalized_SSD(feat_vec,patches_filtered,kernel_shape)

    # Get the indicies of patches that are close enough to point of interest
    candidate_indicies = get_candidate_indices(n_ssd,epsilon)

    # Select an index with some random chance
    selected_index = select_pixel_index(n_ssd, candidate_indicies)
    return selected_index
