import numpy as np
from scipy.fft import fft2,fftshift, ifft2
import cv2 


def cfar(range_doppler_map, cfar_range_guard, cfar_range_training, cfar_doppler_guard, cfar_doppler_training,cfar_pfa, cfar_threshold) :
    """Performs the CFAR process by creating a kernel of size [1 + 2*( cfar_range_guard +  cfar_range_training)] in range axis and
    [1 + 2*( cfar_doppler_guard +  cfar_doppler_training)))] in the doppler axis. It then takes the fft of the kernal and mutliplies that with the 
    frequency response of range_doppler_map to and takes the inverse fft of the result to do a 2D convolution. This slides the kernel around the map to perform
    the CFAR. 
    
        input: range_doppler_map        -> must be a 2D array. It is the output of ard process in dB.
        input: cfar_range_guard         -> number of guard cells in range axis.
        input: cfar_range_training      -> number of training cells in the range axis.
        input: cfar_doppler_guard       -> number of guard cells in doppler axis.
        input: cfar_doppler_training    -> number of training cells in the doppler axis.
        input: cfar_pfa                 -> probability of detection.
        input: cfar_threshold           -> tuning parameter in dB to help set CFAR threshold.

        output: detections          -> 2D matrix of size [nSamples+rangePad,nChirps+dopplerPad] contains value 1 for a detection and 0 for no detection. 
        
    """      

    # create a cfar training kernel
    p_kernel = np.ones(
        (1 + 2*( cfar_range_guard +  cfar_range_training),
            1 + 2*( cfar_doppler_guard +  cfar_doppler_training)))
    
    # set the middle of the kernel to zero to create guard cells
    guard = np.zeros((2* cfar_range_guard+1,2* cfar_doppler_guard+1))
    p_kernel[ cfar_range_training  : cfar_range_training+1+2* cfar_range_guard,
                cfar_doppler_training: cfar_doppler_training+1+2* cfar_doppler_guard] = guard        
    
    # additional cfar params 
    pfa =  cfar_pfa                                             # probability of false alarms
    num_train_cells = np.sum(np.sum(p_kernel))                      # number of training cells
    alpha = num_train_cells * (pfa ** (-1 / num_train_cells) - 1)   # threshold gain
    dims = np.shape(range_doppler_map)
    kernel = np.zeros(dims)

    # squaring the range_doppler_map
    rdm_power = np.square(np.abs(range_doppler_map))
    
    # zero pad kernel for fft
    kernel[0 : np.size(p_kernel, 0), 0 : np.size(p_kernel, 1)] = p_kernel
    kernel = kernel / num_train_cells

    mask = fft2(kernel)                                            # put mask in frequency
    noise = ifft2(np.multiply(np.conj(mask), fft2(rdm_power))) # convolution done in frequency
    row_shift = int( np.floor(np.size(p_kernel, 0) // 2))              # shift that has to be accounted for after the convolution
    noise = np.roll(noise, row_shift, 0)                               # account for shift introduced by convolution

    # threshold exceedance
    indices = rdm_power > (noise * alpha +  cfar_threshold) # does the RD map exceed the cfar threshold at any points?
    detection_indices = np.argwhere(indices)                    # list of [row_idx, col_idx] where there is a possible detections
    n = np.size(detection_indices, 0)                           # number of possible detections
    
    local_max_indices = np.zeros(dims)
    for i in range(n):
        row_col_idx = detection_indices[i, :]
        row = row_col_idx[0]
        col = row_col_idx[1]
        local_max_indices[row, col] = 1

    detections = local_max_indices  # matrix with ones where detection occured
    return detections

def clean_cfar(cfar_map, radius):
    """Finds the clusters within a cfar map and their centroids and replaces the clusters with circles of detections with radius "radius" at the centroids.
    Then it returns the cleaner cfar map and the list of cluster centroids. 
    
        input: cfar_map        -> must be a 2D array. It is the output of cfar process.

        output: detections     -> 2D matrix of size contains value 1 for a detection and 0 for no detection. 
        
    """      

    # find clusters in cfar map
    img = cv2.convertScaleAbs(cfar_map)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find centroids of clusters
    centroid_list = []
    refined_cfar = np.zeros(( cfar_map.shape[0],  cfar_map.shape[1])
                ,dtype=np.uint8)
    
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = [cX,cY]
            centroid_list.append(centroid)
            cv2.circle(refined_cfar, (cX, cY), radius, 1, -1)  
    
    return refined_cfar, centroid_list