from utils.processing.radar_ffts import range_fft
import numpy as np

def get_phase(radar_hdf5,range_time_profile,range_pad,window_function=np.hanning,range_bin_search = (3,3),show_progress = False):

    # get frame numbers
    start_frame, end_frame = radar_hdf5.get_frame_numbers()
    number_of_frames = end_frame - start_frame

    # get radar parameters
    time_between_chirps = radar_hdf5.time_chirp*3
    wavelength = 3e8/radar_hdf5.frequency_centre
    sensitivity = wavelength/(4*np.pi)
    nChirps = radar_hdf5.nChirps
    nVChannels = radar_hdf5.nVChannels

    frame_array = np.zeros((nChirps*number_of_frames,nVChannels),dtype=np.complex128)

    # load frames
    # print("CONSTRUCTING PHASE: ")
    channel_select = np.arange(nVChannels)
    time_vals = []
    previous_timestamp = 0
    for frame_number,i in zip(range(start_frame,end_frame),np.arange(number_of_frames)):
        # get radar frame
        frame_data, timestamp = radar_hdf5.get_frame(frame_number)
        if timestamp < previous_timestamp:
            timestamp = previous_timestamp + np.mean(np.diff(np.array(time_vals)))*nChirps
        previous_timestamp = timestamp
        frame = radar_hdf5.sort_data(frame_data)

        # get radar range profile
        frame = range_fft(frame,range_pad,window_function)

        # select range bin
        range_bin = int(range_time_profile.get_range_bin(timestamp)) 
        frame_array[i*nChirps:(i+1)*nChirps,:] = np.sum(frame[range_bin-range_bin_search[0]:range_bin+range_bin_search[1]+1,:,:],0)

        for chirp in range(nChirps):
            time_vals.append(timestamp+chirp*time_between_chirps)

        # print progress bar
        if show_progress:
            progress = int((frame_number)*20/(number_of_frames+start_frame))
            bar = "".join([u"\u2588"]*progress + [" "]*(20-progress-1))
            print("Progress: %d%%" % ((progress+1)*100/20) + " |" + str(bar) + "| "  ,end="\r") 
    
    if show_progress:
        print()
        print("Done.\n")

    radar_time = np.array(time_vals)
    
    if show_progress:
        print("CALIBRATING PHASE:")
    for channel in channel_select:
        frame_array[:,channel] = frame_array[:,channel] - np.mean(frame_array[:,channel])
    
    chirp_array = np.sum(frame_array[:,channel_select],1)
    
    # below code will result in non-uniform sample time 
    # original_size = len(chirp_array)
    # magnitude_threshold = 2000        
    
    # low_mag_index = np.argwhere(np.abs(chirp_array)<magnitude_threshold)
    # chirp_array = np.delete(chirp_array,low_mag_index)
    # radar_time = np.delete(radar_time,low_mag_index)
    # if show_progress:
    #     print("Removed %.2f%% of samples."%((original_size-len(chirp_array))*100/original_size),end='\r')
        
        
    if show_progress:
        print("Done.\n")
        
        
    phase = np.angle(chirp_array)
    displacement = np.unwrap(phase)*-sensitivity


    return displacement, radar_time, chirp_array

