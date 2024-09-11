## utils
from utils.npy.depth import depthNPY
from utils.hdf5.radar import radarHDF5

from utils.processing.phase import get_phase
from multiprocessing import Pool


from utils.processing.filters import *
from utils.processing.functions import *

## common python imports
import os
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
import json
import glob

def calculate_phase(offset,progress,remove_disconts):
    # global range_time_profile, radar_hdf5, range_pad

    range_pad=416

    range_time_profile.range_bins =  moving_average_filter(range_time_profile.range_bins,30).astype(int) + offset 
    # range_time_profile.range_bins =  290*np.ones(len(range_time_profile.range_bins)) + offset# moving_average_filter(range_time_profile.range_bins,30).astype(int) + offset #

    phase_displacement, phase_time, _ = get_phase(radar_hdf5,range_time_profile,range_pad,show_progress=progress) # (3,3)
    if remove_disconts:
        for i in range(20):
            phase_displacement = remove_discontinuities(phase_time,phase_displacement,(1e-4,0.5),min_time_between_jumps=0.00001)
    
    phase_time = np.sort(phase_time)
    return phase_time,phase_displacement # , iq_samples


# get working dir 
working_dir = os.path.dirname(__file__)

# load config 
f = open(os.path.join(working_dir,'pipeline_config.json'))
pipeline_config = json.load(f)
range_pad = pipeline_config["RANGE_CONFIG"]["RANGE_PAD"]
data_dir_path = pipeline_config["FILE_PATH"]
resample = pipeline_config["RESAMPLE_TIME_TO_PHASE"]
radar_file_name = "Radar_Data*.hdf5"

## Load Radar data
path_to_data = glob.glob(os.path.join(working_dir,data_dir_path,radar_file_name))[0]
radar_hdf5 = radarHDF5(path_to_data)

_, unix_start_time = radar_hdf5.get_frame(0)

print(radar_hdf5.frame_period)
exit()


## Get range time profile
path_to_data = os.path.join(working_dir,data_dir_path,"rs_bins.npy")
range_time_profile = depthNPY(path_to_data)



  
def main():    
    global radar_hdf5
    global range_time_profile

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13

    print("Constructing Phase:")
    
    # settings
    offsets = -np.arange(0,14,2)
    progress = [False]*len(offsets)
    progress[-1] = True
    remove_disconts  = [True]*len(offsets)
    with Pool(10) as p:
        outputs = p.starmap(calculate_phase, zip(offsets,progress,remove_disconts))

    # show phase at each selected offset
    plt.figure()
    try:
        os.makedirs(os.path.join(working_dir,data_dir_path,"phase"))
    except Exception as e:
        pass
        
    for i in range(len(offsets)):
        radar_time= np.array(outputs[i][0])
        displacement= np.array(outputs[i][1])
        
        plt.subplot(211)
        plt.plot(radar_time,displacement,label=i)
        
        plt.subplot(212)
        heart_rate, sos = filter_butter(displacement,"bandpass",(1,2.3),4,np.mean(np.diff(radar_time)))
        plt.plot(radar_time,heart_rate,label=i)

        phase = np.array([radar_time,displacement])
        np.save(os.path.join(working_dir,data_dir_path,"phase","phase_meas_%d"%i), phase)
            
    plt.legend()
    plt.show()
    
    exit()
    
    # code past this point is for getting compensation signal
    # sum offsets together
    outputs = np.sum(outputs,0)/outputs.shape[0]
    

    ## Get compensation signal
    path_to_data = path_to_data = os.path.join(working_dir,data_dir_path,"rs_distance.npy")
    range_time_profile = depthNPY(path_to_data)
    depth_rs = range_time_profile.range_bins # moving_average_filter(range_time_profile.range_bins,10)
    depth_rs_time = range_time_profile.timestamps - unix_start_time

    try:
        path_to_data = os.path.join(working_dir,data_dir_path,"CFAR_distance.npy")
        depth_time_profile = depthNPY(path_to_data)
        depth_cd = depth_time_profile.range_bins # moving_average_filter(range_time_profile.range_bins,10)
        depth_cd_time = depth_time_profile.timestamps - unix_start_time

        path_to_data = os.path.join(working_dir,data_dir_path,"CFAR_velocity.npy")
        velocity_time = depthNPY(path_to_data)
        depth_cv_time = velocity_time.timestamps - unix_start_time
        velocity_time.range_bins = velocity_time.range_bins
        depth_cv = -integrate(velocity_time.range_bins-np.mean(velocity_time.range_bins),np.mean(np.diff(depth_cv_time)))
        depth_cv = depth_cv*(np.max(depth_cd)/np.max(depth_cv))
    except:
        print("No CFAR data available.")

    ## Plot raw data
    fig = plt.figure()
    plt.plot(outputs[0]-unix_start_time,outputs[1], label="Unwrapped Phase")
    plt.plot(depth_rs_time,depth_rs, label="Realsense Depth")
   
    try:
        plt.plot(depth_cd_time,depth_cd, label="CFAR Depth")
        plt.plot(depth_cv_time,depth_cv, label="CFAR Integrated Velocity")
    except:
        pass
    plt.legend()
    plt.title("Phase vs Distance")
    plt.xlabel("Time [s]")
    plt.ylabel("Target Displacement [m]")
    fig.show()

    phase_time = outputs[0]
    phase_displacement = outputs[1]


    ## Truncate time domain and align signals
    print("Choose times for export ->")
    start_time_chosen = int(input("Start time: "))
    end_time_chosen = int(input("End time: "))
    if start_time_chosen >= end_time_chosen:
        print("Invalid selection. Exiting...\n")
        exit()
    print("Selected.\n")

    radar_start_index =   np.argwhere(phase_time-unix_start_time>=start_time_chosen)[0][0]
    radar_end_index =  np.argwhere(phase_time-unix_start_time<=end_time_chosen)[-1][0]

    phase_time = phase_time[radar_start_index:radar_end_index] - unix_start_time
    phase_displacement = phase_displacement[radar_start_index:radar_end_index]
    phase_displacement -= phase_displacement[0]

    np.save(os.path.join(working_dir,data_dir_path,"depth_time"), phase_time)
    np.save(os.path.join(working_dir,data_dir_path,"depth_phase"), phase_displacement)
    
    if  not resample:
        exit()

    ## resample depth 
    print("UPSAMPLE AND TIME-SYNC:")
    print("Realsense:")

    depth_rs, _ = filter_butter(depth_rs,"lowpass",4,4,np.mean(np.diff(depth_rs_time)))
    depth_rs = resample_and_sync(depth_rs_time,depth_rs, phase_time)
    depth_rs, _ = filter_butter(depth_rs,"lowpass",4,4,np.mean(np.diff(phase_time)))

    depth_rs -= depth_rs[0]
    np.save(os.path.join(working_dir,data_dir_path,"depth_rs"), depth_rs)
    
    try:
        print("CFAR Depth:")
        depth_cd, _ = filter_butter(depth_cd,"lowpass",4,4,np.mean(np.diff(depth_cd_time)))
        depth_cd = resample_and_sync(depth_cd_time,depth_cd, phase_time)
        depth_cd, _ = filter_butter(depth_cd,"lowpass",4,4,np.mean(np.diff(phase_time)))
        depth_cd -= depth_cd[0]
        np.save(os.path.join(working_dir,data_dir_path,"depth_cd"), depth_cd)

        print("CFAR Velocity:")
        depth_cv, _ = filter_butter(depth_cv,"lowpass",4,4,np.mean(np.diff(depth_cd_time)))
        depth_cv = resample_and_sync(depth_cd_time,depth_cv, phase_time)
        depth_cv, _ = filter_butter(depth_cv,"lowpass",4,4,np.mean(np.diff(phase_time)))

        depth_cv -= depth_cv[0]
        np.save(os.path.join(working_dir,data_dir_path,"depth_cv"), depth_cv)
    except Exception as e:
        pass
        


    ## plot upsampled depth and phase
    plt.figure()
    try:
        plt.plot(phase_time-unix_start_time,depth_cd, label="CFAR Depth")
        plt.plot(phase_time-unix_start_time,depth_cv, label="CFAR Integrated Velocity")
    except:
        pass

    plt.plot(phase_time-unix_start_time,depth_rs, label="Realsense Depth")
    plt.plot(phase_time-unix_start_time,phase_displacement, label="Unwrapped Phase")

    
    plt.legend()
    plt.title("Phase vs Other Measurements")
    plt.xlabel("Time [s]")
    plt.ylabel("Target Displacement [m]")

    plt.show()




if __name__ == "__main__":
    main()