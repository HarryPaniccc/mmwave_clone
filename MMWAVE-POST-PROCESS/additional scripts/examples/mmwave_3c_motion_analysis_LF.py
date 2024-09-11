## utils
from utils.npy.depth import depthNPY
from utils.hdf5.polar import polarHDF5
from utils.npy.polar import polarNPY
from utils.hdf5.radar import radarHDF5

from utils.processing.functions import *
from utils.processing.filters import *


## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import json
from PyEMD import CEEMDAN

def main():

    # setup plots
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 13
    
    ## Load Config
    print("LOADING CONFIG DATA:")
    working_dir = os.path.dirname(__file__)
    f = open(os.path.join(working_dir,'pipeline_config.json'))
    pipeline_config = json.load(f)

    exp_num = pipeline_config["EXPERIMENT_NUMBER"]
    date = pipeline_config["DATE"]

    
    window_function = signal.windows.hann




    print("Done.\n")


    ## Load Radar Info
    print("LOADING RADAR INFO:")
    path_to_data = "data/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(exp_num,date)
    working_dir = os.path.dirname(__file__)
    radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))
    radar_hdf5.display_radar_performance()
    _, unix_start_time = radar_hdf5.get_frame(0)
    print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())
    print("Done.\n")

    ## Load Phase Data
    print("LOADING PHASE DATA:")

    path_to_data = "data/{0}_Exp{1}/depth_time.npy".format(date,exp_num)
    phase_time = np.load(os.path.join(working_dir,path_to_data)) - unix_start_time
    sample_period  = np.mean(np.diff(phase_time))
    
    path_to_data = os.path.join(working_dir,"data/{0}_Exp{1}/depth_phase.npy".format(date,exp_num))
    phase_data = np.load(path_to_data)
    print("Done.\n")

    # Load depth Data

    ## Load depth Data
    print("LOADING REALSENSE DEPTH DATA:")
    working_dir = os.path.dirname(__file__)
    # hips
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance_hp.npy".format(exp_num,date))
    depth_rs_hp = depthNPY(path_to_data)

    # chest
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance_ch.npy".format(exp_num,date))
    depth_rs_ch = depthNPY(path_to_data)

    # right leg
    # path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance_rl.npy".format(exp_num,date))
    # depth_rs_rl = depthNPY(path_to_data)

    # left leg
    path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance_ll.npy".format(exp_num,date))
    depth_rs_ll = depthNPY(path_to_data)

    rs_time = depth_rs_ll.timestamps - unix_start_time
    sample_period_rs = np.mean(np.diff(rs_time))

        #truncate experiment#
    print("SETTING EXPERIMENT TIME DOMAIN:")
    # additional configs
    segment = 2
    start_time = 50
    end_time = start_time+30
    start_index =   np.argwhere(rs_time>=start_time)[0][0]
    end_index =  np.argwhere(rs_time<=end_time)[-1][0]
    rs_time = rs_time[start_index:end_index]
    depth_rs_hp.range_bins = depth_rs_hp.range_bins[start_index:end_index]
    depth_rs_ch.range_bins = depth_rs_ch.range_bins[start_index:end_index]
    # depth_rs_rl.range_bins = depth_rs_rl.range_bins[start_index:end_index]
    depth_rs_ll.range_bins = depth_rs_ll.range_bins[start_index:end_index]




    plt.figure()
    plt.subplot(121)
    plt.plot(rs_time,depth_rs_hp.range_bins,label="Hips")
    plt.plot(rs_time,depth_rs_ch.range_bins,label="Chest")
    # plt.plot(rs_time,depth_rs_rl.range_bins,label="Right Leg")
    plt.plot(rs_time,depth_rs_ll.range_bins,label="Leg")
    plt.legend(fontsize="10", loc ="upper left")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.title("Body Part Motion Comparison")

    plt.subplot(322)
    plt.title("Body Part Motion Comparison 0.2Hz-1Hz")

    chest_motion = filter_butter(depth_rs_ch.range_bins,"bandpass",(0.2,1),4,sample_period_rs)[0]
    compensated_chest_motion = filter_butter(depth_rs_ch.range_bins-depth_rs_hp.range_bins,"bandpass",(0.2,1),4,sample_period_rs)[0]
    plt.plot(rs_time,chest_motion,label="Chest")
    plt.plot(rs_time,compensated_chest_motion,label="Chest Relative To Hips")
    plt.legend( fontsize="10", loc ="upper left")
    plt.ylabel("Chest")
    plt.ylim(-0.3,0.3)


    plt.subplot(324)
    left_leg_motion = filter_butter(depth_rs_ll.range_bins,"bandpass",(0.2,1),4,sample_period_rs)[0]
    compensated_leg_motion = filter_butter(depth_rs_ll.range_bins-depth_rs_hp.range_bins,"bandpass",(0.2,1),4,sample_period_rs)[0]
    plt.plot(rs_time,left_leg_motion,label="Leg")
    plt.plot(rs_time,compensated_leg_motion,label="Leg Relative To Hips")
    plt.legend( fontsize="10", loc ="upper left")
    plt.ylim(-0.3,0.3)
    plt.ylabel("Leg")

    plt.subplot(326)
    hip_motion = filter_butter(depth_rs_hp.range_bins,"bandpass",(0.2,1),4,sample_period_rs)[0]
    plt.plot(rs_time,hip_motion,label="Hip Motion")
    plt.ylabel("Hips")
    plt.legend( fontsize="10", loc ="upper left")
    plt.ylim(-0.3,0.3)
    plt.xlabel("Time [s]")



    print("Done.\n")



    ## FFT SPECTROGRAM BREATHING -------------------------------------------------------------------------------------------------------------------------------------------
    
    motions = [compensated_leg_motion, compensated_chest_motion, hip_motion]
    # motions = [left_leg_motion, chest_motion, hip_motion]

    motion_title = ["Leg Swing", "Chest Rock","Hip Displacement"]

    for motion, title in zip(motions,motion_title):
        print(title.upper() + " MOTION SPECTROGRAM:")


        window_duration = 10
        num_pts = 15

        time_arr, freq_arr, spectro_map = spectrogram(rs_time,motion,
                                                    sample_period_rs,window_duration,10,0.99,window_func=window_function,normalised=True)
        
        start_frequency = 0
        end_frequency = 4
        freq_start_index = np.argwhere(freq_arr<=start_frequency)[-1][0]
        freq_end_index = np.argwhere(freq_arr<=end_frequency)[-1][0]
        plotted_freq_arr = freq_arr[freq_start_index:freq_end_index]*60


        if start_frequency>=0:
            spectro_map = np.flipud(spectro_map[freq_start_index:freq_end_index,:])
        else:
            spectro_map = spectro_map[freq_start_index:freq_end_index,:]

        extent_list = [time_arr[0],time_arr[-1],start_frequency,end_frequency]
        x_ticks = np.linspace(time_arr[0],time_arr[-1],11)

        freq_inc = 10
        y_ticks = np.arange(start_frequency*60,end_frequency*60+freq_inc,freq_inc)
        

        breathing_fig = plt.figure( figsize=(8, 6))
        img = breathing_fig.add_subplot(111)

        im = img.imshow(spectro_map,extent=extent_list,cmap="plasma",aspect='auto')
        img.set_xticks(x_ticks)
        img.set_yticks(y_ticks/60)
        breathing_fig.colorbar(im,label="Normalised Power [dB]")
        img.set_xticklabels(np.round(x_ticks,2))
        img.set_yticklabels((np.round(y_ticks,2)))
        img.grid(which='minor', color='w', linestyle='-', linewidth=2)
        img.set_ylabel("Frequency [bpm]")
        img.set_xlabel("Window Start Time [s]")
        img.set_title( title + " Spectrogram [%.2fs Window]" % (window_duration))

        img.set_ylim(0,4)
        
        # plot expected heart rate onto spectrogram
        measured_rate = []
        time_index = 0
        for t in time_arr:
            slice_of_spectro = spectro_map[:,time_index]
            measured_rate_index = np.argmax(slice_of_spectro)
            measured_rate.append(240-plotted_freq_arr[measured_rate_index])
            time_index += 1
        
        measured_rate = np.array(measured_rate)
        breathing_rates = moving_average_filter(measured_rate,num_pts)

        # img.plot(time_arr,measured_rate/60,label="Measured Breathing Rate")
        img.plot(time_arr,breathing_rates/60,alpha=0.2,label="Detected Frequency")
        img.legend()

        

        print("Done.\n")

        # plt.show()
        # exit()

   


    plt.show()


    
if __name__ == "__main__":
    main()