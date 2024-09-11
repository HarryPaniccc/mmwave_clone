## utils
from utils.hdf5.radar import radarHDF5
from utils.processing.radar_ffts import range_doppler_fft,range_doppler_sum,azimuth_fft,azimuth_sum
from utils.processing.cfar import cfar, clean_cfar
import utils.processing.phase as PHASE
from utils.processing.functions import range_bins_to_ranges

from utils.npy.depth import depthNPY

## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
import json
import time
from scipy import signal
import cv2


def make_r_theta_vals(range_max,samples,range_pad,
        min_az=-np.pi/3,max_az=np.pi/3, az_bins=8,az_pad=8,
                      ):
    thetas_radians = np.linspace(min_az,max_az,az_bins+az_pad)
    radii = np.flip(np.linspace(0,range_max,samples+range_pad))
    return thetas_radians, radii

def make_polar_plot(azimuth_map,range_max,samples,range_pad,az_pad,fig,subplot):
    ax = fig.add_subplot(subplot,projection="polar")
    thetas_radians, radii = make_r_theta_vals(range_max,samples,range_pad,az_pad=az_pad)
    plot = ax.pcolormesh(thetas_radians,radii,azimuth_map,edgecolors='face',cmap='viridis')
    ax.set_theta_zero_location("N")
    ax.set_rlim(0,5)
    ax.grid(False)
    ax.set_thetalim(thetas_radians[0],thetas_radians[-1])
    ax.set_theta_direction(-1)
    label_position=ax.get_rlabel_position()
    ax.text(np.radians(label_position-110),ax.get_rmax()/2.,'Distance [m]',
        rotation=-(label_position+5),ha='center',va='center')


    return plot, ax


## load config
print("LOADING CONFIG DATA:")
working_dir = os.path.dirname(__file__)
f = open(os.path.join(working_dir,'pipeline_config.json'))
pipeline_config = json.load(f)
exp_num = pipeline_config["EXPERIMENT_NUMBER"]
date = pipeline_config["DATE"]

# processing config
range_pad = pipeline_config["RANGE_CONFIG"]["RANGE_PAD"]
doppler_pad = pipeline_config["DOPPLER_CONFIG"]["DOPPLER_PAD"]
azimuth_pad = pipeline_config["AZIMUTH_CONFIG"]["AZIMUTH_PAD"]
window_func = signal.windows.hann


print("Done.\n")

## Open radar data file
print("LOADING RADAR DATA:")
path_to_data = "data/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(exp_num,date)
working_dir = os.path.dirname(__file__)
radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))
max_velocity = radar_hdf5.velocity_max
max_range = radar_hdf5.range_max
_, unix_start_time = radar_hdf5.get_frame(0)
print("Done.\n")

samples = radar_hdf5.nSamples
chirps = radar_hdf5.nChirps
channels = radar_hdf5.nVChannels
frames = radar_hdf5.nFrames
frame_period = radar_hdf5.frame_period



print("CALIBRATING FOR CLUTTER:")
frame_data, timestamp = radar_hdf5.get_frame(0)
frame = radar_hdf5.sort_data(frame_data)
ard_fft =  range_doppler_fft(frame, range_pad,doppler_pad,window_func,window_func)
raz_fft = azimuth_fft(ard_fft,azimuth_pad=azimuth_pad,azimuth_window=window_func)
raz_map0 = raz_fft

calib_frames = 10
for j in range(1,calib_frames):
    frame_data, timestamp = radar_hdf5.get_frame(j)
    frame = radar_hdf5.sort_data(frame_data)
    ard_fft =  range_doppler_fft(frame, range_pad,doppler_pad,window_func,window_func)
    raz_fft = azimuth_fft(ard_fft,azimuth_pad=azimuth_pad,azimuth_window=window_func)
    raz_map0 = raz_map0 + raz_fft

    increments = 50
    progress = int(j*increments/calib_frames)
    bar = "".join([u"\u2588"]*progress + [" "]*(increments-progress-1))
    print("Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "|"  ,end="\r") 

raz_map0 = raz_map0/calib_frames
print()
print("Done.\n")


frame_data, timestamp = radar_hdf5.get_frame(402)
frame = radar_hdf5.sort_data(frame_data)
ard_fft = range_doppler_fft(frame, range_pad,doppler_pad,window_func,window_func)
raz_fft = azimuth_fft(ard_fft,azimuth_pad=azimuth_pad,azimuth_window=window_func)
raz_map = azimuth_sum(raz_fft-raz_map0)
raz_map_cfar = cfar(raz_map,1,1,2,2,10e-3,-1200)
raz_map_cfar,_raz_map_cfar = clean_cfar(raz_map_cfar,4)

fig = plt.figure()
raz_plot, raz_ax = make_polar_plot(raz_map,max_range,samples,range_pad,az_pad=azimuth_pad,fig=fig,subplot=121)
raz_plot_cfar, raz_ax_cfar = make_polar_plot(raz_map_cfar,max_range,samples,range_pad,az_pad=azimuth_pad,fig=fig,subplot=122)

raz_ax.set_title("RANGE AZIMUTH MAP")
raz_ax_cfar.set_title("RANGE AZIMUTH MAP")


start_time = 0
stop_time = 120

def update_fig(i):
    # global frame_data, frame, ard_fft, ard_map, cfar_map, trajectory
    try:
        q =  calib_frames + 400
        frame_data, timestamp = radar_hdf5.get_frame(q)
        # print(i)
    except:
        print()
        # np.save("CFAFR_time.npy", trajectory_time)
        # np.save("CFAFR.npy", trajectory_cfar)

        print("Done.\n")
        exit()
    
    if (timestamp - unix_start_time) <= start_time:
        return raz_ax,

    if (timestamp - unix_start_time) >= stop_time:
        return raz_ax,

    frame = radar_hdf5.sort_data(frame_data)
    ard_fft =  range_doppler_fft(frame, range_pad,doppler_pad,window_func,window_func)
    raz_fft = azimuth_fft(ard_fft,azimuth_pad=azimuth_pad,azimuth_window=window_func)
    raz_map = azimuth_sum(raz_fft-raz_map0)
    # range_azimuth_fft = raz_fft[:,0,:]
    # raz_map = 20*np.log10(np.abs(range_azimuth_fft)/np.max(np.max(np.abs(range_azimuth_fft))))
    raz_map_cfar = cfar(raz_map,1,1,2,2,10e-3,-1200)

    # calibrate azimuth
    raz_map_cfar = np.roll(raz_map_cfar,10,1)

    raz_map_cfar,_ = clean_cfar(raz_map_cfar,4)

    
    raz_plot.set_array(raz_map)
    raz_plot_cfar.set_array(raz_map_cfar)



    increments = 50 
    progress = int(q*increments/frames)
    bar = "".join([u"\u2588"]*progress + [" "]*(increments-progress-1))
    # try:
    #     print("Frame: %d  || " % (q)  + "Time : %.2fs  || " % (timestamp-unix_start_time) + " Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "| "  ,end="\r") 
    # except:
    #     print("No Target  || " + " Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "|"  ,end="\r") 
    print("Time : %.2fs  || " % (timestamp - unix_start_time) + " Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "|"  ,end="\r") 



    return raz_plot,raz_plot_cfar,

anim = animation.FuncAnimation(fig, update_fig, interval=frame_period, blit=True,frames=frames)



plt.show()


