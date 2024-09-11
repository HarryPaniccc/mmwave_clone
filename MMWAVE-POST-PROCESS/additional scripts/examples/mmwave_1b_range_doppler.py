## utils
from utils.hdf5.radar import radarHDF5
from utils.processing.radar_ffts import range_doppler_fft, range_doppler_sum
from utils.processing.cfar import cfar, clean_cfar
from utils.processing.filters import *

from utils.npy.depth import depthNPY

## common python imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import json
import time
from scipy import signal
import cv2

## load config
print("LOADING CONFIG DATA:")
working_dir = os.path.dirname(__file__)
f = open(os.path.join(working_dir,'pipeline_config.json'))
pipeline_config = json.load(f)
exp_num = pipeline_config["EXPERIMENT_NUMBER"]
date = pipeline_config["DATE"]
range_pad = pipeline_config["RANGE_CONFIG"]["RANGE_PAD"]
doppler_pad = pipeline_config["DOPPLER_CONFIG"]["DOPPLER_PAD"]

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
window_func = signal.windows.hann




print("CALIBRATING FOR CLUTTER:")
frame_data, timestamp = radar_hdf5.get_frame(0)
frame = radar_hdf5.sort_data(frame_data)
ard_fft = range_doppler_fft(frame,range_pad,doppler_pad,window_func,window_func)
ard_map0 = ard_fft
calib_frames = 15
start_time = time.time()
for j in range(1,calib_frames):
    start_time_get_frame = time.time()
    frame_data, timestamp = radar_hdf5.get_frame(j)
    frame = radar_hdf5.sort_data(frame_data)
    time_got_frame = time.time() - start_time_get_frame

    time_start_ard = time.time()
    ard_fft = range_doppler_fft(frame,range_pad,doppler_pad,window_func,window_func)
    time_did_ard = time.time() - time_start_ard
    
    ard_map0 = ard_map0 + ard_fft

    

    increments = 50
    progress = int(j*increments/calib_frames)
    bar = "".join([u"\u2588"]*progress + [" "]*(increments-progress-1))
    print("Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "|"  ,end="\r") 

print()
print("Done.\n")


ard_map0 = ard_map0/calib_frames
# ard_map0 = ard_map0*0

frame_data, timestamp = radar_hdf5.get_frame(11)
frame = radar_hdf5.sort_data(frame_data)
ard_fft = range_doppler_fft(frame,range_pad,doppler_pad,window_func,window_func)
ard_map = range_doppler_sum(ard_fft-ard_map0)
cfar_map = cfar(ard_map,1,1,2,2,1e-2,-100)
# cfar_map = np.roll(cfar_map,7,1)
cfar_map = np.roll(cfar_map,6,1)


cfar_map, centroid = clean_cfar(cfar_map,4)
# print(centroid)


extent = [-max_velocity,max_velocity,0,max_range]

fig = plt.figure()
ard_img = fig.add_subplot(131)
cfar_img = fig.add_subplot(132)
displace_plot = fig.add_subplot(133)


ard_ax = ard_img.imshow(ard_map, interpolation='none', animated=True,extent=extent,aspect="auto")
ard_img.set_title("RANGE DOPPLER MAP")
ard_img.set_xlabel("Velocity [m/s]")
ard_img.set_ylabel("Range [m]")

cfar_ax = cfar_img.imshow(cfar_map, interpolation='none', animated=True,extent=extent,aspect="auto")
cfar_img.set_title("CFAR MAP")
cfar_img.set_xlabel("Velocity [m/s]")


from utils.processing.functions import *
working_dir = os.path.dirname(__file__)
path_to_data = os.path.join(working_dir,"data/{1}_Exp{0}/rs_distance.npy".format(exp_num,date))
depth = depthNPY(path_to_data)
distance_time = depth.timestamps - unix_start_time
depth = depth.range_bins
realsesense_depth, sos = filter_ellip(depth,"lowpass",4,16,np.mean(np.diff(distance_time)),ripple=0.5)

displace_plot.plot(distance_time,realsesense_depth, label = "Realsense", alpha=0.5)
displace_line, = displace_plot.plot([0],[0],  label = "CFAR")

start_time = 13
stop_time = 145


displace_plot.set_xlim(0,stop_time)
displace_plot.set_ylim(0,max_range)

displace_plot.set_title("DISPLACEMENT VS TIME")
displace_plot.set_ylabel("Displacement [m]")
displace_plot.set_xlabel("Time [s]")
displace_plot.legend()

# plt.show()

trajectory_time = []
trajectory_cfar = []
trajectory_cfar_bins = []
trajectory_cfar_velocity = []


def update_fig(i):
    # global frame_data, frame, ard_fft, ard_map, cfar_map, trajector
    try:
        q = i*3 + calib_frames
        frame_data, timestamp = radar_hdf5.get_frame(q)
        # print(i)
    except Exception as e:
        print()
        print(e)
        cfar_depth = np.array(list(zip(trajectory_time,trajectory_cfar)))
        cfar_bins = np.array(list(zip(trajectory_time,trajectory_cfar_bins)))
        cfar_velocity = np.array(list(zip(trajectory_time,trajectory_cfar_velocity)))


        print("Saving to: data/{1}_Exp{0}/____.npy".format(exp_num,date))
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_distance.npy".format(exp_num,date)), cfar_depth)
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_bins.npy".format(exp_num,date)), cfar_bins)
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_velocity.npy".format(exp_num,date)), cfar_velocity)

        print("Done.\n")
        exit()
    
    if (timestamp - unix_start_time) <= start_time:
        return ard_ax,cfar_ax,

    if (timestamp - unix_start_time) >= stop_time:
        print()
        cfar_depth = np.array(list(zip(trajectory_time,trajectory_cfar)))
        cfar_bins = np.array(list(zip(trajectory_time,trajectory_cfar_bins)))
        cfar_velocity = np.array(list(zip(trajectory_time,trajectory_cfar_velocity)))


        print("Saving to: data/{1}_Exp{0}/____.npy".format(exp_num,date))
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_distance.npy".format(exp_num,date)), cfar_depth)
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_bins.npy".format(exp_num,date)), cfar_bins)
        np.save(os.path.join(working_dir,"data/{1}_Exp{0}/CFAR_velocity.npy".format(exp_num,date)), cfar_velocity)

        print("Done.\n")
        exit()

    frame = radar_hdf5.sort_data(frame_data)
    ard_fft = range_doppler_fft(frame, range_pad,doppler_pad,window_func,window_func)
    rdmap = range_doppler_sum(ard_fft-ard_map0)
    cfar_map = cfar(rdmap,1,1,2,2,1e-3,-1400) # -1200

    # calibrate velocity
    cfar_map = np.roll(cfar_map,5,1)

    # calibrate range
    cfar_map = np.roll(cfar_map,21,0)

    # get detections
    cfar_map, centroids = clean_cfar(cfar_map,4)

    # set image plots
    ard_ax.set_array(rdmap)
    cfar_ax.set_array(cfar_map)


    # sort detections
    distance_change_threshold = 0.1
    t = timestamp-unix_start_time
    rs_index =  np.argwhere(distance_time<=t)[-1][0]

    dist = realsesense_depth[rs_index]
    velocity_res = radar_hdf5.velocity_max/(chirps+doppler_pad)

    try:
        # for loop through detections and check if change between current and prev distance is below threshold to ensure continuity
        centroids = np.fliplr(centroids)
        for centroid in centroids:
            distance = (radar_hdf5.range_max)*(samples+range_pad - centroid[1])/(samples+range_pad)
            velocity = velocity_res*(centroid[0]-(chirps+doppler_pad)//2)
            # print()
            # print(chirps+doppler_pad)
            # print(centroid[0])
            # print(centroid[0]-(chirps+doppler_pad)//2)
            # print(velocity)
            if len(trajectory_cfar)!=0:
                # if np.abs(velocity) <= velocity_res:
                #     continue # if target has zero velocity ignore it

                if np.abs(distance - trajectory_cfar[-1])<=distance_change_threshold:
                # i
                    trajectory_cfar_bins.append(centroid[1])
                    trajectory_time.append(timestamp)
                    trajectory_cfar.append(distance)
                    trajectory_cfar_velocity.append(velocity)
                    break
                elif np.abs(distance -dist)<=0.09:
                    trajectory_cfar_bins.append(centroid[1])
                    trajectory_time.append(timestamp)
                    trajectory_cfar.append(distance)
                    trajectory_cfar_velocity.append(velocity)
                    break
            else:
                trajectory_cfar_bins.append(centroid[1])
                trajectory_time.append(timestamp)
                trajectory_cfar.append(distance)
                trajectory_cfar_velocity.append(velocity)
                break

    except:
        distance = 0
        velocity = 0

    displace_line.set_data(np.array(trajectory_time)-unix_start_time,np.array(trajectory_cfar))

    increments = 50 
    progress = int(q*increments/frames)
    bar = "".join([u"\u2588"]*progress + [" "]*(increments-progress-1))
    # try:
    #     # print("Frame: %d  || " % (q) + "Target Range Bin : %d  || " % (range_bin) + "Target Distance : %f  || " % (distance)  +"Time : %.2fs  || " % (trajectory_time[-1]-unix_start_time) + " Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "| "  ,end="\r") 
    # except:
    print("Time : %.2fs  || " % (timestamp - unix_start_time) + " Progress: %d%%" % ((progress+1)*100/increments) + " |" + str(bar) + "|"  ,end="\r") 


    return ard_ax,cfar_ax,displace_line,

anim = animation.FuncAnimation(fig, update_fig, interval=frame_period, blit=True,frames=frames)



plt.show()




