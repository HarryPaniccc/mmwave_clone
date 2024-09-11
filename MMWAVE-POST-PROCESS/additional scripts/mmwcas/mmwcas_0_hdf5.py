from utils.processing.radar_ffts import range_doppler_fft, range_doppler_sum
from utils.processing.functions import sort_data
import numpy as np
import os
import glob
import sys
import h5py

from mmwcas_load_data import mmwcas_load_data_from_bin
import json

from utils.hdf5.radar import radarHDF5

def main():

    
    print()
    print("TI MMWCAS Dataloader\n")

    experiment_name = "hand_wave_data"
    filename= "mmwcas_hand_wave_data.hdf5"
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_dir = os.path.join(script_dir,"data",experiment_name)
    
    # load config file 
    config_file = open(os.path.join(data_dir,'config_radar.json'))
    config = json.load(config_file)
    
    nSamples = config["num_samples"]
    nChirps= config["num_chirps"]
    nChannels = config["num_channels"]
    

    frame_array = mmwcas_load_data_from_bin(data_dir,nSamples,nChirps,nChannels)
    n_frames = frame_array.shape[0]
    
    print("Total Data size:")
    print(frame_array.shape)
    print()

    print("Creating HDF5 File:")
    
    # notes source 
    try:
        with open(os.path.join(data_dir,"description.txt"), 'r') as file:
            data = file.read()     
    except:
        print("No description.txt file found in data directory.")
        data = "No description available."

    # create 
    try:
        out_file = h5py.File(os.path.join(data_dir,filename),'x')
    except:
        print("A file of that name already exists.")
        print()
        
        print("File Overview:")
        radar_hdf5 = radarHDF5(os.path.join(data_dir,"mmwcas_hand_wave_data.hdf5"))
        print(n_frames, " Frames written to file")
        print("Experiment description: ",radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())
        print()

        radar_hdf5.display_radar_performance()
        exit()
        

    # create starting hierarchy
    dataGrp = out_file.create_group("Data")
    paramGrp = out_file.create_group("Parameters")

    
    for key in config:
        paramGrp.create_dataset(key,data=config[key])

    
    # Comments
    commentGrp = out_file.create_group("Comments")
    radar=["AWR1843BOOST".encode("ascii")]
    commentGrp.create_dataset("radar_type", shape=(len(radar),1), data=radar)    
    setup=[data.encode("ascii")]  
    commentGrp.create_dataset("experiment_setup", shape=(len(setup),1), data=setup) 

    start_time = 0
    time = start_time
    for i in range(n_frames):
         
        second, nanosecond = divmod(time,1)
        nanosec = int(nanosecond*1e9)
        sec = int(second)
        
        # store data
        frameGrp = dataGrp.create_group("Frame_%s" % (i))
        timeGrp = frameGrp.create_group("Timestamps")
        timeGrp.create_dataset("nano_seconds" , data = nanosec, dtype = np.uint32)
        timeGrp.create_dataset("seconds" , data = sec, dtype = np.uint32)
        frameGrp.create_dataset("frame_data",data=frame_array[i])
        
        time+= config["frame_period"]*1e-3

        # print progress bar
        progress_points = 50
        progress = int((i)*progress_points/(n_frames))
        bar = "".join([u"\u2588"]*progress + [" "]*(progress_points-progress-1))
        print("Progress: %d%%" % ((progress+1)*100/progress_points) + " |" + str(bar) + "| "  ,end="\r") 
    
    print()
    print("\n")
    
    
    
    print("File Overview:")
    radar_hdf5 = radarHDF5(os.path.join(data_dir,"mmwcas_hand_wave_data.hdf5"))
    print(n_frames, " Frames written to file")
    print("Experiment description: ",radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())
    print()
    radar_hdf5.display_radar_performance()




if __name__ == "__main__":
    main()