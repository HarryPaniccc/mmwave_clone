import numpy as np
import h5py
import os

class radarHDF5():

    # important Parameters
    num_frames         = None # Total Number of Frames Stored
    num_chirps         = None # Number of chirps per frame
    num_samples        = None # Number of samples per chirp
    num_channels      = None # Number of active receivers (a.k.a channels)

    # Intermediate/Useful Parameters
    BYTESPERSAMPLE    = 2    # 16 bit adc - constant value
    frequency_slope   = None # frequency Slope [Hz/s]
    time_sweep        = None # Ramp Time [s]
    time_idle         = None # Time idle between  [s]
    frame_period      = None # Frame Period [s]
    sampling_rate     = None # Sampling rate in [sps]
    frequency_start   = None # Starting frequency of the radar [Hz]
    bandwidth         = None # Sampled bandwidth [hz]
    time_chirp        = None # Total chirp time (idle time + ramp time) [s]
    frequency_centre  = None # Center frequency of chirp (calculated from total bandwidth) [Hz]
    frame_size        = None # Number of bytes per frame [Bytes]

    # performance metrics
    range_max       = None # Maximum unambigious range [m]
    range_res       = None # Range resolution [m]
    doppler_res     = None # Doppler resolution [Hz]
    velocity_max    = None # Maximum unambigious velocity [m/s]
    velocity_res    = None # Velocity resolution [m/s]

    def __init__(self, file_path = None):
        """Radar HDF5 file reader.
        
        This class provides functionality for extracting information hdf5 files created by DCA1000-ROS2 repo. It has functions for
        loading radar configs, calculating radar performance parameters, extracting the frames from the hdf5 file and sorting the data into 
        a radar cube. Radar parameters and configs are loaded during initiation."""
        if not file_path == None:
            self.file = h5py.File(file_path, 'r')
        else:
            print("Error, File path not given")
            exit()
        self.__get_radar_params()

    def get_frame_numbers(self):
        """ 
            Returns the number of the first frame and then number of the last frame
        """
        frame_number_list = []
        for frame in list(self.file["Data"].keys()):
            num = int(str(frame).split("_")[-1])
            frame_number_list.append(num)
        frame_number_list = np.sort(np.array(frame_number_list))
        return frame_number_list[0],frame_number_list[-1]
        
    def display_HDF5_structure(self):
        """Displays the HDF5 file structure if you want detailed break down of contents."""

        print("RADAR HDF5 FILE STRUCTURE:")
        frame_number_list = []
        for frame in list(self.file["Data"].keys()):
            num = int(str(frame).split("_")[-1])
            frame_number_list.append(num)
        frame_number_list = np.sort(np.array(frame_number_list))

        print("Root:")
        print(" " + str(list(self.file.keys())))
        print()

        print("Data:")
        print(" Frame_%d ... Frame_%d" % (frame_number_list[0],frame_number_list[-1]))
        print()

        print("Frame_X:")
        print(" " + str(list(self.file["Data"]["Frame_%d" % frame_number_list[0]].keys())))
        print(" In \"timeStamps\": "  + str(list(self.file["Data"]["Frame_%d" % frame_number_list[0]]["timeStamps"].keys())))
        print()
        
        
        try:
            paramgrp = "Parameters"
            recBitMask =  bin(int(np.array(self.file[paramgrp]["channelCfg"]["rxChannelEn"])))[2:] # convert number to bit mask string with bin()
        except:
            paramgrp = "Params"
            
        print("Parameters:")
        for cmd in (self.file[paramgrp].keys()):
            print(" " + str(cmd) +  ": ")
            for param in (self.file[paramgrp][cmd]):
                print("   " + param + " : " + str(self.file[paramgrp][cmd][param][()]))
            print()
        

        print("Comments:")
        print(list(self.file["Comments"].keys()))
        print()


    def display_radar_performance(self):
        """Displays the Radar's performance calculated from the radar config."""
        print("Max unambigious range: " + str(self.range_max) + " [m]")
        print("Max unambigious velocity: " + str(self.velocity_max) + " [m/s]")
        print("Range resolution: " + str(self.range_res)+ " [m]")
        print("Velocity resolution: " + str(self.velocity_res)+ " [m/s]")
        print("Doppler resolution: " + str(self.doppler_res) + " [Hz]")
        print()

    def get_frame(self, frame_number):
        """Gets frame and timestamp of frame corresponding to frame_number.
        
        input : frame_number                -> The number of the frame to get (integer)

        output : (frame_data, timestamp)    -> Tuple. First element is raw unsorted (int16) radar data. Second element is timestamp (float).
        """
        try:
            frame_group = self.file["Data"]["Frame_"+str(frame_number)]
            frame_data = np.array(frame_group["frame_data"])
            time_group = frame_group["Timestamps"]
            time_nanosecond = time_group["nano_seconds"]
            time_second = time_group["seconds"]
            timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])
        except:
            frame_group = self.file["Data"]["Frame_"+str(frame_number)]
            frame_data = np.array(frame_group["frameData"])
            time_group = frame_group["timeStamps"]
            time_nanosecond = time_group["nanosec"]
            time_second = time_group["seconds"]
            timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])


        return frame_data, timestamp


    
    def __get_radar_params(self):
        # frames
        frame_number_list = []
        for frame in list(self.file["Data"].keys()):
            num = int(str(frame).split("_")[-1])
            frame_number_list.append(num)
        frame_number_list = np.sort(np.array(frame_number_list))
        self.num_frames   = len(frame_number_list) # Total Number of Frames Stored

        # Calculate channels,

        # get number of chirps and samples
        self.num_chirps  = self.file["Parameters"]["num_chirps"][()] 
        self.num_samples = self.file["Parameters"]["num_samples"][()] 
        self.num_channels = self.file["Parameters"]["num_channels"][()] 
        

        # get and calculate intermediate Parameters
        self.frequency_slope  = self.file["Parameters"]["frequency_slope"][()]*1e12 
        self.time_sweep       = self.file["Parameters"]["time_ramp"][()]*1e-6 
        self.time_idle        = self.file["Parameters"]["time_idle"][()]*1e-6
        self.sampling_rate    = self.file["Parameters"]["sampling_rate"][()]*1e3  
        self.frequency_start  = self.file["Parameters"]["frequency_start"][()]*1e9 
        self.frame_period     = self.file["Parameters"]["frame_period"][()]*1e-3
        
        # Calculated Parameters
        self.frame_size       = self.num_chirps*self.num_samples*2*2*self.num_channels
        self.bandwidth        = (self.num_samples/self.sampling_rate)*self.frequency_slope
        self.time_chirp       = (self.time_idle + self.time_sweep)*12 
        self.frequency_centre = (self.frequency_start + (self.frequency_slope*self.time_sweep)/2) 
        wavelength = 3e8/self.frequency_centre

        # performance metrics
        self.range_max = 3e8*self.sampling_rate/(2*self.frequency_slope)
        self.range_res = 3e8/(2*self.bandwidth)
        self.doppler_res = 1/(self.num_chirps*self.time_chirp)
        self.velocity_max = ((wavelength/(4*self.time_chirp)))
        self.velocity_res = self.doppler_res*(wavelength/2)
