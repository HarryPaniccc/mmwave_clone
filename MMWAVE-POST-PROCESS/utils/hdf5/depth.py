import numpy as np
import h5py
import os

class depthHDF5():

    # important Parameters
    height          = None # Total Number of Frames Stored
    width           = None # Number of chirps per frame
    is_big_endian   = None # Number of samples per chirp
    nVChannels      = None # Number of active receivers (a.k.a channels)


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

        self.param = "Parameters"
        try:
            self.height = self.file[self.param]["Image_Size"]["height"]
        except:
            self.param = "Params"
            self.height = self.file[self.param]["image_size"]["height"]

        try:
            self.width = self.file[self.param]["Image_Size"]["width"]
            self.is_big_endian = self.file[self.param]["Image_Encoding_Info"]["is_big_endian"]
            self.row_length = self.file[self.param]["Image_Encoding_Info"]["row_length"]
        except:
            self.width = self.file[self.param]["image_size"]["width"]
            self.is_big_endian = self.file[self.param]["image_encoding_info"]["is_big_endian"]
            self.row_length = self.file[self.param]["image_encoding_info"]["row_length"]


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

        print("DEPTH HDF5 FILE STRUCTURE:")
        frame_number_list = []
        for frame in list(self.file["Image"].keys()):
            num = int(str(frame).split("_")[-1])
            frame_number_list.append(num)
        frame_number_list = np.sort(np.array(frame_number_list))

        print("Root:")
        print(" " + str(list(self.file.keys())))
        print()

        print("Data:")
        print(" Image_%d ... Image_%d" % (frame_number_list[0],frame_number_list[-1]))
        print()

        print("Frame_X:")
        print(" " + str(list(self.file["Data"]["Image_%d" % frame_number_list[0]].keys())))
        print(" In \"timeStamps\": "  + str(list(self.file["Data"]["Image_%d" % frame_number_list[0]]["timeStamps"].keys())))
        print()

        print("Parameters:")
        print(list(self.file[self.param].keys()))
        print()

        
        print("Comments:")
        print(list(self.file["Comments"].keys()))
        print()


    def get_frame(self, frame_number):
        """Gets frame and timestamp of frame corresponding to frame_number.
        
        input : frame_number                -> The number of the frame to get (integer)

        output : (frame_data, timestamp)    -> Tuple. First element is raw unsorted (int16) radar data. Second element is timestamp (float).
        """
        try:
            frame_group = self.file["Data"]["Image_"+str(frame_number)]
            frame_data = np.array(frame_group["image_data"],dtype=np.uint16)
            time_group = frame_group["Timestamps"]
            time_nanosecond = time_group["nano_seconds"]
            time_second = time_group["seconds"]
            timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])
        except:
            frame_group = self.file["Data"]["Frame_"+str(frame_number)]
            frame_data = np.array(frame_group["frameData"],dtype=np.uint16)
            time_group = frame_group["timeStamps"]
            time_nanosecond = time_group["nanosec"]
            time_second = time_group["seconds"]
            timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])
        return frame_data, timestamp

    
   