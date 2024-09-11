import numpy as np
import h5py
import os

class rigHDF5():

    # Samples
    start_time     = None # First time stamp
    nSamples       = None # Total Number of Samples Stored

    def __init__(self, file_path = None):
        """Test Rig HDF5 file reader.
        
        This class provides functionality for extracting information hdf5 files created by STEPPER-ROS2 repo. It has functions for
        loading sample position value and timestamp. Start time of first sample as recorded is loaded in init. Has a function to display. 
        HDF5 file structure."""
        if not file_path == None:
            self.file = h5py.File(file_path, 'r')
        else:
            print("Error, File path not given")
            exit()

        sample_number_list = []
        for sample in list(self.file["Data"].keys()):
            num = int(str(sample).split("_")[-1])
            sample_number_list.append(num)
        sample_number_list = np.sort(np.array(sample_number_list))
        self.nSamples = len(sample_number_list)

        _ , self.start_time = self.get_sample(0)
        
        
    def display_HDF5_structure(self):
        """Displays the HDF5 file structure if you want detailed break down of contents."""

        print("RIG HDF5 FILE STRUCTURE:")
        sample_number_list = []
        for sample in list(self.file["Data"].keys()):
            num = int(str(sample).split("_")[-1])
            sample_number_list.append(num)
        sample_number_list = np.sort(np.array(sample_number_list))

        print("Root:")
        print(" " + str(list(self.file.keys())))
        print()

        print("Data:")
        print(" Sample_%d ... Sample_%d" % (sample_number_list[0],sample_number_list[-1]))
        print()

        print("Sample_X:")
        print(" " + str(list(self.file["Data"]["Sample_%d" % sample_number_list[0]].keys())))
        print(" In \"timeStamps\": "  + str(list(self.file["Data"]["Sample_%d" % sample_number_list[0]]["timeStamps"].keys())))
        print()
        
        print("Trajectory:")
        print(list(self.file["Trajectory"].keys()))
        print()

    def get_sample_numbers(self):
        """ 
            Returns the number of the first frame and then number of the last frame
        """
        frame_number_list = []
        for frame in list(self.file["Data"].keys()):
            num = int(str(frame).split("_")[-1])
            frame_number_list.append(num)
        frame_number_list = np.sort(np.array(frame_number_list))
        return frame_number_list[0],frame_number_list[-1]


    def get_sample(self, sample_number):
        """Gets sample and timestamp of sample corresponding to sample_number.
        
        input : sample_number               -> number of sample to get (integer)

        output : (sample_data, timestamp)   -> tuple of floats representing rig position and timestamp
        """
        sample_group = self.file["Data"]["Sample_"+str(sample_number)]
        sample_data = np.array(sample_group["position"])
        time_group = sample_group["timeStamps"]
        time_nanoseconds = time_group["nanosec"]
        time_seconds = time_group["seconds"]
        timestamp = float(np.array(time_seconds))+float(np.array(time_nanoseconds)*1e-9)

        return sample_data, timestamp
