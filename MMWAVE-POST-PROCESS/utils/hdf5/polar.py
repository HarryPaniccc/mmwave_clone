import numpy as np
import h5py
import os

class polarHDF5():    


    def __init__(self, file_path = None, accelerometer_data=False):
        """Radar HDF5 file reader.
        
        This class provides functionality for extracting information hdf5 files created by DCA1000-ROS2 repo. It has functions for
        loading radar configs, calculating radar performance parameters, extracting the samples from the hdf5 file and sorting the data into 
        a radar cube. Radar parameters and configs are loaded during initiation."""
        if not file_path == None:
            self.file = h5py.File(file_path, 'r')
        else:
            print("Error, File path not given")
            exit()

        print("Getting heart rate values")
        # load heart rate values
        _, _, hr_sample_numbers = self.get_hr_sample_numbers()
        self.heart_timestamps = []
        self.heart_rate = []

        i = 0
        for num in hr_sample_numbers:
            heart_rate, timestamp = self.get_heart_rate(num)
            self.heart_timestamps.append(timestamp)
            self.heart_rate.append(heart_rate)
            
            # print progress bar
            num_increments = 50
            progress = int((i-1)*num_increments/(len(hr_sample_numbers)))
            bar = "".join([u"\u2588"]*progress + [" "]*(num_increments-progress-1))
            print("Progress: %d%%" % ((progress+1)*100/num_increments) + " |" + str(bar) + "|"  ,end="\r")
            i+=1

        print()
        print("Done.\n")
        # print(timestamp)


        self.heart_timestamps = np.array(self.heart_timestamps,dtype=np.float64)
        self.heart_rate = np.array(self.heart_rate,dtype=np.float64)

        # load acceleration values
        if accelerometer_data:
            print("Getting accelerometer values")
            _, _, acc_sample_numbers = self.get_acc_sample_numbers()
            self.accel_timestamps = []
            self.accel_x = []
            self.accel_y = []
            self.accel_z = []

            i = 0
            for num in acc_sample_numbers:
                
                accel_vector, timestamp = self.get_acceleration(num)

                # casuality check
                if not i<=2:
                    if timestamp < self.accel_timestamps[i-1]:
                        timestamp = self.accel_timestamps[i-1] + np.mean(np.diff(self.accel_timestamps))

                self.accel_timestamps.append(timestamp)
                self.accel_x.append(accel_vector[0])
                self.accel_y.append(accel_vector[1])
                self.accel_z.append(accel_vector[2])   

                # print progress bar
                num_increments = 50
                progress = int((i-1)*num_increments/(len(acc_sample_numbers)))
                bar = "".join([u"\u2588"]*progress + [" "]*(num_increments-progress-1))
                print("Progress: %d%%" % ((progress+1)*100/num_increments) + " |" + str(bar) + "|"  ,end="\r")
                i+=1

            print()
            print("Done.\n")

        

            self.accel_timestamps = np.array(self.accel_timestamps,dtype=np.float64)
            self.accel_x = np.array(self.accel_x,dtype=np.float64)
            self.accel_y = np.array(self.accel_y,dtype=np.float64)
            self.accel_z = np.array(self.accel_z,dtype=np.float64)

        

        

    def get_hr_sample_numbers(self):
        """ 
            Returns the number of the first sample, number of the last sample, and list of all sample numbers
        """
        sample_number_list = []
        for sample in list(self.file["Heart Rate Data"].keys()):
            num = int(str(sample).split("_")[-1])
            sample_number_list.append(num)
        sample_number_list = np.sort(np.array(sample_number_list))
        return sample_number_list[0],sample_number_list[-1], sample_number_list
    
    def get_acc_sample_numbers(self):
        """ 
            Returns the number of the first sample and then number of the last sample
        """
        sample_number_list = []
        for sample in list(self.file["Accelerometer Data"].keys()):
            num = int(str(sample).split("_")[-1])
            sample_number_list.append(num)
        sample_number_list = np.sort(np.array(sample_number_list))
        return sample_number_list[0],sample_number_list[-1], sample_number_list
        
    def display_HDF5_structure(self):
        """Displays the HDF5 file structure if you want detailed break down of contents."""

        print("POLAR HDF5 FILE STRUCTURE:")
        sample_number_list = []
        for sample in list(self.file["Heart Rate Data"].keys()):
            num = int(str(sample).split("_")[-1])
            sample_number_list.append(num)
        sample_number_list = np.sort(np.array(sample_number_list))

        print("Root:")
        print(" " + str(list(self.file.keys())))
        print()

        print("Heart Rate Data:")
        print(" Sample_%d ... Sample_%d" % (sample_number_list[0],sample_number_list[-1]))
        print()

        print("Sample_X:")
        print(" " + str(list(self.file["Heart Rate Data"]["Sample_%d" % sample_number_list[0]].keys())))
        print(" In \"timeStamps\": "  + str(list(self.file["Heart Rate Data"]["sample_%d" % sample_number_list[0]]["timeStamps"].keys())))
        print()

        print("Accelerometer Data:")
        print(" Sample_%d ... Sample_%d" % (sample_number_list[0],sample_number_list[-1]))
        print()

        print("Sample_X:")
        print(" " + str(list(self.file["Accelerometer Data"]["Sample_%d" % sample_number_list[0]].keys())))
        print(" In \"timeStamps\": "  + str(list(self.file["Accelerometer Data"]["sample_%d" % sample_number_list[0]]["timeStamps"].keys())))
        print()

        print("Params:")
        print(list(self.file["Params"].keys()))
        print()

        
        print("Comments:")
        print(list(self.file["Comments"].keys()))
        print()


    def get_heart_rate(self, sample_number):
        """Gets sample and timestamp of sample corresponding to sample_number.
        
        input : sample_number                -> The number of the sample to get (integer)

        output : (sample_data, timestamp)    -> Tuple. First element is numpy array x y z accelerations. Second element is timestamp (float).
        """
        sample_group = self.file["Heart Rate Data"]["Sample_"+str(sample_number)]
        heart_rate = np.array(sample_group["heart_rate"],dtype=np.float64)
        time_group = sample_group["timeStamps"]
        time_nanosecond = time_group["nanosec"]
        time_second = time_group["seconds"]
        timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])

        return heart_rate, timestamp

    def get_acceleration(self, sample_number):
        """Gets sample and timestamp of sample corresponding to sample_number.
        
        input : sample_number                -> The number of the sample to get (integer)

        output : (sample_data, timestamp)    -> Tuple. First element is heart rate. Second element is timestamp (float).
        """
        sample_group = self.file["Accelerometer Data"]["Sample_"+str(sample_number)]
        x_accel = np.array(sample_group["x_accel"],dtype=np.int16)
        y_accel = np.array(sample_group["y_accel"],dtype=np.int16)
        z_accel = np.array(sample_group["z_accel"],dtype=np.int16)

        time_group = sample_group["timeStamps"]
        time_nanosecond = time_group["nanosec"]
        time_second = time_group["seconds"]
        # print(np.array(time_second))
        # print(np.array(time_nanosecond))
        # exit()
        timestamp = float(str(np.array(time_second))+str(np.array(time_nanosecond)*1e-9)[1:])

        return np.array([x_accel, y_accel, z_accel]), timestamp
    
   