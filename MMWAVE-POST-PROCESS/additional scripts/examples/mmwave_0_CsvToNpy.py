import os
import json
import numpy as np
import datetime
import time
import glob

from utils.hdf5.radar import radarHDF5
working_dir = os.path.dirname(__file__)
    
f = open(os.path.join(working_dir,'pipeline_config.json'))
pipeline_config = json.load(f)
expNum = pipeline_config["EXPERIMENT_NUMBER"]
date = pipeline_config["DATE"]
dog = pipeline_config["DOG"]

## Open radar data file
working_dir = os.path.dirname(__file__)
path_to_data = "data/{0}_Polar/*.CSV".format(date)
file_paths = glob.glob(os.path.join(working_dir,path_to_data))

polar_data = []

for file_path in file_paths:

    with open(file_path) as csv_file:
        lines = len(csv_file.readlines())
        csv_file.seek(0)
        
        header_1 = csv_file.readline()
        start_time_list = str(csv_file.readline()).split(",")[2:4]
        start_date = start_time_list[0]
        start_time = start_time_list[1]

        date_time = datetime.datetime(int(start_date.split("-")[-1]), 
                                    int(start_date.split("-")[-2]), 
                                    int(start_date.split("-")[-3]), 
                                    int(start_time.split(":")[0]), 
                                    int(start_time.split(":")[1]),
                                    int(start_time.split(":")[2]))
        unix_start_time = time.mktime(date_time.timetuple())

        line_count = 4
        
        # dog_name = str(csv_file.readline()).split(",")[0]
        # header_2 = str(csv_file.readline()).split(",")[0]
        while  "Sample rate" not in str(csv_file.readline()) :
            line_count+=1
        
        for line in range(lines-line_count):
            fields = str(csv_file.readline()).split(",")[1:3]
            sample_time = fields[0]
            sample_time_seconds = np.dot(np.array((fields[0].split(":"))).astype(np.float32),[3600,60,1]) 
            time_val = str(unix_start_time + sample_time_seconds)
            if fields[1] != "":
                heart_rate = fields[1]
            
            polar_data.append([time_val,heart_rate])

polar_data = np.array(polar_data)
print("Saving data.\n")
np.save(os.path.join(working_dir,"data/{2}/{1}_Exp{0}/Polar_Data_Exp{0}".format(expNum,date,dog)), polar_data)

working_dir = os.path.dirname(__file__)
path_to_data = "data/{2}/{1}_Exp{0}/Radar_Data_Exp{0}.hdf5".format(expNum,date,dog)
radar_hdf5 = radarHDF5(os.path.join(working_dir,path_to_data))
samples = radar_hdf5.nSamples
range_max = radar_hdf5.range_max

print(radar_hdf5.file["Comments"]["experiment_setup"][0][0].decode())