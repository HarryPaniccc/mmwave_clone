import numpy as np
import warnings

class polarNPY():

    data = None

    timestamps = None
    start_time = None
    heart_rate = None

    def __init__(self, file_path) -> None:
        self.data = np.load(file_path)
        self.__unpack()

    def __unpack(self):
        time_list = []
        data_list = []
        for i in range(len(self.data)):
            try:
                timestamp,sample_data = self.data[i]
                time_pt = float(timestamp)
                data_pt = float(sample_data)
                data_list.append(data_pt)
                time_list.append(time_pt)
            except:
                pass
        
        self.start_time = time_list[0]
        self.heart_rate = np.array(data_list)
        self.heart_timestamps = np.array(time_list)

    def get_average_in_interval(self,start,end):
        if start > self.timestamps[-1]:
            print("No heart rate recorded over this time.  Heart rate logging ends before start time chosen.")
            return np.nan
        elif end < self.timestamps[0]:
            print("No heart rate recorded over this time. Heart rate logging only starts after end time chosen.")
            return np.nan

        try:
            start_index = np.where(self.timestamps<=start)[0][-1]
        except:
            start_index = 0
            print("Polar Monitor only started after the selected start time by %.2fs. Taking earliest possible index." % (self.start_time-start))

        try:
            end_index = np.where(self.timestamps>=end)[0][0]
        except:
            end_index = -1
            print("Polar Monitor stopped logging before the selected end time by %.2fs. Taking latest possible index." % (end-self.timestamps[-1]))

        mean_heart_rate = np.mean(self.heart_rate[start_index:end_index])
        
        
        return mean_heart_rate