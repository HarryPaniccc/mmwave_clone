import numpy as np

class depthNPY():

    timestamps = None
    range_bins = None

    def __init__(self, file=None) -> None:
        if file == None:
            print("Error no file given.")
            exit()
        else:
            self.unpack(file)

    def unpack(self,file):
        # extract trajectory from file. 
        array = np.load(file)
        time_list = []
        data_list = []
        for i in range(len(array)):
            try:
                timestamp, data = array[i]
                data_list.append(data)
                time_list.append(timestamp)
            except:
                pass

        self.range_bins = np.array(data_list)
        self.timestamps = np.array(time_list)

    def get_range_bin(self, timestamp):
        try:
            index = np.where(self.timestamps<=timestamp)[0][-1]
        except:
            index = 0
        # print(index)
        return self.range_bins[index]