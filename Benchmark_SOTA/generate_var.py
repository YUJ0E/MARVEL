import numpy as np
import func
import time
import os
from fma import FMA_MAP, FMA, LET_path
import csv


if __name__ == '__main__':
    curr_dir = os.getcwd()
    map_dir = curr_dir + '/nn/'
    map_id = 2 #map_id can be integers from 0~7
    '''
    0    Sioux Falls
    1    Anaheim
    2    Winnipeg
    3    Chicago-Sketch
    4    Chengdu-Weekend Off-peak Hour
    5    Chengdu-Weekend Peak Hour
    6    Chengdu-Weekday Off-peak Hour
    7    Chengdu-Weekday Peak Hour
    '''
    mymap = FMA_MAP()
    mymap.generate_real_map(map_id, map_dir, nu=0.4)
    var = mymap.var.tolist()

    with open('data.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in mymap.var:
            writer.writerow(row)
