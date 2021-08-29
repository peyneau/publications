# -*- coding: utf-8 -*-

"""
Written by Pierre-Emmanuel Peyneau
This version: 2021-08-26
License: CC-BY 4.0 <URL: http://creativecommons.org/licenses/by/4.0/ >
"""

"""
This program is intended to read and process sp-ICP-MS data.
It has been tested on raw data obtained with a Agilent 8900 ICP-MS-QQQ mass spectrometer.

Inputs required once the program has been launched:
    Name of the csv file containing the raw data (Agilent's MassHunters format)
    Dwell time (in seconds)
    Level of noise affecting the time scan
"""

import math
import numpy as np



class Event:
    def __init__(self, t1, t2, countcum):
        self.tstart = t1
        self.tend = t2
        self.countcum = countcum
        self.above_threshold = True
    def __str__(self):
        return (f"Starts at: {self.tstart}\nEnds at: {self.tend}\n"
               +f"Cumulated counts: {self.countcum}\n"
               +f"Is above threshold? {str(self.above_threshold)}\n\n")
    def tave(self):
        return 0.5*(self.tstart + self.tend)
    def twidth(self):
        return self.tend - self.tstart
    # Multiplication by 0.001 to compensate for rounding errors
    def n_dwelltime(self):
        return int((self.tend - self.tstart+0.001*_dwelltime)/_dwelltime) +1



def str_to_decimal_part(str_of_digits):
    decimal_number = 0.
    decimal_place = 1
    for digit in str_of_digits:
        decimal_number += int(digit)/(10**decimal_place)
        decimal_place += 1
    return decimal_number


def read_datafile(file_name):
    f = open(file_name, 'r', encoding = 'utf8')
    line_number = 0
    data = []
    for line in f:
        '''
        In Excel files produced by Agilent's MassHunter software, genuine sp-ICP-MS data start from line #5.
        '''
        line_number += 1
        if line_number == 2:
            l = line.strip('\n')
            intensity_mode = l.split(',')[1]
            print("Intensity mode:",intensity_mode,'\n')
        if line[:1].isdigit():
            l = line.strip('\n')
            str_line_data = l.split(',')
            # Case when Excel decimal separator is '.'
            if(len(str_line_data) == 2):
                time, count = float(str_line_data[0]), float(str_line_data[1])
            # Case when Excel decimal separator is ','
            if(len(str_line_data) == 4):
                time = float(str_line_data[0]) + str_to_decimal_part(str_line_data[1])
                count = float(str_line_data[2]) + str_to_decimal_part(str_line_data[3])
            if(intensity_mode == 'Counts'):
                data.append([time, count])
            elif(intensity_mode == 'CPS'):
                 data.append([time, count*_dwelltime])
            else:
                print("Houston, we have a problem!")
    f.close()
    return data


def peak_detection(count, background):
    start, end = [], []
    # count_extended is used to detect spikes at the very beginning or at the very end of a time scan
    count_extended = count[:]
    count_extended[0:0] = [-1]
    count_extended.append(-1)
    for i in range(1, len(count)+1):
        if (count_extended[i] > background and
            count_extended[i-1] <= background):
            start.append(i-1)
        if (count_extended[i] > background and
            count_extended[i+1] <= background):
            end.append(i-1)
    return start, end


def peak_cumulated(count, start, end):
    countcum = []
    for i in range(len(start)):
        cum = 0.
        for j in range(start[i], end[i]+1):
            cum += count[j]
        countcum.append(cum)
    return countcum


def noise_remove(count, threshold):
    for i in range(1,len(count)-1):
        if((count[i] <= threshold) & (count[i-1] == 0) & (count[i+1] == 0)):
            count[i] = 0
    return count

def binarization(count):
    count_bool = [False] * (len(count))
    for i in range(len(count)):
        if(count[i] > 0):
            count_bool[i] = True
    return count_bool



print("Name of the datafile?")
datafile_name = input()
print('\n')

print("Level of noise?")
_noiselevel = float(input())
print('\n')

print("Dwell time (s)?")
_dwelltime = float(input())
print('\n')

# Data lists
data_list = read_datafile(datafile_name)
time_list, count_list = [], []
for i in range(len(data_list)):
    time_list.append(data_list[i][0])
    count_list.append(data_list[i][1])

# Noise elimination below the level <_noiselevel>
count_list_withoutnoise = noise_remove(count_list, _noiselevel)

# Binarization of the denoized signal
count_list_withoutnoise_bool = binarization(count_list_withoutnoise)
count_list_withoutnoise_bool[-1] = False
count_list_withoutnoise_bool.append(False)

# Number of spikes
Nchanges = 0
for i in range(len(count_list)):
    if(count_list_withoutnoise_bool[i+1] ^ count_list_withoutnoise_bool[i]):
        Nchanges += 1
Nspikes = (Nchanges+1)//2

# Spike detection
start_list, end_list = [], []
start_list, end_list = peak_detection(count_list_withoutnoise, 0)

# Cumulated count in each spike
countcum_list = []
countcum_list = peak_cumulated(count_list_withoutnoise, start_list, end_list)

# Events
event_list = []
for i in range(len(start_list)):
    e = Event(time_list[start_list[i]], time_list[end_list[i]], countcum_list[i])
    event_list.append(e)

# Event thresholding
for i in range(len(event_list)):
    if(event_list[i].countcum <= _noiselevel * event_list[i].n_dwelltime()):
        event_list[i].above_threshold = False
    
# Average and (biased) standard deviation of the duration of the detected spikes
peakndwell_list = [event_list[i].n_dwelltime() for i in range(len(event_list)) if event_list[i].above_threshold]
eventthresholded_list = [i for i in range(len(event_list)) if event_list[i].above_threshold]
mean_peak_duration = sum(peakndwell_list) * _dwelltime / len(peakndwell_list)
std_peak_duration = np.std(peakndwell_list) * _dwelltime

# Suspicious spikes (leave a trace over <= 3 dwell times in the time scan)
c = 0
for i in range(len(peakndwell_list)):
    if(peakndwell_list[i] <= 3):
        c += 1
        j = eventthresholded_list[i]
        print(c,"  ",peakndwell_list[i],"  ",event_list[j].countcum,"   ",event_list[j].tstart)
        
        
print("Number of spikes : ", len(peakndwell_list),'\n')
