""" exercise with emg data"""
import numpy as np
import matplotlib.pyplot as plt

# load the data
emg = np.load('demo_data/feature/emg.npy')
rts = np.load('demo_data/feature/rts.npy')
timevec = np.load('demo_data/feature/timevec.npy')

# create base signal vector
signal = np.zeros(150, dtype=float)
emg_idx = 0

for button_time in np.nditer(rts):
    # find in timvec
    vals, idx = np.where(timevec > button_time)
    idx = idx[0]
    signal += emg[emg_idx, idx-150:idx]
    emg_idx +=1
signal = signal/200



