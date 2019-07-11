"""Exercise for feature detection Udemy course in signal processing"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import inter1d
from scipy.signal import savgol_filter

# create log normal distribution
strech = 0.6
signal_base = np.exp(strech*np.random.randn(10000))
signal, time = np.histogram(signal_base, bins=100)
time = time[:-1]
resampled_time = np.linspace(time[0], time[-1], 100*len(time))
interpolator = interp1d(x=time, y=signal, kind='linear')
resampled_signal = interpolator(resampled_time)
resampled_smooth = savgol_filter(resampled_signal, 510, 3)
# find max
max_val = np.max(resampled_smooth)
over_hm_idx = np.where(resampled_smooth>max_val/2)
over_hm_idx = over_hm_idx[0]
fwhm = resampled_time[over_hm_idx[-1]] -  resampled_time[over_hm_idx[0]]

print('FWHM is {}'.format(fwhm))
# plotting
plt.plot(time, signal, 'r.', markersize=5 )
plt.plot(resampled_time, resampled_signal, 'k.', markersize=2)
plt.plot(resampled_time, resampled_smooth, 'k--')
plt.plot(resampled_time[over_hm_idx[0]],
             resampled_smooth[over_hm_idx[0]], 'ro', markersize=7)
plt.plot( resampled_time[over_hm_idx[-1]],
             resampled_smooth[over_hm_idx[-1]], 'go', markersize=7)
plt.show() 

