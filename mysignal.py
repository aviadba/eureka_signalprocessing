"""
signal - Udemy Signal Processing course root file
"""

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.signal import *
#welch, spectrogram, firwin, butter, filtfilt, morlet, ricker


class Signal:
    def __init__(self):
        self.time = None
        self.signal = None
        self._frequency = None
        self.domain = 'time'

    def create_signal(self, stype='poles', sample_frequency=1000, time=3, poles=15,
                      interp='linear', amplitude=1):
        """Create a simulated signal

        Parameters:
            stype : {'poles', 'spikes', 'white noise', 'lintrend', 'polytrend',
            'events', 'sum_frequencies', 'guassian', 'brownian'}
                type of signal to create
            sample_frequency :  float
                signal rate in Hz
            time : float
                signal length in seconds
            poles : int
                number of random poles in 'poles' signal, or number of spikes in spikes mode or deg of polynomial
            interp {'linear', 'cubic'}
                modes of interpolation
            amplitude : float
                 amplitude of max pole signal
        """
        # create time vector
        self.time = np.arange(0, time, 1/sample_frequency)
        self.sample_frequency = sample_frequency
        # save amplitude as an object property
        self.amplitude = amplitude
        if stype == 'poles':
            # create a signal by interpolating values between randomly selected
            # poles
            if interp == 'linear':
                self.signal = np.interp(np.linspace(0, poles-1, len(self.time)),  # interpolate at these values
                                              # define poles
                                              np.arange(0, poles),
                                              np.random.rand(poles)*amplitude)
            elif interp == 'cubic':
                signal_fun = interp1d(np.arange(0, poles),
                                      np.random.rand(poles)*amplitude,
                                      kind='cubic')
                signal_x = np.linspace(0, poles-1, len(self.time))
                self.signal = signal_fun(signal_x)
        elif stype == 'spikes':
            # create randomly distributes spikes
            inter_spike_distance = np.exp(np.random.randn(poles))
            inter_spike_distance = \
                np.round(sample_frequency*time/inter_spike_distance.sum()
                         * inter_spike_distance)
            distribution_tail = inter_spike_distance.sum() - \
                (sample_frequency*time - 1)
            if distribution_tail > 0:
                inter_spike_distance[0] = inter_spike_distance[0] -\
                    distribution_tail
            signal = np.zeros(int(sample_frequency*time))
            idx = 0
            for distance in inter_spike_distance:
                idx += int(distance)
                signal[idx] = 1
            self.signal = signal * amplitude
        elif stype == 'white noise':
            self.signal = amplitude*np.random.randn(int(sample_frequency*time))
        elif stype == 'lintrend':
            trend = 10*amplitude*np.random.random()
            self.signal = np.linspace(-trend,
                                            trend, int(sample_frequency*time))
        elif stype == 'polytrend':
            base_signal = np.zeros(int(sample_frequency*time))
            for deg in range(poles):
                base_signal += np.random.randn()*self.time**deg
            self.signal = base_signal
        elif stype == 'events':
            # create and scale an event
            event_ratio = 0.3  # ratio of signal occupied by 'events'
            event_length = int(sample_frequency*time/poles*event_ratio)
            event = np.diff(np.exp(np.linspace(-2, 2, event_length+1)**2))
            event = event/max(event)*amplitude
            event_start_idx = \
                np.random.permutation(
                    range(int(sample_frequency*time)-event_length))[:poles]
            base_signal = np.zeros(int(sample_frequency*time))
            for idx in event_start_idx:
                base_signal[idx:idx+event_length] = event
            self.signal = base_signal
        elif stype == 'sum_frequencies':
            base_signal = np.zeros(int(sample_frequency*time))
            freq = np.linspace(0, time, time*sample_frequency)
            # select frequencies to include
            selected_signal_frequencies = \
                np.random.permutation(range(1, int(sample_frequency/2)))[:poles]
            for freq in selected_signal_frequencies:
                base_signal += np.random.randn()*np.sin(2*np.pi*freq*_freq)
            self.signal = base_signal
        elif stype == 'guassian':
            # create randomly distributes centers
            guass_center_distance = np.exp(np.random.randn(poles))
            if len(guass_center_distance) == 1:
                guass_center_distance =  \
                    np.round(sample_frequency*time/3 * abs(guass_center_distance))
            else:
                guass_center_distance =  \
                    np.round(sample_frequency*time/guass_center_distance.sum()
                             * guass_center_distance)
                distribution_tail = guass_center_distance.sum() - \
                    (sample_frequency*time - 1)
                if distribution_tail > 0:
                    guass_center_distance[0] = guass_center_distance[0] -\
                        distribution_tail
            guass_center_distance = np.cumsum(guass_center_distance)
            base_signal = np.zeros(int(sample_frequency*time))
            signal_idx = range(int(sample_frequency*time))
            for distance in guass_center_distance:
                instance_signal_idx = [
                    i_idx - distance for i_idx in signal_idx]
                instance_signal_idx = np.array(instance_signal_idx)
                instance_sigma = 0.1*sample_frequency*time*np.random.random()
                instance_guassian = \
                    1/(instance_sigma*(2*np.pi)**0.5) * \
                    np.exp(-0.5*(instance_signal_idx/instance_sigma)**2)
                base_signal += instance_guassian
            self.signal = base_signal
        elif stype == 'brownian':
            base_signal = np.random.randn(int(sample_frequency*time))
            base_signal = np.cumsum(base_signal)
            self.signal = base_signal


class Noise:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = 'time'

    def add_noise(self, noise_amplitude=1, ntype='rand'):
        """
        Add noise to signal

        Parameters:
            ntype : {'rand', 'normal' , 'pink'}
                Noise type to add. 'rand' for randomly distributed noise,
                'nrand' for normally distributed noise
                'pink' noise distributed with 1/alpha*f distribution (pink noise)
            noise_amplitude : float
                noise max level in 'standard deviation' units
            """
        # get standard deviation of signal
        noise_amplitude = noise_amplitude*np.std(self.source_signal.signal)
        # create noise
        if ntype == 'rand':
            noise = \
                noise_amplitude*np.random.rand(len(self.source_signal.signal))
        elif ntype == 'normal':
            noise = \
                noise_amplitude*np.random.randn(len(self.source_signal.signal))
        elif ntype == 'pink':
            # create frquency vector
            breakpoint()
            frequency = np.linspace(-len(self.source_signal.signal)//2,
                        len(self.source_signal.signal)//2,
                        len(self.source_signal.signal))
            pink = 1/frequency
            # set pink nan to zero
            phase = 2* np.pi * np.random.rand(len(self.source_signal.signal)//2)
            # pink = pink * np.exp(-1j 
        self.signal = self.source_signal.signal + noise



# Class for applying time domain filters
class Filter:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = 'time'

    def running_filter(self, ftype='mean', order=3, edge='copy'):
        """Sets the value in a filtered signal according to the values of
        neighbors
        Arguments
        ---------
        ftype <str> type of filter. Options are: 'mean' 'gaussian' , tkeo
        (Teager-Kaiser energy operator), 'median'
        order <int> the number of neighbors to consider when calculating the
        mean value
        edge <string> - defines how to deal with edges. possibilities are
        'copy', 'crop'
        """
        # find length of signal
        signal_length = len(self.source_signal.time)
        # allocate memory for for signals and time
        signal = np.zeros(signal_length)
        if ftype == 'mean':
            for idx in range(order, signal_length-order):
                signal[idx] = np.mean(
                    self.source_signal.signal[idx-order:idx+order+1])
        elif ftype == 'gaussian':
            gaussain_ker = np.arange(-(order), (order+1), 1)
            gaussain_ker = np.exp(-(4*np.log(2)*gaussain_ker**2) / (order)**2)
            gaussain_ker = gaussain_ker/gaussain_ker.sum()
            for idx in range(order, signal_length-order):
                signal[idx] = \
                    (self.source_signal.signal[idx -
                                               order:idx+order+1]*gaussain_ker).sum()
        elif ftype == 'tkeo':
            # set order to 1 for edge effect actions
            order = 1
            for idx in range(1, signal_length-1):
                signal[idx] = self.source_signal.signal[idx]**2 -\
                    self.source_signal.signal[idx-1] * \
                    self.source_signal.signal[idx+1]
            signal = signal/np.max(signal)*np.max(self.source_signal.signal)
        elif ftype == 'median':
            for idx in range(order, signal_length-order):
                signal[idx] = np.median(
                    self.source_signal.signal[idx-order:idx+order+1])
        if edge == 'copy':
            signal[:order] = self.source_signal.signal[:order]
            signal[signal_length-order:] = \
                self.source_signal.signal[signal_length-order:]
            self.signal = signal
            self.time = self.source_signal.time
        elif edge == 'crop':
            self.signal = signal[order:signal_length-order]
            self.time = self.source_signal.time[order:signal_length-order]

# Class to remove several types of trends from data
class Detrend:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = 'time'

    def detrend(self, method='linear'):
        """detrend signals
        Arguments
        ---------
        method <str> detrending method. Allowed values are' 'linear' (least
        square fit of the data, 'polynomial' polynomial fitting with Bayes
        information criterion to find best deg
        """
        if method == 'linear':
            self.signal = scipy.signal.detrend(
                self.source_signal.signal, type='linear')
            self.time = self.source_signal.time
        elif method == 'polynomial':
            # use Bayes information criterion
            deg_max = 10
            bic = []
            bic_score = []
            signal_l = len(self.source_signal.time)
            for deg in range(deg_max):
                ifit = np.polyfit(self.source_signal.time,
                                  self.source_signal.signal, deg=deg, full=True)
                score = signal_l*np.log(ifit[1])+(deg+1)*np.log(signal_l)
                bic.append(ifit)
                bic_score.append(score)
            # find indices of min val
            deg = np.argmin(bic_score)
            # get fitting polynomial coeffecients
            pol_coeff = bic[deg][0]
            detrend = np.zeros(len(self.source_signal.signal))
            for idx, coef in enumerate(pol_coeff):
                detrend += coef*self.source_signal.time**(deg-idx)
            self.signal = self.source_signal.signal - detrend
            self.time = self.source_signal.time

# Class to load signals from saved data sets of type npy
class Loadsignal:

    def __init__(self):
        self.time = None
        self .signal = None
        self.source_signal = None
        self.domain = 'time'

    def load_signal(self, path_to_signal, sigtype='orig', domain='time'):
        """
        Load saved signal from .npy files. Supports loading a reference file
        to practice cascade processing with a reference target. 
        Parameters
            path_to_signal : string
                relative path to data file
                sigtype : {'orig', 'ref'}
                    the type of signal being loaded. Can take the value of 'orig' (original signal) or 'ref' (reference signal after corrections)
        """
        raw_signal = np.squeeze(np.load(path_to_signal))
        if len(raw_signal.shape) == 2:  # array with time and counts
            # find smaller dimension
            main_data_axis = np.argmin(raw_signal.shape)
            if main_data_axis == 0:
                time = np.squeeze(raw_signal[0,:])
                signal = np.squeeze(raw_signal[1,:])
            else:
                time = np.squeeze(raw_signal[:,0])
                signal = np.squeeze(raw_signal[:,1])
        else:  # time vector not available, default to 1 sec per data4TFnpy
            signal = raw_signal
            time = np.arange(0, signal.shape[0])
        if domain == 'time':
            sample_frequency = 1/np.mean(np.diff(time))
        if sigtype == 'orig':
            self.signal = signal
            self.time = time
            self.domain = domain
            self.sample_frequency = sample_frequency
        elif sigtype == 'ref':
            self.source_signal = Signal()
            self.source_signal.signal = signal
            self.source_signal.time = time
            self.source_signal.domain = domain
            self.source_signal.sample_frequency = sample_frequency


class FFTsignal:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = source_signal.domain
        self.time_scaling = 1  # for dealing with frequency sub sampling

    def fft_signal(self):
        """Create the frequency domain representation of a time domain or the
        time domain representation f a frequency domain signal
        """
        if self.source_signal.domain == 'time':
            # create time vector
            self.signal = np.fft.fft(self.source_signal.signal, norm='ortho')
            frequencies = np.fft.fftfreq(self.time.size,
                                         d=np.gradient(self.time).mean())
            self.time = frequencies
            self.domain = 'frequency'
        elif self.source_signal.domain == 'frequency':
            self.signal = np.fft.ifft(self.source_signal.signal, norm='ortho')
            self.signal = self.signal.real
            time = np.linspace(0,
                               len(self.time)/(2*max(abs(self.time))) *
                               self.source_signal.time_scaling,
                               len(self.time))
            self.time = time
            self.domain = 'time'
        # save amplitude as an object property
        self.amplitude = None

    def welch_signal(self, window_size=1024):
        """calculate the Fourier Transform using Welch's method of
        averaging multiple spectra
        Arguments
        ---------
        window_size <int> the size of window over which a FFT is calculated
        """
        welch_freq, welch_signal = welch(self.source_signal.signal, window='hanning', nperseg=window_size,
                                         return_onesided=False)
        frequencies = np.fft.fftfreq(welch_signal.size,
                                     d=np.gradient(self.time).mean())
        self.signal = welch_signal
        self.time = frequencies
        self.time_scaling = self.source_signal.time.size/window_size
        self.domain = 'frequency'

    def spectrogram_signal(self, window_size=1024):
        """Copute the time dependant spectrogram of a signal
        Arguments
        ---------
        window_size <int> size of window to use
        """
        spectro_freq, spectro_time, spectro_vals = \
            spectrogram(x=self.source_signal.signal, nperseg=window_size)
        self.signal = spectro_vals
        self.time = spectro_time
        self.spectrogram_frequencies = spectro_freq
        self.domain = 'spectrogram'

    def time_frequency(self, window_size=1024):
        """
        time - frequency analysis (similar to spectrograph

        Parameters
            window_size : int
        """
        # define frequency resollution
        num_freq = 40
        min_freq = 2;
        max_freq = self.source_signal.sample_frequency//2
        freqs = np.linspace(min_freq, max_freq, num_freq)
        # define widths of wavelets
        fwhms = np.linspace(5, 15, num_freq)
#        tidx = mp.round(np.linspace(1, length(times)
        # define morlet wavelets
        wavet = np.arange(-10, 10, 1/self.source_signal.sample_frequency) 
        base_wavelets = np.zeros((len(wavet), num_freq), dtype=np.complex)
        for i in range(num_freq):
            base_wavelets[:, i] = np.exp(2 * 1j * np.pi * freqs[i] * wavet) * \
                np.exp(-(4 * np.log(2) * wavet ** 2)/fwhms[i]**2)
        
        # 
                                    


class Freqfilter:
    def __init__(self, source_signal):
        """Filters defined in the frequency domain. Operate in the time domain.
        Class allows comparing the resulting filter to the expected filter.
        Filters are then applied using filtfilt (zero phase shift) to the
        source_signal data"""
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = 'time'
        self.filter = None  # the filter representation in the time domain
        self.filterfreq = None  # the filter representation in the frequqnecy domain

    def fir_filter(self, window='boxcar', order=73, frange=[None, None]):
        """Apply the FIR (Finite impulse response) filter using the specified method
        Arguments
        ---------
        window <str> - tapering window to use. Allowed values are 'none',
        'hann', 'hamming', 'guassian'
        order <int> - size of filter 
        frange <lst(2)>: band pass rise and fall or rise/fall for high/low pass
        """
        # create window
        if window == 'gaussian':
            window = ('guassian', order/6)

        # create filter
        if frange[0]:
            if frange[1]:  # bandpass filter
                if frange[0] > frange[1]:
                    fir_filter = firwin(numtaps=order,
                                        cutoff=[frange[1], frange[0]],
                                        pass_zero=True,
                                        window=window,
                                        scale=True)
                elif frange[1] > frange[0]:
                    fir_filter = firwin(numtaps=order,
                                        cutoff=frange,
                                        pass_zero=False,
                                        window=window,
                                        scale=True)
                self.filterfreq = ['bandpass', frange]
            else:  # highpass filter
                fir_filter = firwin(
                    numtaps=order, cutoff=frange[0], pass_zero=False,
                    window=window, scale=True)
                self.filterfreq = ['highpass', frange[0]]
        else:
            if frange[1]:  # lowpass filter
                fir_filter = firwin(
                    numtaps=order, cutoff=frange[1], pass_zero=True,
                    window=window, scale=True)

            self.filterfreq = ['lowpass', frange[1]]
        # save filter
        self.filter = fir_filter
        # filtered signal
        self.signal = np.convolve(
            self.source_signal.signal, fir_filter, mode='same')
        self.time = self.source_signal.time

    def iir_filter(self, order=9, frange=[None, None]):
        """Implemetation of IIR filter
        Arguments
        ---------
        order <int> - the order of the filter
        frange <lst(2)>: band pass rise and fall or rise/fall for high/low pass
        ftype <str> - filter tyepe. Allowed options are 'lowpass'
        , 'highpass', 'bandpass'
        """
        # create filter
        if frange[0]:
            if frange[1]:  # bandpass filter
                if frange[0] > frange[1]:
                    iir_filter = butter(N=order, Wn=[frange[1], frange[0]], btype='bandstop')
                elif frange[1] > frange[0]:
                    iir_filter = butter(N=order, Wn=frange, btype='bandpass')
                self.filterfreq = ['butterpass', frange]
            else:  # highpass filter
                iir_filter = butter(N=order, Wn=frange[0], btype='highpass')
                self.filterfreq = ['butterhigh', frange[0]]
        else:
            if frange[1]:  # lowpass filter
                iir_filter = butter(N=order, Wn=frange[1], btype='lowpass')
                self.filterfreq = ['butterlow', frange[1]]
        # create impulse
        impulse = np.zeros(len(iir_filter[0])*len(iir_filter[1]))
        impulse[(len(iir_filter[0])*len(iir_filter[1]))//2] = 1
        impulse_response = filtfilt(iir_filter[0], iir_filter[1], impulse)
        # save filter
        self.filter = impulse_response
        # filtered signal
        self.signal = filtfilt(iir_filter[0], iir_filter[1],
                               self.source_signal.signal)
        self.time  = self.source_signal.time


class Convolution:
    def __init__(self, source_signal):
        """Convolve signal with a kernel
        """
        self.source_signal = source_signal
        self.signal = source_signal.signal
        self.time = source_signal.time
        self.domain = 'time'
        self.filter = None  # the filter representation in the time domain

    def convolve(self, ktype='guassian', kradius=None):
        """Apply a convolution kernel on the signal
        
        Parameters:
            ktype : {'guassian', 'mean', 'linear'}
                The convolution kernel type.
                'guassian', 'linear' (linear decay function),
                'mean', 'morlet' (real), 'ricker' (mexican hat)

            kradius : int
                radius of convolution kernel in sampling units. Actual width is
                2*kradius+1


        """

        kernel_base = np.arange(-kradius, kradius+1)
        # create the convolution kernel
        if ktype == 'guassian':
            kernel = np.exp(- (kernel_base**2)/(2 * (0.3*kradius)**2))
            # normalize kernel    
            self.filter = kernel/np.sum(kernel)
        elif ktype == 'mean':
            self.filter = np.ones(2*kradius+1)/(2*kradius+1)
        elif ktype == 'linear':
            kernel = np.linspace(start=1, stop=0.1, num=(2*kradius+1),
                                 endpoint=True)
            kernel = kernel - np.mean(kernel)
            self.filter = kernel
        elif ktype == 'morlet':
            self.filter = morlet(M=(2*kradius+1))
        elif ktype == 'ricker':
            self.filter = ricker(points=(2*kradius+1), a=0.3*kradius)
        # apply convolution
        self.signal = np.convolve(self.source_signal.signal,
                                  np.real(self.filter),
                                  mode='same')
 
