"""
udemysignal - Udemy Signal Processing course root file - contains
all the implemented signal processing classes. For GUI, use through udemysignal
tabs classes and generic GUI framework
"""

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.signal import *
#welch, spectrogram, firwin, butter, filtfilt, morlet, ricker

# Base Signal generating class


class Signal:
    def __init__(self):
        self.time = None
        self.signal = None
        self.sample_frequency = None
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
            freqs = np.linspace(0, time, time*sample_frequency)
            # select frequencies to include
            selected_signal_frequencies = \
                np.random.permutation(
                    range(1, int(sample_frequency/2)))[:poles]
            for instance_freq in selected_signal_frequencies:
                base_signal += np.random.randn()*np.sin(2*np.pi*instance_freq*freqs)
            self.signal = base_signal
        elif stype == 'guassian':
            # create randomly distributes centers
            guass_center_distance = np.exp(np.random.randn(poles))
            if len(guass_center_distance) == 1:
                guass_center_distance =  \
                    np.round(sample_frequency*time/3 *
                             abs(guass_center_distance))
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


# Base load signal class
class Loadsignal:

    def __init__(self):
        self.time = None
        self .signal = None
        self.source_signal = None
        self.sample_frequency = None
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
                time = np.squeeze(raw_signal[0, :])
                signal = np.squeeze(raw_signal[1, :])
            else:
                time = np.squeeze(raw_signal[:, 0])
                signal = np.squeeze(raw_signal[:, 1])
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

# Add simulated noise of several types to a base signal


class Noise:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = 'time'
        self.sample_frequency = source_signal.sample_frequency

    def add_noise(self, noise_amplitude=1, ntype='rand',
                  noise_sample_ratio=0.1):
        """
        Add noise or other kind of signal interference to the signal

        Parameters:
            ntype : {'rand', 'normal' , 'pink'}
                Noise type to add. 'rand' for randomly distributed noise,
                'nrand' for normally distributed noise
                'pink' noise distributed with 1/alpha*f distribution (pink noise)
                'irregular' downsample the data in irregular intervals 
                'gap' create a gap in the data
            noise_amplitude : float
                noise max level in 'standard deviation' units
            noise_sample_ratio : float
                ratio of point final/points original in creating irregular
                sequence or the ratio of gapped data to total signal length
            """
        # get standard deviation of signal
        noise_amplitude = noise_amplitude*np.std(self.source_signal.signal)
        # create noise
        if ntype in {'rand', 'normal', 'pink'}:
            if ntype == 'rand':
                noise = \
                    noise_amplitude * \
                    np.random.rand(len(self.source_signal.signal))
            elif ntype == 'normal':
                noise = \
                    noise_amplitude * \
                    np.random.randn(len(self.source_signal.signal))
            elif ntype == 'pink':
                # create frquency vector
                frequency = np.linspace(-len(self.source_signal.signal)//2,
                                        len(self.source_signal.signal)//2,
                                        len(self.source_signal.signal))
                pink = 1/frequency
                # set pink nan to zero
                phase = 2 * np.pi * \
                    np.random.rand(len(self.source_signal.signal)//2)
                # pink = pink * np.exp(-1j
            self.signal = self.source_signal.signal + noise
        elif ntype == 'irregular':
            num_points = \
                np.round(irregular_sample_ratio*len(self.source_signal.signal))
            intervals = np.exp(np.random.randn(int(num_points)))
            intervals = np.cumsum(intervals)
            intervals = intervals * \
                (len(self.source_signal.signal)-1)/intervals[-1]
            intervals = (np.ceil(intervals)).astype(int)
            intervals[-1] = len(self.source_signal.signal)-1
            self.signal = self.source_signal.signal[intervals]
            self.time = self.source_signal.time[intervals]
            self.sample_frequency = \
                self.sample_frequency*irregular_sample_ratio
        elif ntype == 'gap':
            gap_length = int(len(self.source_signal.time)*noise_sample_ratio)
            gap_start_idx = int(len(self.source_signal.time)/2 - gap_length/2)
            gap_end_idx = int(len(self.source_signal.time)/2 + gap_length/2)
            time = self.source_signal.time
            signal = self.source_signal.signal
            time = np.delete(time, range(gap_start_idx, gap_end_idx))
            signal = np.delete(signal, range(gap_start_idx, gap_end_idx))
            self.time = time
            self.signal = signal

# Time domain filters


class Filter:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = 'time'
        self.sample_frequency = source_signal.sample_frequency

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

# Linear and polynomial detrending of time domain data


class Detrend:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = 'time'
        self.sample_frequency = source_signal.sample_frequency

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

# FFT


class FFTsignal:
    def __init__(self, source_signal):
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = source_signal.domain
        self.sample_frequency = source_signal.sample_frequency
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

    def wavelet_cwt(self, wavelet='morlet'):
        """
        Preform a continuous wavelet transform of the data

        Parameters
            wavelet : {'morlet', 'ricker'}
                the type of wavelet to use. Morlet or Ricker (mexican hat)
        """
        # define frequency resolution
        if wavelet == 'morlet':
            wavelet = morlet
        elif wavelet == 'ricker':
            wavelet = ricker
        widths = np.arange(2, self.source_signal.sample_frequency//2)
        cwt_matr = np.abs(cwt(self.source_signal.signal, wavelet, widths))**2
        self.signal = cwt_matr
        self.spectrogram_frequencies = widths
        self.domain = 'spectrogram'

# Filters operating in the frequency domain


class Freqfilter:
    def __init__(self, source_signal):
        """Filters defined in the frequency domain. Operate in the time domain.
        Class allows comparing the resulting filter to the expected filter.
        Filters are then applied using filtfilt (zero phase shift) to the
        source_signal data"""
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = 'time'
        self.sample_frequency = source_signal.sample_frequency
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
                    iir_filter = butter(
                        N=order, Wn=[frange[1], frange[0]], btype='bandstop')
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
        self.time = self.source_signal.time

# Convolution


class Convolution:
    def __init__(self, source_signal):
        """Convolve signal with a kernel
        """
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.domain = 'time'
        self.filter = None  # the filter representation in the time domain
        self.sample_frequency = source_signal.sample_frequency

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

# Resample


class Resample:
    def __init__(self, source_signal):
        """Resample signal
        """
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.sample_frequency = source_signal.sample_frequency
        self.domain = 'time'

    def resample(self, factor=None, new_sample_rate=None, method='linear'):
        """Resample data
            Parameters
                factor : int
                    new data sampling rate will be factor * sampeling. if
                    factor>1 data is upsampled, if factor<1 data is
                    downsampled. If data is downsampled, original if initially
                    low-pass filtered at the new nyquist freq to avoid
                    aliasing. Use this value or new_sample_rate
                new_sample_rate : float
                    new target sampling rate. User either this value or factor
                method : {'linear', 'cubic', 'nearest'}
                    method of interpolation
       """
        if factor:
            new_sample_rate = self.source_signal.sample_frequency * factor
        elif new_sample_rate:
            factor = new_sample_rate/self.source_signal.sample_frequency
        else:  # default to max sample rate in data
            delta_time = np.diff(self.source_signal.time)
            new_sample_rate = 1/np.min(delta_time)
            factor = 1
        self.sample_frequency = new_sample_rate
        #downsampling: anitaliasing
        if self.sample_frequency < self.source_signal.sample_frequency:
            num_samples_resample = \
                int(self.sample_frequency/self.source_signal.sample_frequency *
                    len(self.source_signal.signal))
            self.signal, self.time = resample(x=self.source_signal.signal,
                                              num=num_samples_resample,
                                              t=self.source_signal.time)

        else:
            # create new time series
            self.time = np.arange(self.source_signal.time[0],
                                  self.source_signal.time[-1],
                                  1/new_sample_rate)
            # interpolate new points
            interpolation_function = interp1d(self.source_signal.time,
                                              self.source_signal.signal, kind=method)
            # calculate new signal
            self.signal = interpolation_function(self.time)

    def fill_gaps(self):
        """fill gaps in data
                fill_gaps for gapped data, uses the mean power spectrum of data
                before gap and after gap (representing sequences the same
                length of the gap, detrending and adding a trend to
                represent last/first known points
         """
        # find of gap
        time_gaps = np.diff(self.source_signal.time)
        gap_start_idx = np.argmax(time_gaps)
        # find gap border values
        gap_start_value = self.source_signal.signal[gap_start_idx]
        gap_end_value = self.source_signal.signal[gap_start_idx+1]
        # gap_length = self.source_signal.time[gap_start_idx +
        #                                     1] - self.source_signal.time[gap_start_idx]
        # create time sequence
        simulated_series_time = np.arange(start=self.source_signal.time[gap_start_idx],
                                          stop=self.source_signal.time[gap_start_idx+1],
                                          step=1/self.source_signal.sample_frequency)
        len_gap = len(simulated_series_time)
        # get FFT of sequence before gap
        pre_gap_fft = \
            np.fft.fft(self.source_signal.signal[gap_start_idx-len_gap:gap_start_idx],
                       norm='ortho')
        # get FFT of sequences after the gap
        post_gap_fft = \
            np.fft.fft(self.source_signal.signal[gap_start_idx+1:gap_start_idx+1+len_gap],
                       norm='ortho')
        # find mean spectrum
        mean_gap_spectrum = (pre_gap_fft + post_gap_fft)/2
        gap_sequence = np.abs(np.fft.ifft(mean_gap_spectrum, norm='ortho'))
        # detrend gap sequence
        gap_sequence = detrend(gap_sequence, type='linear')
        # add trend to data
        base_gap_signal = np.linspace(gap_start_value-gap_sequence[0],
                                      gap_end_value-gap_sequence[-1],
                                      len(simulated_series_time))
        gap_sequence += base_gap_signal
        self.signal = np.concatenate((self.source_signal.signal[:gap_start_idx-1],
                                      gap_sequence,
                                      self.source_signal.signal[gap_start_idx+2:]))
        self.time = \
            np.concatenate((self.source_signal.time[:gap_start_idx-1],
                            simulated_series_time,
                            self.source_signal.time[gap_start_idx+2:]))

    def remove_nan(self):
        """remove nan values from signal and time data"""
        nan_bool = np.isnan(self.source_signal.signal)
        # keep only non nan
        self.signal = self.source_signal.signal[~nan_bool]
        self.time = self.source_signal.time[~nan_bool]


class Outliers:
    def __init__(self, source_signal):
        """Outliers
        """
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.sample_frequency = source_signal.sample_frequency
        self.domain = 'time'

    def remove_outliers(self, method='static', metric='std', factor=3, kind='linear',
                        window_ratio=0.05):
        """remove outliesrs from data
            Parameters
                method : {'static', 'rolling'}
                    method of outlier identification
                metric : {'std', 'rms'}
                factor : int
                    outlier definition boundary factor (SD or RMS units)
                kind : {'linear', 'nearest', 'previous', 'cubic', 'quadratic'}
                    interpolation method
                window : float
                    the relative size of window for rolling calculations
       """
        if method == 'static':
            if metric == 'std':
                metric = np.std(self.source_signal.signal)
            elif metric == 'rms':
                metric = self.source_signal.signal - \
                    np.mean(self.source_signal.signal)
                metric = (np.mean(metric**2))**0.5
            mean = np.mean(self.source_signal.signal)
            outliers = np.logical_or(self.source_signal.signal > mean+factor*metric,
                                     self.source_signal.signal < mean-factor*metric)
        elif method == 'rolling':
            window = int(len(self.source_signal.signal)*window_ratio)
            outliers = np.array([False]*len(self.source_signal.signal))
            last_data_idx = len(self.source_signal.signal)-1
            for idx in range(len(self.source_signal.signal)):
                upper_limit = np.amin((idx+window, last_data_idx))
                lower_limit = np.amax((idx-window, 0))
                instance_range = self.source_signal.signal[lower_limit:upper_limit]
                # np.concatenate([self.source_signal.signal[idx-window:idx],
                #             self.source_signal.signal[idx+1:idx+window]])
                value = self.source_signal.signal[idx]
                if metric == 'std':
                    upper_b = np.mean(instance_range) + \
                        factor*np.std(instance_range)
                    lower_b = np.mean(instance_range) - \
                        factor*np.std(instance_range)
                elif metric == 'rms':
                    metric = instance_range - np.mean(instance_range)
                    metric = (np.mean(metric**2))**0.5
                    upper_b = np.mean(instance_range) + factor*metric
                    lower_b = np.mean(instance_range) - factor*metric
                if value > upper_b or value < lower_b:
                    outliers[idx] = True
        interpolator = interp1d(self.source_signal.time[~outliers],
                                self.source_signal.signal[~outliers], kind=kind,
                                bounds_error=False, fill_value='extrapolate')
        for idx in np.nditer(np.argwhere(outliers)):
            idx = int(idx)
            self.signal[idx] = interpolator(self.time[idx])

        self.domain = 'time'

    def map_noise_regions(self, metric='rms'):
        """Create a map of RMS values vs window size used. Calculations are
        carried out for windows ranging from 0.01 to 0.2 of the data length in
        0.01 increments

        Parameters
            metric : {'rms', 'std'}
       """
        # get rms for the entire data signal
        num_test_windows = 20
        window_increment = 0.01
        ratio_ranges = np.arange(start=0.01,
                                 stop=(num_test_windows+1)*window_increment,
                                 step=window_increment)
        # initialize empty results window
        result_map = np.full((num_test_windows,
                              len(self.source_signal.signal)), np.nan)
        last_data_idx = len(self.source_signal.signal)-1
        window_selection_idx = num_test_windows-1
        for win_size_ratio in np.nditer(ratio_ranges):
            window = int(len(self.source_signal.signal)*win_size_ratio)
            for idx in range(len(self.source_signal.signal)):
                upper_limit = np.amin((idx+window, last_data_idx))
                lower_limit = np.amax((idx-window, 0))
                instance_range = self.source_signal.signal[lower_limit:upper_limit]
                value = self.source_signal.signal[idx]
                if metric == 'std':
                    result_map[window_selection_idx,
                               idx] = np.std(instance_range)
                elif metric == 'rms':
                    # mean center instance range
                    instance_range = instance_range - np.mean(instance_range)
                    result_map[window_selection_idx, idx] = (
                        np.mean(instance_range**2))**0.5
            window_selection_idx -= 1
        self.domain = 'spectrogram'
        self.spectrogram_frequencies = ratio_ranges
        self.signal = result_map


class Features:
    def __init__(self, source_signal):
        """Features
        """
        self.source_signal = source_signal
        self.signal = source_signal.signal.copy()
        self.time = source_signal.time.copy()
        self.sample_frequency = source_signal.sample_frequency
        self.domain = 'time'

    def find_extrema(self, extrema='max', method='global', order=100):
        """Find maxima/minima in signal

            Parameters
                extrema : {'max', 'min'}
                method : {'global', 'local'}
                    method of extrema detection
                order : int
                    how many points on each side to consider for local extrema
       """
        if method == 'global':
            if extrema == 'max':
                extremaindx = np.argmax(self.source_signal.signal)
            elif extrema == 'min':
                extremaindx = np.argmin(self.source_signal.signal)
            self.signal = np.array([self.source_signal.signal[extremaindx]])
            self.time = np.array([self.source_signal.time[extremaindx]])
        elif method == 'local':
            if extrema == 'max':
                comparator = np.greater
            elif extrema == 'min':
                comparator = np.less
            # signal.signal.argrelextrema
            extremaindx = argrelextrema(data=self.source_signal.signal,
                                        comparator=comparator,
                                        order=order)
            # create a point set signal of extrema
            self.signal = self.source_signal.signal[extremaindx]
            self.time = self.source_signal.time[extremaindx]

    def find_envelope(self, method='hilbert_transform', order=99, cutoff=0.1):
        """find the envelope of a signal. Note: to asses the different
        envelopes, calculate R^2 with the rectified signal.

        Parameters
            method : {'hilbert_transform', 'varaince_envelope'}
            cutoff : float
                the cutoff of the low pass filter in nyquist rate ratio
            order : float
                the size of the lowpass filter
        """
        if method == 'hilbert_transform':
            # Hilbert transform - rectify, lowpass filter
            hilbert_transform = hilbert(self.source_signal.signal)
            self.signal = np.abs(hilbert_transform)
            self.time = self.source_signal.time
        elif method == 'variance_envelope':
            # rectify signal
            signal = np.abs(self.source_signal.signal)
            # create low pass filter
            fir_filter = firwin(
                numtaps=order, cutoff=cutoff, pass_zero=True,
                window='hamming', scale=True)
            self.signal = np.convolve(fir_filter, signal,
                                      mode='same')
            self.time = self.source_signal.time

    def feature_by_wavelets(self, wavelet='DoG'):
        """Convolve signal with a wavelet to find features
        Parameters
            wavelet : {'DoG'}
                name of wavelet
            width: int
                fwhm 
        """
        if wavelet == 'DoG':
            breakpoint()
            wavelet_time = np.linspace(-3, 3,
                                       int(self.source_signal.sample_frequency/2))
            wavelet = np.diff(np.exp(-wavelet_time**2))
        elif wavelet == 'ricker':
            wavelet = ricker(100, width)
