"""Signal object GUI tab"""

import numpy as np
import inspect
from mysignal import Signal, Noise, Filter, Detrend, Loadsignal, FFTsignal, Freqfilter, Convolution
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
# required to embed matplotlib figures in tkinter
import matplotlib
matplotlib.use("TkAgg")

# custom toolbar


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, plot_canvas, parent):
        NavigationToolbar2Tk.__init__(self, plot_canvas, parent)
    toolitems = (('Home', 'Reset original view', 'home', 'home'),
                 ('Back', 'Back to previous view', 'back', 'back'),
                 ('Forward', 'Forward to next view', 'forward', 'forward'),
                 ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'))


class Guitab(ttk.Frame):
    tab_type = 'tab'

    def __init__(self, parent, controller, previous=None):
        """Class for creating a generic GUI tab
        Arguments
        ---------
        parent <ttk.Notebook> - notebook to which tab will be inserted
        controller <Tk> - root GUI
        previous is the previous common object (i.e Signal NOT Signaltab)"""
        ttk.Frame.__init__(self, parent)
        # link to controller
        self.controller = controller
        # unique name identifier assigned by controller
        self.name = controller.assign_name(self.tab_type)
        # link to literal previous element in precessing cascade (NOT the gui tab )
        self.previous = previous
        self.change_flag = False  # flag for changes in gui tab. Calls for downstream refresh

    def layout_tab(self):
        # iterate over tab['parts']
        for labelrow, labelframe in enumerate(self.tab['parts']):
            # create signal generation frame
            iframe = tk.LabelFrame(self,
                                   text=labelframe['name'])
            iframe.grid(row=labelrow, column=0, sticky='nswe')
            # create a plot
            if 'axes' in labelframe.keys():
                # define figure
                temp_figure = Figure(figsize=(3.5, 2.5), dpi=100)
                num_subplots = labelframe['axes']
                # add subplot for number of axes
                subplot = []
                for plot_index in range(num_subplots):
                    instance_axes = temp_figure.add_subplot(1,
                                                            num_subplots,
                                                            plot_index+1)
                    subplot.append(instance_axes)
                labelframe['axes'] = subplot
                # add canvas (rendering target)
                labelframe['canvas'] = FigureCanvasTkAgg(temp_figure, iframe)
                canvas_row_span = 5
                labelframe['canvas'].get_tk_widget().grid(row=0, column=0,
                                                          rowspan=canvas_row_span,
                                                          sticky='nsw')
                toolbar_frame = tk.Frame(master=iframe)
                toolbar_frame.grid(row=canvas_row_span+1, column=0,
                                   sticky='nsw')
                toolbar = Toolbar(labelframe['canvas'],
                                  toolbar_frame)
                toolbar.update()
                toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            if labelframe['comps']:  # section contains interactive elements
                # add checkbox
                tk.Checkbutton(iframe, text='Activate', variable=labelframe['active'][0]).grid(
                    row=0, column=1, sticky='nw')
                if type(labelframe['comps']) == dict:
                    for entryrow, entrykey in enumerate(labelframe['comps'].keys()):
                        tk.Label(iframe,
                                 text=labelframe['comps'][entrykey][0]).grid(row=entryrow+1,
                                                                             column=1,
                                                                             sticky='nw')
                        if type(labelframe['comps'][entrykey][1]) is list:  # comboBox
                            labelframe['comps'][entrykey][2] = ttk.Combobox(iframe,
                                                                            values=labelframe['comps'][entrykey][1])
                            # set value to top values
                            labelframe['comps'][entrykey][2].current(0)
                        elif inspect.ismethod(labelframe['comps'][entrykey][1]):
                            labelframe['comps'][entrykey][2] = tk.Button(iframe,
                                                                         text=labelframe['comps'][entrykey][0],
                                                                         command=labelframe['comps'][entrykey][1])
                        else:
                            labelframe['comps'][entrykey][2] = tk.Entry(iframe)
                            labelframe['comps'][entrykey][2].delete(0, tk.END)
                            labelframe['comps'][entrykey][2].insert(0,
                                                                    labelframe['comps'][entrykey][1])
                        labelframe['comps'][entrykey][2].grid(row=entryrow+1,
                                                              column=2, sticky='nwe')
            refresh_button = ttk.Button(self, text='Refresh',
                                        command=lambda: self.refresh_tab())
        refresh_button.grid(
            row=len(self.tab['parts']), column=0, sticky='nswe')
        # refresh the tab for first display
        self.refresh_tab()

    def refresh_tab(self):
        """Refresh tab and update with new values in input boxes"""
        # iterate all over all tab values and check if changed
        # set activated flag
        self.activated = False
        # iterate over panels
        for labelframe in self.tab['parts']:
            if labelframe['comps']:
                # check checkbox for if active
                current_state = labelframe['active'][0].get()
                if current_state != labelframe['active'][1]:
                    self.change_flag = True
                    labelframe['active'][1] = current_state
                if current_state == True:
                    # iterate over entry boxes
                    self.activated = True
                    if type(labelframe['comps']) == dict:
                        for entry in labelframe['comps'].values():
                            if inspect.ismethod(entry[1]):
                                continue
                            # entry[2]: entry box, entry[3]: previous val, entry[4] :type cast
                            current_val = entry[4](entry[2].get())
                            if current_val != entry[3]:  # values has changed
                                entry[3] = current_val  # store new values
                                self.change_flag = True  # set flag
        if self.change_flag or self.issource:
            self.refresh_action()
            self.update_controller()

    def refresh_action(self):
        pass

    def update_controller(self):
        """method to notify controller gui a tab has been modified so changes
        can propogate downstream"""
        self.controller.propogate_signal(self)
        # reset change flag
        self.change_flag = False

    # Auxiliary methods for plotting
    def plot_signal(self, axes_handle, canvas_handle, stype='source'):
        """Plot the signal type specified by stype on the axes handles
        Arguments
        ---------
        stype <str> - signal type to plot. Allowed values are: 'source' for
        source signal, 'results' for resulting signal and 'combined' for both
        'source' and 'results' plots 
        axes_handle <reference> - reference to the axes of the figure
        canvas_handle <reference> - a reference to the canvas handle (required
        for refresh (i.e 'draw')
        """
        tab_name = self.name
        # plot clean signal
        # required for axes handles with multiple subplots
        axes_handle = axes_handle[0]
        axes_handle.clear()
        if stype == 'source':
            # select plotting mode - time or frequency domain
            if self.controller.common[tab_name].source_signal.domain == 'time':
                axes_handle.plot(
                    self.controller.common[tab_name].source_signal.time, self.controller.common[tab_name].source_signal.signal)
                axes_handle.set_yscale('linear')
                #plt.xlabel('time [sec|a.u]')
                #plt.ylabel('Amplitude [au]')
            elif self.controller.common[tab_name].source_signal.domain == 'frequency':
                # plot frequency domain plot
                fft_signal = self.controller.common[tab_name].source_signal.signal
                time = self.controller.common[tab_name].source_signal.time
                fft_signal = 2*np.abs(fft_signal)/len(fft_signal)
                axes_handle.plot(time[time >= 0], fft_signal[time >= 0])
                axes_handle.set_yscale('log')
                #plt.xlabel('frequency [Hz]')
                #plt.ylabel('PSD [V**2/Hz]')
        elif stype == 'results':
            # select plotting mode - time or frequency domain or spectrogram
            if self.controller.common[tab_name].domain == 'time':
                axes_handle.plot(
                    self.controller.common[tab_name].time, self.controller.common[tab_name].signal)
                axes_handle.set_yscale('linear')
                #plt.xlabel('time [sec|a.u]')
                #plt.ylabel('Amplitude [au]')
            elif self.controller.common[tab_name].domain == 'frequency':
                # plot frequency domain plot
                fft_signal = self.controller.common[tab_name].signal
                time = self.controller.common[tab_name].time
                fft_signal = 2*np.abs(fft_signal)/len(fft_signal)
                axes_handle.plot(time[time >= 0], fft_signal[time >= 0])
                axes_handle.set_yscale('log')
                #plt.xlabel('frequency [Hz]')
                #plt.ylabel('PSD [V**2/Hz]')
            elif self.controller.common[tab_name].domain == 'spectrogram':
                spectrogram = self.controller.common[tab_name].signal
                time = self.controller.common[tab_name].time
                frequencies = self.controller.common[tab_name].spectrogram_frequencies
                axes_handle.pcolormesh(time, frequencies, np.log(spectrogram))
                # axes_handle.imshow(spectrogram)
                axes_handle.set_yscale('linear')
        elif stype == 'combined':
             # select plotting mode - time or frequency domain
            if self.controller.common[tab_name].domain == 'time':
                axes_handle.plot(
                    self.controller.common[tab_name].source_signal.time,
                    self.controller.common[tab_name].source_signal.signal)
                axes_handle.plot(
                    self.controller.common[tab_name].time, self.controller.common[tab_name].signal)
                axes_handle.set_yscale('linear')
                #plt.xlabel('time [sec|a.u]')
                #plt.ylabel('Amplitude [au]')
            elif self.controller.common[tab_name].domain == 'frequency':
                # plot frequency domain plot
                fft_signal = self.controller.common[tab_name].signal
                fft_source_signal = \
                    self.controller.common[tab_name].source_signal.signal
                time_signal = self.controller.common[tab_name].time
                time_source_signal = \
                    self.controller.common[tab_name].source_signal.time
                fft_signal = 2*np.abs(fft_signal)/len(fft_signal)
                fft_source_signal = 2 * \
                    np.abs(fft_source_signal)/len(fft_source_signal)
                axes_handle.plot(
                    time_signal[time_signal >= 0], fft_signal[time_signal >= 0])
                axes_handle.plot(time_source_signal[time_source_signal > 0],
                                 fft_source_signal[time_source_signal > 0])
                axes.set_yscale('log')
                #plt.xlabel('frequency [Hz]')
                #plt.ylabel('PSD [V**2/Hz]')

        canvas_handle.draw()

    def plot_filter(self, axes_handle, canvas_handle):
        """Plot the filter and its frequency representation on the axes handles
        Arguments
        ---------
        axes_handle <reference> - reference to the axes of the figure. Must be
        axes handle to two subplots
        canvas_handle <reference> - a reference to the canvas handle (required
        for refresh (i.e 'draw')
        """
        tab_name = self.name
        # plot
        frequency_axes_handle = axes_handle[0]
        time_axes_handle = axes_handle[1]
        # clean axes
        frequency_axes_handle.clear()
        time_axes_handle.clear()
        # plot time domain
        time_axes_handle.plot(np.real(self.controller.common[tab_name].filter))
        time_axes_handle.set_yscale('linear')
        # draw ideal filter
        frequency_response = np.fft.fft(
            self.controller.common[tab_name].filter)
        nyquist = len(frequency_response)/2
        frequency_response = \
            np.abs(frequency_response[:int(nyquist)])
        frequency_axes_handle.plot(frequency_response)
        # check if filter is from object with support to 'ideal' shape
        if hasattr(self.controller.common[tab_name], 'filterfreq'):
            if self.controller.common[tab_name].filterfreq[0] in {'bandpass',
                                                                  'butterpass'}:
                band_limit_freqs = self.controller.common[tab_name].filterfreq[1]
                band_limit_freqs = [nyquist*freq for freq in band_limit_freqs]
                # differentiate between pass and stop filters
                if band_limit_freqs[0] > band_limit_freqs[1]:
                    filter_in, filter_out = 0, 1
                else:
                    filter_in, filter_out = 1, 0
                ideal_filter = np.array([[0, filter_out],
                                         [band_limit_freqs[0], filter_out],
                                         [band_limit_freqs[0], filter_in],
                                         [band_limit_freqs[1], filter_in],
                                         [band_limit_freqs[1], filter_out],
                                         [nyquist, filter_out]])
                frequency_axes_handle.plot(ideal_filter[:, 0], ideal_filter[:, 1])
            elif self.controller.common[tab_name].filterfreq[0] in {'highpass',
                                                                    'butterhigh'}:
                band_limit_freqs = self.controller.common[tab_name].filterfreq[1]
                band_limit_freqs = nyquist*band_limit_freqs
                ideal_filter = np.array([[0, 0],
                                         [band_limit_freqs, 0],
                                         [band_limit_freqs, 1],
                                         [nyquist, 1]])
                frequency_axes_handle.plot(ideal_filter[:, 0], ideal_filter[:, 1])
            elif self.controller.common[tab_name].filterfreq[0] in {'lowpass',
                                                                    'butterlow'}:
                band_limit_freqs = self.controller.common[tab_name].filterfreq[1]
                band_limit_freqs = nyquist*band_limit_freqs
                ideal_filter = np.array([[0, 1],
                                         [band_limit_freqs, 1],
                                         [band_limit_freqs, 0],
                                         [nyquist, 0]])
            frequency_axes_handle.plot(ideal_filter[:, 0], ideal_filter[:, 1])
        frequency_axes_handle.set_yscale('log')
        # redraw canvas
        canvas_handle.draw()


class Signaltab(Guitab):
    tab_type = 'signal'

    def __init__(self, parent, controller, previous=None):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = True
        self.tab = {'name': 'signal',
                    'type': 'plot',
                    'parts': [{'name': 'Generate',
                               'axes': 1,
                               'canvas': None,
                               # set to active since this is the root panel
                               'active': [tk.IntVar(value=1), None],
                               'comps': {'stype': ['type', ['poles', 'spikes',
                                                            'white noise',
                                                            'lintrend', 'polytrend',
                                                            'events',
                                                            'sum_frequencies',
                                                            'guassian', 'brownian'],
                                                   None, None, str],
                                         'interp':['interpolate', ['linear',
                                                                   'cubic'],
                                                   None, None, str],
                                         'sample_frequency': ['frequency [Hz]', 1000, None,
                                                  None, int],
                                         'length':['length [sec]', 3, None,
                                                   None, float],
                                         'poles':['pole/spike/deg/event', 15, None, None, int],
                                         'amplitude':['amplitude', 1, None,
                                                      None, float]}}]}
        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = Signal()
        self.controller.common[tab_name].create_signal(stype=self.tab['parts'][0]['comps']['stype'][3],
                                                       sample_frequency=self.tab['parts'][0]['comps']['sample_frequency'][3],
                                                       time=self.tab['parts'][0]['comps']['length'][3],
                                                       poles=self.tab['parts'][0]['comps']['poles'][3],
                                                       interp=self.tab['parts'][0]['comps']['interp'][3],
                                                       amplitude=self.tab['parts'][0]['comps']['amplitude'][3])

        # plot signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'],
                         stype='results')


class Noisetab(Guitab):
    tab_type = 'noise'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'noise',
                    'type': 'plot',
                    'parts': [{'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'Noise',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'amplitude': ['amplitude',
                                                       1, None, None,
                                                       float],
                                         'ntype': ['type', ['rand', 'normal',
                                                            'pink'],
                                                   None, None, str]}},
                              {'name': 'Combined',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}
        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = \
            Noise(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'], stype='source')
        if self.tab['parts'][1]['active'][0].get():
            # create noise
            self.controller.common[tab_name].add_noise(noise_amplitude=self.tab['parts'][1]['comps']['amplitude'][3],
                                                       ntype=self.tab['parts'][1]['comps']['ntype'][3])
            # plot noise time and frequency domain representaion
            self.plot_signal(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'],
                             stype='results')
        else:
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
        self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                         canvas_handle=self.tab['parts'][2]['canvas'],
                         stype='combined')


class Loadsignaltab(Guitab):
    tab_type = 'load'

    def __init__(self, parent, controller, previous=None, signal_path=None,
                 reference_path=None):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = True
        self.signal_path = signal_path
        self.reference_path = reference_path
        self.tab = {'name': 'load',
                    'type': 'plot',
                    'parts': [{'name': 'Signal',
                               'axes': 1,
                               'canvas': None,
                               # set to active since this is the root panel
                               'active': [tk.IntVar(value=1), None],
                               'comps': {'signal': ['Select...',
                                                    self.load_signal,
                                                    None, None, str]}},
                              {'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'ref': ['Select...',
                                                 self.load_reference_signal,
                                                 None, None, str]}},
                              {'name': 'Combined',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}
        self.layout_tab()

    def load_signal(self):
        """action for loading signal clicked"""
        # start file select gui
        self.signal_path = tk.filedialog.askopenfilename(
            title="Select signal file", filetypes=(('numpy files',
                                                    '*.npy'), ('all files',
                                                               '*.*')))
        tab_name = self.name
        self.controller.common[tab_name].load_signal(path_to_signal=self.signal_path,
                                                     sigtype='orig')

    def load_reference_signal(self):
        """action for loading reference signal clicked"""
        # start file select gui
        self.signal_path = tk.filedialog.askopenfilename(
            title="Select reference signal file", filetypes=(('numpy files',
                                                              '*.npy'), ('all files',
                                                                         '*.*')))
        tab_name = self.name
        self.controller.common[tab_name].load_signal(path_to_signal=self.signal_path,
                                                     sigtype='ref')

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        if tab_name not in self.controller.common.keys():
            self.controller.common[tab_name] = Loadsignal()
            if not self.signal_path:
                self.load_signal()
            if self.reference_path:
                self.controller.common[tab_name].load_signal(path_to_signal=self.reference_path,
                                                             sigtype='ref')
        # plot clean signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'],
                         stype='results')
        if self.tab['parts'][1]['active'][0].get() and self.controller.common[tab_name].source_signal is not None:
            self.plot_signal(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'],
                             stype='source')
            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas'],
                             stype='combined')
        else:
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas'],
                             stype='results')


class Filtertab(Guitab):
    tab_type = 'filter'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'filter',
                    'type': 'plot',
                    'parts': [{'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'Filter',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'ftype': ['type', ['mean', 'gaussian',
                                                            'tkeo', 'median'], None,
                                                   None, str],
                                         'order': ['order', 2, None, None,
                                                   int],
                                         'edge': ['edge', ['copy', 'crop'],
                                                  None, None, str]}},
                              {'name': 'Combined',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}
        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = \
            Filter(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'], stype='source')
        if self.tab['parts'][1]['active'][0].get():
            # activate filter
            self.controller.common[tab_name].running_filter(ftype=self.tab['parts'][1]['comps']['ftype'][3],
                                                            order=self.tab['parts'][1]['comps']['order'][3],
                                                            edge=self.tab['parts'][1]['comps']['edge'][3])
            # plot filtered signal
            self.plot_signal(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'],
                             stype='results')
        else:
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
        self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                         canvas_handle=self.tab['parts'][2]['canvas'],
                         stype='combined')


class Detrendtab(Guitab):
    tab_type = 'detrend'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'detrend',
                    'type': 'plot',
                    'parts': [{'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'Detrend',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'method': ['method', ['linear',
                                                               'polynomial'], None,
                                                    None, str]}},
                              {'name': 'Combined',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}
        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = \
            Detrend(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'],
                         stype='source')
        if self.tab['parts'][1]['active'][0].get():
            # activate detrending
            self.controller.common[tab_name].detrend(
                method=self.tab['parts'][1]['comps']['method'][3])
            # draw plots
            self.plot_signal(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'],
                             stype='results')
        else:
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
        # combined signal
        self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                         canvas_handle=self.tab['parts'][2]['canvas'],
                         stype='combined')


class FFTtab(Guitab):
    tab_type = 'fft'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'fft',
                    'type': 'plot',
                    'parts': [{'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'FFT/iFFT',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(value=1), None],
                               'comps': {'ffttype': ['method', ['FFT', 'Welch',
                                                                'spectrogram',
                                                               'time-freq'],
                                                     None, None, str],
                                         'window': ['window size', 1024, None,
                                                    None, int]}}]}

        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = FFTsignal(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'], stype='source')
        # plot FFT/iFFT signal
        if self.tab['parts'][1]['active'][0].get():
            # select type of fft to preform
            if self.tab['parts'][1]['comps']['ffttype'][3] == 'FFT':
                self.controller.common[tab_name].fft_signal()
            elif self.tab['parts'][1]['comps']['ffttype'][3] == 'Welch':
                self.controller.common[tab_name].welch_signal(
                    window_size=self.tab['parts'][1]['comps']['window'][3])
            elif self.tab['parts'][1]['comps']['ffttype'][3] == 'spectrogram':
                self.controller.common[tab_name].spectrogram_signal(
                    window_size=self.tab['parts'][1]['comps']['window'][3])
            elif self.tab['parts'][1]['comps']['ffttype'][3] == 'time-freq':
                self.controller.common[tab_name].time_frequency(
                    window_size=self.tab['parts'][1]['comps']['window'][3])
        self.plot_signal(axes_handle=self.tab['parts'][1]['axes'],
                         canvas_handle=self.tab['parts'][1]['canvas'], stype='results')


class Freqfiltertab(Guitab):
    tab_type = 'freqfilter'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'freq_filter',
                    'type': 'plot',
                    'parts': [{'name': 'Source',  # 0
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'Filter',  # 1
                               'axes': 2,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'ftype': ['type', ['lowpass',
                                                            'highpass',
                                                            'bandpass',
                                                            'butterlow',
                                                            'butterhigh',
                                                            'butterband'], None,
                                                   None, str],
                                         'cutin': ['cutin (0-1 highpass)', 0.4, None, None,
                                                   float],
                                         'cutout': ['cutout (0-1 lowpass)', 0.6, None,
                                                    None, float],
                                         'order': ['order', 73, None, None,
                                                   int],
                                         'window': ['window', ['boxcar',
                                                               'hann',
                                                               'hamming',
                                                               'guassian'], None,
                                                    None, str]}},
                              {'name': 'Combined',  # 3
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}
        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = Freqfilter(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'], stype='source')
        if self.tab['parts'][1]['active'][0].get():
            if self.tab['parts'][1]['comps']['ftype'][3] == 'lowpass':
                self.controller.common[tab_name].fir_filter(window=self.tab['parts'][1]['comps']['window'][3],
                                                            order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[None,
                                                                    self.tab['parts'][1]['comps']['cutout'][3]])
            elif self.tab['parts'][1]['comps']['ftype'][3] == 'highpass':
                self.controller.common[tab_name].fir_filter(window=self.tab['parts'][1]['comps']['window'][3],
                                                            order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[self.tab['parts'][1]['comps']['cutin'][3],
                                                                    None])
            elif self.tab['parts'][1]['comps']['ftype'][3] == 'bandpass':
                self.controller.common[tab_name].fir_filter(window=self.tab['parts'][1]['comps']['window'][3],
                                                            order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[self.tab['parts'][1]['comps']['cutin'][3],
                                                                    self.tab['parts'][1]['comps']['cutout'][3]])
            elif self.tab['parts'][1]['comps']['ftype'][3] == 'butterlow':
                self.controller.common[tab_name].iir_filter(order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[None,
                                                                    self.tab['parts'][1]['comps']['cutout'][3]])
            elif self.tab['parts'][1]['comps']['ftype'][3] == 'butterhigh':
                self.controller.common[tab_name].iir_filter(order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[self.tab['parts'][1]['comps']['cutin'][3],
                                                                    None])
            elif self.tab['parts'][1]['comps']['ftype'][3] == 'butterband':
                self.controller.common[tab_name].iir_filter(order=self.tab['parts'][1]['comps']['order'][3],
                                                            frange=[self.tab['parts'][1]['comps']['cutin'][3],
                                                                    self.tab['parts'][1]['comps']['cutout'][3]])

            # plot filtered signal
            self.plot_filter(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'])
            # plot combo signals
            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas'],
                             stype='combined')

        else:
            # clear filter axes
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
            # plot source signal as results signal
            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas '], stype='results')


class Convolutiontab(Guitab):
    tab_type = 'convolution'

    def __init__(self, parent, controller, previous):
        Guitab.__init__(self, parent, controller, previous)
        self.issource = False
        self.tab = {'name': 'convolution',
                    'type': 'plot',
                    'parts': [{'name': 'Source',
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None},
                              {'name': 'convolution kernel',
                               'axes': 2,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': {'ktype': ['type', ['guassian', 'mean',
                                                            'linear', 'morlet',
                                                           'ricker'],
                                                   None, None, str],
                                         'kradius': ['kernel radius', 9, None,
                                                     None, int]}},
                              {'name': 'Convolved',  # 3
                               'axes': 1,
                               'canvas': None,
                               'active': [tk.IntVar(), None],
                               'comps': None}]}

        self.layout_tab()

    def refresh_action(self):
        tab_name = self.name
        # create signal memory in controller
        self.controller.common[tab_name] = Convolution(self.previous)
        # plot source signal
        self.plot_signal(axes_handle=self.tab['parts'][0]['axes'],
                         canvas_handle=self.tab['parts'][0]['canvas'], stype='source')
        # convolve
        if self.tab['parts'][1]['active'][0].get():
            self.controller.common[tab_name].convolve(ktype=self.tab['parts'][1]['comps']['ktype'][3],
                                                      kradius=self.tab['parts'][1]['comps']['kradius'][3])
            self.plot_filter(axes_handle=self.tab['parts'][1]['axes'],
                             canvas_handle=self.tab['parts'][1]['canvas'])

            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas'],
                             stype='combined')
        else:
            # clear filter axes
            for axes in self.tab['parts'][1]['axes']:
                axes.clear()
            self.tab['parts'][1]['canvas'].draw()
            # plot source signal as results signal
            self.plot_signal(axes_handle=self.tab['parts'][2]['axes'],
                             canvas_handle=self.tab['parts'][2]['canvas'], stype='results')

