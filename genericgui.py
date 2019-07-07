"""GenericGUI is a gui for viewing and manipulating tabbed objects of type
ttk.Frame"""

from tkinter import ttk
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk) 
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


class GenericGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Generic GUI tool")
        # create common memory dict
        self.common = dict()
        # create a presistent tab counter
        self.tab_counter = 0
        # create main container frame
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        # create a static left panel with labelFrame, plot area a Refresh
        # button
        main_frame = tk.LabelFrame(container, text='Main')
        main_frame.grid(row=0, column = 0, sticky='nswe')
        # define figure
        temp_figure = Figure(figsize=(3.5, 2.5), dpi=100)
        # add subplot
        self.axes = temp_figure.add_subplot(111)
        # add canvas (rendering target)
        self.canvas = FigureCanvasTkAgg(temp_figure, main_frame)
        #canvas_row_span = 3
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nw')
        toolbar_frame= tk.Frame(master=main_frame)
        toolbar_frame.grid(row=1, column =0, sticky='nsw')
        toolbar = Toolbar(self.canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        update_button = ttk.Button(main_frame, text='Update', command=lambda:
                                   self.update_gui())
        update_button.grid(row=2, column=0, sticky='nswe')

        # create notebook (tabs) in main container frame
        self.notebook_tabs = []
        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=0, column=1, sticky="nsew")

    def assign_name(self, base_name):
        """assign a unique name to tab. Function is called by child tabs to
        generate unique names
        Parameters:
            base_name : string
                A unique three digit extension is concatenated to base name to
                create a unique name"""
        counter = 0
        available = False
        while not available:
            name = base_name+str(counter).zfill(3)
            if name in [tab.name for tab in self.notebook_tabs]:
                counter+=1
            else:
                available = True
        return name

    def add_tabs(self, notebook_tabs):
        """Sequentially add tabs in notebook_tabs dict to gui        
        Parameters:
            notebook_tabs : list
                classes for creating ttk.Frame objects
        """
        previous_tab_name=None
        for tab in notebook_tabs:
            # create tab
            if previous_tab_name:
                instance_tab = tab(self.notebook, self,
                                   self.common[previous_tab_name])
            else:
                instance_tab = tab(self.notebook, self, previous_tab_name)
            # update previous tab name
            previous_tab_name = instance_tab.name
            self.notebook_tabs.append(instance_tab)
            # add tab
            self.notebook.add(self.notebook_tabs[-1],
                              text=self.notebook_tabs[-1].name)


    def update_gui(self):
        """Callback for main panel 'Update' button. Cascade through all tabs to
        refresh input parameters
        """
        tab_names = [itab.name for itab in self.notebook_tabs]
        self.axes.clear()
        self.axes.plot(self.common[tab_names[0]].time, self.common[tab_names[0]].signal)
        self.axes.plot(self.common[tab_names[-1]].time,
                       self.common[tab_names[-1]].signal)
        try:
            self.axes.plot(self.common[tab_names[0]].time,
                           self.common[tab_names[0]].ref_signal)
        except (ValueError, AttributeError) as error:  # reference signal does not exist
            pass
        self.canvas.draw()

    def propogate_signal(self, sourcetab):
        # find name of source tab
        sourcetab_name = sourcetab.name
        # find index of name in notebook_tabs names
        tab_names = [itab.name for itab in self.notebook_tabs]
        try:
            tab_idx = tab_names.index(sourcetab_name)
            next_tab = self.common[tab_names[tab_idx+1]]
            # update new source signal
            next_tab.source_signal = self.common[tab_names[tab_idx]]
            # call next tab update
            self.notebook_tabs[tab_idx+1].change_flag = True
            self.notebook_tabs[tab_idx+1].previous = self.common[tab_names[tab_idx]]
            self.notebook_tabs[tab_idx+1].refresh_tab()
        except (ValueError, IndexError) as error:
            # result of propagating signal at initial layout refresh sequence -
            # normal
            pass

