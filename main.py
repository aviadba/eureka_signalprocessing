#!/usr/bin/python

# IMPORTANT!!
# append path to common packages sys.path. Must reflect path to genericgui
# module.
import sys
sys.path.insert(0, '/home/aviad/sketchbook/python/common/genericgui')
import genericgui


# import project specific tabs
from udemySignalProcessingTabs import *

# avialable tabs
# Signaltab, Noisetab, Loadsignaltab, Filtertab, Detrendtab, FFTtab,
# Freqfiltertab, Convolutiontab, Resampletab, Outlierstab, Featurestab

# create tabs using specific tab definitions examples:
#tabs = [Loadsignaltab, Freqfiltertab, Freqfiltertab, FFTtab]
#tabs = [Loadsignaltab, Resampletab, Resampletab, Outlierstab]
#tabs = [Loadsignaltab, Freqfiltertab, Freqfiltertab, Detrendtab,  FFTtab]
#tabs = [signaltabs.Signaltab, signaltabs.Noisetab]
tabs = [Signaltab, Noisetab, Featurestab]

# initialize gui
app = genericgui.GenericGUI()
# add tabs to generic gui
app.add_tabs(tabs)

app.mainloop()

