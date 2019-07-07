import genericgui
from signaltabs import *
# Signaltab, Noisetab, Loadsignaltab, Filtertab, Detrendtab, FFTtab, Freqfiltertab, Convolutiontab

# create tabs

tabs = [Loadsignaltab, FFTtab]
#tabs = [signaltabs.Signaltab, signaltabs.Freqfiltertab, signaltabs.Freqfiltertab]
#tabs = [Loadsignaltab, Freqfiltertab, Freqfiltertab, Detrendtab,  FFTtab]
#tabs = [signaltabs.Signaltab, signaltabs.Noisetab]
#tabs = [signaltabs.Loadsignaltab]

# create gui

app = genericgui.GenericGUI()

app.add_tabs(tabs)

app.mainloop()


# create gui

