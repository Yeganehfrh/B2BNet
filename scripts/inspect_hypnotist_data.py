# open hypnotist data

import mne
import matplotlib.pyplot as plt
import numpy as np

eeg_path = 'data/EEG/hypnotist/Relaxation_whole.vhdr'
order = np.arange(1, 62, 1)
order = np.delete(order, 30)

raw = mne.io.read_raw_brainvision(eeg_path)
raw.plot(n_channels=62,
         order=order,
         highpass=1,
         lowpass=50,
         remove_dc=True)

plt.show()
