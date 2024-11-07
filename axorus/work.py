#%%
import numpy as np

f = r"F:\Axorus\ex_vivo_series_3\241024_A\processed\sorted\spike_templates.npy"

data = np.load(f)

#%%

print(data.shape)


#%%

from phylib.io.model import load_model

m = load_model(r"F:\Axorus\ex_vivo_series_3\241024_A\processed\sorted\params.py")
ch = m.get_cluster_channels(1)
m.

#%%

print(ch)
