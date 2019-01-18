import OpenEXR
import numpy as np
import matplotlib.pyplot as plt

file = OpenEXR.InputFile("label0001.exr")
r = file.channel("R")
dw = file.header()['dataWindow']
sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

data = np.frombuffer(r, dtype=np.float32)
data.resize(sz[::-1])

plt.imshow(np.ma.masked_where(data == 1e10, data), cmap="jet")
plt.colorbar()
plt.show()
