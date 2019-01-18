import OpenEXR
import numpy as np
import matplotlib.pyplot as plt

file = OpenEXR.InputFile("flow0001.exr")
(r, g, b, a) = file.channels("RGBA")
dw = file.header()['dataWindow']
sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

r = np.frombuffer(r, dtype=np.int32)
r.resize(sz[::-1])

g = np.frombuffer(g, dtype=np.int32)
g.resize(sz[::-1])

b = np.frombuffer(b, dtype=np.int32)
b.resize(sz[::-1])

a = np.frombuffer(a, dtype=np.int32)
a.resize(sz[::-1])

print(np.array_equal(r, g) and np.array_equal(r, b))

plt.imshow(np.ma.masked_where(r == 0, r), cmap="jet")
plt.colorbar()
