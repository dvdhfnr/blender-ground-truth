import OpenEXR
import numpy as np
import matplotlib.pyplot as plt


def read_depth(filename):
    f = OpenEXR.InputFile(filename)

    (r, a) = f.channels("RA")
    dw = f.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    data = np.frombuffer(r, dtype=np.float32)
    data.resize(sz[::-1])

    d = np.frombuffer(a, dtype=np.float32)
    d.resize(sz[::-1])

    return data, d


if __name__ == "__main__":
    print("Test")

    # depth
    depth, d = read_depth("depth0001.exr")

    plt.title("depth")
    plt.imshow(np.ma.masked_where(d >= 1e10, d), cmap="jet")
