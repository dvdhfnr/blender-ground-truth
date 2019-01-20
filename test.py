import OpenEXR
import numpy as np
import matplotlib.pyplot as plt


def read_normal(filename):
    f = OpenEXR.InputFile(filename)
    (r, g, b) = f.channels("RGB")
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    n_x = np.frombuffer(r, dtype=np.float32)
    n_x.resize(sz[::-1])

    n_y = np.frombuffer(g, dtype=np.float32)
    n_y.resize(sz[::-1])

    n_z = np.frombuffer(b, dtype=np.float32)
    n_z.resize(sz[::-1])

    return n_x, n_y, n_z


if __name__ == "__main__":
    print("Blender-Ground-Truth")

    (n_x, n_y, n_z) = read_normal("normal0001.exr")

    valid = np.logical_not(np.logical_and(np.logical_and(n_x == 0, n_y == 0), n_z == 0))

    rgb = 0.5 * np.ones(np.append(n_x.shape, 3))
    rgb[valid, 0] = n_x[valid]
    rgb[valid, 1] = n_y[valid]
    rgb[valid, 2] = n_z[valid]

    plt.imshow(rgb)
    plt.show()
