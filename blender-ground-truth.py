import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolors
from copy import copy
import argparse


def read_depth(filename):
    f = OpenEXR.InputFile(filename)
    r = f.channel("R")
    dw = f.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    data = np.frombuffer(r, dtype=np.float32)
    data.resize(sz[::-1])

    return data


def read_label(filename):
    f = OpenEXR.InputFile(filename)
    r = f.channel("R")
    dw = f.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    data = np.frombuffer(r, dtype=np.int32)
    data.resize(sz[::-1])

    return data


def read_normal(filename):
    f = OpenEXR.InputFile(filename)
    (r, g, b) = f.channels("RGB")
    dw = f.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    n_x = np.frombuffer(r, dtype=np.float32)
    n_x.resize(sz[::-1])

    n_y = np.frombuffer(g, dtype=np.float32)
    n_y.resize(sz[::-1])

    n_z = np.frombuffer(b, dtype=np.float32)
    n_z.resize(sz[::-1])

    return n_x, n_y, n_z


def read_flow(filename):
    f = OpenEXR.InputFile(filename)
    (r, g, b, a) = f.channels("RGBA")
    dw = f.header()["dataWindow"]
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    u_bw = np.frombuffer(r, dtype=np.float32)
    u_bw.resize(sz[::-1])

    v_bw = np.frombuffer(g, dtype=np.float32)
    v_bw.resize(sz[::-1])

    u_fw = -np.frombuffer(b, dtype=np.float32)
    u_fw.resize(sz[::-1])

    v_fw = -np.frombuffer(a, dtype=np.float32)
    v_fw.resize(sz[::-1])

    return u_bw, v_bw, u_fw, v_fw


def normal_to_rgb(n_x, n_y, n_z):
    rgb = 0.5 * np.ones(np.append(n_x.shape, 3))

    valid = np.logical_not(np.logical_and(np.logical_and(n_x == 0, n_y == 0), n_z == 0))

    rgb[valid, 0] = (n_x[valid] + 1) / 2
    rgb[valid, 1] = (n_y[valid] + 1) / 2
    rgb[valid, 2] = n_z[valid]

    return rgb


def flow_to_rgb(u, v):
    r = np.sqrt(u * u + v * v)

    if r.max() > 0:
        r /= r.max()

    phi = np.arctan2(v, u)
    phi = (np.where(phi < 0, 2 * np.pi, 0) + phi) / (2 * np.pi)

    invalid = r == 0
    v = np.where(invalid, 0.5, 1)
    r[invalid] = 0

    hsv = np.stack((phi, r, v), axis=2)
    rgb = mpcolors.hsv_to_rgb(hsv)

    return rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender-Ground-Truth")
    parser.add_argument(
        "-i", "--inpath", type=str, help="input path", default="output/scene"
    )
    args = parser.parse_args()

    palette = copy(plt.cm.jet)
    palette.set_bad("gray", 1.0)

    # image
    image = mpimg.imread(args.inpath + "/image0002.png")

    plt.subplot(321)
    plt.title("image")
    plt.imshow(image)

    # label
    label = read_label(args.inpath + "/label0002.exr")

    plt.subplot(322)
    plt.title("label")
    plt.imshow(np.ma.masked_where(label == 0, label), cmap=palette)

    # depth
    depth = read_depth(args.inpath + "/depth0002.exr")

    plt.subplot(323)
    plt.title("depth")
    plt.imshow(np.ma.masked_where(depth >= 1e10, depth), cmap=palette)

    # normal
    (n_x, n_y, n_z) = read_normal(args.inpath + "/normal0002.exr")

    plt.subplot(324)
    plt.title("normal")
    normal = normal_to_rgb(n_x, n_y, n_z)
    plt.imshow(normal)

    # flow
    (u_bw, v_bw, u_fw, v_fw) = read_flow(args.inpath + "/flow0002.exr")

    plt.subplot(325)
    plt.title("backward flow")
    flow_bw = flow_to_rgb(u_bw, v_bw)
    plt.imshow(flow_bw)

    plt.subplot(326)
    plt.title("forward flow")
    flow_fw = flow_to_rgb(u_fw, v_fw)
    plt.imshow(flow_fw)

    plt.tight_layout()
    plt.show()
