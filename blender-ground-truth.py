import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolors

def read_depth(filename):
    f = OpenEXR.InputFile(filename)
    r = f.channel("R")
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    data = np.frombuffer(r, dtype=np.float32)
    data.resize(sz[::-1])

    return data


def read_label(filename):
    f = OpenEXR.InputFile(filename)
    r = f.channel("R")
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    data = np.frombuffer(r, dtype=np.int32)
    data.resize(sz[::-1])

    return data


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


def read_flow(filename):
    f = OpenEXR.InputFile(filename)
    (r, g, b, a) = f.channels("RGBA")
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    u_fw = np.frombuffer(r, dtype=np.float32)
    u_fw.resize(sz[::-1])

    v_fw = np.frombuffer(g, dtype=np.float32)
    v_fw.resize(sz[::-1])

    u_bw = np.frombuffer(b, dtype=np.float32)
    u_bw.resize(sz[::-1])

    v_bw = np.frombuffer(a, dtype=np.float32)
    v_bw.resize(sz[::-1])

    return u_fw, v_fw, u_bw, v_bw


def normal_to_rgb(n_x, n_y, n_z):
    rgb = np.empty(np.append(n_x.shape, 3))

    rgb[:, :, 0] = (n_x + 1) / 2
    rgb[:, :, 1] = (n_y + 1) / 2
    rgb[:, :, 2] = n_z

    return rgb


def flow_to_rgb(u, v):
    r = u * u + v * v

    if r.max() > 0:
        r /= r.max()

    phi = np.arctan2(v, u)
    phi = (phi / np.pi + 1) / 2

    valid = r
    valid[valid > 0] = 1

    hsv = np.stack((r, valid, phi), axis=2)
    rgb = mpcolors.hsv_to_rgb(hsv)

    return rgb


if __name__ == "__main__":
    print("Blender-Ground-Truth")

    # image
    image = mpimg.imread("image0001.png")

    plt.subplot(321)
    plt.title("image")
    plt.imshow(image)

    # label
    label = read_label("label0001.exr")

    plt.subplot(322)
    plt.title("label")
    plt.imshow(np.ma.masked_where(label == 0, label), cmap="jet")

    # depth
    depth = read_depth("depth0001.exr")

    plt.subplot(323)
    plt.title("depth")
    plt.imshow(np.ma.masked_where(depth == 1e10, depth), cmap="jet")

    # normal
    (n_x, n_y, n_z) = read_normal("normal0001.exr")

    plt.subplot(324)
    plt.title("normal")
    normal = normal_to_rgb(n_x, n_y, n_z)
    plt.imshow(normal, cmap="jet")

    # flow
    (u_fw, v_fw, u_bw, v_bw) = read_flow("flow0001.exr")

    plt.subplot(325)
    plt.title("forward flow")
    flow_fw = flow_to_rgb(u_fw, v_fw)
    plt.imshow(flow_fw)

    plt.subplot(326)
    plt.title("backward flow")
    flow_bw = flow_to_rgb(u_bw, v_bw)
    plt.imshow(flow_bw)

    plt.show()
