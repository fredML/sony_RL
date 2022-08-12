from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import glob
import imageio as io


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_halfsphere(fig, z0, R, N_theta, N_phi):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = z0+R * np.outer(np.ones(np.size(u)), np.cos(v))

    u = np.arange(0, 2 * np.pi+N_phi, N_phi)
    v = np.arange(0, np.pi+N_theta, N_theta)
    x2 = R * np.outer(np.cos(u), np.sin(v))
    y2 = R * np.outer(np.sin(u), np.sin(v))
    z2 = z0 + R * np.outer(np.ones(np.size(u)), np.cos(v))

    ax = fig.gca(projection='3d')

    #ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
    ax.plot_wireframe(x2, y2, z2, color="r", alpha=.2)
    return ax


def plot_cam(ax, theta, phi, R, z0, r):
    x_c = R*np.sin(theta)*np.cos(phi)
    y_c = R*np.sin(theta)*np.sin(phi)
    z_c = z0+R*np.cos(theta)

    x_a = x_c - r*np.sin(theta)*np.cos(phi)
    y_a = y_c - r*np.sin(theta)*np.sin(phi)
    z_a = z_c - r*np.cos(theta)

    a = Arrow3D([x_c, x_a], [y_c, y_a], [z_c, z_a], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")

    ax.add_artist(a)
    ax.scatter([x_c], [y_c], [z_c], color="r", s=100)
    return ax


def plot_traj(pts, theta, phi, N_theta, N_phi, svg):

    R = 50
    r = 20
    z0 = 50

    fig = plt.figure(figsize=(12, 7))
    ax = plot_halfsphere(fig, z0, R, N_theta, N_phi)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="k", alpha=.3, s=1)

    ax = plot_cam(ax, theta, phi, R, z0, r)

    plt.axis("off")
    plt.savefig(svg, bbox_inches="tight")


def mk_anim(svg_folder,name):
    ims = glob.glob(svg_folder+"/*")
    writer = io.get_writer(name+'.mp4', fps=2)
    images = []
    for im in ims:
        writer.append_data(io.imread(im))
    writer.close()
