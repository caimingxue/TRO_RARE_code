#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : plot_mag.py
# @Time : 2022/5/6 下午4:52
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import sys
sys.path.append('../../')
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from loguru import logger
# from mayavi import mlab
import magpylib as magpy
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.plot_utils import Frame
from mpl_toolkits.mplot3d import Axes3D

import math

from scipy.integrate import odeint

try:
    plt.style.use('ggplot')
except OSError:
    pass


# support chinese style
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']

class PlotTool(object):

    def __init__(self):
        # self.fig_1 = plt.figure(figsize=(30, 20))
        # spec_1 = gridspec.GridSpec(figure=self.fig_1, nrows=1, ncols=1, width_ratios=[1, 1])

        # self.ax_mag_move_3D = make_3d_axis(ax_s=20, pos=spec_1[0], unit="mm", n_ticks=15)
        self.ax_coils = make_3d_axis(ax_s=20, pos=111, unit="mm", n_ticks=20)
        self.ax_coils.tick_params(labelsize=10, bottom =0)

        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 12,
        #          }
        # self.ax_coils.set_xlabel('round', font2, fontweight ='bold')
        # self.ax_coils.set_ylabel('round', font2)
        # self.ax_coils.set_zlabel('round', font2)

        self.fig_2 = plt.figure(figsize=(15, 8))
        spec_2 = gridspec.GridSpec(figure=self.fig_2, nrows=1, ncols=2, width_ratios=[1, 1])
        self.ax_3D_mag_field = make_3d_axis(ax_s=15, pos=spec_2[0], unit="mm", n_ticks=15)
        self.ax_strm = self.fig_2.add_subplot(spec_2[1])

        # self.fig_3 = plt.figure(figsize=(15, 8))
        # spec_3 = gridspec.GridSpec(figure=self.fig_3, nrows=1, ncols=2, width_ratios=[1, 1])
        # self.ax_2D_mag_field = self.fig_3.add_subplot(spec_3[0])

        # self.quiver_mag_3D = self.ax_mag_move_3D.quiver(*(0, 0, 0, 0, 0, 0))
        # self.quiver_mag_3D_normal = self.ax_mag_move_3D.quiver(*(0, 0, 0, 0, 0, 0))

        self.quiver_mag_3D = self.ax_coils.quiver(*(0, 0, 0, 0, 0, 0))
        self.quiver_mag_3D_normal = self.ax_coils.quiver(*(0, 0, 0, 0, 0, 0))

        self.quiver_3D_anim = self.ax_3D_mag_field.quiver(*(0, 0, 0, 0, 0, 0))
        self.quiver_3D_anim_normal = self.ax_3D_mag_field.quiver(*(0, 0, 0, 0, 0, 0))

        self._flag_ = False

    def axis3d_equal(self, X, Y, Z, ax):
        max_range = np.array([X.max() - X.min(), Y.max()
                              - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                               - 1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                               - 1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                               - 1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    def quiver_3D_mag_field(self, x, y, z, Bx, By, Bz):
        self.quiver_3D_anim.remove()
        # for stopping simulation with the esc key.
        self.fig_2.canvas.mpl_connect('key_release_event',
                                      lambda event: [exit(0) if event.key == 'escape' else None])
        self.quiver_3D_anim = self.ax_3D_mag_field.quiver(*(x, y, z, Bx, By, Bz),
                                                          length=3, cmap=plt.cm.jet,
                                                          pivot='tail', normalize=False)

    # def streamplot(self,
    #                plane_name: str = None,
    #                plane_grid=None,
    #                B=None,
    #                density=[1, 2],
    #                ):
    #
    #     if plane_name == "xy":
    #         dim_1 = plane_grid[:, :, 0]
    #         dim_2 = plane_grid[:, :, 1]
    #         B_dim_1 = B[:, :, 0]
    #         B_dim_2 = B[:, :, 1]
    #
    #     elif plane_name == "xz":
    #         dim_1 = plane_grid[:, :, 0]
    #         dim_2 = plane_grid[:, :, 2]
    #         B_dim_1 = B[:, :, 0]
    #         B_dim_2 = B[:, :, 2]
    #
    #     elif plane_name == "yz":
    #         dim_1 = plane_grid[:, :, 1]
    #         dim_2 = plane_grid[:, :, 2]
    #         B_dim_1 = B[:, :, 1]
    #         B_dim_2 = B[:, :, 2]
    #
    #     self.ax_strm.collections = []  # clear lines streamplot
    #     self.ax_strm.patches = []  # clear arrowheads streamplot
    #
    #     Bamp = np.linalg.norm(B, axis=2)
    #     self.ax_strm.contourf(dim_1, dim_2, Bamp, 100, cmap='rainbow')
    #
    #     speed = np.hypot(B_dim_1, B_dim_2)
    #     lw = 3 * speed / speed.max() + .5
    #
    #     strm = self.ax_strm.streamplot(dim_1, dim_2, B_dim_1, B_dim_2,
    #                                    density=density,
    #                                    color=speed,
    #                                    linewidth=lw,
    #                                    cmap=plt.cm.jet)
    #     # self.fig_2.colorbar(strm.lines) unknown
    #     logger.warning("complete streamplot")
    #
    #     return strm,
    #
    # def quiver_2D_mag_field(self,
    #                         plane_name: str = None,
    #                         plane_grid=None,
    #                         B=None,
    #                         ):
    #     if plane_name == "xy":
    #         dim_1 = plane_grid[:, :, 0]
    #         dim_2 = plane_grid[:, :, 1]
    #         B_dim_1 = B[:, :, 0]
    #         B_dim_2 = B[:, :, 1]
    #
    #     elif plane_name == "xz":
    #         dim_1 = plane_grid[:, :, 0]
    #         dim_2 = plane_grid[:, :, 2]
    #         B_dim_1 = B[:, :, 0]
    #         B_dim_2 = B[:, :, 2]
    #
    #     elif plane_name == "yz":
    #         dim_1 = plane_grid[:, :, 1]
    #         dim_2 = plane_grid[:, :, 2]
    #         B_dim_1 = B[:, :, 1]
    #         B_dim_2 = B[:, :, 2]
    #
    #     if not self._flag_:
    #         self.quiver = self.ax_2D_mag_field.quiver(dim_1, dim_2,
    #                                                   np.zeros_like(dim_1),
    #                                                   np.zeros_like(dim_1),
    #                                                   units='xy',
    #                                                   pivot='mid',
    #                                                   alpha=0.8
    #                                                   )
    #         self._flag_ = True
    #
    #     color = np.sin(B_dim_1)
    #
    #     self.quiver.set_UVC(B_dim_1, B_dim_2, C=color)

    def mlab_anim_3D_mag(self, x_data, y_data, z_data, Bx, By, Bz, B_norm):

        mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(480, 480))
        mlab.clf()

        mag_filed = mlab.pipeline.vector_field(x_data,
                                               y_data,
                                               z_data,
                                               np.zeros_like(x_data),
                                               np.zeros_like(y_data),
                                               np.zeros_like(z_data),
                                               scalars=np.zeros_like(x_data))
        vectors = mlab.pipeline.vectors(mag_filed,
                                        scale_factor=x_data[1, 0, 0] - x_data[0, 0, 0])

        # Mask random points, to have a lighter visualization.
        vectors.glyph.mask_input_points = True
        vectors.glyph.mask_points.on_ratio = 2

        vcp = mlab.pipeline.vector_cut_plane(mag_filed)
        vcp.glyph.glyph.scale_factor = 5 * (x_data[1, 0, 0] - x_data[0, 0, 0])

        ms = vectors.mlab_source
        delayer = 1000

        @mlab.animate(delay=delayer)
        def anim_loc():
            i = 0
            for u, v, w, norm in zip(Bx, By, Bz, B_norm):
                ms.set(
                    u=u,
                    v=v,
                    w=w,
                    scalars=norm,
                    colormap='jet')
                print("========================", i)
                i = i + 1
            yield

        anim_loc()
        mlab.axes(xlabel='x', ylabel='y', zlabel='z')
        # mlab.outline()
        mlab.colorbar()
        # Mask random points, to have a lighter visualization.
        # mlab.view(39, 74, 0.59, [.008, .0007, -.005])
        mlab.show()

    def mag_motion_update(self, mag_position, mag_vector, n):
        '''
        :param mag_position: the magnetic generation position, i.e., target point position
        :param mag_vector: the magnetic field vector at desired mag_position
        :param n: the normal vector of the rotating field
        :return:
        '''
        self.quiver_mag_3D.remove()
        self.quiver_mag_3D_normal.remove()
        # ax.collections = []  # clear lines streamplot
        # ax.patches = []  # clear arrowheads streamplot
        # for stopping simulation with the esc key.
        self.fig_1.canvas.mpl_connect('key_release_event',
                                      lambda event: [exit(0) if event.key == 'escape' else None])
        # 'tail', 'middle', 'tip'
        self.quiver_mag_3D = self.ax_coils.quiver(mag_position[0], mag_position[1], mag_position[2],
                                                  mag_vector[0], mag_vector[1], mag_vector[2],
                                                  color='r', pivot='tail', length=8, arrow_length_ratio=0.3,
                                                  normalize=False)
        self.quiver_mag_3D_normal = self.ax_coils.quiver(mag_position[0], mag_position[1], mag_position[2],
                                                         n[0], n[1], n[2],
                                                         color='blue', pivot='tail', length=15,
                                                         arrow_length_ratio=0.3, normalize=False)
        plt.grid(True)
        plt.title("Time[s]:" + str(round(2, 2)) +
                  ", accel[m/s]:" + str(round(2, 2)) +
                  ", speed[km/h]:" + str(round(1 * 3.6, 2)))

    def plot_3D_trajectory(self, trajectory_waypoints):
        for waypoint in trajectory_waypoints:
            self.ax_mag_move_3D.scatter(waypoint[0], waypoint[1], waypoint[2],
                                        color='black', label="point")

    def get_MagRobotSim_class(self, magrobot_sim):
        self.sim = magrobot_sim

    '''
    coils system frame animation
    '''

    # def build_coil_frames(self):

        # self.coila_frame = Frame(self.sim.coil_system.coila.coil_coords.T(), label="coila", s=3)
        # self.coila_frame.add_frame(self.ax_mag_move_3D)
        #
        # self.coilb_frame = Frame(self.sim.coil_system.coilb.coil_coords.T(), label="coilb", s=3)
        # self.coilb_frame.add_frame(self.ax_mag_move_3D)
        #
        # self.coilc_frame = Frame(self.sim.coil_system.coilc.coil_coords.T(), label="coilc", s=3)
        # self.coilc_frame.add_frame(self.ax_mag_move_3D)

    # def update_coils_frame(self, num):
    #     self.coila_frame.set_data(self.sim.coil_system.coila.coil_coords.T())
    #     self.coilb_frame.set_data(self.sim.coil_system.coilb.coil_coords.T())
    #     self.coilc_frame.set_data(self.sim.coil_system.coilc.coil_coords.T())

        # self.mag_motion_update(self.sim.desired_rotating_mag.mag_position,
        #                        self.sim.mag_on_target,
        #                        self.sim.desired_rotating_mag.n)
        # self.line.set_data(self.traj_waypoints[0:2,: num])
        # self.line.set_3d_properties(self.traj_waypoints[2, :num])
        #
        # self.ax_mag_move_3D.scatter(self.traj_waypoints[0, : self.waypoints_index],
        #                                        self.traj_waypoints[1, : self.waypoints_index],
        #                                        self.traj_waypoints[2, : self.waypoints_index],
        #                                        c='gray')

    '''
    coils stream_mag animation
    '''

    def update_stream_mag_field(self, num):
        if len(self.sim.B_on_plane) > 0:
            self.streamplot('yz', self.sim.coil_system.yz_grid, self.sim.B_on_plane)

    def update_3D_mag_field(self, x, y, z, Bx, By, Bz):
        self.quiver_3D_mag_field(x, y, z, Bx, By, Bz)

    '''
    coils quiver_2D_mag anim5ation
    '''

    def update_quiver_2D_mag_field(self, num):
        self.quiver_2D_mag_field('yz', self.sim.coil_system.yz_grid, self.sim.B_on_plane)

    def animation(self, x, y, z, Bx, By, Bz):
        '''
        函数FuncAnimation(fig,func,frames,init_func,interval,blit)是绘制动图的主要函数，其参数如下：
            a.fig 绘制动图的画布名称
            b.func自定义动画函数，即下边程序定义的函数update
            c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
            d.init_func自定义开始帧，即传入刚定义的函数init,初始化函数
            e.interval更新频率，以ms计
            f.blit选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
        '''

        animation_type = 'polar'

        # self.build_coil_frames()
        # animator_mag = animation.FuncAnimation(self.fig_1,
        #                                        self.update_coils_frame,
        #                                        frames=2000,
        #                                        interval=1.0 / self.sim.dt,
        #                                        blit=False,
        #                                        repeat=True)


        animator_3D_strm_mag = animation.FuncAnimation(self.fig_2,
                                                          self.update_3D_mag_field(x, y, z, Bx, By, Bz),
                                                          frames=2000,
                                                          interval=1.0 / 0.1,
                                                          blit=False,
                                                          repeat=True)

        # animator_coils_strm_mag = animation.FuncAnimation(self.fig_2,
        #                                                   self.update_stream_mag_field,
        #                                                   frames=2000,
        #                                                   interval=1.0 / self.sim.dt,
        #                                                   blit=False,
        #                                                   repeat=True)
        # print("===================dt", self.sim.dt)
        # animator_quiver_2D_mag = animation.FuncAnimation(self.fig_3,
        #                                                  self.update_quiver_2D_mag_field,
        #                                                  frames=2000,
        #                                                  interval=1.0 / self.sim.dt,
        #                                                  blit=False,
        #                                                  repeat=True)
        plt.show()

    def mlab_plot(self):
        mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(480, 480))
        mlab.clf()

        mag_filed = mlab.pipeline.vector_field(self.sim.coil_system.xyz_grid[0],
                                               self.sim.coil_system.xyz_grid[1],
                                               self.sim.coil_system.xyz_grid[2],
                                               np.zeros_like(self.sim.coil_system.xyz_grid[0]),
                                               np.zeros_like(self.sim.coil_system.xyz_grid[0]),
                                               np.zeros_like(self.sim.coil_system.xyz_grid[0]),
                                               scalars=np.zeros_like(self.sim.coil_system.xyz_grid[0]))
        vectors = mlab.pipeline.vectors(mag_filed,
                                        scale_factor=self.sim.coil_system.xyz_grid[0][1, 0, 0] -
                                                     self.sim.coil_system.xyz_grid[0][0, 0, 0])

        # Mask random points, to have a lighter visualization.
        vectors.glyph.mask_input_points = True
        vectors.glyph.mask_points.on_ratio = 1

        vcp = mlab.pipeline.vector_cut_plane(mag_filed)
        vcp.glyph.glyph.scale_factor = 5 * (self.sim.coil_system.xyz_grid[0][1, 0, 0] -
                                            self.sim.coil_system.xyz_grid[0][0, 0, 0])
        ms = vectors.mlab_source
        delayer = 250

        @mlab.animate(delay=delayer)
        def anim_loc():
            while True:
                ms.set(
                    u=self.sim.Bx,
                    v=self.sim.By,
                    w=self.sim.Bz,
                    scalars=self.sim.B_norm,
                    colormap='jet')
                print("========================")
                mlab.points3d(1, 1, 0.2, colormap="copper", scale_factor=1)
                yield

        anim_loc()
        mlab.axes(xlabel='x', ylabel='y', zlabel='z')
        # mlab.outline()
        mlab.colorbar()
        # Mask random points, to have a lighter visualization.
        mlab.view(39, 74, 0.59, [.008, .0007, -.005])
        mlab.show()

    def mlab_plot_real(self):

        mlab.figure("RoboMag", bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(480, 480))
        mlab.clf()

        mag_filed = mlab.pipeline.vector_field(self.sim.coil_system.xyz_grid_real[0],
                                               self.sim.coil_system.xyz_grid_real[1],
                                               self.sim.coil_system.xyz_grid_real[2],
                                               np.zeros_like(self.sim.coil_system.xyz_grid_real[0]),
                                               np.zeros_like(self.sim.coil_system.xyz_grid_real[0]),
                                               np.zeros_like(self.sim.coil_system.xyz_grid_real[0]),
                                               scalars=np.zeros_like(self.sim.coil_system.xyz_grid_real[0]))
        vectors = mlab.pipeline.vectors(mag_filed, line_width=3, mask_points=1,
                                        scale_factor=self.sim.coil_system.xyz_grid_real[0][1, 0, 0] -
                                                     self.sim.coil_system.xyz_grid_real[0][0, 0, 0])
        # Mask random points, to have a lighter visualization.
        vectors.glyph.mask_input_points = True
        # vectors.glyph.mask_points.on_ratio = 2

        vcp = mlab.pipeline.vector_cut_plane(mag_filed)
        vcp.glyph.glyph.scale_factor = 1 * (self.sim.coil_system.xyz_grid_real[0][1, 0, 0] -
                                            self.sim.coil_system.xyz_grid_real[0][0, 0, 0])

        ms = vectors.mlab_source
        delayer = 200

        xs = self.sim.x_hist_all
        ys = self.sim.y_hist_all
        zs = np.zeros_like(xs) + 30

        plt = mlab.points3d(xs[:1], ys[:1], zs[:1], mode='cylinder', color=(0, 0, 1), scale_factor=2.0)
        trajectory = mlab.plot3d(xs[:1], ys[:1], zs[:1], color=(1, 0, 0), tube_radius=0.1, line_width=0.2)
        self.x_all = []
        self.y_all = []
        self.z_all = []

        # 必须在 @ mlab.animated函数中更改mlab_source中的数据
        @mlab.animate(delay=delayer)
        def anim_loc():
            # while True:
            # mlab.points3d(0, 0, 0, mode='cylinder', colormap="Dark2", scale_factor=1)
            mlab.plot3d(0, 0, 0, color=(1, 0, 0), tube_radius=0.1, line_width=0.2)
            # for (x, y, z) in zip(xs, ys, zs):
            while True:
                plt.mlab_source.set(x=self.sim.x_hist[len(self.sim.x_hist) - 1],
                                    y=self.sim.y_hist[len(self.sim.y_hist) - 1],
                                    z=self.sim.z_hist[len(self.sim.z_hist) - 1])

                # self.x_all.append(self.sim.x_hist)
                # self.y_all.append(self.sim.y_hist)
                # self.z_all.append(self.sim.z_hist)
                trajectory.mlab_source.reset(x=self.sim.x_hist, y=self.sim.y_hist, z=self.sim.z_hist)

                ms.set(
                    u=self.sim.Bx_real,
                    v=self.sim.By_real,
                    w=self.sim.Bz_real,
                    scalars=self.sim.B_norm_real,
                    colormap='jet')
                logger.info("=================")
                time.sleep(0.01)
                yield

        anim_loc()
        mlab.axes(xlabel='x', ylabel='y', zlabel='z')
        # mlab.outline(name="test")
        mlab.colorbar()
        # Mask random points, to have a lighter visualization.
        mlab.view(39, 74, 0.59, [.008, .0007, -.005])

        mlab.show()


class Ball:
    def __init__(self, ax, init_pos, size=10, shape='o'):
        self.scatter, = ax.plot([init_pos[0]], [init_pos[1]], [init_pos[2]], shape, markersize=size, animated=True)

    def update(self, pos):
        # draw ball
        self.scatter.set_data_3d(pos)


class Line:
    def __init__(self, ax, init_pos, size=3, color='g'):
        self.line = ax.plot([init_pos[0]], [init_pos[1]], [init_pos[2]], linewidth=size, color=color)[0]

    def update(self, pos):
        # draw line
        print("==============", pos)
        self.line.set_data(pos[0:2])
        self.line.set_3d_properties(pos[2])
