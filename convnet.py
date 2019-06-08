from manimlib.imports import *

import numpy as np

from colour import Color

from torchvision.datasets import MNIST


class ThreeDSceneSquareGrid(ThreeDScene):


    def create_grid(self, xx, yy, fill_colors, fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=1):

        grid = []

        class SquareCell:

            def __init__(self, square, x, y):
                self.square = square
                self.x = x
                self.y = y


        for x in xx:
            for y in yy:

                    # RGB image (scaled to [0,1])
                    if type(fill_colors) == np.ndarray and fill_colors.ndim == 3:
                        fill_color_rgb = tuple(fill_colors[x,y, :])
                    # Grayscalage image (scaled to [0.1])
                    elif type(fill_colors) == np.ndarray and fill_colors.ndim == 2:
                        fill_color_rgb = (fill_colors[x,y], fill_colors[x,y], fill_colors[x,y])
                    # solid color, RGB Tuple
                    elif type(fill_colors) == tuple and len(fill_colors) == 3:
                        fill_color_rgb = fill_colors


                    if type(stroke_colors) == np.ndarray and stroke_colors.ndim == 3:
                        stroke_color_rgb = tuple(stroke_colors[x,y, :])
                    elif type(stroke_colors) == np.ndarray and stroke_colors.ndim == 2:
                        stroke_color_rgb = (stroke_colors[x,y], stroke_colors[x,y], fill_colors[x,y])
                    elif type(stroke_colors) == tuple:
                        stroke_color_rgb = stroke_colors

                    if type(fill_opacities) == int or type(fill_opacities) == float:
                        fill_opacity = (fill_opacities, fill_opacities, fill_opacities)
                    elif type(fill_opacities) == np.ndarray:
                        fill_opacity = fill_opacities[x,y]

                    square = Square(side_length=side_length, fill_color=Color(rgb=fill_color_rgb), fill_opacity=fill_opacity, stroke_color=Color(rgb=stroke_color_rgb))

                    cell = SquareCell(square, x, y)

                    cell.square.set_x(x*side_length)
                    cell.square.set_y(y*side_length)

                    grid.append(cell)

        return grid


    def shift_grid(self, grid, step_size, x_increment=None, y_increment=None, z_increment=None):
        for cell in grid:
            if x_increment:
                current_x = cell.square.get_x()
                cell.square.set_x(current_x+x_increment*step_size)
            if y_increment:
                current_y = cell.square.get_y()
                cell.square.set_y(current_y+y_increment*step_size)
            if z_increment:
                current_z = cell.square.get_z()
                cell.square.set_z(current_z+z_increment*step_size)


class SimpleGrid(ThreeDSceneSquareGrid):

    def construct(self):

        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        self.side_length = 0.9

        r_channel = self.create_grid(xx, yy, fill_colors=(1,0.0,0.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=1)

        for cell in r_channel:
            self.add(cell.square)

        self.shift_grid(r_channel, step_size=self.side_length, x_increment=1)


class RGBConv(ThreeDSceneSquareGrid):

    def construct(self):
        self.side_length = 0.9

        xx = np.arange(-3,3)
        yy = np.arange(-3,3)

        self.move_camera(phi=0, theta=0, frame_center=(0*self.side_length, 0*self.side_length, 10*self.side_length))

        r_channel = self.create_grid(xx, yy, fill_colors=(1,0.0,0.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)

        g_channel = self.create_grid(xx, yy, fill_colors=(0.0, 1.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)

        b_channel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 1.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)

        # Display channels for the first time
        for channel in (r_channel, g_channel, b_channel):
            for cell in channel:
                self.add(cell.square)



        self.shift_grid(g_channel, x_increment=0.5, y_increment=0.5, z_increment=1, step_size=self.side_length)
        self.shift_grid(b_channel, x_increment=1, y_increment=1, z_increment=2, step_size=self.side_length)

        self.wait(1)
        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        #  self.side_length = 0.7

        r_kernel = self.create_grid(xx, yy, fill_colors=(0.5,0.0,0.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)
        g_kernel = self.create_grid(xx, yy, fill_colors=(0.0,0.5,0.0), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)
        b_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 0.5), fill_opacities=0.8, stroke_colors=(1.0,1.0,1.0), side_length=self.side_length)

        for kernel in (r_kernel, g_kernel, b_kernel):
            for cell in kernel:
                self.add(cell.square)

        self.shift_grid(r_kernel, x_increment=-4, y_increment=0, z_increment=0, step_size=self.side_length)
        self.shift_grid(g_kernel, x_increment=-3.5, y_increment=0.5, z_increment=1, step_size=self.side_length)
        self.shift_grid(b_kernel, x_increment=-3, y_increment=1, z_increment=2, step_size=self.side_length)
        self.move_camera(phi=PI/6, theta=0, frame_center=(0*self.side_length, 0*self.side_length, 1*self.side_length))

        self.wait(2)

        self.move_camera(phi=PI/6, theta=0, frame_center=(2*self.side_length, 2*self.side_length, 20*self.side_length))

        self.wait(2)

        # Separate kernels and channels
        for _ in range(18):
            self.shift_grid(r_kernel, self.side_length, y_increment=-0.5)
            self.shift_grid(r_channel, self.side_length, y_increment=-0.5)
            self.shift_grid(b_kernel, self.side_length, y_increment=0.5)
            self.shift_grid(b_channel, self.side_length, y_increment=0.5)
            self.wait(0.08)


        for _ in range(8):
            self.shift_grid(r_kernel, x_increment=0.5, step_size=self.side_length)
            self.shift_grid(g_kernel, x_increment=0.5, step_size=self.side_length)
            self.shift_grid(b_kernel, x_increment=0.5, step_size=self.side_length)
            self.wait(0.08)

        self.wait(1)

        for _ in range(3):
            self.shift_grid(r_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(g_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(b_kernel, y_increment=1, step_size=self.side_length)
            self.wait(0.5)

        self.shift_grid(r_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(g_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(b_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.wait(0.5)

        for _ in range(3):
            self.shift_grid(r_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(g_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(b_kernel, y_increment=1, step_size=self.side_length)
            self.wait(0.5)

        self.shift_grid(r_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(g_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(b_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.wait(0.5)

        for _ in range(3):
            self.shift_grid(r_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(g_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(b_kernel, y_increment=1, step_size=self.side_length)
            self.wait(0.5)

        self.shift_grid(r_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(g_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.shift_grid(b_kernel, x_increment=1, y_increment=-3, step_size=self.side_length)
        self.wait(0.5)

        for _ in range(3):
            self.shift_grid(r_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(g_kernel, y_increment=1, step_size=self.side_length)
            self.shift_grid(b_kernel, y_increment=1, step_size=self.side_length)
            self.wait(0.5)

        self.wait(1)


class ConvNet(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        xx = np.arange(-3,3)
        yy = np.arange(-3,3)

        grid = self.create_grid(xx, yy, fill_colors=(0.1, 0.1, 0.1), side_length=self.side_length)

        for cell in grid:
            self.add(cell.square)

        self.move_camera(phi=3*PI/8,gamma=0)

        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.4,0.4,0.4), side_length=self.side_length)

        for cell in kernel:
            self.add(cell.square)

        self.wait(2)

        for _ in range(3):
            self.shift_grid(kernel, step_size=self.side_length, x_increment=1)
            self.wait(1)

        self.move_camera(phi=0,gamma=0, distance=50)
        self.shift_grid(kernel, step_size=self.side_length, x_increment=-3, y_increment=1)
        self.wait(1)

        for _ in range(3):
            self.shift_grid(kernel, step_size=self.side_length, x_increment=1)
            self.wait(1)
        self.begin_ambient_camera_rotation()


class MnistConvNet(ThreeDSceneSquareGrid):

    def construct(self):

        self.mnist_dataset = MNIST(root='../data',
                              train=True,
                              download=True)

        # Get a sample mnist image with digit 8
        self.sample_image = np.array(self.mnist_dataset[17][0]) / 255

        self.side_length = 0.4

        xx = np.arange(0,28)
        yy = np.arange(0,28)


        self.move_camera(phi=0, theta=0, frame_center=(14*self.side_length, 14*self.side_length, 40*self.side_length))

        # Define grid + show mnist digit
        mnist_grid = self.create_grid(xx, yy, fill_colors=self.sample_image, fill_opacities=1, side_length=self.side_length)
        for cell in mnist_grid:
            self.add(cell.square)

        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.8, 0.4), side_length=self.side_length)

        for cell in kernel:
            self.add(cell.square)

        self.shift_grid(kernel, step_size=self.side_length, x_increment=1)
        self.wait(2)

        for _ in range(28):
            self.shift_grid(kernel, step_size=self.side_length, y_increment=1)
            self.wait(0.1)
        #
        self.shift_grid(kernel, step_size=self.side_length, x_increment=1, y_increment=-14)
        self.shift_grid(kernel, step_size=self.side_length, x_increment=14)
        self.wait(1)

        self.move_camera(phi=PI/6, theta=0, frame_center=(14*self.side_length, 14*self.side_length, 1*self.side_length))
        for _ in range(10):
            self.shift_grid(kernel, step_size=self.side_length, y_increment=1)
            self.wait(0.1)
        self.wait(2)
