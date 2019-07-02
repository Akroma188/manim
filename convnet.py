from manimlib.imports import *

import numpy as np

from colour import Color

from torchvision.datasets import MNIST


class ThreeDSceneSquareGrid(ThreeDScene):

    def create_grid(self, xx, yy, fill_colors, fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=1):

        class Grid:

            def __init__(self, xx, yy, fill_colors, fill_opacities, stroke_colors, side_length):

                class SquareCell:

                    def __init__(self, square, x, y):
                        self.square = square
                        self.x = x
                        self.y = y

                self.xx = xx
                self.yy = yy
                self.fill_colors = fill_colors
                self.fill_opacities = fill_opacities
                self.stroke_colors = stroke_colors
                self.side_length = side_length

                self.grid = []

                for x in xx:
                    for y in yy:

                            square = Square(side_length=side_length)

                            cell = SquareCell(square, x, y)

                            cell.square.set_x(x * side_length)
                            cell.square.set_y(y * side_length)

                            self.grid.append(cell)

                self = self.update_colors(fill_colors=self.fill_colors, stroke_colors=self.stroke_colors)
                self = self.update_opacities(fill_opacities=self.fill_opacities)

            def colors2rgb(self, colors, x, y):
                # RGB image (scaled to [0,1])
                if type(colors) == np.ndarray and colors.ndim == 3:
                    color_rgb = tuple(fill_colors[x, y, :])
                # Grayscalage image (scaled to [0.1])
                elif type(colors) == np.ndarray and colors.ndim == 2:
                    color_rgb = (colors[x, y], colors[x, y], colors[x, y])
                # solid color, RGB Tuple
                elif type(colors) == tuple and len(colors) == 3:
                    color_rgb = colors
                else:
                    raise ValueError("Check color type")

                return color_rgb

            def update_colors(self, fill_colors=None, stroke_colors=None):
                cell_count = 0
                for x in xx:
                    for y in yy:

                        fill_color = Color(rgb=self.colors2rgb(fill_colors, x, y)) if fill_colors is not None else None

                        stroke_color = Color(rgb=self.colors2rgb(stroke_colors, x, y)) if stroke_colors is not None else None

                        self.grid[cell_count].square.set_fill(color=fill_color)
                        self.grid[cell_count].square.set_stroke(color=stroke_color)
                        cell_count += 1

                self.fill_colors = fill_colors
                self.stroke_colors = stroke_colors

                return self

            def update_opacities(self, fill_opacities):
                cell_count = 0
                for x in xx:
                    for y in yy:

                        if type(fill_opacities) == int or type(fill_opacities) == float:
                            fill_opacity = (fill_opacities, fill_opacities, fill_opacities)
                        elif type(fill_opacities) == np.ndarray:
                            fill_opacity = fill_opacities[x, y]

                        self.grid[cell_count].square.set_opacity(fill_opacity)
                        cell_count += 1

                self.fill_opacities = fill_opacities

                return self

            def shift_grid(self, x_increment=None, y_increment=None, z_increment=None, step_size=self.side_length):
                for cell in self.grid:
                    if x_increment:
                        current_x = cell.square.get_x()
                        cell.square.set_x(current_x + x_increment * step_size)
                    if y_increment:
                        current_y = cell.square.get_y()
                        cell.square.set_y(current_y + y_increment * step_size)
                    if z_increment:
                        current_z = cell.square.get_z()
                        cell.square.set_z(current_z + z_increment * step_size)

        return Grid(xx, yy, fill_colors, fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)


class SimpleGridExample(ThreeDSceneSquareGrid):

    def construct(self):

        # Meshgrid defining bounds of grid
        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        # Side length of each square cell in the grid
        self.side_length = 0.9

        # Create a grid and display each cell in the grid
        simple_grid = self.create_grid(xx, yy, fill_colors=(1.0, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        # Necessary to display grids initially
        for cell in simple_grid.grid:
            self.add(cell.square)

        self.wait(1)

        self.move_camera(phi=PI/3, theta=0)

        self.wait(1)

        simple_grid.update_colors(fill_colors=(0.7, 0.7, 0.2))

        self.wait(1)


class RGBConv(ThreeDSceneSquareGrid):

    def construct(self):
        self.side_length = 0.9

        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        self.move_camera(phi=0, theta=0, frame_center=(0 * self.side_length, 0 * self.side_length, 10 * self.side_length))

        r_channel = self.create_grid(xx, yy, fill_colors=(1.0, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        g_channel = self.create_grid(xx, yy, fill_colors=(0.0, 1.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        b_channel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 1.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        convolution_output = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 0.0), side_length=self.side_length)

        # Initialize the displaying of channels for the first time
        for channel in (r_channel, g_channel, b_channel):
            for cell in channel.grid:
                self.add(cell.square)

        g_channel.shift_grid(x_increment=0.5, y_increment=0.5, z_increment=1)
        b_channel.shift_grid(x_increment=1, y_increment=1, z_increment=2)

        self.wait(1)
        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        r_kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        g_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.5, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        b_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 0.5), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        for kernel in (r_kernel, g_kernel, b_kernel):
            for cell in kernel.grid:
                self.add(cell.square)

        r_kernel.shift_grid(x_increment=-4, y_increment=0, z_increment=0)
        g_kernel.shift_grid(x_increment=-3.5, y_increment=0.5, z_increment=1)
        b_kernel.shift_grid(x_increment=-3, y_increment=1, z_increment=2)
        self.move_camera(phi=PI/6, theta=0, frame_center=(0 * self.side_length, 0 * self.side_length, 1 * self.side_length))

        self.wait(2)

        self.move_camera(phi=PI/6, theta=0, frame_center=(2 * self.side_length, 2 * self.side_length, 20 * self.side_length))

        self.wait(2)

        # Separate kernels and channels, overlap everything
        for _ in range(18):
            r_kernel.shift_grid(y_increment=-0.5)
            r_channel.shift_grid(y_increment=-0.5)
            b_kernel.shift_grid(y_increment=0.5)
            b_channel.shift_grid(y_increment=0.5)
            self.wait(0.08)
        for _ in range(8):
            r_kernel.shift_grid(x_increment=0.5)
            g_kernel.shift_grid(x_increment=0.5)
            b_kernel.shift_grid(x_increment=0.5)
            self.wait(0.08)

        self.wait(1)

        # Perform the convolutions
        for _ in range(3):
            r_kernel.shift_grid(y_increment=1)
            g_kernel.shift_grid(y_increment=1)
            b_kernel.shift_grid(y_increment=1)
            self.wait(0.5)

        r_kernel.shift_grid(x_increment=1, y_increment=-3)
        g_kernel.shift_grid(x_increment=1, y_increment=-3)
        b_kernel.shift_grid(x_increment=1, y_increment=-3)
        self.wait(0.5)

        for _ in range(3):
            r_kernel.shift_grid(y_increment=1)
            g_kernel.shift_grid(y_increment=1)
            b_kernel.shift_grid(y_increment=1)
            self.wait(0.5)

        r_kernel.shift_grid(x_increment=1, y_increment=-3)
        g_kernel.shift_grid(x_increment=1, y_increment=-3)
        b_kernel.shift_grid(x_increment=1, y_increment=-3)
        self.wait(0.5)

        for _ in range(3):
            r_kernel.shift_grid(y_increment=1)
            g_kernel.shift_grid(y_increment=1)
            b_kernel.shift_grid(y_increment=1)
            self.wait(0.5)

        r_kernel.shift_grid(x_increment=1, y_increment=-3)
        g_kernel.shift_grid(x_increment=1, y_increment=-3)
        b_kernel.shift_grid(x_increment=1, y_increment=-3)
        self.wait(0.5)

        for _ in range(3):
            r_kernel.shift_grid(y_increment=1)
            g_kernel.shift_grid(y_increment=1)
            b_kernel.shift_grid(y_increment=1)
            self.wait(0.5)

        self.wait(1)

class Conv2D(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0.1, 0.1, 0.1), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)

        self.move_camera(phi=3*PI/8, gamma=0)

        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.4, 0.4, 0.4), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)

        for _ in range(3):
            kernel.shift_grid(x_increment=1)
            self.wait(1)

        self.move_camera(phi=0, gamma=0, distance=50)
        kernel.shift_grid(x_increment=-3, y_increment=1)
        self.wait(1)

        for _ in range(3):
            kernel.shift_grid(x_increment=1)
            self.wait(1)
        self.begin_ambient_camera_rotation()


class Conv2D(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0.1, 0.1, 0.1), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)

        self.move_camera(phi=3*PI/8, gamma=0)

        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.4, 0.4, 0.4), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)

        for _ in range(3):
            kernel.shift_grid(x_increment=1)
            self.wait(1)

        self.move_camera(phi=0, gamma=0, distance=50)
        kernel.shift_grid(x_increment=-3, y_increment=1)
        self.wait(1)

        for _ in range(3):
            kernel.shift_grid(x_increment=1)
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

        xx = np.arange(0, 28)
        yy = np.arange(0, 28)


        self.move_camera(phi=0, theta=0, frame_center=(14*self.side_length, 14*self.side_length, 40*self.side_length))

        # Define grid + show mnist digit
        mnist_grid = self.create_grid(xx, yy, fill_colors=self.sample_image, fill_opacities=1, side_length=self.side_length)
        for cell in mnist_grid.grid:
            self.add(cell.square)

        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.8, 0.4), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        kernel.shift_grid(x_increment=1)
        self.wait(2)

        for _ in range(28):
            kernel.shift_grid(y_increment=1)
            self.wait(0.1)
        #
        kernel.shift_grid(x_increment=1, y_increment=-14)
        kernel.shift_grid(x_increment=14)
        self.wait(1)

        self.move_camera(phi=PI/6, theta=0, frame_center=(14 * self.side_length, 14 * self.side_length, 1 * self.side_length))
        for _ in range(10):
            kernel.shift_grid(y_increment=1)
            self.wait(0.1)
        self.wait(2)
