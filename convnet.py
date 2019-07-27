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

                for y in reversed(yy):
                    for x in xx:

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
                    color_rgb = tuple(colors[x, y, :])
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
                for x in range(len(xx)):
                    for y in range(len(yy)):

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

                        else:
                            raise ValueError("check opacities value")

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

        return Grid(xx, yy, fill_colors, fill_opacities=fill_opacities, stroke_colors=stroke_colors, side_length=self.side_length)


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

        self.move_camera(frame_center=(0 * self.side_length, 5 * self.side_length, 50 * self.side_length))

        r_channel = self.create_grid(xx, yy, fill_colors=(1.0, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        g_channel = self.create_grid(xx, yy, fill_colors=(0.0, 1.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        b_channel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 1.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        r_output = self.create_grid(xx, yy, fill_colors=(0.8,  0.8, 0.1), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        g_output = self.create_grid(xx, yy, fill_colors=(0.1, 0.8, 0.8), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        b_output = self.create_grid(xx, yy, fill_colors=(0.8, 0.1, 0.8), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        # Initialize the displaying of channels for the first time
        for channel in (r_channel, g_channel, b_channel):
            for cell in channel.grid:
                self.add(cell.square)

        r_channel.shift_grid(y_increment=10)
        g_channel.shift_grid(x_increment=0.5, y_increment=10, z_increment=1)
        b_channel.shift_grid(x_increment=1, y_increment=10, z_increment=2)

        r_output.shift_grid(x_increment= -9, y_increment=0)
        g_output.shift_grid(x_increment=0, y_increment=0, z_increment=1)
        b_output.shift_grid(x_increment=9, y_increment=0, z_increment=2)

        self.wait(1)
        xx = np.arange(-4, -1)
        yy = np.arange(-3, 0)

        r_kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        g_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.5, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        b_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 0.5), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

        for kernel in (r_kernel, g_kernel, b_kernel):
            for cell in kernel.grid:
                self.add(cell.square)

        r_kernel.shift_grid(x_increment=0, y_increment=6, z_increment=0)
        g_kernel.shift_grid(x_increment=0.5, y_increment=6, z_increment=1)
        b_kernel.shift_grid(x_increment=1, y_increment=6, z_increment=2)

        self.wait(1)

        # Separate kernels and channels, overlap everything
        for _ in range(18):
            r_kernel.shift_grid(x_increment=-0.5)
            r_channel.shift_grid(x_increment=-0.5)
            b_kernel.shift_grid(x_increment=0.5)
            b_channel.shift_grid(x_increment=0.5)
            self.wait(0.08)
        for _ in range(16):
            r_kernel.shift_grid(y_increment=0.5)
            g_kernel.shift_grid(y_increment=0.5)
            b_kernel.shift_grid(y_increment=0.5)
            self.wait(0.08)

        self.wait(1)

        # Perform the convolutions
        count = 0
        for _ in range(6):
            for jj in range(6):

                if count < len(b_output.grid):

                    self.add(b_output.grid[count].square)
                    self.add(g_output.grid[count].square)
                    self.add(r_output.grid[count].square)
                    count += 1
                    self.wait(0.35)
                    if jj != 5:
                        r_kernel.shift_grid(x_increment=1)
                        g_kernel.shift_grid(x_increment=1)
                        b_kernel.shift_grid(x_increment=1)

            if count < len(b_output.grid):
                r_kernel.shift_grid(x_increment=-5, y_increment=-1)
                g_kernel.shift_grid(x_increment=-5, y_increment=-1)
                b_kernel.shift_grid(x_increment=-5, y_increment=-1)

        # Put back output everything together kernels and channels, overlap everything
        for _ in range(18):
            r_output.shift_grid(x_increment=0.5)
            b_output.shift_grid(x_increment=-0.5)
            self.wait(0.08)

        self.wait(3)


class RGB_vol2vol(ThreeDSceneSquareGrid):

    def construct(self):
        self.side_length = 0.9

        xx = np.arange(-3, 3)
        yy = np.arange(-3, 3)

        self.move_camera(frame_center=(0 * self.side_length, 5 * self.side_length, 50 * self.side_length))

        r_channel = self.create_grid(xx, yy, fill_colors=(1.0, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        g_channel = self.create_grid(xx, yy, fill_colors=(0.0, 1.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
        b_channel = self.create_grid(xx, yy, fill_colors=(0.0, 0.0, 1.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)


        # Initialize the displaying of channels for the first time
        for channel in (r_channel, g_channel, b_channel):
            for cell in channel.grid:
                self.add(cell.square)

        r_channel.shift_grid(y_increment=10)
        g_channel.shift_grid(x_increment=0.5, y_increment=10, z_increment=1)
        b_channel.shift_grid(x_increment=1, y_increment=10, z_increment=2)


        self.wait(1)



        for n in range(4):

            # Add kernels
            xx = np.arange(-9, -6)
            yy = np.arange(-3, 0)
            r_kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.0, 0.0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
            g_kernel = self.create_grid(xx, yy, fill_colors=(0.0, 0.5, 0), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)
            b_kernel = self.create_grid(xx, yy, fill_colors=(0, 0.0, 0.5), fill_opacities=0.8, stroke_colors=(1.0, 1.0, 1.0), side_length=self.side_length)

            for kernel in (r_kernel, g_kernel, b_kernel):
                for cell in kernel.grid:
                    self.add(cell.square)

            r_kernel.shift_grid(x_increment=0 + 5*n, y_increment=6, z_increment=0)
            g_kernel.shift_grid(x_increment=0.5 + 5*n, y_increment=6, z_increment=1)
            b_kernel.shift_grid(x_increment=1 + 5*n, y_increment=6, z_increment=2)
            self.wait(0.5)

            # Add conv result
            #  conv_result_n = self.create_grid(np.arange(-2, 4), np.arange(-4, 2), fill_colors=(0.2*n, 153/255 + 0.1*n, 76/255 + 0.1*n), side_length=self.side_length)
            conv_result_n = self.create_grid(np.arange(-4, 2), np.arange(-4, 2), fill_colors=(204/255, 204/255 - 0.1*n, 76/255 + 0.1*n), side_length=self.side_length)
            conv_result_n.shift_grid(x_increment=0.5*(n+2), z_increment=1*(n+2))

            for cell in conv_result_n.grid:
                self.add(cell.square)
            self.wait(0.5)

        self.wait(2)

class RGB_vol2vol_2(ThreeDSceneSquareGrid):

    def construct(self):
        self.side_length = 0.9
        self.move_camera(frame_center=(5 * self.side_length, 5 * self.side_length, 60 * self.side_length))

        # Place input volume
        for n in range(4):

            input_volume_n = self.create_grid(np.arange(-4, 2), np.arange(6, 12), fill_colors=(204/255, 204/255 - 0.1*n, 76/255 + 0.1*n), side_length=self.side_length)
            input_volume_n.shift_grid(x_increment=0.5*(n+2), z_increment=1*(n+2))

            for cell in input_volume_n.grid:
                self.add(cell.square)


        self.wait(1)
        for j in range(8):
            # Add kernels
            for n in range(4):
                kernel_n = self.create_grid(np.arange(-20, -17), np.arange(0, 3), fill_colors=(204/255, 204/255 - 0.1*n, 76/255 + 0.1*n), side_length=self.side_length)
                kernel_n.shift_grid(x_increment=0.5*(n+2)+j*6, z_increment=1*(n+2))

                for cell in kernel_n.grid:
                    self.add(cell.square)

            self.wait(0.25)
            # Add output
            output_volume_n = self.create_grid(np.arange(-4, 2), np.arange(-9, -3), fill_colors=(51/255, 1 - 0.1*j, 1 - 0.1*j), side_length=self.side_length)
            output_volume_n.shift_grid(x_increment=0.5*(j+2), z_increment=1*(j+2))

            for cell in output_volume_n.grid:
                self.add(cell.square)

            self.wait(0.5)
        self.wait(2)

class Conv1D(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        xx = np.arange(-5, 5)
        yy = np.arange(0, 1)

        fill_colors = np.random.random((len(xx), 1, 3))

        self.move_camera(phi=0, gamma=0, frame_center=0, distance=0.1)
        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), fill_opacities=1, side_length=self.side_length)

        conv_result = self.create_grid(np.arange(-4,4), np.arange(-3, -2), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)


        kernel = self.create_grid(np.arange(-5, -2), np.arange(2, 3), fill_colors=(1, 1, 0), fill_opacities=0.1, side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)

        kernel.shift_grid(y_increment=-2)
        self.wait(0.5)

        self.add(conv_result.grid[0].square)
        self.wait(1)

        for count in range(1, len(conv_result.grid)):
            self.add(conv_result.grid[count].square)
            kernel.shift_grid(x_increment=1)
            self.wait(0.8)

        self.wait(1)



class Conv2D(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-6, 0)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        conv_result = self.create_grid(np.arange(3, 7), np.arange(-2, 2), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)


        xx = np.arange(-6, -3)
        yy = np.arange(0, 3)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)
        self.move_camera(phi=0, gamma=0, distance=50)

        count = 0
        for _ in range(4):
            for jj in range(4):

                if count < len(conv_result.grid):

                    self.add(conv_result.grid[count].square)
                    count += 1
                    self.wait(0.5)
                    if jj != 3:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-3, y_increment=-1)

        self.wait(2)


class Conv2D_dilated(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-6, 0)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        conv_result = self.create_grid(np.arange(4, 6), np.arange(-1, 1), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)

        # Dilated kernel
        xx = np.arange(-6, 0, step=2)
        yy = np.arange(-2, 4, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)


        self.move_camera(phi=0, gamma=0, distance=50)
        self.wait(1)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(1)

        count = 0
        for _ in range(2):
            for jj in range(2):

                if count < len(conv_result.grid):

                    self.add(conv_result.grid[count].square)
                    count += 1
                    self.wait(0.5)
                    if jj != 1:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-1, y_increment=-1)

        self.wait(2)


class Conv2D_strided(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-6, 0)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        conv_result = self.create_grid(np.arange(3, 7), np.arange(-2, 2), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)


        xx = np.arange(-6, -3)
        yy = np.arange(0, 3)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)
        self.move_camera(phi=0, gamma=0, distance=50)

        count = 0
        for _ in range(4):
            for jj in range(4):

                if count < len(conv_result.grid):

                    self.add(conv_result.grid[count].square)
                    count += 1
                    self.wait(0.5)
                    if jj != 3:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-3, y_increment=-1)

        self.wait(2)

class Conv2D_depth(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-6, 0)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        conv_result = self.create_grid(np.arange(3, 7), np.arange(-2, 2), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        conv_result_2 = self.create_grid(np.arange(3, 7), np.arange(-2, 2), fill_colors=(0, 153/255, 76/255), side_length=self.side_length)
        conv_result_2.shift_grid(x_increment=0.5, y_increment=0.5, z_increment=0.5)

        for cell in simple_grid.grid:
            self.add(cell.square)


        xx = np.arange(-6, -3)
        yy = np.arange(0, 3)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)
        self.move_camera(phi=0, gamma=0, distance=50)

        # First depth conv
        count = 0
        for _ in range(4):
            for jj in range(4):

                if count < len(conv_result.grid):

                    self.add(conv_result.grid[count].square)
                    count += 1
                    self.wait(0.15)
                    if jj != 3:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-3, y_increment=-1)


        self.wait(1)

        kernel.shift_grid(x_increment=-3, y_increment=3)
        kernel.update_colors(fill_colors=(1, 0.5, 0))

        self.wait(2)

        # Second depth conv
        count = 0
        for _ in range(4):
            for jj in range(4):

                if count < len(conv_result.grid):

                    self.add(conv_result_2.grid[count].square)
                    count += 1
                    self.wait(0.15)
                    if jj != 3:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-3, y_increment=-1)

        self.wait(2)

        # Remove kernel
        for cell in kernel.grid:
            self.remove(cell.square)

        # Add a bunch of depth layers

        self.move_camera(phi=0, gamma=0, distance=50, frame_center=(0, 0, 20*self.side_length))
        for n in range(4):
            conv_result_n = self.create_grid(np.arange(3, 7), np.arange(-2, 2), fill_colors=(0.2*n, 153/255 + 0.1*n, 76/255 + 0.1*n), side_length=self.side_length)
            conv_result_n.shift_grid(x_increment=0.5*(n+2), y_increment=0.5*(n+2), z_increment=0.5*(n+2))

            for cell in conv_result_n.grid:
                self.add(cell.square)
            self.wait(0.5)

        self.wait(2)



class Conv2D_stride(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)

        xx = np.arange(-9, 0)
        yy = np.arange(-5, 4)

        stride1 = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)
        stride2 = self.create_grid(xx + 11, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        for cell1, cell2 in zip(stride1.grid, stride2.grid):
            self.add(cell1.square)
            self.add(cell2.square)


        xx = np.arange(-9, -6)
        yy = np.arange(1, 4)

        kernel1 = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)
        kernel2 = self.create_grid(xx + 11, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        for cell1, cell2 in zip(kernel1.grid, kernel2.grid):
            self.add(cell1.square)
            self.add(cell2.square)

        self.wait(2)
        self.move_camera(phi=0, gamma=0, frame_center=(0, 0, 10*self.side_length))

        for jj in range(49):
            #  kk = jj*2

                #  if count < len(conv_result.grid):

                    #  self.add(conv_result.grid[count].square)
                    #  count += 1

            # Stride 1
            if jj % 7 == 0 and jj != 0:
                kernel1.shift_grid(x_increment=-6, y_increment=-1)
            elif jj != 0:
                kernel1.shift_grid(x_increment=1)

            # Stride 2

            if jj % 2 == 0 and jj < 32:
                kk = jj / 2
                if kk % 4 == 0 and kk != 0:
                    kernel2.shift_grid(x_increment=-6, y_increment=-2)
                elif kk != 0:
                    kernel2.shift_grid(x_increment=2)

            self.wait(0.5)
            #  if count < len(conv_result.grid):

        self.wait(2)


class Conv2D_zeropad(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-6, 0)
        yy = np.arange(-3, 3)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)
        conv_result = self.create_grid(np.arange(2, 8), yy, fill_colors=(0, 0.8, 0), side_length=self.side_length)

        xx = np.arange(-7, 1)
        yy = np.arange(-4, 4)

        #  zero_colors = np.zeros((
        zero_pad = self.create_grid(xx, yy, fill_colors=(0.5, 0.5, 0.5))


        for cell in zero_pad.grid:
            self.add(cell.square)

        for cell in simple_grid.grid:
            self.add(cell.square)


        xx = np.arange(-7, -4)
        yy = np.arange(1, 4)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)
        self.move_camera(phi=0, gamma=0, frame_center=(0, 0, 10*self.side_length))

        self.wait(0.5)

        count = 0
        for _ in range(6):
            for jj in range(6):

                if count < len(conv_result.grid):

                    self.add(conv_result.grid[count].square)
                    count += 1
                    self.wait(0.3)
                    if jj != 5:
                        kernel.shift_grid(x_increment=1)

            if count < len(conv_result.grid):
                kernel.shift_grid(x_increment=-5, y_increment=-1)

        self.wait(2)


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


        self.move_camera(frame_center=(14*self.side_length, 14*self.side_length, 50*self.side_length))

        # Define grid + show mnist digit
        mnist_grid = self.create_grid(xx, yy, fill_colors=self.sample_image, fill_opacities=1, side_length=self.side_length)
        for cell in mnist_grid.grid:
            self.add(cell.square)

        xx = np.arange(-1, 2)
        yy = np.arange(14, 17)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)

        kernel = self.create_grid(xx, yy, fill_colors=(0.5, 0.8, 0.4), side_length=self.side_length)

        for cell in kernel.grid:
            self.add(cell.square)

        self.wait(2)

        self.move_camera(phi=0.1*3*PI/8, frame_center=(14*self.side_length, 14*self.side_length, 10*self.side_length))

        self.wait(1)

        for _ in range(10):
            kernel.shift_grid(x_increment=1)
            self.wait(0.1)

        for _ in range(7):
            kernel.shift_grid(x_increment=1)
            self.wait(0.3)

        for _ in range(10):
            kernel.shift_grid(x_increment=1)
            self.wait(0.1)

        self.wait(2)


class Pooling(ThreeDSceneSquareGrid):

    def construct(self):

        self.side_length = 0.9

        self.move_camera(phi=3*PI/8, gamma=0)
        xx = np.arange(-4, 4)
        yy = np.arange(-4, 4)

        simple_grid = self.create_grid(xx, yy, fill_colors=(0, 0, 0.9), side_length=self.side_length)

        pool_result = self.create_grid(np.arange(6, 10), np.arange(-2, 2), fill_colors=(0, 0.8, 0), side_length=self.side_length)

        for cell in simple_grid.grid:
            self.add(cell.square)


        xx = np.arange(-4, -2)
        yy = np.arange(2, 4)

        kernel = self.create_grid(xx, yy, fill_colors=(1, 1, 0), side_length=self.side_length)

        self.move_camera(phi=0, gamma=0, frame_center=(-8, 0, 10))
        self.wait(1)

        for cell in kernel.grid:
            self.add(cell.square)


        self.wait(2)
        #  self.add(conv_result.grid[0].square)

        count = 0
        for _ in range(4):
            for jj in range(4):

                if count < len(pool_result.grid):

                    self.add(pool_result.grid[count].square)
                    count += 1
                    self.wait(0.5)
                    if jj != 3:
                        kernel.shift_grid(x_increment=2)

            if count < len(pool_result.grid):
                kernel.shift_grid(x_increment=-6, y_increment=-2)

            else:
                print("here")
                break
                #  count+=1

        self.wait(2)
