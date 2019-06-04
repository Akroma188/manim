from manimlib.imports import *

import numpy as np

from colour import Color

from torchvision.datasets import MNIST


class ConvNet(ThreeDScene):

    def construct(self):

        self.side_length = 0.9

        xx = np.arange(-3,3)
        yy = np.arange(-3,3)

        grid = []

        #  self.set_camera_orientation(distance=100)
        for x in xx:
            for y in yy:
                temp_square = Square(side_length=self.side_length, fill_color=Color(rgb=(0.1,0.22,0.3)), fill_opacity=1, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*self.side_length)
                temp_square.set_y(y*self.side_length)
                self.add(temp_square)
                grid.append(temp_square)


        self.move_camera(phi=3*PI/8,gamma=0)

        kernel = []
        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)
        for x in xx:
            for y in yy:
                temp_square = Square(side_length=self.side_length, fill_color=Color(rgb=(0.7,0.72,0.3)), fill_opacity=0.5, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*self.side_length)
                temp_square.set_y(y*self.side_length)
                temp_square.set_z(3*self.side_length)
                self.add(temp_square)
                kernel.append(temp_square)
        self.wait(2)

        for _ in range(3):
            shift_kernel(kernel, step_size=self.side_length, x_increment=1)
            self.wait(1)

        self.move_camera(phi=0,gamma=0, distance=50)
        shift_kernel(kernel, step_size=self.side_length, x_increment=-3, y_increment=1)
        self.wait(1)

        for _ in range(3):
            shift_kernel(kernel, step_size=self.side_length, x_increment=1)
            self.wait(1)
        self.begin_ambient_camera_rotation()


class MnistConvNet(ThreeDScene):

    def construct(self):

        self.shift_kernel = shift_kernel
        self.mnist_dataset = MNIST(root='../data',
                              train=True,
                              download=True)

        self.sample_image = np.array(self.mnist_dataset[17][0])

        self.side_length = 0.4

        xx = np.arange(0,28)
        yy = np.arange(0,28)


        self.move_camera(phi=0, theta=0, frame_center=(14*self.side_length, 14*self.side_length, 40*self.side_length))

        # Define meshgrid + show mnist digit
        grid = []
        for x in xx:
            for y in yy:
                intensity = self.sample_image[x,y] / 255
                temp_square = Square(side_length=self.side_length, fill_color=Color(rgb=(intensity, intensity, intensity)), fill_opacity=1, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*self.side_length)
                temp_square.set_y(y*self.side_length)
                self.add(temp_square)
                grid.append(temp_square)

        # Define kernel used for convolution
        kernel = []
        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)
        for x in xx:
            for y in yy:
                temp_square = Square(side_length=self.side_length, fill_color=Color(rgb=(0.7,0.72,0.3)), fill_opacity=0.6, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*self.side_length)
                temp_square.set_y(y*self.side_length)
                temp_square.set_z(5*self.side_length)
                self.add(temp_square)
                kernel.append(temp_square)
        self.wait(2)


        for _ in range(28):
            shift_kernel(kernel, step_size=self.side_length, y_increment=1)
            self.wait(0.1)

        shift_kernel(kernel, step_size=self.side_length, x_increment=1, y_increment=-14)
        shift_kernel(kernel, step_size=self.side_length, x_increment=14)
        self.wait(1)
        self.move_camera(phi=PI/6, theta=0, frame_center=(14*self.side_length, 14*self.side_length, 1*self.side_length))
        for _ in range(10):
            shift_kernel(kernel, step_size=self.side_length, y_increment=1)
            self.wait(0.1)
        self.wait(2)


def shift_kernel(kernel, step_size, x_increment=None, y_increment=None):
    for k in kernel:
        if x_increment:
            current_x = k.get_x()
            k.set_x(current_x+x_increment*step_size)
        if y_increment:
            current_y = k.get_y()
            k.set_y(current_y+y_increment*step_size)
