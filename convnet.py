from manimlib.imports import *

import numpy as np

from colour import Color


class ConvNet(ThreeDScene):

    def construct(self):

        side_length = 0.9

        xx = np.arange(-3,3)
        yy = np.arange(-3,3)

        grid = []

        #  self.set_camera_orientation(distance=100)
        for x in xx:
            for y in yy:
                temp_square = Square(side_length=side_length, fill_color=Color(rgb=(0.1,0.22,0.3)), fill_opacity=1, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*side_length)
                temp_square.set_y(y*side_length)
                self.add(temp_square)
                #  self.play(GrowFromCenter(temp_square))
                grid.append(temp_square)


        #  self.move_camera(phi=PI/4,gamma=PI/2, distance=50)
        self.move_camera(phi=3*PI/8,gamma=0)

        kernel = []
        xx = np.arange(-3, 0)
        yy = np.arange(-3, 0)

        # Dilated kernel
        #  xx = np.arange(-3, 3, step=2)
        #  yy = np.arange(-3, 3, step=2)
        for x in xx:
            for y in yy:
                temp_square = Square(side_length=side_length, fill_color=Color(rgb=(0.7,0.72,0.3)), fill_opacity=0.5, stroke_color=Color(rgb=(1.0, 1.0, 1.0)))
                temp_square.set_x(x*side_length)
                temp_square.set_y(y*side_length)
                temp_square.set_z(3*side_length)
                self.add(temp_square)
                #  self.play(GrowFromCenter(temp_square))
                kernel.append(temp_square)
        self.wait(2)

        def shift_kernel(kernel, x_increment=None, y_increment=None):
            for k in kernel:
                if x_increment:
                    current_x = k.get_x()
                    k.set_x(current_x+x_increment*side_length)
                if y_increment:
                    current_y = k.get_y()
                    k.set_y(current_y+y_increment*side_length)

        for _ in range(3):
            shift_kernel(kernel, x_increment=1)
            self.wait(1)

        self.move_camera(phi=0,gamma=0, distance=50)
        shift_kernel(kernel, x_increment=-3, y_increment=1)
        self.wait(1)
        for _ in range(3):
            shift_kernel(kernel, x_increment=1)
            self.wait(1)
        self.begin_ambient_camera_rotation()
