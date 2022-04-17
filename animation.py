# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:17:57 2022

@author: hbomb
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def Gen_Line(path, dims=2):
    """
    Create a line

    path - a list which contains 3D position
    dims is the number of dimensions the line has
    attention: the xyz axis of the graph is not corresponding to xyz value of the position
    z-:  front
    y+:  height
    x+:  
    """
    lineData = np.empty((dims, len(path)))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(len(path)):
        lineData[0, index] = path[index][0]
        lineData[1, index] = -path[index][2]
        lineData[2, index] = path[index][1]
    print(lineData)

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

