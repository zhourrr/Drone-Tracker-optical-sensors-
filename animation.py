# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:17:57 2022

@author: hbomb
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def decompress_path(path_info,step_num):
    path_len = {}
    path_s = {}
    path_e = {}
    path_t = {}
    for pos in path_info:
        if pos[0] not in path_t.keys():
            path_len[pos[0]] = 0
            path_s[pos[0]] = [pos[1],pos[2]]
            path_e[pos[0]] = [pos[1],pos[2]]
            path_t[pos[0]] = {}
        else:
            path_len[pos[0]] += 1
            path_e[pos[0]] = [pos[1],pos[2]]
            path_t[pos[0]][pos[2]] = pos[1]
            
    #print(self.path_len)
    #print(self.path_t)
    
    dele = []
    for ids in path_len.keys():
        if path_len[ids] < 20:
            dele.append(ids)
        
    for i in dele:
        del path_len[i]
        del path_s[i]
        del path_e[i]
        del path_t[i]
    return [path_s,path_e,path_t]


def Gen_Line(path,step_num,idv,dims=2):
    """
    Create a line

    path - a list which contains 3D position
    dims is the number of dimensions the line has
    attention: the xyz axis of the graph is not corresponding to xyz value of the position
    z-:  front
    y+:  height
    x+:  
    """
    path_s = path[0]
    path_e = path[1]
    path_t = path[2]
    lineData = np.zeros((dims, step_num))
    temp_pos = path_s[idv][0]
    #lineData[:, 0] = np.random.rand(dims)
    for index in range(step_num):
        if index <= path_s[idv][1]:
            lineData[0, index] = path_s[idv][0][0]
            lineData[1, index] = -path_s[idv][0][2]
            lineData[2, index] = path_s[idv][0][1]            
        elif index in path_t[idv].keys():            
            lineData[0, index] = path_t[idv][index][0]
            lineData[1, index] = -path_t[idv][index][2]
            lineData[2, index] = path_t[idv][index][1]
            temp_pos = path_t[idv][index]
        else:
            lineData[0, index] = temp_pos[0]
            lineData[1, index] = -temp_pos[2]
            lineData[2, index] = temp_pos[1]   
            
    #print(lineData)
    return lineData



def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


