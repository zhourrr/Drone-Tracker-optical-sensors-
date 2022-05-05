# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:17:18 2022

@author: 45242
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

class Ray:
    def __init__(self, origin = np.array([]), direction = np.array([])):
        self.origin = origin
        self.direction = direction
    def pointAtT(self, t):
        return self.origin + self.direction * t

class Camera:
    def __init__(self, width = 1920, height = 1080, front = np.array([0, 0, -1]), up = np.array([0, 1, 0]), cam_center = np.array([0, 0, 0]), offset = 0, angle_degree = 61):
        self.dist_e2s = 0.5 / np.tan(angle_degree * np.pi / 360)
        self.front = front / np.linalg.norm(front)
        self.e = cam_center - offset * self.front
        self.right = np.cross(front, up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.width = width
        self.height = height
    
    def get_ray(self, x, y):
        ### how to count pixel position ? 
        ray_dir = self.front * self.dist_e2s + self.right * (x + 0.5 - self.width / 2) / self.width - self.up * (y + 0.5 - self.height / 2) / self.width
        ray_dir /= np.linalg.norm(ray_dir)
        return Ray(self.e, ray_dir)
    


class Model:
    def __init__(self, cam_list = []):
        self.n_cam = len(cam_list)
        self.cameras = [None] * self.n_cam
        for i in range(len(cam_list)):
            self.cameras[i] = cam_list[i]
            
    def get_coord_basic(self, pixel_pos_tuple_list = []):
        ray_list = [None] * self.n_cam
        for i in range(self.n_cam):
            ray_list[i] = self.cameras[i].get_ray(pixel_pos_tuple_list[i][0],pixel_pos_tuple_list[i][1])
        ### now only consider 2-cam situation
        Ray1 = ray_list[0]
        Ray2 = ray_list[1]
        n = np.cross(Ray1.direction, Ray2.direction)
        n /= np.linalg.norm(n)
        n_plane_1 = np.cross(Ray1.direction, n)
        n_plane_2 = np.cross(Ray2.direction, n)
        t2 = (n_plane_1.dot(Ray1.origin) - n_plane_1.dot(Ray2.origin)) / n_plane_1.dot(Ray2.direction)
        t1 = (n_plane_2.dot(Ray2.origin) - n_plane_2.dot(Ray1.origin)) / n_plane_2.dot(Ray1.direction)
        return (Ray1.pointAtT(t1) + Ray2.pointAtT(t2)) / 2

          
class Locate:
    def __init__(self):
        self.trajectory = []
        self.img_counter = 0
        self.plt_color = ['cx','gx','mx','rx','yx','kx']
        self.color_index = 0
        self.id_color = {}
        self.warnU = warningUnit()
        self.camera_pos = [20,50]
        self.path_step = 0
        
    def locate(self,m,ids):
            
        if (len(ids[0])==0) or (len(ids[1])==0):
            return
        
        #check if the same id appear in both tracker. 
        path = [] 
        for id1 in ids[0]:
            for id2 in ids[1]:
                if  id2[0] == id1[0]:
                    path.append([id1,id2])
                    #give each id a different color for plotting (but we just have 6 colors)
                    if id1[0] not in self.id_color.keys():
                        self.id_color[id1[0]] = self.plt_color[self.color_index%6]
                        self.color_index += 1
                    self.warnU.checkAndAddNew(id1[0])
                    break
        #pos_3d store the position of each object in one frame
        pos_3d = [None]*len(path)
        for i in range(len(path)):
            x0 = path[i][0][1] + path[i][0][3]/ 2
            y0 = path[i][0][2] + path[i][0][4]/ 2
            x1 = path[i][1][1] + path[i][1][3]/ 2
            y1 = path[i][1][2] + path[i][1][4]/ 2
            pos_3d[i] = m.get_coord_basic([(x0, y0), (x1, y1)])
            # if invalid position
            if pos_3d[i][2] >= 0: 
                print("position error")
            else:
                self.trajectory.append([path[i][0][0],pos_3d[i],self.path_step])
                self.warnU.updateXandT(path[i][0][0], pos_3d[i])
            #print(path)
        
        #path_step is the total length of all path
        #print(self.id_color)
        self.path_step += 1
        self.img_show([path,pos_3d])
        self.warnU.checkWarning()
        
    def img_show(self,path_info):
        """
        plot real time positioning 
        """
        path = path_info[1]
        ids = path_info[0]
        if len(path) == 0:
            return
        self.img_counter += 1
        
        if self.img_counter % 50 == 0:
            plt.cla()
        g_xy = plt.subplot(1, 2, 1) 
        plt.plot(-self.camera_pos[0]/2,self.camera_pos[1],"bo")
        plt.plot(self.camera_pos[0]/2,self.camera_pos[1],"bo") 
        
        if len(path) == 1:
            plt.plot(path[0][0],path[0][1],self.id_color[ids[0][0][0]])
            
        elif len(path) == 2:
            plt.plot(path[0][0],path[0][1],self.id_color[ids[0][0][0]])
            plt.plot(path[1][0],path[1][1],self.id_color[ids[1][0][0]])
            
        elif len(path) == 3:
            plt.plot(path[0][0],path[0][1],self.id_color[ids[0][0][0]])
            plt.plot(path[1][0],path[1][1],self.id_color[ids[1][0][0]])
            plt.plot(path[2][0],path[2][1],self.id_color[ids[2][0][0]])
        
        plt.xlim([-100,100])
        plt.ylim([0,200])
        plt.ylabel('height')
        plt.xlabel('x')
        
        if self.img_counter % 50 == 0:
            plt.cla()
        g_xz = plt.subplot(1, 2, 2)
        plt.plot(-self.camera_pos[0]/2,0,"bo")
        plt.plot(self.camera_pos[0]/2,0,"bo")
        
        if len(path) == 1:
            plt.plot(path[0][0],-path[0][2], self.id_color[ids[0][0][0]])
        elif len(path) == 2:  
            plt.plot(path[0][0],-path[0][2], self.id_color[ids[0][0][0]])
            plt.plot(path[1][0],-path[1][2], self.id_color[ids[1][0][0]])
        elif len(path) == 3:
            plt.plot(path[0][0],-path[0][2], self.id_color[ids[0][0][0]])
            plt.plot(path[1][0],-path[1][2], self.id_color[ids[1][0][0]])
            plt.plot(path[2][0],-path[2][2], self.id_color[ids[2][0][0]])            
        
        plt.xlim([-100,100])
        plt.ylim([0,300])
        plt.ylabel('distance')
        plt.xlabel('x')
        plt.draw()   


class queueBoundedX:
    def __init__(self, queue_size = 8):
        self.qX = []
        self.qT = []
        self.fillcount = 0
        self.size = queue_size
        self.newFlag = False
        
    def push(self, x, t):
        if (self.fillcount == self.size):
            self.qX.pop(0)
            self.qT.pop(0)
        else:
            self.fillcount += 1
        self.qX.append(x)
        self.qT.append(t)
        self.newFlag = True
        
    def get_velocity(self):
        #print((self.qX[-1] - self.qX[0]) / (self.qT[-1] - self.qT[0]))
        return (self.qX[-1] - self.qX[0]) / (self.qT[-1] - self.qT[0])

    def setNotNew(self):
        self.newFlag = False
    
    def getNewFlag(self):
        return self.newFlag
        
    def get_position(self):
        return self.qX[-1]
    
class warningUnit:
    def __init__(self, CONST_QUEUE_SIZE = 8, TIME_PREDICT = 5, RADIUS = 50):
        self.CONST_QUEUE_SIZE = CONST_QUEUE_SIZE
        self.TIME_PREDICT_RANGE = TIME_PREDICT
        self.RADIUS = RADIUS
        self.id_q_dict = {}
        
    def checkAndAddNew(self, id1):
        if id1 not in self.id_q_dict.keys():
            self.id_q_dict[id1] = queueBoundedX(self.CONST_QUEUE_SIZE)
    
    def updateXandT(self, id1, x):
        self.id_q_dict[id1].push(x, time.time())
    
    def checkWarning(self):
        vec_pos_List = []
        for id1 in self.id_q_dict:
            if self.id_q_dict[id1].getNewFlag():
                vec_pos_List.append((id1, self.id_q_dict[id1].get_velocity(), self.id_q_dict[id1].get_position())) 
                self.id_q_dict[id1].setNotNew()
        warning_pair_list = []
        if(len(vec_pos_List) > 1):
            for i in range(len(vec_pos_List) - 1):
                for j in range(i + 1, len(vec_pos_List)):
                    rel_pos = vec_pos_List[j][2] - vec_pos_List[i][2]
                    rel_vt = (vec_pos_List[j][1] - vec_pos_List[i][1]) * self.TIME_PREDICT_RANGE
                    rel_vt_value = np.linalg.norm(rel_vt)
                    v_dir = rel_vt / rel_vt_value
                    proj_pos_vt_dist = rel_pos.dot(v_dir)
                    if proj_pos_vt_dist >= 0:
                        continue
                    min_dist = 0
                    proj_pos_vt_dist *= -1
                    if proj_pos_vt_dist > rel_vt_value:
                        min_dist = np.linalg.norm(rel_vt + rel_pos)
                    else:
                        min_dist = np.linalg.norm(v_dir * proj_pos_vt_dist + rel_pos)
                    if min_dist < self.RADIUS:
                        warning_pair_list.append((vec_pos_List[i][0], vec_pos_List[j][0]))
        for warning in warning_pair_list:
            print("Warning of Collision for: %d and %d." % (warning[0], warning[1]))

'''

Cam1 = Camera(cam_center = np.array([0, 20, 0]))
Cam2 = Camera(cam_center = np.array([55, 20, 0]))

m = Model([Cam1, Cam2])

pos_3d = m.get_coord_basic([(440,193), (200,193)])
        
'''     
    

    




