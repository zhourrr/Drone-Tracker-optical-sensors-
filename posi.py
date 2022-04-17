# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:17:18 2022

@author: 45242
"""

import numpy as np

class Ray:
    def __init__(self, origin = np.array([]), direction = np.array([])):
        self.origin = origin
        self.direction = direction
    def pointAtT(self, t):
        return self.origin + self.direction * t

class Camera:
    def __init__(self, width = 480, height = 640, front = np.array([0, 0, -1]), up = np.array([0, 1, 0]), cam_center = np.array([0, 0, 0]), offset = 0, angle_degree = 61):
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

            

Cam1 = Camera(cam_center = np.array([0, 20, 0]))
Cam2 = Camera(cam_center = np.array([55, 20, 0]))

m = Model([Cam1, Cam2])

pos_3d = m.get_coord_basic([(440,193), (200,193)])
        
        
    
    
    




