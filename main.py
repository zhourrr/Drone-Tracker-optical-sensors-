import cv2
import numpy as np
import matplotlib.pyplot as plt
from detector import *
from posi import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from animation import *


Cam1 = Camera(cam_center=np.array([0, 50, 0]), angle_degree=40)
Cam2 = Camera(cam_center=np.array([14, 50, 0]), angle_degree=40)
my_ins = MyDetector(captures=["test.mp4", "test1.mp4"], cameras=[Cam1, Cam2], wt=40)
my_ins.detect()


# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_Line(my_ins.trajectory, 3)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-25, 75])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 100])
ax.set_ylabel('Z')

ax.set_zlim3d([0.0, 100])
ax.set_zlabel('Height')

ax.set_title('3D Test')

# Creating the Animation object
'''
interval is the speed of each step
'''
line_ani = animation.FuncAnimation(fig, update_lines, len(my_ins.trajectory_opt), fargs=(data, lines),
                                   interval=200, blit=False)

plt.show()