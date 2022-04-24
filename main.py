import cv2
import numpy as np
import matplotlib.pyplot as plt
from detector import *
from posi import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from animation import *
from coordinator import *

Cam1 = Camera(cam_center=np.array([-10, 50, 0]), angle_degree=100)
Cam2 = Camera(cam_center=np.array([10, 50, 0]), angle_degree=100)
# detector = MyDetector(captures=[0, 1], cameras=[Cam1, Cam2], wt=30)
# coordinator = Coordinator(detector)
# detector.tracker_init(coordinator)
# detector.detect()
detector = MyDetector(captures=[0, 1], cameras=[Cam1, Cam2], wt=30)
coordinator = Coordinator(detector)
detector.tracker_init(coordinator)
detector.init_background()

pos_model = Model([Cam1, Cam2])
locate = Locate()
fig = plt.figure(figsize=(10, 5))

while True:
    res = detector.detect()
    if not res:
        break
    else:
        locate.locate(pos_model,res)
                
        #print(res)



### animation


#decompress the path for animation
path = decompress_path(locate.trajectory,locate.path_step)


fig = plt.figure(figsize=(8, 8))
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
print(path[0].keys())
data = [Gen_Line(path,locate.path_step,idv,3) for idv in path[0].keys()]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-25, 25])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 100])
ax.set_ylabel('Z')

ax.set_zlim3d([0.0, 100])
ax.set_zlabel('Height')

ax.set_title('3D Test')

# Creating the Animation object
#interval is the speed of each step

line_ani = animation.FuncAnimation(fig, update_lines, locate.path_step, fargs=(data, lines),
                                    interval=80, blit=False)

plt.show()
