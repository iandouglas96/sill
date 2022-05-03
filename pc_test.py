#!/usr/bin/env python3
# demo from https://vispy.org/gallery/scene/point_cloud.html#sphx-glr-gallery-scene-point-cloud-py
import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# generate data
pos = np.random.normal(size=(100000, 3), scale=0.2)
# one could stop here for the data generation, the rest is just to make the
# data look more interesting. Copied over from magnify.py
centers = np.random.normal(size=(50, 3))
indexes = np.random.normal(size=100000, loc=centers.shape[0]/2.,
                           scale=centers.shape[0]/3.)
indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
pos *= scales
pos += centers[indexes]

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)

view.camera = vispy.scene.PanZoomCamera(aspect=1)

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

color = 0
def update(ev):
    global scatter, color
    scatter.set_data(pos, edge_color=None, face_color=(1, color, 1, .5), size=5)
    color += 0.01
    if color > 1:
        color = 0

timer = app.Timer()
timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()

