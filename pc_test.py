#!/usr/bin/env python3
# demo from https://vispy.org/gallery/scene/point_cloud.html#sphx-glr-gallery-scene-point-cloud-py
import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app, scene

class Canvas(scene.SceneCanvas):
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()
        #
        # Make a canvas and add simple view
        #
        self.view = self.central_widget.add_view()

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
        # initialize all white
        colors = vispy.color.ColorArray(np.ones((pos.shape[0], 3)))
        colors[:50000] = vispy.color.Color('red')

        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(pos, edge_color=None, face_color=colors, size=5)

        self.view.add(scatter)

        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        self.view.camera._viewbox.events.mouse_move.disconnect(
            self.view.camera.viewbox_mouse_event)
        self.view.camera._viewbox.events.mouse_wheel.disconnect(
            self.view.camera.viewbox_mouse_event)

        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=self.view.scene)
        self.show()

        #color = 0
        #def update(ev):
        #    global scatter, color
        #    scatter.set_data(pos, edge_color=None, face_color=(1, color, 1, .5), size=5)
        #    color += 0.01
        #    if color > 1:
        #        color = 0

        #timer = app.Timer()
        #timer.connect(update)
        #timer.start(0)
        self.freeze()

    def on_mouse_move(self, event):
        # only care if click and drag
        if event.button == 1:
            self.view.camera._viewbox.events.mouse_move.connect(
                self.view.camera.viewbox_mouse_event)
            print("mouse drag")

    def on_mouse_wheel(self, event):
        self.view.camera._viewbox.events.mouse_wheel.connect(
            self.view.camera.viewbox_mouse_event)
        print("scroll delta: " + str(event.delta))

    def on_key_press(self, event):
        print("key: " + event.text)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        canvas = Canvas()
        vispy.app.run()

