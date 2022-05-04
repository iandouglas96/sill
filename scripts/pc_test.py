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
        self.pos = np.random.normal(size=(100000, 3), scale=0.2)
        # one could stop here for the data generation, the rest is just to make the
        # data look more interesting. Copied over from magnify.py
        centers = np.random.normal(size=(50, 3))
        indexes = np.random.normal(size=100000, loc=centers.shape[0]/2.,
                                   scale=centers.shape[0]/3.)
        indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
        scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
        self.pos *= scales
        self.pos += centers[indexes]
        # initialize all white
        self.colors = vispy.color.ColorArray(np.ones((self.pos.shape[0], 3)), alpha=1)

        self.target_z = 0
        self.pan_zoom_mode = False
        self.compute_visible()

        # create scatter object and fill in the data
        self.scatter = visuals.Markers()
        self.redraw()

        self.view.add(self.scatter)

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
        #    scatter.set_data(self.pos, edge_color=None, face_color=(1, color, 1, .5), size=5)
        #    color += 0.01
        #    if color > 1:
        #        color = 0

        #timer = app.Timer()
        #timer.connect(update)
        #timer.start(0)
        self.freeze()

    def compute_visible(self):
        visible = self.pos[:, 2] <= self.target_z
        if np.any(visible):
            self.colors[visible] = vispy.color.ColorArray(self.colors[visible], alpha=1)
        if not np.all(visible):
            self.colors[np.invert(visible)] = vispy.color.ColorArray(self.colors[np.invert(visible)], alpha=0)

    def redraw(self):
        self.scatter.set_data(self.pos, edge_color=None, face_color=self.colors, size=5)

    def label(self, pos, eps):
        in_range = np.linalg.norm(self.pos[:, :2] - pos, axis=1) < eps
        if np.any(in_range):
            self.colors[in_range] = vispy.color.Color('red')

    def on_mouse_move(self, event):
        # only care if click and drag
        if event.button == 1 and not self.pan_zoom_mode:
            tr = self.scene.node_transform(self.scatter)
            pos = tr.map(event.pos)
            self.label(pos[:2], eps=0.1)
            self.compute_visible()
            self.redraw()

    def on_mouse_wheel(self, event):
        if not self.pan_zoom_mode:
            self.target_z -= event.delta[1]*0.1
            self.compute_visible()
            self.redraw()

    def on_key_press(self, event):
        if event.key == 'M':
            self.pan_zoom_mode = True
            self.view.camera._viewbox.events.mouse_move.connect(
                self.view.camera.viewbox_mouse_event)
            self.view.camera._viewbox.events.mouse_wheel.connect(
                self.view.camera.viewbox_mouse_event)

    def on_key_release(self, event):
        if event.key == 'M':
            self.pan_zoom_mode = False
            self.view.camera._viewbox.events.mouse_move.disconnect(
                self.view.camera.viewbox_mouse_event)
            self.view.camera._viewbox.events.mouse_wheel.disconnect(
                self.view.camera.viewbox_mouse_event)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        canvas = Canvas()
        vispy.app.run()

