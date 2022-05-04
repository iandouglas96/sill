#!/usr/bin/env python3

import argparse
import numpy as np
from vispy import app, scene
from queue import Queue
from threading import Thread
from integrated_cloud import IntegratedCloud

class SillCanvas(scene.SceneCanvas):
    def __init__(self, bagpath):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()

        self.view_ = self.central_widget.add_view(bgcolor='white')
        self.cloud_ = IntegratedCloud(bagpath)
        self.cloud_render_ = scene.visuals.Markers()
        # dummy data or cannot add to render
        self.cloud_render_.set_data(np.zeros((1, 3)))
        # bug with borders and antialiasing, see https://github.com/vispy/vispy/issues/1583
        self.cloud_render_.antialias = 0
        self.view_.add(self.cloud_render_)

        #self.view_.camera = scene.PanZoomCamera(aspect=1)
        self.view_.camera = 'turntable'
        axis = scene.visuals.XYZAxis(parent=self.view_.scene)
        self.redraw()
        self.show()
        self.freeze()

    def redraw(self):
        self.cloud_render_.set_data(self.cloud_.cloud(), 
                edge_color=None, edge_width=0, face_color=self.cloud_.colors(), size=5)

    def on_key_press(self, event):
        if event.key == 'N':
            self.cloud_.add_new()    
            self.redraw()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    args = parser.parse_args()

    sc = SillCanvas(args.bag)
    app.run()
