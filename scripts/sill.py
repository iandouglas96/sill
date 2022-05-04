#!/usr/bin/env python3

import argparse
import numpy as np
from vispy import app, scene
from queue import Queue
from threading import Thread
from integrated_cloud import IntegratedCloud
from time import perf_counter, time

class SillCanvas(scene.SceneCanvas):
    def __init__(self, bagpath):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()

        self.view_ = self.central_widget.add_view(bgcolor='white')
        self.cloud_ = IntegratedCloud(bagpath)
        self.cloud_render_ = {}
        self.last_mouse_point_ = np.zeros(2)

        # disconnect camera, cannot pan/zoom by default
        self.view_.camera = scene.PanZoomCamera(aspect=1)
        self.view_.camera._viewbox.events.mouse_move.disconnect(
            self.view_.camera.viewbox_mouse_event)
        self.view_.camera._viewbox.events.mouse_wheel.disconnect(
            self.view_.camera.viewbox_mouse_event)
        self.pan_zoom_mode_ = False

        self.axis_ = scene.visuals.XYZAxis(parent=self.view_.scene)
        self.redraw()
        self.show()
        self.freeze()

    def redraw(self, pt=None):
        if pt is None:
            for ind in self.cloud_.get_indices():
                if ind not in self.cloud_render_.keys():
                    new_cloud_render = scene.visuals.Markers(edge_width = 0)
                    # settings to make things render properly
                    new_cloud_render.antialias = 0
                    new_cloud_render.update_gl_state(depth_test = False)
                    self.view_.add(new_cloud_render)
                    self.cloud_render_[ind] = new_cloud_render

                self.cloud_render_[ind].set_data(self.cloud_.cloud(ind), 
                        edge_color=None, edge_width=0, face_color=self.cloud_.colors(ind), size=5)
        else:
            inds = self.cloud_.get_block_neighborhood(pt)
            for ind in inds:
                if ind in self.cloud_render_.keys():
                    self.cloud_render_[ind].set_data(self.cloud_.cloud(ind), 
                            edge_color=None, edge_width=0, face_color=self.cloud_.colors(ind), size=5)

    def on_key_press(self, event):
        if event.key == 'R':
            self.redraw()
        elif event.key == 'N':
            for _ in range(10):
                self.cloud_.add_new()    
            self.redraw()
        elif event.key == 'M':
            self.pan_zoom_mode_ = True
            self.view_.camera._viewbox.events.mouse_move.connect(
                self.view_.camera.viewbox_mouse_event)
            self.view_.camera._viewbox.events.mouse_wheel.connect(
                self.view_.camera.viewbox_mouse_event)

    def on_key_release(self, event):
        if event.key == 'M':
            self.pan_zoom_mode_ = False
            self.view_.camera._viewbox.events.mouse_move.disconnect(
                self.view_.camera.viewbox_mouse_event)
            self.view_.camera._viewbox.events.mouse_wheel.disconnect(
                self.view_.camera.viewbox_mouse_event)

    def on_mouse_wheel(self, event):
        if not self.pan_zoom_mode_:
            self.cloud_.adjust_z(-event.delta[1]*0.1)

    def on_mouse_move(self, event):
        # only care if click and drag
        if event.button == 1 and not self.pan_zoom_mode_:
            tr = self.scene.node_transform(self.axis_)
            pos = tr.map(event.pos)[:2]
            if np.linalg.norm(pos - self.last_mouse_point_) > 1:
                start_t = perf_counter()
                self.cloud_.label(pos, 1)
                label_t = perf_counter()
                self.redraw(pos)
                redraw_t = perf_counter()
                
                print(f"label: {label_t - start_t}")
                print(f"render: {redraw_t - label_t}")

                self.last_mouse_point_ = pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    args = parser.parse_args()

    sc = SillCanvas(args.bag)
    app.run()
