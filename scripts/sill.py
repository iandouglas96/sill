#!/usr/bin/env python3

import argparse
import numpy as np
from vispy import app, scene
from vispy.color import Color, ColorArray
from queue import Queue
from threading import Thread
from integrated_cloud import IntegratedCloud
from time import perf_counter, time

class SillCanvas(scene.SceneCanvas):
    def __init__(self, bagpath, start_ind = 0, load = False):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()

        self.view_ = self.central_widget.add_view(bgcolor='white')
        self.cloud_ = IntegratedCloud(bagpath, start_ind, load)
        self.cloud_render_ = {}
        self.last_mouse_point_ = np.zeros(2)
        self.current_class_ = 1
        self.index_ = start_ind
        self.updated_z_ = True

        self.text_ = scene.visuals.Text("Status",
                                        color='black',
                                        anchor_x='left',
                                        parent=self.view_,
                                        pos=(20, 50))
        self.update_text()

        self.cursor_size_ = 50
        self.cursor_ = scene.visuals.Ellipse(center = (0, 0), radius = (self.cursor_size_,)*2, 
                color=Color(alpha=0), border_color=Color('black'), border_width=5, parent=self.view_)

        # disconnect camera, cannot pan/zoom by default
        self.view_.camera = scene.PanZoomCamera(aspect=1)
        self.view_.camera._viewbox.events.mouse_move.disconnect(
            self.view_.camera.viewbox_mouse_event)
        self.mode_ = 'label'

        self.axis_ = scene.visuals.XYZAxis(parent=self.view_.scene)
        self.redraw()
        self.show()
        self.freeze()

    def redraw(self, pt=None):
        if pt is None:
            self.cloud_.adjust_z(0, update=True)
            for ind in self.cloud_.get_render_indices():
                if ind not in self.cloud_render_.keys():
                    new_cloud_render = scene.visuals.Markers(edge_width = 0)
                    # settings to make things render properly
                    new_cloud_render.antialias = 0
                    new_cloud_render.update_gl_state(depth_test = False)
                    self.view_.add(new_cloud_render)
                    self.cloud_render_[ind] = new_cloud_render

                self.cloud_render_[ind].set_data(self.cloud_.cloud(ind), 
                        edge_color=None, edge_width=0, face_color=self.cloud_.colors(ind), size=5)
            self.updated_z_ = True
            self.update_text()
        else:
            inds = self.cloud_.get_render_block_neighborhood(pt)
            for ind in inds:
                if ind in self.cloud_render_.keys():
                    self.cloud_render_[ind].set_data(self.cloud_.cloud(ind), 
                            edge_color=None, edge_width=0, face_color=self.cloud_.colors(ind), size=5)

    def update_text(self):
        self.text_.text = f"Current class: {self.current_class_}: {self.cloud_.class_lut_[self.current_class_]}\n" + \
                          f"Current elevation: {self.cloud_.get_z()}\n" + \
                          f"Updated elev: {self.updated_z_}\n" + \
                          f"Index: {self.index_}"

    def set_class(self, cls):
        if cls < len(self.cloud_.class_lut_):
            self.current_class_ = cls
            self.update_text()

    def on_key_press(self, event):
        if event.key == 'R':
            self.redraw()
        elif event.key == 'N':
            for _ in range(10):
                self.cloud_.add_new()    
            self.index_ += 10
            self.redraw()
            self.update_text()
        elif event.key == 'Space' and self.mode_ == 'label':
            self.mode_ = 'view2' 
            self.view_.camera._viewbox.events.mouse_move.connect(
                self.view_.camera.viewbox_mouse_event)
        elif event.key == 'W':
            self.cloud_.write()
        elif event.key == 'X':
            self.cloud_.reset()
            # clear screen
            for key in self.cloud_render_.keys():
                self.cloud_render_[key].parent = None
            self.redraw()
            self.cloud_render_ = {}
        elif event.key == 'F':
            self.cloud_.voxel_filter()
            self.redraw()
        elif event.key == 'PageUp':
            self.cloud_.adjust_z(0.5, update=False)
            self.updated_z_ = False
            self.update_text()
        elif event.key == 'PageDown':
            self.cloud_.adjust_z(-0.5, update=False)
            self.updated_z_ = False
            self.update_text()
        elif event.key == 'O':
            self.mode_ = 'view3'
            self.view_.camera = scene.TurntableCamera()
        elif event.key == 'L':
            self.mode_ = 'label'
            self.view_.camera = scene.PanZoomCamera(aspect=1)
            self.view_.camera._viewbox.events.mouse_move.disconnect(
                self.view_.camera.viewbox_mouse_event)
        elif event.text.isnumeric():
            self.set_class(int(event.text))

    def on_key_release(self, event):
        if event.key == 'Space' and self.mode_ == 'view2':
            self.mode_ = 'label'
            self.view_.camera._viewbox.events.mouse_move.disconnect(
                self.view_.camera.viewbox_mouse_event)

    def on_mouse_move(self, event):
        # only care if click and drag
        if event.button == 1 and self.mode_ == 'label':
            tr = self.scene.node_transform(self.axis_)
            pos = tr.map(event.pos)[:2]
            eps = np.linalg.norm(tr.map(event.pos + np.array([self.cursor_size_, 0]))[:2] - pos)

            if np.linalg.norm(pos - self.last_mouse_point_) > eps*0.2:
                start_t = perf_counter()
                self.cloud_.label(pos, self.current_class_, eps)
                label_t = perf_counter()
                self.redraw(pos)
                redraw_t = perf_counter()
                
                print(f"label: {label_t - start_t}")
                print(f"render: {redraw_t - label_t}")

                self.last_mouse_point_ = pos

        self.cursor_.center = event.pos[:2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    sc = SillCanvas(args.bag, args.start, args.load)
    app.run()
