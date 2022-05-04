#!/usr/bin/env python3

import numpy as np
from vispy import app, scene
from queue import Queue
from threading import Thread
from sill_ros import SillROS

class SillCanvas(scene.SceneCanvas):
    def __init__(self, pc_q):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()

        self.pc_q_ = pc_q
        self.show()

if __name__ == '__main__':
    pc_q = Queue()
    sr = SillROS(pc_q)
    ros_thread = Thread(target=sr.spin)
    ros_thread.start()
    sc = SillCanvas(pc_q)
    app.run()
