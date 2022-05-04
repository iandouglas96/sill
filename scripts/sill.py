#!/usr/bin/env python3

import argparse
import numpy as np
from vispy import app, scene
from queue import Queue
from threading import Thread
from dataloader import DataLoader

class SillCanvas(scene.SceneCanvas):
    def __init__(self, bagpath):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()

        self.loader_ = DataLoader(bagpath)
        self.pc_q_ = pc_q
        self.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    args = parser.parse_args()

    sc = SillCanvas(args.bag)
    app.run()
