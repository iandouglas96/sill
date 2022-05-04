import numpy as np
from dataloader import DataLoader
from vispy.color import Color, ColorArray

class IntegratedCloud:
    def __init__(self, bagpath):
        self.loader_ = DataLoader(bagpath).__iter__()
        self.cloud_ = np.empty([0, 4], dtype=np.float32)
        self.labels_ = np.empty([0, 1], dtype=np.int8)
        self.colors_ = None
        self.add_new()

    def add_new(self):
        pc, pose, img, info = self.loader_.__next__()
        # transform cloud
        pc_trans = pc.copy()
        pc_trans[:, :3] = pose['R'].apply(pc[:, :3]) + pose['T']
        self.cloud_ = np.vstack((self.cloud_, pc_trans))
        self.labels_ = np.vstack((self.labels_, np.zeros((self.cloud_.shape[0], 1))))
        new_colors = ColorArray(np.repeat(pc[:, 3, None]/1000, 3, axis=1), alpha=1)

        if self.colors_ is None:
            self.colors_ = new_colors
        else:
            self.colors_.extend(new_colors)

    def cloud(self):
        return self.cloud_[:, :3]

    def colors(self):
        return self.colors_
