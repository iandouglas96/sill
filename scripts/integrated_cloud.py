import numpy as np
from dataloader import DataLoader
from vispy.color import Color, ColorArray

class IntegratedCloud:
    def __init__(self, bagpath):
        self.loader_ = DataLoader(bagpath).__iter__()
        self.cloud_ = np.empty([0, 4], dtype=np.float32)
        # label, elevation when labelled
        self.labels_ = np.empty([0, 2], dtype=np.float32)
        self.render_block_indices_ = np.empty([0, 1], dtype=np.int32)
        self.block_size_ = 1
        self.colors_ = None
        self.target_z_ = 0
        self.add_new()

    def add_new(self):
        pc, pose, img, info = self.loader_.__next__()
        # transform cloud
        pc_trans = pc.copy()
        pc_trans[:, :3] = pose['R'].apply(pc[:, :3]) + pose['T']
        self.cloud_ = np.vstack((self.cloud_, pc_trans))
        self.labels_ = np.vstack((self.labels_, -1000*np.ones((pc.shape[0], 2))))
        new_colors = ColorArray(np.repeat(np.clip(pc[:, 3, None]/1000, 0, 1), 3, axis=1), alpha=1)

        self.render_block_indices_ = np.vstack((self.render_block_indices_, 
            self.get_block_ind(pc_trans[:, :2])[:, None]))

        if self.colors_ is None:
            self.colors_ = new_colors
        else:
            self.colors_.extend(new_colors)

        self.adjust_z(0)

    def get_block_ind(self, pts):
        inds = (pts / self.block_size_).astype(np.int32)
        return inds[:,0] + inds[:,1] << 16

    def get_block_neighborhood(self, pt):
        pts = np.stack((pt,
                        pt + np.array([0, self.block_size_]),
                        pt + np.array([self.block_size_, 0]),
                        pt + np.array([0, -self.block_size_]),
                        pt + np.array([-self.block_size_, 0])))
        return self.get_block_ind(pts)

    def get_indices(self):
        return np.unique(self.render_block_indices_)

    def adjust_z(self, delta):
        self.target_z_ += delta
        visible = self.cloud_[:, 2] <= self.target_z_
        labelled_below = np.logical_and(self.labels_[:, 1] < self.target_z_, self.labels_[:, 0] > 0)
        if np.any(visible):
            self.colors_[visible] = ColorArray(self.colors_[visible], alpha=1)
        if np.any(labelled_below):
            self.colors_[labelled_below] = ColorArray(self.colors_[labelled_below], alpha=0.3)
        if not np.all(visible):
            self.colors_[np.invert(visible)] = ColorArray(self.colors_[np.invert(visible)], alpha=0)

    def label(self, pos, label, eps=1):
        to_update = np.logical_and(
                np.logical_and(self.cloud_[:, 2] > self.labels_[:, 1], self.cloud_[:, 2] <= self.target_z_),
                np.linalg.norm(self.cloud_[:, :2] - pos, axis=1) < eps)
        if np.any(to_update):
            self.labels_[to_update] = np.array([label, self.target_z_])
            self.colors_[to_update] = Color('red')

    def cloud(self, block):
        return self.cloud_[self.render_block_indices_[:,0] == block, :2]

    def colors(self, block):
        return self.colors_[self.render_block_indices_[:,0] == block]
