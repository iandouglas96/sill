import numpy as np
from dataloader import DataLoader
from vispy.color import Color, ColorArray
from scipy.spatial.transform import Rotation
import cv2
from pathlib import Path

CLASS_LUT = ['unlabelled',
             'road',
             'tree',
             'building', 
             'grass',
             'vehicle',
             'human']

COLOR_LUT = [Color([0, 0, 0]), # unlabelled
             Color([0, 0, 1]), # road
             Color([0, 1, 0]), # tree
             Color([1, 0, 0]), # building
             Color([0, 0.4, 0]), # grass
             Color([0, 1, 1]), # vehicle
             Color([1, 0, 1])] # human

def get_cv_lut(lut):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for ind, color in enumerate(COLOR_LUT):
        # flip to bgr
        lut[ind, 0, :] = np.flip(color.RGB)
    return lut

class IntegratedCloud:
    def __init__(self, bagpath, start_ind, load, directory = None):
        self.loader_ = DataLoader(bagpath).__iter__()
        self.block_size_ = 10
        self.start_ind_ = start_ind
        self.load_ = load
        if directory is not None:
            self.directory_ = Path(directory)
        else:
            self.directory_ = Path(bagpath).with_name(Path(bagpath).stem + '_labels')
        self.directory_.mkdir(exist_ok = True)
        self.cv_lut_ = get_cv_lut(COLOR_LUT)
        self.reset()
        self.add_new()

    def go_to_start(self):
        for _ in range(self.start_ind_):
            self.loader_.__next__()

    def reset(self):
        self.cloud_ = np.empty([0, 4], dtype=np.float32)
        # label, elevation when labelled
        self.labels_ = np.empty([0, 2], dtype=np.float32)
        self.inds_ = np.empty([0, 1], dtype=np.int32)
        self.imgs_ = []
        self.info_ = []
        self.render_block_indices_ = np.empty([0, 1], dtype=np.int32)
        self.colors_ = None
        self.target_z_ = 0
        self.root_transform_ = None 
        self.go_to_start()

    def write(self):
        label_dir = self.directory_ / 'labels'
        label_dir.mkdir(exist_ok = True)
        scan_dir = self.directory_ / 'scans'
        scan_dir.mkdir(exist_ok = True)

        for ind, img in enumerate(self.imgs_):
            img_labels = self.labels_[self.inds_[:, 0] == ind, 0].reshape(img.shape[:2])
            # destagger
            img_undist = img.copy()
            label_undist = img_labels.copy()
            for row, shift in enumerate(self.info_[0].D):
                img_undist[row, :, :] = np.roll(img[row, :, :], int(shift), axis=0)
                label_undist[row, :] = np.roll(img_labels[row, :], int(shift), axis=0)

            # use tiff since can handle a 4 channel floating point image
            stamp = self.info_[ind].header.stamp.to_nsec()
            cv2.imwrite((scan_dir / f'{stamp}.tiff').as_posix(), img_undist)

            cv2.imwrite((label_dir / f'{stamp}.png').as_posix(), label_undist)
            range_img = np.linalg.norm(img_undist[:, :, :3]*10, axis=2).astype(np.uint8)
            range_img_color = cv2.cvtColor(range_img, cv2.COLOR_GRAY2BGR)
            label_undist_color = cv2.cvtColor(label_undist.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            label_viz = cv2.LUT(label_undist_color, self.cv_lut_)
            cv2.imwrite((label_dir / f'viz_{stamp}.png').as_posix(), 
                    cv2.addWeighted(range_img_color, 0.5, label_viz, 0.5, 0))
        print(f"Labels written to disk at {self.directory_.as_posix()}")

    def add_new(self):
        try:
            pc, pose, img, info = self.loader_.__next__()
        except StopIteration:
            print("REACHED END OF BAG")
            return

        if self.root_transform_ is None:
            self.root_transform_ = pose

        scan_ind = len(self.imgs_)
        self.imgs_.append(img)
        self.info_.append(info)

        # transform cloud
        pc_trans = pc.copy()
        comb_pose = {'R': self.root_transform_['R'].inv() * pose['R'], 
                     'T': self.root_transform_['R'].inv().apply(pose['T'] - self.root_transform_['T'])}
        pc_trans[:, :3] = comb_pose['R'].apply(pc[:, :3]) + comb_pose['T']
        self.cloud_ = np.vstack((self.cloud_, pc_trans))
        self.inds_ = np.vstack((self.inds_, np.ones([pc.shape[0], 1])*scan_ind))
        # initialize to high z because can overwrite with low z
        self.labels_ = np.vstack((self.labels_, np.repeat(np.array([[0, 1000]]), pc.shape[0], axis=0)))
        new_colors = ColorArray(np.repeat(np.clip(pc[:, 3, None]/1000, 0, 1), 3, axis=1), alpha=1)

        if self.load_:
            label_dir = self.directory_ / 'labels'
            label_img = cv2.imread((label_dir / f'{info.header.stamp.to_nsec()}.png').as_posix())
            if label_img is not None:
                flattened_labels = label_img[:,:,0].flatten()
                self.labels_[self.inds_[:, 0] == scan_ind, 0] = flattened_labels
                for label in np.unique(label_img):
                    new_colors[flattened_labels == label] = COLOR_LUT[label]

        self.render_block_indices_ = np.vstack((self.render_block_indices_, 
            self.get_block_ind(pc_trans[:, :2])[:, None]))

        if self.colors_ is None:
            self.colors_ = new_colors
        else:
            self.colors_.extend(new_colors)

        self.adjust_z(0)

    def get_block_ind(self, pts):
        inds = (pts / self.block_size_).astype(np.int32)
        return inds[:,0] + (inds[:,1] << 16)

    def get_block_neighborhood(self, pt):
        pts = np.stack((pt,
                        pt + np.array([0, self.block_size_]),
                        pt + np.array([self.block_size_, 0]),
                        pt + np.array([0, -self.block_size_]),
                        pt + np.array([-self.block_size_, 0])))
        return self.get_block_ind(pts)

    def get_indices(self):
        return np.unique(self.render_block_indices_)

    def adjust_z(self, delta, update=True):
        self.target_z_ += delta
        if not update:
            return

        visible = self.cloud_[:, 2] <= self.target_z_
        labelled_below = np.logical_and(self.labels_[:, 1] < self.target_z_, self.labels_[:, 0] > 0)
        if np.any(visible):
            self.colors_[visible] = ColorArray(self.colors_[visible], alpha=1)
        if np.any(labelled_below):
            self.colors_[labelled_below] = ColorArray(self.colors_[labelled_below], alpha=0.1)
        if not np.all(visible):
            self.colors_[np.invert(visible)] = ColorArray(self.colors_[np.invert(visible)], alpha=0)

    def label(self, pos, label, eps=1):
        to_update = np.logical_and(
                np.logical_and(self.target_z_ <= self.labels_[:, 1], 
                               self.cloud_[:, 2] <= self.target_z_),
                np.linalg.norm(self.cloud_[:, :2] - pos, axis=1) < eps)
        if np.any(to_update):
            self.labels_[to_update] = np.array([label, self.target_z_])
            self.colors_[to_update] = COLOR_LUT[label]

    def cloud(self, block):
        return self.cloud_[self.render_block_indices_[:,0] == block, :2]

    def colors(self, block):
        return self.colors_[self.render_block_indices_[:,0] == block]

    def get_z(self):
        return self.target_z_
