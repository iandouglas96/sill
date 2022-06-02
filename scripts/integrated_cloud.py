import numpy as np
from dataloader import DataLoader
from vispy.color import Color, ColorArray
from scipy.spatial.transform import Rotation
import cv2
from pathlib import Path
import rospkg
import yaml
import tqdm

def get_cv_lut(color_lut):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for ind, color in enumerate(color_lut):
        # flip to bgr
        lut[ind, 0, :] = np.flip(color.RGB)
    return lut

class IntegratedCloud:
    def __init__(self, bagpath, start_ind, load, directory = None, config_path = None):
        self.loader_ = DataLoader(bagpath).__iter__()

        if config_path is None:
            rospack = rospkg.RosPack()
            package_path = Path(rospack.get_path('sill'))
            config_path = package_path / Path('config') / Path('config.yaml')
        else:
            config_path = Path(config_path)
        config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)

        self.render_block_size_ = config['render_block_size']
        self.label_grid_res_xy_ = config['label_grid']['res_xy']
        self.label_grid_res_z_ = config['label_grid']['res_z']
        self.label_grid_origin_z_ = config['label_grid']['origin_z']
        self.label_grid_size_xy_ = config['label_grid']['size_xy']
        self.label_grid_size_z_ = config['label_grid']['size_z']
        self.voxel_filter_size_ = 1./config['voxel_filter_res']
        self.pano_info_thresh_ = config['pano_info_thresh']

        # label, elevation when labelled
        # only need to allocate mem once
        self.labels_ = np.zeros([self.label_grid_size_xy_*self.label_grid_res_xy_,
                                 self.label_grid_size_xy_*self.label_grid_res_xy_,
                                 self.label_grid_size_z_*self.label_grid_res_z_, 
                                 2], dtype=np.float32)
        self.grid_centers_ = self.get_grid_centers()

        self.start_ind_ = start_ind
        self.load_ = load
        if directory is not None:
            self.directory_ = Path(directory)
        else:
            self.directory_ = Path(bagpath).with_name(Path(bagpath).stem + '_labels')
        self.directory_.mkdir(exist_ok = True)
        self.class_lut_, self.color_lut_ = self.load_luts()
        self.cv_lut_ = get_cv_lut(self.color_lut_)
        self.reset()
        self.add_new()

    def load_luts(self):
        rospack = rospkg.RosPack()
        package_path = Path(rospack.get_path('sill'))
        config = yaml.load(open(package_path / Path('config') / Path('classes.yaml'), 'r'),
                           Loader=yaml.SafeLoader)
        class_lut = []
        color_lut = []

        for cls in config:
            class_lut.append(list(cls.keys())[0])
            color_lut.append(Color(list(cls.values())[0]))

        return class_lut, color_lut

    def go_to_start(self):
        for ind in range(self.start_ind_):
            try:
                self.loader_.__next__()
            except StopIteration:
                print(f"STARTED PAST END, LAST INDEX IS {ind-1}")
                return

    def reset(self):
        self.cloud_ = np.empty([0, 3], dtype=np.float32)
        # initilize elevation to high z value so easily overwritten
        self.labels_[:, :, :, 0] = 0
        self.labels_[:, :, :, 1] = 1000
        # keep track of grid indices for the current pc
        self.grid_indices_ = np.empty([0, 1], dtype=np.int32)
        self.panos_ = []
        self.sweeps_ = []
        self.render_block_indices_ = np.empty([0, 1], dtype=np.int32)
        self.colors_ = None
        self.target_z_ = 0
        self.root_transform_ = None 
        self.go_to_start()

    def write(self):
        print(f"Writing panos...")
        for pano in tqdm.tqdm(self.panos_):
            pose = np.eye(4)
            pose[:3, :] = np.array(pano[1].P).reshape((3, 4))

            # project and transform
            pc_orig = self.project_pano(pano)
            pc = pc_orig.copy()
            #rot[:3, :3] = Rotation.from_rotvec(np.array([0,0,np.pi])).as_dcm()
            pc[:, :3] = (pose[:3, :3] @ pc_orig[:, :3].T).T + pose[:3, 3] - self.root_transform_[:3, 3]

            indices = self.compute_grid_indices(pc)
            pano_labels = self.labels_.reshape(-1, 2)[indices, 0]
            pano_labels[np.logical_or(~np.isfinite(pc_orig[:, 0]), 
                                      pc_orig[:, 0] == 0, indices < 0)] = 0
            pano_labels = pano_labels.reshape(pano[0].shape[:2])
            # don't label points with low confidence
            pano_labels[pano[0][:,:,2] < self.pano_info_thresh_] = 0

            # actually save
            stamp = pano[1].header.stamp.to_nsec()
            pano_img = pc_orig.reshape(*pano[0].shape[:2], 4).astype(np.float32)
            # add info channel
            pano_img = np.concatenate((pano_img, pano[0][:, :, 2, None]), axis=-1)
            self.save_scan(stamp, pano_img, pano_labels, "pano")

        print(f"Writing sweeps...")
        for sweep in tqdm.tqdm(self.sweeps_):
            sweep[0][:, :3] -= self.root_transform_[None, :3, 3]
            indices = self.compute_grid_indices(sweep[0])
            sweep_labels = self.labels_.reshape(-1, 2)[indices, 0]
            sweep_labels[np.logical_or(~np.isfinite(sweep[0][:, 0]), 
                                       sweep[0][:, 0] == 0, indices < 0)] = 0
            sweep_labels = sweep_labels.reshape(sweep[1].shape[:2])
            
            # extract depth and intensity channels, which are each 16 bits, splitting
            # the last 32 bit float
            depth_intensity = np.frombuffer(sweep[1][:, :, 3].tobytes(), dtype=np.uint16).reshape(
                                *(sweep[1].shape[:2]), -1)

            # destagger
            img_undist = sweep[1].copy()
            img_undist[:, :, 3] = depth_intensity[:, :, 1].astype(np.float32)
            label_undist = sweep_labels.copy()
            for row, shift in enumerate(sweep[2].D):
                img_undist[row, :, :] = np.roll(img_undist[row, :, :], int(shift), axis=0)
                label_undist[row, :] = np.roll(label_undist[row, :], int(shift), axis=0)
             
            # use tiff since can handle a 4 channel floating point image
            stamp = sweep[2].header.stamp.to_nsec()
            self.save_scan(stamp, img_undist, label_undist, "sweep")
        print(f"Labels written to disk at {self.directory_.as_posix()}")

    def save_scan(self, stamp, scan, labels, prefix=''):
        label_dir = self.directory_ / 'labels'
        label_dir.mkdir(exist_ok = True)
        scan_dir = self.directory_ / 'scans'
        scan_dir.mkdir(exist_ok = True)

        if len(prefix) > 0:
            prefix = f"{prefix}_"

        cv2.imwrite((scan_dir / f"{prefix}{stamp}.tiff").as_posix(), scan[:, :, :4])
        if scan.shape[-1] > 4:
            cv2.imwrite((scan_dir / f"{prefix}{stamp}_supp.tiff").as_posix(), scan[:, :, 4:])

        label_path = (label_dir / f"{prefix}{stamp}.png").as_posix()
        if self.load_:
            old_label_img = cv2.imread(label_path)
            if old_label_img is not None:
                # don't write unknown to file
                # if we load an old file we haven't set the grid cells properly, so don't want
                # to wipe everything
                labels.ravel()[labels.ravel() == 0] = \
                    old_label_img[:, :, 0].ravel()[labels.ravel() == 0]

        cv2.imwrite(label_path, labels)
        range_img = np.linalg.norm(scan[:, :, :3]*10, axis=2).astype(np.uint8)
        range_img_color = cv2.cvtColor(range_img, cv2.COLOR_GRAY2BGR)
        label_undist_color = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        label_viz = cv2.LUT(label_undist_color, self.cv_lut_)
        cv2.imwrite((label_dir / f"viz_{prefix}{stamp}.png").as_posix(), 
                cv2.addWeighted(range_img_color, 0.5, label_viz, 0.5, 0))

    def add_new(self):
        try:
            pano, sweeps = self.loader_.__next__()
        except StopIteration:
            print("REACHED END OF BAG")
            return

        pose = np.eye(4)
        pose[:3, :] = np.array(pano[1].P).reshape((3, 4))
        if self.root_transform_ is None:
            self.root_transform_ = pose

        scan_ind = len(self.panos_)
        self.panos_.append(pano)
        self.sweeps_ += sweeps

        # project pano into cloud
        pc = self.project_pano(pano)

        # transform cloud
        pc[:, :3] = (pose[:3, :3] @ pc[:, :3].T).T + pose[:3, 3] - self.root_transform_[:3, 3]

        new_colors = ColorArray(np.repeat(np.clip(pc[:, 3, None]/1000, 0, 1), 3, axis=1), alpha=1)
        if self.load_:
            label_dir = self.directory_ / 'labels'
            label_img = cv2.imread((label_dir / f'pano_{pano[1].header.stamp.to_nsec()}.png').as_posix())
            if label_img is not None:
                flattened_labels = label_img[:,:,0].ravel()
                for label in np.unique(label_img):
                    if label != 0:
                        new_colors[flattened_labels == label] = self.color_lut_[label]

        # remove extra points
        valid_points = np.logical_and(np.isfinite(pc[:, 0]), pc[:, 0] != 0)
        pc = pc[valid_points, :]
        new_colors = new_colors[valid_points]

        self.cloud_ = np.vstack((self.cloud_, pc[:, :3]))
        self.grid_indices_ = self.compute_grid_indices(self.cloud_)

        self.render_block_indices_ = np.vstack((self.render_block_indices_, 
            self.get_render_block_ind(pc[:, :2])[:, None]))

        if self.colors_ is None:
            self.colors_ = new_colors
        else:
            self.colors_.extend(new_colors)

        self.adjust_z(0)

    def voxel_filter(self, voxel_size=None):
        import open3d as o3d

        if voxel_size is None:
            voxel_size = self.voxel_filter_size_
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(self.cloud_[:, :3])
        pc_o3d.colors = o3d.utility.Vector3dVector(self.colors_.RGB)
        down_pc_o3d = pc_o3d.voxel_down_sample(voxel_size=voxel_size)

        self.cloud_ = np.asarray(down_pc_o3d.points)
        self.colors_ = ColorArray(np.asarray(down_pc_o3d.colors)/255, alpha=1)

        # recomupte what needs to be recomputed
        self.grid_indices_ = self.compute_grid_indices(self.cloud_)
        self.render_block_indices_ = self.get_render_block_ind(self.cloud_[:, :2])[:, None]

    def project_pano(self, pano, keep_shape=False, vfov=np.pi/2):
        azis = np.arange(2*np.pi, 0, -np.pi * 2 / pano[0].shape[1])
        elevs = np.arange(vfov/2, -vfov/2-0.0001, -vfov / (pano[0].shape[0] - 1))

        pc_full = np.empty((*pano[0].shape[:2], 4))
        # x,y,z
        pc_full[:,:,0] = pano[0][:,:,0]/pano[1].R[0] * np.cos(elevs[:, None]) * np.cos(azis)
        pc_full[:,:,1] = pano[0][:,:,0]/pano[1].R[0] * np.cos(elevs[:, None]) * np.sin(azis)
        pc_full[:,:,2] = pano[0][:,:,0]/pano[1].R[0] * np.sin(elevs[:, None])
        # intensity
        pc_full[:,:,3] = pano[0][:,:,1]

        return pc_full.reshape(-1, pc_full.shape[-1])

    def compute_grid_indices(self, pc):
        pc_ind = -np.ones((pc.shape[0], 3), dtype=np.int32)
        pc_ind[:,:2] = pc[:,:2] * self.label_grid_res_xy_ + self.labels_.shape[0]/2
        pc_ind[:,2] = (pc[:,2] - self.label_grid_origin_z_) * self.label_grid_res_z_

        flattened_ind = np.ravel_multi_index((pc_ind[:,0], pc_ind[:,1], pc_ind[:,2]), 
                self.labels_.shape[:3], mode='clip')

        # check bounds
        # let z below 0 just get clipped
        flattened_ind[np.any(pc_ind[:,:2] < 0, axis=1)] = -1
        flattened_ind[np.any(pc_ind[:,:2] >= self.labels_.shape[0], axis=1)] = -1
        flattened_ind[pc_ind[:,2] >= self.labels_.shape[2]] = -1

        return flattened_ind

    def get_grid_centers(self):
        axis = np.arange(self.labels_.shape[0])
        axis = axis / self.label_grid_res_xy_
        axis -= np.mean(axis)
        return np.stack(np.meshgrid(axis, axis, indexing='ij'))

    def get_render_block_ind(self, pts):
        inds = (pts / self.render_block_size_).astype(np.int32)
        return inds[:,0] + (inds[:,1] << 16)

    def get_render_block_neighborhood(self, pt):
        pts = np.stack((pt,
                        pt + np.array([0, self.render_block_size_]),
                        pt + np.array([self.render_block_size_, 0]),
                        pt + np.array([0, -self.render_block_size_]),
                        pt + np.array([-self.render_block_size_, 0])))
        return self.get_render_block_ind(pts)

    def get_render_indices(self):
        return np.unique(self.render_block_indices_)

    def adjust_z(self, delta, update=True):
        self.target_z_ += delta / self.label_grid_res_z_
        if not update:
            return

        visible = self.cloud_[:, 2] <= self.target_z_
        cells_labelled_below = np.logical_and(self.labels_[:, :, :, 1] < self.target_z_, 
                                              self.labels_[:, :, :, 0] > 0)
        if np.any(visible):
            self.colors_[visible] = ColorArray(self.colors_[visible], alpha=1)
        if np.any(cells_labelled_below):
            pt_indices = np.isin(self.grid_indices_, np.where(cells_labelled_below.ravel()))
            self.colors_[pt_indices] = ColorArray(self.colors_[pt_indices], alpha=0.1)
        if not np.all(visible):
            self.colors_[np.invert(visible)] = ColorArray(self.colors_[np.invert(visible)], alpha=0)

    def label(self, pos, label, eps=1):
        xy_to_update = np.where(np.linalg.norm(self.grid_centers_ - pos[:,None,None], axis=0) < eps)
        if xy_to_update[0].shape[0] < 1:
            # zoomed in, just pick closest cell
            xy_to_update = np.unravel_index(np.argmin(np.linalg.norm(self.grid_centers_ - pos[:,None,None], axis=0)), 
                    self.grid_centers_.shape[1:])
            xy_to_update = np.array(xy_to_update)[:, None]

        max_z_ind = (self.target_z_ - self.label_grid_origin_z_) * self.label_grid_res_z_ - 0.001
        max_z_ind = np.minimum(max_z_ind, self.label_grid_size_z_ * self.label_grid_res_z_)

        z_to_update = np.arange(max_z_ind).astype(np.int32)
        # get all combinations
        xyz_to_update = np.tile(xy_to_update, (1, z_to_update.shape[0]))
        xyz_to_update = np.vstack((xyz_to_update, np.repeat(z_to_update, xy_to_update[0].shape[0])))

        update_ind = np.ravel_multi_index((xyz_to_update[0,:], xyz_to_update[1,:], xyz_to_update[2,:]),
                self.labels_.shape[:3])

        # filter points that have not yet been set
        update_ind = update_ind[self.target_z_ <= self.labels_.reshape(-1, 2)[update_ind, 1]]

        self.labels_.reshape(-1, 2)[update_ind, :] = np.array([label, self.target_z_])
        self.colors_[np.isin(self.grid_indices_, update_ind)] = self.color_lut_[label]

    def cloud(self, block):
        return self.cloud_[self.render_block_indices_[:,0] == block, :3]

    def colors(self, block):
        return self.colors_[self.render_block_indices_[:,0] == block]

    def get_z(self):
        return self.target_z_
