#!/usr/bin/env python3

'''
To pair with the integrate_sem branch of rofl, which integrates semantics in the
panoramas.  In this way we can use a trained model for sweeps and use this to
supervise training for panos.  The performance probably isn't quite as good,
but it is a relatively quick and dirty solution that doesn't require more
labelling.
'''

from pathlib import Path
import numpy as np
import yaml
from vispy.color import Color
import cv2
import rospy
import rospkg
from sensor_msgs.msg import Image, CameraInfo

class SaveIntegratedPanos:
    def __init__(self):
        self.class_lut_, self.color_lut_ = self.load_luts()
        self.cv_lut_ = self.get_cv_lut(self.color_lut_)
        self.directory_ = Path(rospy.get_param('directory', 'labelled_panos'))

        self.scale_ = None
        self.pano_sub_ = rospy.Subscriber("/os_node/rofl_odom/pano/img", Image, self.pano_cb)
        self.pano_info_sub_ = rospy.Subscriber("/os_node/rofl_odom/pano/camera_info", CameraInfo, self.pano_info_cb)

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

    def get_cv_lut(self, color_lut):
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for ind, color in enumerate(color_lut):
            # flip to bgr
            lut[ind, 0, :] = np.flip(color.RGB)
        return lut

    def pano_info_cb(self, msg):
        self.scale_ = msg.R[0]

    def pano_cb(self, msg):
        if self.scale_ is None:
            return

        img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)

        pc = self.project_pano(img).astype(np.float32)
        labels = img[:, :, -1].astype(np.uint8)

        self.save_scan(msg.header.stamp.to_nsec(), pc, labels, "pano")
        rospy.loginfo(f"Saved pano {msg.header.stamp.to_sec()}")

    def project_pano(self, pano, vfov=np.pi/2):
        azis = np.arange(2*np.pi, 0, -np.pi * 2 / pano.shape[1])
        elevs = np.arange(vfov/2, -vfov/2-0.0001, -vfov / (pano.shape[0] - 1))

        pc_full = np.empty((*pano.shape[:2], 4))
        # x,y,z
        pc_full[:,:,0] = pano[:,:,0]/self.scale_ * np.cos(elevs[:, None]) * np.cos(azis)
        pc_full[:,:,1] = pano[:,:,0]/self.scale_ * np.cos(elevs[:, None]) * np.sin(azis)
        pc_full[:,:,2] = pano[:,:,0]/self.scale_ * np.sin(elevs[:, None])
        # intensity
        pc_full[:,:,3] = pano[:,:,1]

        return pc_full

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

        cv2.imwrite(label_path, labels)
        range_img = np.linalg.norm(scan[:, :, :3]*10, axis=2).astype(np.uint8)
        range_img_color = cv2.cvtColor(range_img, cv2.COLOR_GRAY2BGR)
        label_undist_color = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        label_viz = cv2.LUT(label_undist_color, self.cv_lut_)
        cv2.imwrite((label_dir / f"viz_{prefix}{stamp}.png").as_posix(),
                cv2.addWeighted(range_img_color, 0.5, label_viz, 0.5, 0))

if __name__ == '__main__':
    rospy.init_node('save_integrated_panos')
    sip = SaveIntegratedPanos()
    rospy.spin()
