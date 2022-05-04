import numpy as np
from scipy.spatial.transform import Rotation
import rosbag
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

def ros_pose_to_numpy(msg):
    R = Rotation.from_quat([msg.rotation.x,
                            msg.rotation.y,
                            msg.rotation.z,
                            msg.rotation.w])
    T = np.array([msg.translation.x,
                  msg.translation.y,
                  msg.translation.z])
    return {'R': R, 'T': T}

class DataLoader:
    def __init__(self, bagpath, spacing_sec=1):
        self.bag_ = rosbag.Bag(bagpath, 'r')
        self.spacing_sec_ = spacing_sec

    def __iter__(self):
        # reset iterator
        self.bag_iter_ = self.bag_.read_messages()
        self.frame_pose_ = None
        self.last_stamp_ = 0
        return self

    def __next__(self):
        stamp = 0
        data = [None, None, None]
        # this assumes that all messages with the same stamp arrive sequentially
        # before any messages of the next timestamp
        while True:
            topic, msg, t = self.bag_iter_.__next__()

            if topic == '/tf':
                for trans in msg.transforms:
                    if trans.header.stamp.to_nsec() > stamp - 1e5 and \
                       trans.header.frame_id == 'odom' and \
                       trans.child_frame_id == 'pano':
                        self.frame_pose_ = ros_pose_to_numpy(trans.transform)
            else:
                if msg.header.stamp.to_nsec() > self.last_stamp_ + self.spacing_sec_*1e9:
                    if msg.header.stamp.to_nsec() > stamp + 1e5:
                        # reset
                        stamp = msg.header.stamp.to_nsec()
                        data = [None, None, None]

                if abs(msg.header.stamp.to_nsec() - stamp) < 1e5:
                    if topic == '/os_node/llol_odom/sweep':
                        # initially organized as x,y,z,space,intensity,space,...
                        pc = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8).copy()
                        pc[:, 3] = pc[:, 4]
                        data[0] = pc[:, :4]
                    elif topic == '/os_node/image':
                        data[1] = np.frombuffer(msg.data, dtype=np.float32).reshape(64, 1024, -1)
                    elif topic == '/os_node/camera_info':
                        data[2] = msg

                    if all(d is not None for d in data) and self.frame_pose_ is not None:
                        # we have all elements
                        self.last_stamp_ = stamp
                        return [data[0], self.frame_pose_, data[1], data[2]]

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    # test
    d = DataLoader('/media/ian/ResearchSSD/xview_collab/callisto/twojackalquad2_labeltest.bag')
    for pc, pose, img, info in d:
        print(pc.shape)
