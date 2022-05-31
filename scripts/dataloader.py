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
    def __init__(self, bagpath):
        self.bag_ = rosbag.Bag(bagpath, 'r')

    def __iter__(self):
        # reset iterator
        self.bag_iter_ = self.bag_.read_messages()
        return self

    def __next__(self):
        stamp = 0
        sweep_data = [None, None, None]
        sweeps = []
        pano_data = [None, None]
        # this assumes that all messages with the same stamp arrive sequentially
        # before any messages of the next timestamp
        while True:
            topic, msg, t = self.bag_iter_.__next__()

            if msg.header.stamp.to_nsec() > stamp + 1e5:
                # reset
                stamp = msg.header.stamp.to_nsec()
                sweep_data = [None, None, None]

            if 'pano' in topic:
                if topic == '/os_node/rofl_odom/pano/image':
                    pano_data[0] = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                            msg.height, msg.width, -1) 
                elif topic == '/os_node/rofl_odom/pano/camera_info':
                    pano_data[1] = msg

                if all(d is not None for d in pano_data):
                    return pano_data, sweeps
            else:
                if topic == '/os_node/rofl_odom/sweep/cloud':
                    # initially organized as x,y,z,intensity
                    sweep_data[0] = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4).copy()
                elif topic == '/os_node/image':
                    sweep_data[1] = np.frombuffer(msg.data, dtype=np.float32).reshape(
                            msg.height, msg.width, -1)
                elif topic == '/os_node/camera_info':
                    sweep_data[2] = msg

                if all(d is not None for d in sweep_data):
                    # we have all elements
                    sweeps.append(sweep_data)
                    sweep_data = [None, None, None]

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    # test
    d = DataLoader('/media/ian/ResearchSSD/xview_collab/callisto/twojackalquad2_labeltest_rofl.bag')
    for pano, sweeps in d:
        print(pano[0].shape)
        print(len(sweeps))
