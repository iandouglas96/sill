import numpy as np
from scipy.spatial.transform import Rotation
import rosbag
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

def ros_pose_to_numpy(msg):
    R = Rotation.from_quat([msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w])
    T = np.array([msg.pose.position.x, 
                  msg.pose.position.y, 
                  msg.pose.position.z])
    return (T, R)

class DataLoader:
    def __init__(self, bagpath):
        self.bag_ = rosbag.Bag(bagpath, 'r')

    def __iter__(self):
        # reset iterator
        self.bag_iter_ = self.bag_.read_messages()
        self.last_stamp_ = 0
        return self

    def __next__(self):
        stamp = 0
        data = [None, None, None, None]
        # this assumes that all messages with the same stamp arrive sequentially
        # before any messages of the next timestamp
        while True:
            topic, msg, t = self.bag_iter_.__next__()
            if msg.header.stamp.to_nsec() > self.last_stamp_:
                if msg.header.stamp.to_nsec() > stamp:
                    data = [None, None, None, None]
                stamp = msg.header.stamp.to_nsec()

            if msg.header.stamp.to_nsec() == stamp:
                if topic == '/os_node/llol_odom/sweep':
                    data[0] = np.frombuffer(msg.data, dtype=np.float32).reshape(4, -1)
                elif topic == '/os_node/llol_odom/pose':
                    data[1] = ros_pose_to_numpy(msg)
                elif topic == '/os_node/image':
                    data[2] = np.frombuffer(msg.data, dtype=np.float32).reshape(1024, 64, -1)
                elif topic == '/os_node/camera_info':
                    data[3] = msg

                if all(d is not None for d in data):
                    # we have all elements
                    self.last_stamp_ = stamp
                    return data

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    # test
    d = DataLoader('/media/ian/ResearchSSD/xview_collab/callisto/twojackalquad2_labeltest.bag')
    for pc, pose, img, info in d:
        print(pc.shape)
