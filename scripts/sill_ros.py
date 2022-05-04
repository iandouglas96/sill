import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

class SillROS:
    def __init__(self, pc_q):
        rospy.init_node('sill', disable_signals=True)
        self.pc_q_ = pc_q

        self.pc_sub_ = message_filters.Subscriber('/os_node/llol_odom/sweep', PointCloud2)
        self.pose_sub_ = message_filters.Subscriber('/os_node/llol_odom/pose', PoseStamped)
        self.pc_pose_sync_ = message_filters.TimeSynchronizer([self.pc_sub_, self.pose_sub_], 10)
        self.pc_pose_sync_.registerCallback(self.pc_pose_cb)

    def pc_pose_cb(self, pc_msg, pose_msg):
        
        pass

    def spin(self):
        rospy.spin()
