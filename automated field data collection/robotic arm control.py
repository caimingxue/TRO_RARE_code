#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : field_calibartion_M_arm.py
# @Time : 2022/7/13 下午3:35
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import sys
import moveit_commander
import numpy as np
import rospy
import time
from import ArmBase
from std_msgs.msg import *
from copy import deepcopy
from utils.pandas2csv import Pandas_2_CSV

np.set_printoptions(precision=5)
np.set_printoptions(suppress=False)


class MagControl():
    def __init__(self):
        rospy.init_node('mag_control')
        ctr_freq = 10
        self.rate = rospy.Rate(ctr_freq)
        self.dt = 1. / ctr_freq

        group_name = "M_manipulator"
        self.single_arm = ArmBase(group_name)

        self.magrobo_state_sub = rospy.Subscriber("magrobo_state", Float64MultiArray, self.magrobo_state_callback, queue_size=10)

        self.single_arm.go_to_home_cartesian_single_arm()
        time.sleep(1)

        # exit()


        self.current_joint_value = self.single_arm.group.get_current_joint_values()

        self.arm_pose = self.single_arm.get_current_pose_base_link()

        self.mag_sensor_value = Float64MultiArray()
        self.mag_sensor_pub = rospy.Subscriber("mag_sensor_data", Float64MultiArray, self.recv_mag_data, queue_size=10)

        self.csv_file = Pandas_2_CSV(file_name="auto_field_calibration", header=["x", "y", "z", "mag_x", "mag_y", "mag_z"])


    def magrobo_state_callback(self, magrobo_state_info):
        self.magrobo_state = magrobo_state_info.data
        self.recv_magrobo_state_flag = True

    def recv_mag_data(self, mag_data_info):
        self.mag_data = mag_data_info.data
        # print("*********************", self.mag_data)
    def run(self):
        # main thread

        start_pose = self.single_arm.get_current_pose_base_link().pose
        waypoints = []
        target_initial_pose = deepcopy(start_pose)
        target_initial_pose.position.x = 0.417-0.01
        target_initial_pose.position.y = -0.033
        target_initial_pose.position.z = 0.189  ###z:0.183 means senror-tip distance:50mm
        target_initial_pose.orientation.w = 1
        temp = PoseStamped()
        temp.header.frame_id = "M_base_link"
        temp.header.stamp = rospy.Time.now()
        temp.pose = target_initial_pose
        self.single_arm.go_to_pose_goal(temp)
        time.sleep(2)
        exit()

        task_flag = False

        while not rospy.is_shutdown():
            current_pose = self.single_arm.get_current_pose_base_link().pose
            print("==============current pose================", current_pose.position)
            waypoints = []
            wpose = deepcopy(current_pose)

            if current_pose.position.y <= -0.05 and task_flag == False:
                wpose.position.z += 0.003
                robot_in_left_border = True
                task_flag = True


            if robot_in_left_border == True:
                wpose.position.y += 0.003
                robot_in_right_border = False


            if current_pose.position.y >= 0.05  and task_flag == False:
                wpose.position.z += 0.003
                robot_in_right_border = True
                task_flag = True


            if robot_in_right_border == True:
                wpose.position.y -= 0.003
                robot_in_left_border = False

            if abs(current_pose.position.y) <= 0.01:
                task_flag = False

            #
            # if current_pose.position.y < 0.04:
            #     wpose.position.y -= 0.002

            waypoints.append(deepcopy(wpose))
            plan, friction = self.single_arm.plan_cartesian_path(waypoints)
            self.single_arm.execute_plan(plan)
            time.sleep(1)
            recorded_data = np.array([0, current_pose.position.y, current_pose.position.z - 0.123,
                                      self.mag_data[0], self.mag_data[1], self.mag_data[2]])
            # self.csv_file.write_csv(recorded_data)

            if current_pose.position.z >=0.333:
                break



            print("=======================================================LOOP============================================================================")
            # break


def main():
    robo_mag=MagControl()
    robo_mag.run()



if __name__ == "__main__":
    main()
