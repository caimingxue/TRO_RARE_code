#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : magrobo_rviz_env.py
# @Time : 2022/7/16 上午10:51
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import time
import numpy as np
import math
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import *
from std_msgs.msg import *
import rospy
import visualization_msgs.msg
from nav_msgs.msg import Path
from coordinates.base import make_coords
from magrobot import orient_coords_to_axis
from visualization_msgs.msg import Marker
# np.set_printoptions(precision=5)
# np.set_printoptions(suppress=False)
from utils.transformations import euler_from_quaternion, quaternion_from_euler
from magrobot import make_coords
from coordinates.geo import orient_coords_to_axis
from utils.transformations import unit_vector
from ekf import EKF

class MagroboRvizEnv():

    def __init__(self):

        self.x_range = np.array([-0.2, 0.2])
        self.y_range = np.array([-0.5, 0.5])
        self.z_range = np.array([0, 0.2])

        self.node_name = "magrobo_rviz_node"
        rospy.init_node(self.node_name)
        self.rate = rospy.Rate(10)

        self.dt = 0.1  # secondes between state updates
        self.threshold = 0.005  # unit m

        self.fluid_viscosity = 1 * 1e-3  # Pa.s ** original 4.5 * 1e-3
        self.fluid_density = 1.05 * 1e3  # Kg/m^3
        self.mass_robo = 0.895 * 1e-3
        self.radius_robot = 0.01  # m max length 16mm, diameter 8mm

        self.magrobo_momentum = 0.08  # A*m^2

        self.F_buoyancy = 8.376 * 1e-3  # N

        # magrobo_pose_init = np.array([0.220, 0.00, 0.0, 0, 0, 0])
        self.magrobo_pose_goal = np.array([0.00, 0.02, 0.24, 0, 0, 0, 1])

        self.magrobo_state_info = Float64MultiArray()

        self.use_B_direction_align = True
        self.B_direction = np.array([0, 0, 1])

        self.recv_mag_action_flag = False

        self.scene = PlanningSceneInterface()
        self.path_pub = rospy.Publisher('trajectory', Path, queue_size=50)
        self.path_record = Path()

        # filteration function relization
        self.mag_ekf_x = EKF()
        self.mag_ekf_y = EKF()
        self.mag_ekf_z = EKF()

        self.mag_arrow_pose_sub = rospy.Subscriber("mag_arrow_pose", PoseStamped, self.recv_arrow, queue_size=10)
        self.arrow_info = PoseStamped()

        self.mag_action = Float64MultiArray()
        self.mag_action_sub = rospy.Subscriber("mag_action", Float64MultiArray, self.recv_action, queue_size=10)

        ############################################################################
        self.mag_sensor_value = Float64MultiArray()
        self.mag_sensor_pub = rospy.Publisher("mag_sensor_data", Float64MultiArray, queue_size=10)
        #######################################################################################

        self.magrobo_state_info_pub = rospy.Publisher("magrobo_state", Float64MultiArray, queue_size=10)

        self.landmarker_arrow = rospy.Publisher("visualization_marker_arrow", visualization_msgs.msg.Marker,
                                                queue_size=10)
        self.landmarker_capsule = rospy.Publisher("visualization_marker_capsule", visualization_msgs.msg.Marker,
                                                  queue_size=10)
        self.landmarker_human = rospy.Publisher("visualization_marker_human", visualization_msgs.msg.Marker,
                                                queue_size=10)
        self.landmarker_doctor = rospy.Publisher("visualization_marker_doctor", visualization_msgs.msg.Marker,
                                                 queue_size=10)
        self.landmarker_camera = rospy.Publisher("visualization_marker_camera", visualization_msgs.msg.Marker,
                                                 queue_size=10)
        self.landmarker_machine_chair = rospy.Publisher("visualization_marker_chair", visualization_msgs.msg.Marker,
                                                        queue_size=10)
        self.landmarker_laptop = rospy.Publisher("visualization_marker_laptop", visualization_msgs.msg.Marker,
                                                 queue_size=10)
        self.landmarker_cube = rospy.Publisher("visualization_marker_cube", Marker, queue_size=10)
        self.landmarker_helical_magrobot = rospy.Publisher("visualization_marker_helical_magrobot", Marker,
                                                           queue_size=10)
        self.mag_field_direction = rospy.Publisher("mag_field_direction", PoseStamped, queue_size=10)

    def recv_action(self, action_info):

        self.recv_mag_action_flag = True
        ax, ay, az = self.compute_acceleration(Fx, Fy, Fz)
        
         # update:
        self.magrobo_state[7] = ax * self.dt
        self.magrobo_state[8] = ay * self.dt
        self.magrobo_state[9] = az * self.dt


        magrobo_linear_vel = 0.0012
  

        self.magrobo_state[0] += magrobo_linear_vel * np.cos(action_info.data[6]) * self.dt
        self.magrobo_state[1] += magrobo_linear_vel * np.sin(action_info.data[6]) * self.dt
        self.magrobo_state[2] += 0


        self.B_direction = np.array([action_info.data[3], action_info.data[4], action_info.data[5]])

        if self.use_B_direction_align is True:
            magrobo_cs = make_coords(pos=np.array([self.magrobo_state[0],
                                                   self.magrobo_state[1],
                                                   self.magrobo_state[2]]))

            self.B_direction_norm_vector = action_info.data[0:3]
            orient_coords_to_axis(magrobo_cs, self.B_direction_norm_vector, 'x')
            magrobo_orientation = orient_coords_to_axis(magrobo_cs, list(self.B_direction), 'z').quaternion

            if (round(magrobo_cs.z_axis[0], 2) != round(self.B_direction[0], 2)) or \
                    (round(magrobo_cs.z_axis[1], 2) != round(self.B_direction[1], 2)) or \
                    (round(magrobo_cs.z_axis[2], 2) != round(self.B_direction[2], 2)):
                return

            angles = euler_from_quaternion(magrobo_orientation)
            # print("******************euler angle", angles, magrobo_orientation)

            quater = quaternion_from_euler(angles[0], angles[1], angles[2], 'sxyz')
            # print("******************quater", quater)
            ####### quaternion order: rx, ry, rz, w #######
            self.magrobo_state[3] = quater[1]
            self.magrobo_state[4] = quater[2]
            self.magrobo_state[5] = quater[3]
            self.magrobo_state[6] = quater[0]

        rospy.loginfo("===================== Received the magnetic action info ===================")

        # self.limit_motion_space()

    def recv_arrow(self, arrow_info):

        print("+++++++++++++++++++++++++++++++++")
        self.arrow_info = arrow_info

    def seed(self, seed=None):
        pass

    def reset(self):
        self.magrobo_state = np.array([0.0, -0.0, 0.24, 0, 0, 0, 1,  # [0-6] x, y, z, quaternion(x, y, z, w)
                                       0, 0, 0, 0, 0, 0,  # [7-12] vel: x, y, z, angular_vel(w_r, w_p, w_y)
                                       self.magrobo_momentum,  # [13]
                                       0, 0, 0, 0  # [14-17] error x, y, z, sum
                                       ])

    def set_target_pose(self, magrobo_pose_goal):
        self.magrobo_pose_goal = magrobo_pose_goal

    def drag_force(self):
        # ref (2.27) from Micro-Scale Mobile Robotics - Eric Diller and Metin Sitti
        # approximation for a sphere at low Reynolds number
        F_drag_x = self.radius_robot * 6 * math.pi * self.fluid_viscosity * self.magrobo_state[7]
        F_drag_y = self.radius_robot * 6 * math.pi * self.fluid_viscosity * self.magrobo_state[8]
        F_drag_z = self.radius_robot * 6 * math.pi * self.fluid_viscosity * self.magrobo_state[9]
        F_drag = np.array([F_drag_x, F_drag_y, F_drag_z])
        return F_drag

    def compute_acceleration(self, Fx, Fy, Fz):
        # magnetic forces applied to agent
        F_gravity = self.gravity_force()

        F_drag = self.drag_force()
        ax = (Fx - F_drag[0]) / self.mass_robo
        ay = (Fy - F_drag[1]) / self.mass_robo
        az = (Fz - F_gravity + self.F_buoyancy - F_drag[2]) / self.mass_robo
        # rospy.loginfo("F_buoyancy-F_gravity {}".format(self.F_buoyancy - F_gravity- F_drag[2]))
        return ax, ay, az

    def gravity_force(self):
        self.F_gravity = self.mass_robo * 9.8  # kg * (N/kg)
        return self.F_gravity

    def limit_motion_space(self):
        if self.magrobo_state[0] < self.x_range[0]:
            self.magrobo_state[0] = self.x_range[0]
            rospy.logwarn("robot is on the lower x_range border")
        if self.magrobo_state[0] > self.x_range[1]:
            self.magrobo_state[0] = self.x_range[1]
            rospy.logwarn("robot is on the upper x_range border")
        if self.magrobo_state[1] < self.y_range[0]:
            self.magrobo_state[1] = self.y_range[0]
            rospy.logwarn("robot is on the lower y_range border")
        if self.magrobo_state[1] > self.y_range[1]:
            self.magrobo_state[1] = self.y_range[1]
            rospy.logwarn("robot is on the upper y_range border")
        if self.magrobo_state[2] < self.z_range[0]:
            self.magrobo_state[2] = self.z_range[0]
            rospy.logwarn("robot is on the lower z_range border")
        if self.magrobo_state[2] > self.z_range[1]:
            self.magrobo_state[2] = self.z_range[1]
            rospy.logwarn("robot is on the upper z_range border")

    def update(self):

        self.reset()
        count_flag = 0
        reset_flag = False
        time.sleep(0.5)

        while not rospy.is_shutdown():
            if reset_flag is True:
                self.reset()

            # if self.magrobo_state[2] < 0.21:  # 0.235
            #     self.magrobo_state[2] = 0.21
            #     self.magrobo_state[9] = 0

            if (self.B_direction[0] == 0) and (self.B_direction[1] == 0) and (self.B_direction[2] == 0):
                rospy.logwarn("Keeping the last pose")
                self.magrobo_state_info_pub.publish(self.magrobo_state_info)
                self.mag_field_direction.publish(self.magrobo_marker_arrow)
                self.rate.sleep()
                continue

            self.magrobo_marker_arrow = PoseStamped()
            self.magrobo_marker_arrow.header.frame_id = "world"
            self.magrobo_marker_arrow.header.stamp = rospy.Time.now()
            self.magrobo_marker_arrow.pose.position.x = self.magrobo_state[0]
            self.magrobo_marker_arrow.pose.position.y = self.magrobo_state[1]
            self.magrobo_marker_arrow.pose.position.z = self.magrobo_state[2]
            self.magrobo_marker_arrow.pose.orientation.x = self.magrobo_state[3]
            self.magrobo_marker_arrow.pose.orientation.y = self.magrobo_state[4]
            self.magrobo_marker_arrow.pose.orientation.z = self.magrobo_state[5]
            self.magrobo_marker_arrow.pose.orientation.w = self.magrobo_state[6]
            # self.mag_field_direction.publish(self.magrobo_marker_arrow)

            # update the magrobo pose trajectory info
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'world'
            pose.pose.position.x = self.magrobo_state[0]
            pose.pose.position.y = self.magrobo_state[1]
            pose.pose.position.z = self.magrobo_state[2]
            pose.pose.orientation.w = 1
            self.path_record.header.stamp = rospy.Time.now()
            self.path_record.header.frame_id = 'world'
            self.path_record.poses.append(pose)
            self.path_pub.publish(self.path_record)

            ############### If Not Mag Action, the robot position reset automatically##############
            if self.recv_mag_action_flag == False:
                count_flag += 1
            else:
                count_flag = 0
                reset_flag = False
            if count_flag > 10:
                reset_flag = True
            self.recv_mag_action_flag = False

            ######################################################################################
            ### error change
            self.magrobo_state[14] = self.magrobo_pose_goal[0] - self.magrobo_state[0]
            self.magrobo_state[15] = self.magrobo_pose_goal[1] - self.magrobo_state[1]
            self.magrobo_state[16] = self.magrobo_pose_goal[2] - self.magrobo_state[2]
            self.magrobo_state[17] = np.sqrt(
                self.magrobo_state[14] ** 2 + self.magrobo_state[15] ** 2 + self.magrobo_state[16] ** 2)
            self.magrobo_state_info.data = self.magrobo_state
            self.magrobo_state_info_pub.publish(self.magrobo_state_info)

            # self.makeBox()
            self.makeHuman()
            self.makeDoctor()
            self.makemachine_chair()
            self.makelaptop()
            self.makecamera()
            self.makeArrow()
            # self.makeCapsule()
            self.make_helical_magrobot()

            self.rate.sleep()
            rospy.loginfo("!!!!!!!!!!!!!!!!!!! Update the Magrobo Info !!!!!!!!!!!!!!!!!!!!!!!")

    # Arrow visualization in Rviz
    def makeArrow(self):
        self.Arrow_marker = visualization_msgs.msg.Marker()
        self.Arrow_marker.type = visualization_msgs.msg.Marker.ARROW
        self.Arrow_marker.action = self.Arrow_marker.ADD
        self.Arrow_marker.header.frame_id = "world"
        self.Arrow_marker.header.stamp = rospy.Time.now()
        self.Arrow_marker.pose.position.x = self.arrow_info.pose.position.x
        self.Arrow_marker.pose.position.y = self.arrow_info.pose.position.y
        self.Arrow_marker.pose.position.z = self.arrow_info.pose.position.z

        magfield_cartesian_x = self.arrow_info.pose.orientation.x
        magfield_cartesian_y = self.arrow_info.pose.orientation.y
        magfield_cartesian_z = self.arrow_info.pose.orientation.z

        Filtred_magfield_cartesian_x = self.mag_ekf_x.predict_updata(magfield_cartesian_x)
        Filtred_magfield_cartesian_y = self.mag_ekf_y.predict_updata(magfield_cartesian_y)
        Filtred_magfield_cartesian_z = self.mag_ekf_z.predict_updata(magfield_cartesian_z)

        self.mag_sensor_value.data = [Filtred_magfield_cartesian_x, Filtred_magfield_cartesian_y, Filtred_magfield_cartesian_z]

        # self.mag_sensor_value.data = [magfield_cartesian_x, magfield_cartesian_y, magfield_cartesian_z]


        self.mag_sensor_pub.publish(self.mag_sensor_value)


        magfield_vector = np.array(
            [-Filtred_magfield_cartesian_x, -Filtred_magfield_cartesian_y, -Filtred_magfield_cartesian_z])
        magfield_norm = np.linalg.norm(magfield_vector)
        direction_vector = unit_vector(magfield_vector)
        print("magfield_vector++++++++++++++++++++", magfield_vector)
        arrow_cs = make_coords(pos=np.array([self.Arrow_marker.pose.position.x, self.Arrow_marker.pose.position.y,
                                             self.Arrow_marker.pose.position.z]))
        self.arrow_orientation_result = orient_coords_to_axis(arrow_cs, direction_vector,
                                                              'x').quaternion

        self.Arrow_marker.pose.orientation.w = self.arrow_orientation_result[0]
        self.Arrow_marker.pose.orientation.x = self.arrow_orientation_result[1]
        self.Arrow_marker.pose.orientation.y = self.arrow_orientation_result[2]
        self.Arrow_marker.pose.orientation.z = self.arrow_orientation_result[3]
        print("self.Arrow_marker.pose.orientation++++++++++++++++++++", self.Arrow_marker.pose.orientation)

        # regulate the length of the arrow
        self.Arrow_marker.scale.x = magfield_norm / 20
        self.Arrow_marker.scale.y = 0.008
        self.Arrow_marker.scale.z = 0.008
        self.Arrow_marker.color.g = 1.0
        self.Arrow_marker.color.a = 1.0

        self.landmarker_arrow.publish(self.Arrow_marker)

    def makecamera(self):
        self.camera_marker = visualization_msgs.msg.Marker()
        self.camera_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        self.camera_marker.action = self.camera_marker.ADD
        self.camera_marker.header.frame_id = "M_base_link"
        self.camera_marker.header.stamp = rospy.Time.now()
        self.camera_marker.pose.position.x = 1.05
        self.camera_marker.pose.position.y = 0
        self.camera_marker.pose.position.z = 0.2

        self.camera_marker.pose.orientation.x = 0
        self.camera_marker.pose.orientation.y = 0.7071068
        self.camera_marker.pose.orientation.z = 0
        self.camera_marker.pose.orientation.w = 0.7071068
        self.camera_marker.scale.x = 1
        self.camera_marker.scale.y = 1
        self.camera_marker.scale.z = 1
        self.camera_marker.color.r = 0.2
        self.camera_marker.color.g = 0.4
        self.camera_marker.color.b = 0.6
        self.camera_marker.color.a = 1  # Don't forget to set the alpha!
        self.camera_marker.mesh_resource = "camera.STL"
        self.landmarker_camera.publish(self.camera_marker)

    def makeCapsule(self):
        self.magrobo_capsule_marker = visualization_msgs.msg.Marker()
        self.magrobo_capsule_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        self.magrobo_capsule_marker.action = self.magrobo_capsule_marker.ADD
        self.magrobo_capsule_marker.header.frame_id = "world"
        self.magrobo_capsule_marker.header.stamp = rospy.Time.now()
        self.magrobo_capsule_marker.pose.position.x = self.magrobo_state[0]
        self.magrobo_capsule_marker.pose.position.y = self.magrobo_state[1] + 0.03
        self.magrobo_capsule_marker.pose.position.z = self.magrobo_state[2]

        # self.magrobo_capsule_marker.pose.orientation.x = self.magrobo_state[3]
        # self.magrobo_capsule_marker.pose.orientation.y = self.magrobo_state[4]
        # self.magrobo_capsule_marker.pose.orientation.z = self.magrobo_state[5]
        # self.magrobo_capsule_marker.pose.orientation.w = self.magrobo_state[6]
        self.magrobo_capsule_marker.pose.orientation.x = 0.
        self.magrobo_capsule_marker.pose.orientation.y = 0.7068252
        self.magrobo_capsule_marker.pose.orientation.z = 0
        self.magrobo_capsule_marker.pose.orientation.w = 0.7068252
        self.magrobo_capsule_marker.scale.x = 1.1
        self.magrobo_capsule_marker.scale.y = 1.1
        self.magrobo_capsule_marker.scale.z = 1.1
        self.magrobo_capsule_marker.color.r = 0.0
        self.magrobo_capsule_marker.color.g = 1.0
        self.magrobo_capsule_marker.color.b = 0.0
        self.magrobo_capsule_marker.color.a = 1  # Don't forget to set the alpha!
        self.magrobo_capsule_marker.mesh_resource = "capsule.STL"
        self.landmarker_capsule.publish(self.magrobo_capsule_marker)

    def makeHuman(self):
        self.Human_marker = visualization_msgs.msg.Marker()
        self.Human_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        self.Human_marker.action = self.Human_marker.ADD
        self.Human_marker.header.frame_id = "world"
        self.Human_marker.header.stamp = rospy.Time.now()
        self.Human_marker.pose.position.x = 0.
        self.Human_marker.pose.position.y = 0.
        self.Human_marker.pose.position.z = 0.152

        self.Human_marker.pose.orientation.x = 0
        self.Human_marker.pose.orientation.y = 0
        self.Human_marker.pose.orientation.z = 0
        self.Human_marker.pose.orientation.w = 1
        self.Human_marker.scale.x = 0.6
        self.Human_marker.scale.y = 0.6
        self.Human_marker.scale.z = 0.7
        self.Human_marker.color.r = 0.4
        self.Human_marker.color.g = 0.75
        self.Human_marker.color.b = 0.9
        self.Human_marker.color.a = 0.4  # Don't forget to set the alpha!
        self.Human_marker.mesh_resource = "phantom.STL"
        # self.landmarker_human.publish(self.Human_marker)

    def make_helical_magrobot(self):
        self.helical_magrobot_marker = visualization_msgs.msg.Marker()
        self.helical_magrobot_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        self.helical_magrobot_marker.action = self.helical_magrobot_marker.ADD
        self.helical_magrobot_marker.header.frame_id = "world"
        self.helical_magrobot_marker.header.stamp = rospy.Time.now()
        self.helical_magrobot_marker.pose.position.x = self.magrobo_state[0]
        self.helical_magrobot_marker.pose.position.y = self.magrobo_state[1]
        self.helical_magrobot_marker.pose.position.z = self.magrobo_state[2]

        self.helical_magrobot_marker.pose.orientation.x = 0.
        self.helical_magrobot_marker.pose.orientation.y = 0.7068252
        self.helical_magrobot_marker.pose.orientation.z = 0.7068252
        self.helical_magrobot_marker.pose.orientation.w = 0.0
        self.helical_magrobot_marker.scale.x = 2.0
        self.helical_magrobot_marker.scale.y = 3
        self.helical_magrobot_marker.scale.z = 3
        self.helical_magrobot_marker.color.r = 0.67
        self.helical_magrobot_marker.color.g = 0.67
        self.helical_magrobot_marker.color.b = 0.67
        self.helical_magrobot_marker.color.a = 1  # Don't forget to set the alpha!
        self.helical_magrobot_marker.mesh_resource = "helix_swimmer.STL"
        self.landmarker_helical_magrobot.publish(self.helical_magrobot_marker)


    def makeDoctor(self):
        self.Doctor_marker = visualization_msgs.msg.Marker()
        self.Doctor_marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        self.Doctor_marker.action = self.Doctor_marker.ADD
        self.Doctor_marker.header.frame_id = "M_base_link"
        self.Doctor_marker.header.stamp = rospy.Time.now()
        self.Doctor_marker.pose.position.x = 1.0763
        self.Doctor_marker.pose.position.y = 1.59624
        self.Doctor_marker.pose.position.z = 0.77421

        self.Doctor_marker.pose.orientation.x = 0
        self.Doctor_marker.pose.orientation.y = 0
        self.Doctor_marker.pose.orientation.z = -0.7071068
        self.Doctor_marker.pose.orientation.w = 0.7071068
        self.Doctor_marker.scale.x = 1
        self.Doctor_marker.scale.y = 1
        self.Doctor_marker.scale.z = 1
        self.Doctor_marker.color.r = 1
        self.Doctor_marker.color.g = 1
        self.Doctor_marker.color.b = 1
        self.Doctor_marker.color.a = 1  # Don't forget to set the alpha!
        self.Doctor_marker.mesh_resource = "doctor.STL"
        self.landmarker_doctor.publish(self.Doctor_marker)

 

    def get_quaternion_norm(self, q):
        r"""
        Return the norm of a quaternion: :math:`|q| = \sqrt(x^2 + y^2 + z^2 + w^2)`
        Args:
            q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)
        Returns:
            float: norm of a quaternion
        References:
            - [1] https://www.3dgep.com/understanding-quaternions/#Quaternions
        """
        return np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)

    def normalize_quaternion(self, q):
        r"""
        Return the normalized quaternion; the quaternion divided by its norm.
        Args:
            q (np.array[float[4]], quaternion.quaternion): quaternion (it doesn't have to be a unit quaternion)
        Returns:
            np.array[float[4]], quaternion.quaternion: normalized quaternion.
        """
        return q / self.get_quaternion_norm(q)


def main():
    magrobo_rviz_env = MagroboRvizEnv()
    try:
        magrobo_rviz_env.update()
    except KeyboardInterrupt:
        print("Shutting down node")


if __name__ == "__main__":
    main()
