#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @File : arm_base.py
# @Time : 2022/5/9 下午5:55
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import sys, os
from copy import deepcopy
import moveit_msgs.msg
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetStateValidityRequest, GetStateValidity
import controller_manager_msgs.srv as cm_srv
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from sensor_msgs.msg import JointState
import numpy as np
from moveit_commander import MoveGroupCommander
import moveit_commander
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import PoseStamped, Pose
import tf
from src.probot_g604.probot_tools import helpers, spalg, conversions, transformations
from src.probot_g604.probot_tools.pyquaternion import Quaternion

class ArmBase():
    """ Base methods for any arm controlled via MoveIt """
    def __init__(self, group_name):
        # First initialize `moveit_commander`_:
        moveit_commander.roscpp_initialize(sys.argv)
        self.group = MoveGroupCommander(group_name)

        # 设置目标位置所使用的参考坐标系
        if group_name == 'L_manipulator':
            self.group.set_pose_reference_frame("L_base_link")
        if group_name == 'M_manipulator':
            self.group.set_pose_reference_frame("M_base_link")
        if group_name == 'R_manipulator':
            self.group.set_pose_reference_frame("R_base_link")
        ##### for test
        if group_name == 'manipulator':
            self.group.set_pose_reference_frame("base_link")
        # MoveIt! Setup
        self.group.set_planning_time(5)
        # Set the number of times the motion plan is to be computed, the default value is 1
        self.group.set_num_planning_attempts(2) # 10

        # setup planner
        # Specify a planner to be used for further planning
        self.group.set_planner_id("RRTConnect")
        self.group.allow_replanning(True)

        # self.group.set_workspace()
        self.robot = moveit_commander.RobotCommander()

        # 获取终端link的名称
        self.end_effector_link = self.group.get_end_effector_link()
        rospy.loginfo(self.group.get_pose_reference_frame())

        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        self.goal_position_tolerance = 0.003
        self.goal_orientation_tolerance = 0.01
        self.group.set_goal_position_tolerance(self.goal_position_tolerance)
        self.group.set_goal_orientation_tolerance(self.goal_orientation_tolerance)
        # 设置允许的最大速度和加速度
        self.group.set_max_acceleration_scaling_factor(0.5)
        self.group.set_max_velocity_scaling_factor(0.5)

        self.tf_listener = tf.TransformListener()

        try:
            self.tf_listener.waitForTransform("world", self.group.get_pose_reference_frame(), rospy.Time(0), rospy.Duration(1))
            self.transform2target = self.tf_listener.fromTranslationRotation(
                *self.tf_listener.lookupTransform("world", self.group.get_pose_reference_frame(), rospy.Time(0)))
        except Exception as e:
            print(e)

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory, queue_size=10)
        rospy.wait_for_service('compute_ik')
        rospy.wait_for_service('compute_fk')

        self.moveit_ik_srv = rospy.ServiceProxy('/compute_ik', moveit_msgs.srv.GetPositionIK)
        self.moveit_fk_srv = rospy.ServiceProxy('/compute_fk', moveit_msgs.srv.GetPositionFK)

    def get_current_pose_stamped(self):
        return self.group.get_current_pose()

    def get_current_pose_base_link(self):
        pose_stamped = self.get_current_pose_stamped()
        try:
            pose_goal_base_link = self.tf_listener.transformPose(self.group.get_pose_reference_frame(), pose_stamped)
        except:
            return False
        return pose_goal_base_link

    def world_frame2base_link(self, pose_stamped):
        pose_base_link = conversions.transform_pose(self.group.get_pose_reference_frame(),
                                                    np.linalg.inv(self.transform2target),
                                                    pose_stamped)
        return pose_base_link

    def base_link2world_frame(self, pose_stamped):
        pose_world_frame = conversions.transform_pose("world", self.transform2target, pose_stamped)
        return pose_world_frame

    def check_goal_pose_reached(self, goal_pose):
        current_pose = self.get_current_pose_base_link()
        return helpers.all_close(goal_pose.pose, current_pose.pose, 0.005)

    def check_new_goal_pose(self, goal_pose, tolerance):
        current_pose = self.get_current_pose_base_link()
        return helpers.all_close(goal_pose.pose, current_pose.pose, tolerance)

    def joint_angle(self, joint):
        """
        Return the requested joint angle.

        @type joint: str
        @param joint: name of a joint
        @rtype: float
        @return: angle in radians of individual joint
        """
        return self.joint_traj_controller.get_joint_positions()[joint]

    def joint_angles(self):
        """
        Return all joint angles.

        @rtype: dict({str:float})
        @return: unordered dict of joint name Keys to angle (rad) Values
        """
        return self.joint_traj_controller.get_joint_positions()

    def joint_velocity(self, joint):
        """
        Return the requested joint velocity.

        @type joint: str
        @param joint: name of a joint
        @rtype: float
        @return: velocity in radians/s of individual joint
        """
        return self.joint_traj_controller.get_joint_velocities()[joint]

    def joint_velocities(self):
        """
        Return all joint velocities.

        @rtype: dict({str:float})
        @return: unordered dict of joint name Keys to velocity (rad/s) Values
        """
        return self.joint_traj_controller.get_joint_velocities()

    ### Basic Control Methods ###

    def get_jacobian_world_frame(self, jacobian):
        base_link2world_frame_rotation_matrix = np.linalg.inv(self.transform2target)[0:3, 0:3]
        construct_matrix = np.vstack((np.hstack((base_link2world_frame_rotation_matrix, np.zeros((3, 3)))),
                                      np.hstack((np.zeros((3, 3)), base_link2world_frame_rotation_matrix))))

        return np.matmul(construct_matrix, jacobian)


    def set_joint_positions(self,
                            position,
                            velocities=None,
                            accelerations=None,
                            wait=False,
                            t=5.0):
        self.joint_traj_controller.add_point(positions=position,
                                             time=t,
                                             velocities=velocities,
                                             accelerations=accelerations)
        self.joint_traj_controller.start(delay=0.01, wait=wait)
        self.joint_traj_controller.clear_points()
        return True

    def set_joint_trajectory(self, trajectory, velocities=None, accelerations=None, t=5.0):
        dt = float(t) / float(len(trajectory))

        vel = None
        acc = None

        if velocities is not None:
            vel = [velocities] * 6
        if accelerations is not None:
            acc = [accelerations] * 6

        for i, q in enumerate(trajectory):
            self.joint_traj_controller.add_point(positions=q,
                                                 time=(i + 1) * dt,
                                                 velocities=vel,
                                                 accelerations=acc)
        self.joint_traj_controller.start(delay=0.01, wait=True)
        self.joint_traj_controller.clear_points()

##########################  Moveit Methods Starts #######################

    def go_to_joint_state(self, joint_goal):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        # Planning to a Joint Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the joint values from the group and adjust some of the values:
        arm_joint_goal = self.group.get_current_joint_values()
        arm_joint_goal[0] = joint_goal[0] 
        arm_joint_goal[1] = joint_goal[1]
        arm_joint_goal[2] = joint_goal[2]
        arm_joint_goal[3] = joint_goal[3]
        arm_joint_goal[4] = joint_goal[4]
        arm_joint_goal[5] = joint_goal[5]



        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.group.go(arm_joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        # self.group.stop()
        current_joints = self.group.get_current_joint_values()
        return helpers.all_close(arm_joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self, pose_goal):
        "pose_goal: frame_id is ns+ base_link"
        move_group = self.group
        assert pose_goal.header.frame_id == move_group.get_pose_reference_frame()
        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        move_group.set_pose_target(pose_goal)

        # Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()
        rospy.loginfo("success")

    def go_to_home_cartesian_single_arm(self, wait=True):
        home_pose = Pose()
        home_pose.position.x = 0.2800158067
        home_pose.position.y = -0.000563784282564
        home_pose.position.z = 0.413496428357
        home_pose.orientation.w = 1

        plan, friction = self.plan_cartesian_path(home_pose)
        self.execute_plan(plan)

        rospy.loginfo("========= go to home cartesian successfully =========")

    def plan_cartesian_path(self, waypoint):
        fraction = 0.0  # 路径规划覆盖率
        maxtries = 100  # 最大尝试规划次数
        attempts = 0  # 已经尝试规划次数

        # # 获取当前的pose作为机械臂运动的起始位姿
        # start_pose = self.get_current_pose_base_link().pose
        waypoints = []
        # 如果添加下面的初始路径点：Trajectory message contains waypoints that are not strictly increasing in time
        # waypoints.append(start_pose)
        waypoints.append(waypoint)

        if isinstance(waypoints[0], list):
            waypoints = waypoints[0]
        else:
            waypoints = waypoints

        self.group.set_start_state_to_current_state()

        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.group.compute_cartesian_path(
                waypoints,  # waypoint poses，路点列表
                self.goal_position_tolerance,  # eef_step，终端步进值
                0.0,  # jump_threshold，跳跃阈值
                True)  # avoid_collisions，避障规划

            # 尝试次数累加
            attempts += 1

            # 打印运动规划进程
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        # if fraction != 1.0:
        #     raise ValueError("Unable to plan path")
        return plan, fraction

    def display_trajectory(self, plan):
        # You can ask RViz to visualize a plan (aka trajectory) for you. But the
        # group.plan() method does this automatically so this is not that useful
        # here (it just displays the same trajectory again):
        ##
        # A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        # We populate the trajectory_start with our current robot state to copy over
        # any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.group.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, plan):

        # Executing a Plan
        # Use execute if you would like the robot to follow
        # the plan that has already been computed:
        success = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        print("Done!")
        return success

        # **Note:** The robot's current joint state must be within some tolerance of the
        # first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail

    def get_ik_client(self, request):
        inverse_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        ret = inverse_ik(request)
        return ret

    def compute_ik(self, target_pose, joints_seed=None, timeout=0.01, retry=False, allow_collisions=False):
        """
                    Compute the Inverse Kinematics for a move group the MoveIt service
                    target_pose: PoseStamped
                    joints_seed: list, must be in the same order as defined for that group
                    timeout: float, overrides the timeout for the IK solver (higher is sometimes better)
                    retry: bool, for 10 secs send the same request until success or timeout
                    allow_collisions: bool, compute IK with or without considering collisions with other objects (Likely self-collisions are always considered)
                    return
                    solution: `list`: the joint values are in the same order as defined for that group
        """
        if isinstance(target_pose, PoseStamped):
            ik_request = moveit_msgs.msg.PositionIKRequest()
            ik_request.avoid_collisions = not allow_collisions
            ik_request.timeout = rospy.Duration(timeout)
            ik_request.pose_stamped = target_pose
            ik_request.group_name = self.group.get_name()
            ik_request.ik_link_name = self.group.get_end_effector_link()
            ik_request.robot_state.joint_state.name = self.group.get_active_joints()
            # 这句话会增加耗时
            # ik_request.robot_state.joint_state.position = joints_seed if joints_seed is not None else self.group.get_current_joint_values()
        else:
            rospy.logerr("Unsupported type of target_pose %s" % type(target_pose))
            raise

        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request = ik_request
        res = self.moveit_ik_srv.call(req)

        if retry:
            start_time = rospy.get_time()
            while res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS \
                    and not rospy.is_shutdown() and (rospy.get_time() - start_time < 10):
                res = self.moveit_ik_srv.call(req)

        if res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS:
            rospy.logwarn("compute IK failed with code: %s" % res.error_code.val)
            error_str = self.moveit_error_string(res.error_code.val)
            rospy.logwarn("compute IK failed with: %s" % error_str)
            return None

        solution = []
        for joint_name in self.group.get_active_joints():
            solution.append(res.solution.joint_state.position[res.solution.joint_state.name.index(joint_name)])
        return solution

    def compute_fk(self, robot_state=None, tcp_link=None, frame_id=None):
        """
            Compute the Forward kinematics for a move group using the MoveIt service
            robot_state: list, tuple, or moveit_msgs.msg.RobotState
                         if passed as `list` or `tuple`: assumes that the joint values are in the same order as defined for that group
        """
        if robot_state:
            if isinstance(robot_state, moveit_msgs.msg.RobotState):
                robot_state_ = robot_state
            elif isinstance(robot_state, (list, tuple, np.ndarray)):
                robot_state_ = moveit_msgs.msg.RobotState()
                robot_state_.joint_state.name = self.group.get_active_joints()
                robot_state_.joint_state.position = list(robot_state)
            else:
                rospy.logerr("Unsupported type of robot_state %s" % type(robot_state))
                raise
        else:
            return self.compute_fk(robot_state=self.group.get_current_joint_values())
        req = moveit_msgs.srv.GetPositionFKRequest()
        req.fk_link_names = [tcp_link if tcp_link else self.group.get_end_effector_link()]
        req.robot_state = robot_state_
        res = self.moveit_fk_srv.call(req)
        if res.error_code.val != moveit_msgs.msg.MoveItErrorCodes.SUCCESS:
            rospy.logwarn("compute FK failed with code: %s" % res.error_code.val)
            return False
        else:
            if frame_id:
                return self.listener.transformPose(frame_id, res.pose_stamped[0])
            return res.pose_stamped[0]

    def joint_configuration_changes(self, start, end, tolerance=0.1):
        """ Returns True if the sign of any joint angle changes during the motion,
            and the joint angle is not near 0 (0.01 rad =~ 0.5 deg tolerance).
        """
        signs = np.sign(np.array(start) * np.array(end))

        if np.all(signs > 0):
            return False  # = all OK

        joint_changes_small = True
        for i in range(len(signs)):

            if signs[i] < 0:
                if abs(start[i] < tolerance) or abs(end[i] < tolerance):
                    rospy.logdebug("Joint changes sign, but the change is small. Ignoring.")
                    rospy.logdebug("start[i] = %d6, end[i] = %d6", (start[i], end[i]))
                    continue
                rospy.logerr("Joint angle " + str(i) + " would change sign!")
                print("start[i] = %d6, end[i] = %d6", (start[i], end[i]))
                joint_changes_small = False
        if joint_changes_small:
            return False  # = all OK
        else:
            return True  # Joints change

    ##########################  Moveit Methods Ends #######################

    def move_relative(self, delta, relative_to_ee=False, wait=True, t=5.):
        """
            Move relative to the current pose of the robot
            delta: array[6], translations and rotations(euler angles) from the current pose
            relative_to_ee: bool, whether to consider the delta relative to the robot's base or its end-effector (TCP)
            wait: bool, wait for the motion to be completed
            t: float, duration of the motion (how fast it will be)
        """
        cpose = self.end_effector()
        cmd = transformations.pose_euler_to_quaternion(cpose, delta, ee_rotation=relative_to_ee)
        return self.set_target_pose(cmd, wait=True, t=t)

    def move_linear(self, pose, eef_step=0.01, t=5.0):
        """
            CAUTION: simple linear interpolation
            pose: array[7], target translation and rotation
            granularity: int, number of point for the interpolation
            t: float, duration in seconds
        """
        joint_trajectory = self.compute_cartesian_path(pose, eef_step, t)
        self.set_joint_trajectory(joint_trajectory, t=t)

    def compute_cartesian_path(self, pose, eef_step=0.01, t=5.0):
        """
            CAUTION: simple linear interpolation
            pose: array[7], target translation and rotation
            granularity: int, number of point for the interpolation
            t: float, duration in seconds
        """
        cpose = self.get_current_pose()
        translation_dist = np.linalg.norm(cpose[:3])
        rotation_dist = Quaternion.distance(transformations.vector_to_pyquaternion(cpose[3:]), transformations.vector_to_pyquaternion(pose[3:])) / 2.0

        steps = int((translation_dist + rotation_dist) / eef_step)

        points = np.linspace(cpose[:3], pose[:3], steps)
        rotations = Quaternion.intermediates(transformations.vector_to_pyquaternion(cpose[3:]), transformations.vector_to_pyquaternion(pose[3:]), steps, include_endpoints=True)

        joint_trajectory = []

        for i, (point, rotation) in enumerate(zip(points, rotations)):
            cmd = np.concatenate([point, transformations.vector_from_pyquaternion(rotation)])
            q_guess = None if i < 2 else np.mean(joint_trajectory[:-1], 0)
            q = self._solve_ik(cmd, q_guess)
            if q is not None:  # ignore points with no IK solution, can we do better?
                joint_trajectory.append(q)

        dt = t/float(len(joint_trajectory))
        # TODO(cambel): is this good enough to catch big jumps due to IK solutions?
        return spalg.jump_threshold(np.array(joint_trajectory), dt, 2.5)

    def moveit_error_string(self, val):
        """Returns a string associated with a MoveItErrorCode.

        Args:
            val: The val field from moveit_msgs/MoveItErrorCodes.msg

        Returns: The string associated with the error value, 'UNKNOWN_ERROR_CODE'
            if the value is invalid.
        """
        if val == MoveItErrorCodes.SUCCESS:
            return 'SUCCESS'
        elif val == MoveItErrorCodes.FAILURE:
            return 'FAILURE'
        elif val == MoveItErrorCodes.PLANNING_FAILED:
            return 'PLANNING_FAILED'
        elif val == MoveItErrorCodes.INVALID_MOTION_PLAN:
            return 'INVALID_MOTION_PLAN'
        elif val == MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
            return 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE'
        elif val == MoveItErrorCodes.CONTROL_FAILED:
            return 'CONTROL_FAILED'
        elif val == MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA:
            return 'UNABLE_TO_AQUIRE_SENSOR_DATA'
        elif val == MoveItErrorCodes.TIMED_OUT:
            return 'TIMED_OUT'
        elif val == MoveItErrorCodes.PREEMPTED:
            return 'PREEMPTED'
        elif val == MoveItErrorCodes.START_STATE_IN_COLLISION:
            return 'START_STATE_IN_COLLISION'
        elif val == MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS:
            return 'START_STATE_VIOLATES_PATH_CONSTRAINTS'
        elif val == MoveItErrorCodes.GOAL_IN_COLLISION:
            return 'GOAL_IN_COLLISION'
        elif val == MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS:
            return 'GOAL_VIOLATES_PATH_CONSTRAINTS'
        elif val == MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED:
            return 'GOAL_CONSTRAINTS_VIOLATED'
        elif val == MoveItErrorCodes.INVALID_GROUP_NAME:
            return 'INVALID_GROUP_NAME'
        elif val == MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS:
            return 'INVALID_GOAL_CONSTRAINTS'
        elif val == MoveItErrorCodes.INVALID_ROBOT_STATE:
            return 'INVALID_ROBOT_STATE'
        elif val == MoveItErrorCodes.INVALID_LINK_NAME:
            return 'INVALID_LINK_NAME'
        elif val == MoveItErrorCodes.INVALID_OBJECT_NAME:
            return 'INVALID_OBJECT_NAME'
        elif val == MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
            return 'FRAME_TRANSFORM_FAILURE'
        elif val == MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE:
            return 'COLLISION_CHECKING_UNAVAILABLE'
        elif val == MoveItErrorCodes.ROBOT_STATE_STALE:
            return 'ROBOT_STATE_STALE'
        elif val == MoveItErrorCodes.SENSOR_INFO_STALE:
            return 'SENSOR_INFO_STALE'
        elif val == MoveItErrorCodes.NO_IK_SOLUTION:
            return 'NO_IK_SOLUTION'
        else:
            return 'UNKNOWN_ERROR_CODE'

class ControlSwitcher:
    # Class to switch between controllers in ROS
    def __init__(self, controllers, controller_manager_node='/controller_manager'):
        self.controllers = controllers
        rospy.wait_for_service(controller_manager_node + "/switch_controller")
        rospy.wait_for_service(controller_manager_node + "/list_controllers")

        self.switcher_srv = rospy.ServiceProxy(controller_manager_node + "/switch_controller", cm_srv.SwitchController)
        self.lister_srv = rospy.ServiceProxy(controller_manager_node + "/list_controller", cm_srv.ListControllers)

    def switch_controllers(self, start_controller_names):
        rospy.sleep(0.5)
        # Get list of controller full names to start and stop
        start_controllers = [self.controllers[start_controller] for start_controller in start_controller_names]
        stop_controllers = [self.controllers[n] for n in self.controllers if n not in start_controller_names]

        controller_switch_msg = cm_srv.SwitchControllerRequest()
        controller_switch_msg.strictness = 1
        controller_switch_msg.start_controllers = start_controllers
        controller_switch_msg.stop_controllers = stop_controllers

        result = self.switcher_srv(controller_switch_msg).ok
        if result:
            rospy.logdebug('Successfully switched to controllers {} ({})'.format(start_controllers, start_controller_names))
            return result
        else:
            rospy.logdebug("Failed switching controllers")
            return False

    def stop_controllers(self):
        self.switch_controllers([])

class StateValidity():
    def __init__(self):
        # prepare msg to interface with moveit
        self.rs = RobotState()
        self.rs.joint_state.name = JOINT_NAMES
        self.rs.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_states_received = False
        # subscribe to joint joint states
        self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.jointStatesCB, queue_size=1)
        # prepare service for collision check
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is avaiable')

    def jointStatesCB(self, msg):
        '''
        update robot state
        '''
        self.rs.joint_state.position = [msg.position[3], msg.position[2], msg.position[0], msg.position[4],
                                        msg.position[5], msg.position[6]]
        self.joint_states_received = True

    def start_collision_checker(self):
        while not self.joint_states_received:
            rospy.sleep(0.1)
        rospy.loginfo('joint states received! continue')
        self.checkCollision()
        rospy.spin()

    def checkCollision(self):
        '''
        check if robotis in collision
        '''
        if self.getStateValidity().valid:
            rospy.loginfo('robot not in collision, all ok!')
        else:
            rospy.logwarn('robot in collision')

    def getStateValidity(self, group_name='manipulator', constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints

        # State validity Service call
        result = self.sv_srv.call(gsvr)
        return result