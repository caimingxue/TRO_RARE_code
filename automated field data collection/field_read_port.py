#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : field_read_port.py
# @Time : 2022/7/16 下午2:43
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import numpy as np
import serial  #
import threading
import rospy
from src.probot_g604.probot_tools.arm_base import ArmBase
from src.probot_g604.probot_tools.conversions import *
from magrobot.controls.mag_manipulability import Manipulability
from std_msgs.msg import *

np.set_printoptions(precision=5)
np.set_printoptions(suppress=False)

STRGLO = ""  # 
BOOL = True  #

import binascii


class Uart_operation():
    def __init__(self):
        rospy.init_node('Uart')
        self.portx = "/dev/ttyACM4"
        self.bps = 9600
        self.timeout = None
        self.pub_mag_arrow_pose = rospy.Publisher("mag_arrow_pose", PoseStamped, queue_size=10)

    # open the port
    def DOpenPort(self):
        ret = False
        try:
            ser = serial.Serial(self.portx, self.bps, timeout=self.timeout)
            if (ser.is_open):
                ret = True
                threading.Thread(target=self.ReadData, args=(ser,)).start()
        except Exception as e:
            print("------：", e)
        return ser, ret

    # Close the Port
    def DColsePort(self, ser):
        global BOOL
        BOOL = False
        ser.close()

    # write the data
    def DWritePort(self, ser, text):
        result = ser.write(text.encode("gbk"))  

    def str_to_hexStr(self, string):
        str_bin = string.encode('utf-8')
        return binascii.hexlify(str_bin).decode('utf-8')

    # read mag data
    def ReadData(self, ser):
        global STRGLO, BOOL, strl, strl1, strl2, strl3
        strl = ""
        strl1 = ""
        strl2 = ""
        strl3 = ""
        cnt = 0
        cnt_all = []

        while BOOL:
            if ser.in_waiting:
                STRGLO = " "  
                # STRGLO = ser.read()
                # temp = str_to_hexStr(STRGLO)
                STRGLO = ser.readline()
                strl = str(STRGLO)
                print("byte length：", STRGLO)

                while cnt < len(strl):
                    if strl[cnt] == b';':
                        cnt_all.append(cnt)
                    cnt = cnt + 1
                cnt = 0
                strl1 = strl[0:cnt_all[0]]
                strl2 = strl[cnt_all[0] + 2:cnt_all[1]]
                strl3 = strl[cnt_all[1] + 2:]

                magfield_pose = PoseStamped()
                magfield_pose.header.frame_id = "world"
                magfield_pose.header.stamp = rospy.Time.now()
                magfield_pose.pose.position.x = 0.160
                magfield_pose.pose.position.y = 0
                magfield_pose.pose.position.z = 0.005

                magfield_pose.pose.orientation.x = float(strl1)
                magfield_pose.pose.orientation.y = float(strl2)
                magfield_pose.pose.orientation.z = float(strl3)
                self.pub_mag_arrow_pose.publish(magfield_pose)

                strl = ""
                cnt_all = []


def main():
    uart = Uart_operation()
    ser, ret = uart.DOpenPort()


if __name__ == "__main__":
    main()
