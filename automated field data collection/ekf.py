#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : ekf.py
# @Time : 2022/7/17 下午2:10
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import numpy as np
import math

class EKF(object):
    def __init__(self):
        self.xhat = [0]
        self.P = [1.0]
        self.xhatminus = [0]  # a priori estimate of x
        self.Pminus = [0]  # a priori error estimate
        self.K = [0]  # a priori error estimate
        self.R = 0.4 ** 2  # estimate of measurement variance, change to see effect
        self.k = 1
        self.Q = 1e-5

    def predict_updata(self, sensor_data):
        self.sensor_data = sensor_data
        ############predict#############
        self.xhatminus.append(self.xhat[-1])  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        self.Pminus.append(self.P[- 1] + self.Q)  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
        ############updata#############
        self.K.append(self.Pminus[-1] / (self.Pminus[-1] + self.R))  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        self.xhat.append(self.xhatminus[-1] + self.K[-1] * (self.sensor_data - self.xhatminus[-1]))  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        self.P.append((1 - self.K[-1]) * self.Pminus[-1])  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
        self.filter_sensor_data = self.xhat[-1]
        return self.filter_sensor_data

if __name__ == "__main__":
    EKF()

