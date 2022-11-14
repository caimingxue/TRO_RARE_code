#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : pandas_to_csv.py
# @Time : 2022/9/2 下午8:27
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import numpy as np
import pandas as pd
from datetime import datetime
import os

timestr = datetime.now().strftime("%Y%m%d-%H%M%S")


class Pandas_2_CSV():
    def __init__(self, file_name, header):
        self.file_name = file_name + '_' + timestr + '.csv'
        self.header = header

    def write_csv(self, data):
        # 字典中的key值即为csv中列名
        self.df = pd.DataFrame({self.header[0]: [data[0]], self.header[1]: [data[1]], self.header[2]: [data[2]],
                                self.header[3]: [data[3]], self.header[4]: [data[4]], self.header[5]: [data[5]],
                                # self.header[6]: [data[6]], self.header[7]: [data[7]], self.header[8]: [data[8]],
                                # self.header[9]: [data[9]], self.header[10]: [data[10]], self.header[11]: [data[11]]
                                })
        if not os.path.exists(self.file_name):
            self.df.to_csv(self.file_name, mode='a', index=False, index_label=True)
        else:
            self.df.to_csv(self.file_name, mode='a', index=False, index_label=True, header=False)


class Read_CSV():
    def __init__(self, file_name):
        self.file = file_name

    # self.file = os.path.join("./", self.file)

    def read_csv(self):
        pd_data = pd.read_csv(self.file)
        position = np.array(["field_x", "field_y", "field_z"])

        field_x = pd_data[position[0]].T.tolist()

        field_y = pd_data[position[1]].T.tolist()

        field_z = pd_data[position[2]].T.tolist()

        return field_x, field_y, field_z


# file_name = 'optim_data'
# header = ["coil_current1", "coil_current2", "coil_current3"]
# test = Pandas_2_CSV(file_name, header)
