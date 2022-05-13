# -*- coding: utf-8 -*-

import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import time

plt.rcParams['figure.figsize'] = (6.0, 4.02)
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})
output_file = './results'
data = torch.load('./time_list_rep_celebA.pt', map_location=torch.device('cpu'))


# %% 绘制直方图
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# bplot1 = ax1.boxplot(data[0]['sort'],
#                       vert=True,  # vertical box alignment
#                       patch_artist=True)  # will be used to label x-ticks

# bplot2 = ax2.boxplot(data[0]['keep_status'],
#                       notch=True,  # notch shape
#                       vert=True,  # vertical box alignment
#                       patch_artist=True)  # will be used to label x-ticks
# plt.show()

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# bplot1 = ax1.boxplot(data[1]['sort'],
#                       vert=True,  # vertical box alignment
#                       patch_artist=True)  # will be used to label x-ticks
# bplot2 = ax2.boxplot(data[1]['keep_status'],
#                       notch=True,  # notch shape
#                       vert=True,  # vertical box alignment
#                       patch_artist=True)  # will be used to label x-ticks
# plt.show()

# %% 绘制箱线图
# 处理数据
all_data = [data[0]['sort'], data[0]['keep_status'], data[1]['sort'], data[1]['keep_status']]
labels = ['1', '2', '3', '4']
colors = ['pink', 'lightyellow', 'lightblue', 'lightgreen']
save_path = './time_list_rep_cifar10.pdf'

# plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
bplot = ax.boxplot(all_data, notch=True, vert=True, patch_artist=True, labels=labels)
# ax.set_title('time')
ax.set_ylabel('Time/s', fontsize=20)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.grid(linestyle="--", alpha=0.5)
ax.yaxis.grid(True)
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
plt.show()
