import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np

# plt.rcParams['figure.figsize'] = (6.0, 4.02)
# plt.rcParams['savefig.dpi'] = 900  # 图片像素
plt.rcParams['figure.dpi'] = 200  # 分辨率
# plt.rcParams['errorbar.capsize'] = 1.0
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 5})

y1 = [1, 2, 3, 4]
y2 = [5, 6, 7, 8]
y3 = [9, 10, 11, 12]
color1 = 'red'
color2 = 'green'
color3 = 'blue'

fig, ax = plt.subplots(2, 4)
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Boston dataset
x1_00 = [4.25, 2.92, 2.72, 3.40]
x2_00 = [2.77, 2.82, 2.79, 2.80]
x3_00 = [3.01, 2.85, 3.01, 3.77]
xerr1_00 = [0.31, 0.32, 0.31, 0.32]
xerr2_00 = [0.22, 0.22, 0.33, 0.22]
xerr3_00 = [0.22, 0.31, 0.25, 0.25]
ax[0][0].errorbar(x1_00, y1, xerr=xerr1_00, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[0][0].errorbar(x2_00, y2, xerr=xerr2_00, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[0][0].errorbar(x3_00, y3, xerr=xerr3_00, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[0][0].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29),
             yticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), title='Boston  (N=506  D=13)',
             yticklabels=(
                 'NOVI DGP 2', 'NOVI DGP 3', 'NOVI DGP 4', 'NOVI DGP 5', 'SGHMC DGP 2', 'SGHMC DGP 3', 'SGHMC DGP 4',
                 'SGHMC DGP 5', 'DSVI DGP 2', 'DSVI DGP 3', 'DSVI DGP 4', 'DSVI DGP 5'))
ax[0][0].set_aspect(0.25)

# Energy dataset
x1_01 = [4.25, 2.92, 2.72, 3.40]
x2_01 = [2.77, 2.82, 2.79, 2.80]
x3_01 = [3.01, 2.85, 3.01, 3.77]
xerr1_01 = [0.31, 0.32, 0.21, 0.32]
xerr2_01 = [0.22, 0.32, 0.23, 0.32]
xerr3_01 = [0.22, 0.31, 0.25, 0.25]
ax[0][1].errorbar(x1_01, y1, xerr=xerr1_01, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[0][1].errorbar(x2_01, y2, xerr=xerr2_01, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[0][1].errorbar(x3_01, y3, xerr=xerr3_01, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[0][1].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Energy  (N=768  D=8)')
ax[0][1].set_aspect(0.25)

# Power dataset
x1_02 = [4.25, 2.92, 2.72, 3.40]
x2_02 = [2.77, 2.82, 2.79, 2.80]
x3_02 = [3.01, 2.85, 3.01, 3.77]
xerr1_02 = [0.31, 0.32, 0.21, 0.32]
xerr2_02 = [0.22, 0.32, 0.23, 0.32]
xerr3_02 = [0.22, 0.31, 0.25, 0.25]
ax[0][2].errorbar(x1_02, y1, xerr=xerr1_02, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[0][2].errorbar(x2_02, y2, xerr=xerr2_02, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[0][2].errorbar(x3_02, y3, xerr=xerr3_02, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[0][2].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Power  (N=9568  D=4)')
ax[0][2].set_aspect(0.25)

# Concrete dataset
x1_03 = [4.25, 2.92, 2.72, 3.40]
x2_03 = [2.77, 2.82, 2.79, 2.80]
x3_03 = [3.01, 2.85, 3.01, 3.77]
xerr1_03 = [0.31, 0.32, 0.21, 0.32]
xerr2_03 = [0.22, 0.32, 0.23, 0.32]
xerr3_03 = [0.22, 0.31, 0.25, 0.25]
ax[0][3].errorbar(x1_03, y1, xerr=xerr1_03, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[0][3].errorbar(x2_03, y2, xerr=xerr2_03, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[0][3].errorbar(x3_03, y3, xerr=xerr3_03, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[0][3].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Concrete  (N=1030  D=8)')
ax[0][3].set_aspect(0.25)

# Yacht dataset
x1_10 = [4.25, 2.92, 2.72, 3.40]
x2_10 = [2.77, 2.82, 2.79, 2.80]
x3_10 = [3.01, 2.85, 3.01, 3.77]
xerr1_10 = [0.31, 0.32, 0.31, 0.32]
xerr2_10 = [0.22, 0.22, 0.33, 0.22]
xerr3_10 = [0.22, 0.31, 0.25, 0.25]
ax[1][0].errorbar(x1_10, y1, xerr=xerr1_10, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[1][0].errorbar(x2_10, y2, xerr=xerr2_10, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[1][0].errorbar(x3_10, y3, xerr=xerr3_10, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[1][0].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29),
             yticks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), title='Yacht  (N=308  D=6)',
             yticklabels=(
                 'NOVI DGP 2', 'NOVI DGP 3', 'NOVI DGP 4', 'NOVI DGP 5', 'SGHMC DGP 2', 'SGHMC DGP 3', 'SGHMC DGP 4',
                 'SGHMC DGP 5', 'DSVI DGP 2', 'DSVI DGP 3', 'DSVI DGP 4', 'DSVI DGP 5'))
ax[1][0].set_aspect(0.25)

# Qsar dataset
x1_11 = [4.25, 2.92, 2.72, 3.40]
x2_11 = [2.77, 2.82, 2.79, 2.80]
x3_11 = [3.01, 2.85, 3.01, 3.77]
xerr1_11 = [0.31, 0.32, 0.21, 0.32]
xerr2_11 = [0.22, 0.32, 0.23, 0.32]
xerr3_11 = [0.22, 0.31, 0.25, 0.25]
ax[1][1].errorbar(x1_11, y1, xerr=xerr1_11, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[1][1].errorbar(x2_11, y2, xerr=xerr2_11, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[1][1].errorbar(x3_11, y3, xerr=xerr3_11, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[1][1].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Qsar  (N=546  D=8)')
ax[1][1].set_aspect(0.25)

# Protein dataset
x1_12 = [4.25, 2.92, 2.72, 3.40]
x2_12 = [2.77, 2.82, 2.79, 2.80]
x3_12 = [3.01, 2.85, 3.01, 3.77]
xerr1_12 = [0.31, 0.32, 0.21, 0.32]
xerr2_12 = [0.22, 0.32, 0.23, 0.32]
xerr3_12 = [0.22, 0.31, 0.25, 0.25]
ax[1][2].errorbar(x1_12, y1, xerr=xerr1_12, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[1][2].errorbar(x2_12, y2, xerr=xerr2_12, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[1][2].errorbar(x3_12, y3, xerr=xerr3_12, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[1][2].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Protein  (N=45730  D=9)')
ax[1][2].set_aspect(0.25)

# Kin8nm dataset
x1_13 = [4.25, 2.92, 2.72, 3.40]
x2_13 = [2.77, 2.82, 2.79, 2.80]
x3_13 = [3.01, 2.85, 3.01, 3.77]
xerr1_13 = [0.31, 0.32, 0.21, 0.32]
xerr2_13 = [0.22, 0.32, 0.23, 0.32]
xerr3_13 = [0.22, 0.31, 0.25, 0.25]
ax[1][3].errorbar(x1_13, y1, xerr=xerr1_13, fmt='o', ecolor=color1, elinewidth=1.0, ms=3, mfc=color1, mec=color1)
ax[1][3].errorbar(x2_13, y2, xerr=xerr2_13, fmt='o', ecolor=color2, elinewidth=1.0, ms=3, mfc=color2, mec=color2)
ax[1][3].errorbar(x3_13, y3, xerr=xerr3_13, fmt='o', ecolor=color3, elinewidth=1.0, ms=3, mfc=color3, mec=color3)
ax[1][3].set(xlim=(2.50, 4.50), ylim=(0, 13), xticks=(2.75, 3.50, 4.29), yticks=[], title='Kin8nm  (N=8192  D=8)')
ax[1][3].set_aspect(0.25)

plt.show()
