# Visualization code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_mode = 'quartile'
#plot_mode = 'stdev'

if plot_mode == 'quartile':
    filename_a = 'Q_graph.pdf'
if plot_mode == 'stdev':
    filename_a = 'Q_stdev_graph.pdf'

cooperation = 3.0
temptation = 2.0
sucker = 1.0
punishment = 0.0



filename0 = '_gamma_0.95_lr_0.1_RTSP_3.0_2.0_1.0_0.0_epsilon_1.0_1e-06_0.002_Qinit_zeros_1.0_seed_42_1000trials.tsv'
filename1 = '_gamma_0.95_lr_0.1_RTSP_3.0_2.0_1.0_0.0_epsilon_0.02_1e-06_0.002_Qinit_'
filename2 = [
    'zeros',
    'wsls',
    'f_handshake',
]
filename3 = '_1.0_seed_42_1000trials.tsv'

label_a = 'mutual learning'
label_b = [
    r'(a) $Q_{init} = Q_{zero}, \ \epsilon_{init} = 1$',
    r'(b) $Q_{init} = Q_{zero}, \ \epsilon_{init} = 0.02$',
    r'(c) $Q_{init} = Q_{WSLS}, \ \epsilon_{init} = 0.02$',
    r'(d) $Q_{init} = Q_{FH}, \ \epsilon_{init} = 0.02$',
]


filename_Q = (
    ['Qfunc0' + filename0]
    + ['Qfunc0' + filename1 + filename2[i_fig] + filename3 for i_fig in range(3)]
)
df_Q = [pd.read_csv(fname_each, sep='\t', header=None) for fname_each in filename_Q]
value_Q = [df_each.values for df_each in df_Q]

filename_Q_th = (
    ['Qfunc_theory' + filename0]
    + ['Qfunc_theory' + filename1 + filename2[i_fig] + filename3 for i_fig in range(3)]
)
df_Q_th = [pd.read_csv(fname_each, sep='\t', header=None) for fname_each in filename_Q_th]
chosen_mode = [0, 0, 1, 2]
value_Q_th = [df_Q_th[i_fig].values[chosen_mode[i_fig]] for i_fig in range(4)]

color_list = ['red', 'darkorange', 'green', 'cyan', 'blue', 'purple', 'magenta', 'brown']
Q_list = [
    r'$Q _A \left( \left\{ C, C \right\}, C \right)$',
    r'$Q _A \left( \left\{ C, C \right\}, D \right)$',
    r'$Q _A \left( \left\{ C, D \right\}, C \right)$',
    r'$Q _A \left( \left\{ C, D \right\}, D \right)$',
    r'$Q _A \left( \left\{ D, C \right\}, C \right)$',
    r'$Q _A \left( \left\{ D, C \right\}, D \right)$',
    r'$Q _A \left( \left\{ D, D \right\}, C \right)$',
    r'$Q _A \left( \left\{ D, D \right\}, D \right)$',
]

xmin = np.min(value_Q[0][:,0])
xmax = np.max(value_Q[0][:,0])
dash_width=2.4

spine_width = 1.2
fig, axes = plt.subplots(2, 2, tight_layout=True, figsize=[8.0, 7.2])
#fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=[4.0,3.6])
plt.subplots_adjust(hspace=0.3)

for axes_row in axes:
    for ax in axes_row:
        ax.tick_params(labelsize=12.5)
        ax.set_facecolor ('white')
        for p in ['left','top','right','bottom']:
            ax.spines[p].set_visible(True)
            ax.spines[p].set_color('black')
            ax.spines[p].set_visible(spine_width)
        ax.grid(True, color='lightgray')
        ax.set_xlim(xmin, xmax)
        ax.set_xscale("log")
        ax.set_xlabel("steps", family="serif", weight="normal", size=15, labelpad=6)
        ax.set_ylabel("Q", family="serif", weight="normal", size=15, labelpad=6)


axes[0, 0].set_ylim(50, 62)
axes[0, 1].set_ylim(50, 62)
axes[1, 0].set_ylim(50, 62)
axes[1, 1].set_ylim(25, 31)

for i_fig in range(4):
    y_fig = i_fig // 2
    x_fig = i_fig % 2
    axes[y_fig, x_fig].text(
        x=0.5, y=1.03, s=label_b[i_fig], 
        color='black', ha='center', 
        family="serif", fontsize='16',
        transform=axes[y_fig, x_fig].transAxes,
    )
    for i_Q in range(8):
        if plot_mode == 'quartile':
            axes[y_fig, x_fig].plot(
                value_Q[i_fig][:,0],
                value_Q[i_fig][:,5 * i_Q + 4],
                color=color_list[i_Q], marker='o', markersize=0, ls='-',
                label=Q_list[i_Q] if i_fig == 0 else None, linewidth=1.4,
            )
            axes[y_fig, x_fig].fill_between(
                value_Q[i_fig][:,0],
                value_Q[i_fig][:,5 * i_Q + 3],
                value_Q[i_fig][:,5 * i_Q + 5],
                alpha=0.15, color=color_list[i_Q],
            )
        if plot_mode == 'stdev':
            axes[y_fig, x_fig].plot(
                value_Q[i_fig][:,0],
                value_Q[i_fig][:,5 * i_Q + 1],
                color=color_list[i_Q], marker='o', markersize=0, ls='-',
                label=Q_list[i_Q] if i_fig == 0 else None, linewidth=1.4,
            )
            axes[y_fig, x_fig].fill_between(
                value_Q[i_fig][:,0],
                value_Q[i_fig][:,5 * i_Q + 1] + value_Q[i_fig][:,5 * i_Q + 2],
                value_Q[i_fig][:,5 * i_Q + 1] - value_Q[i_fig][:,5 * i_Q + 2],
                alpha=0.15, color=color_list[i_Q],
            )
        axes[y_fig, x_fig].hlines(
            value_Q_th[i_fig][i_Q + 1],
            xmin, xmax, colors=color_list[i_Q],
            linestyle='dotted', linewidth=dash_width,
        )


fig.legend(loc="upper right", bbox_to_anchor=(1.25, 0.65), fontsize=12)
if False:
    fig.text(
        x=0.5, y=1.0, s=label_a,
        color='black', ha='center', 
        family="serif", fontsize='18',
        transform=fig.transFigure,
    )

#plt.show()
fig.savefig(filename_a, bbox_inches="tight", pad_inches=0.05)
