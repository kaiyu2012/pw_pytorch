from matplotlib import pyplot as plt
import numpy as np

def calculate(numpy_array):
    mean = numpy_array.mean(axis=0)
    minimum = np.amin(numpy_array,axis=0)
    maximum = np.amax(numpy_array,axis=0)
    return mean, minimum, maximum
ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

top_k_singleshot_run1 = np.array([0.596, 0.602, 0.602, 0.604, 0.608, 0.616, 0.616, 0.602, 0.566, 0.518, 0.488, 0.452, 0.232, 0.228, 0.228, 0.22, 0.22, 0.22, 0.232, 0.232, 0.232])
top_k_singleshot_run2 = np.array([0.598, 0.614, 0.614, 0.614, 0.612, 0.612, 0.616, 0.604, 0.612, 0.574, 0.526, 0.412, 0.254, 0.228, 0.228, 0.232, 0.258, 0.218, 0.232, 0.232, 0.232])
top_k_singleshot_run3 = np.array([0.6, 0.61, 0.604, 0.598, 0.612, 0.6, 0.6, 0.604, 0.594, 0.544, 0.482, 0.35, 0.242, 0.24, 0.236, 0.244, 0.226, 0.244, 0.244, 0.244, 0.216])

top_k_singleshot_mean, top_k_singleshot_min, top_k_singleshot_max = calculate(np.array([top_k_singleshot_run1,top_k_singleshot_run2,top_k_singleshot_run3]))

top_k_multishot_l3_run1 = np.array([0.624, 0.648, 0.65, 0.64, 0.646, 0.646, 0.628, 0.638, 0.624, 0.616, 0.594, 0.482, 0.418, 0.222, 0.234, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232])
top_k_multishot_l3_run2 = np.array([0.59, 0.632, 0.624, 0.636, 0.628, 0.638, 0.646, 0.64, 0.626, 0.624, 0.58, 0.55, 0.44, 0.21, 0.22, 0.22, 0.226, 0.244, 0.244, 0.244, 0.212])
top_k_multishot_l3_run3 = np.array([0.598, 0.634, 0.642, 0.642, 0.642, 0.628, 0.634, 0.634, 0.604, 0.6, 0.5740000000000001, 0.454, 0.396, 0.356, 0.252, 0.244, 0.244, 0.212, 0.232, 0.232, 0.234])

top_k_multishot_mean, top_k_multishot_min, top_k_multishot_max = calculate(np.array([top_k_multishot_l3_run1,top_k_multishot_l3_run2,top_k_multishot_l3_run3]))

synflow_singleshot_run1 = np.array([0.63, 0.624, 0.622, 0.618, 0.6, 0.594, 0.564, 0.512, 0.418, 0.32, 0.266, 0.292, 0.218, 0.218, 0.204, 0.164, 0.192, 0.192, 0.17, 0.17, 0.192])
synflow_singleshot_run2 = np.array([0.624, 0.62, 0.632, 0.63, 0.628, 0.63, 0.604, 0.58, 0.554, 0.52, 0.38, 0.26, 0.26, 0.242, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232])
synflow_singleshot_run3 = np.array([0.616, 0.62, 0.612, 0.624, 0.62, 0.602, 0.556, 0.486, 0.454, 0.468, 0.402, 0.262, 0.242, 0.21, 0.212, 0.14, 0.154, 0.15, 0.15, 0.15, 0.15])

synflow_singleshot_mean, synflow_singleshot_min, synflow_singleshot_max = calculate(np.array([synflow_singleshot_run1,synflow_singleshot_run2,synflow_singleshot_run3]))

synflow_multishot_l3_run1 = np.array([0.62, 0.654, 0.658, 0.646, 0.654, 0.644, 0.638, 0.626, 0.618, 0.552, 0.506, 0.264, 0.41000000000000003, 0.23600000000000002, 0.226, 0.226, 0.202, 0.194, 0.194, 0.194, 0.194])
synflow_multishot_l3_run2 = np.array([0.604, 0.626, 0.636, 0.642, 0.626, 0.63, 0.632, 0.608, 0.588, 0.584, 0.542, 0.44, 0.364, 0.28400000000000003, 0.268, 0.202, 0.22, 0.194, 0.218, 0.212, 0.194])
synflow_multishot_l3_run3 = np.array([0.616, 0.66, 0.666, 0.662, 0.672, 0.658, 0.624, 0.648, 0.624, 0.58, 0.604, 0.45, 0.364, 0.34, 0.22, 0.22, 0.192, 0.232, 0.258, 0.232, 0.232])

synflow_multishot_mean, synflow_multishot_min, synflow_multishot_max = calculate(np.array([synflow_multishot_l3_run1,synflow_multishot_l3_run2,synflow_multishot_l3_run3]))

ugs_run1 = np.array([0.724, 0.678, 0.53, 0.506, 0.322, 0.35, 0.358, 0.242, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232])
ugs_run2 = np.array([0.7, 0.608, 0.502, 0.418, 0.464, 0.358, 0.342, 0.258, 0.24, 0.24, 0.218, 0.218, 0.218, 0.218, 0.212, 0.212, 0.212, 0.212, 0.212, 0.212, 0.212])
ugs_run3 = np.array([0.708, 0.61, 0.44, 0.344, 0.264, 0.244, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232])

ugs_mean, ugs_min, ugs_max = calculate(np.array([ugs_run1, ugs_run2, ugs_run3]))


random_run1 = np.array([0.698, 0.652, 0.554, 0.428, 0.304, 0.244, 0.242, 0.22, 0.188, 0.232, 0.232, 0.216, 0.232, 0.214, 0.212, 0.21, 0.232, 0.232, 0.21, 0.28, 0.22])
random_run2 = np.array([0.73, 0.636, 0.504, 0.55, 0.28, 0.226, 0.188, 0.216, 0.188, 0.188, 0.188, 0.188, 0.188, 0.188, 0.222, 0.224, 0.188, 0.188, 0.256, 0.214, 0.232])
random_run3 = np.array([0.702, 0.688, 0.566, 0.468, 0.236, 0.278, 0.286, 0.232, 0.232, 0.232, 0.232, 0.232, 0.244, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232, 0.232])

random_mean, random_min, random_max = calculate(np.array([random_run1, random_run2, random_run3]))

plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1)
ax.plot(ratios, top_k_multishot_mean, color='purple', ls='solid', linewidth = 0.5, marker = "*", label = "IGRP (Multiple Shot)")
ax.plot(ratios, top_k_singleshot_mean, color='magenta', ls='solid', linewidth = 0.5, marker = "x", label = "IGRP (Single Shot)")
ax.plot(ratios, synflow_multishot_mean, color='blue', ls='solid', linewidth = 0.5, marker = "^", label = "SF (Multiple Shot)")
ax.plot(ratios, synflow_singleshot_mean, color='orange', ls='solid', linewidth = 0.5, marker = "v", label = "SF (Single Shot)")


ax.plot(ratios, ugs_mean, color='red', ls='solid', linewidth = 0.5, marker = "s", label = "UGS")
#ax.fill_between(ratios, ugs_min, ugs_max, facecolor='lightcoral', alpha=0.2)

ax.plot(ratios, random_mean, color='green', ls='solid', linewidth = 0.5, marker = "o", label = "Random")
#ax.fill_between(ratios, random_min, random_max, facecolor='lightgreen', alpha=0.2)
ax.set_facecolor("whitesmoke")
plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)

title = 'Weight Pruning (Citeseer)'
ax.set_title(title)
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('Top-1 Accuracy')
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
ax.set_xticks((0, 1, 2, 3, 4, 5))
plt.legend(loc = 1, prop={'size':12})
plt.savefig('figure.png')
plt.show()
