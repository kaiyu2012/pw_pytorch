from matplotlib import pyplot as plt
import numpy as np

def calculate(numpy_array):
    mean = numpy_array.mean(axis=0)
    minimum = np.amin(numpy_array,axis=0)
    maximum = np.amax(numpy_array,axis=0)
    return mean, minimum, maximum
ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

top_k_singleshot_run1 = np.array([0.752, 0.746, 0.748, 0.752, 0.75, 0.744, 0.7, 0.66, 0.586, 0.502, 0.412, 0.344, 0.256, 0.226, 0.162, 0.166, 0.29, 0.316, 0.316, 0.316, 0.316])
top_k_singleshot_run2 = np.array([0.732, 0.75, 0.756, 0.732, 0.696, 0.684, 0.65, 0.568, 0.498, 0.434, 0.42, 0.3, 0.258, 0.372, 0.37, 0.372, 0.348, 0.166, 0.3, 0.3, 0.3])
top_k_singleshot_run3 = np.array([0.758, 0.752, 0.76, 0.756, 0.748, 0.732, 0.72, 0.704, 0.646, 0.572, 0.326, 0.224, 0.254, 0.162, 0.156, 0.172, 0.172, 0.172, 0.306, 0.306, 0.306])

top_k_singleshot_mean, top_k_singleshot_min, top_k_singleshot_max = calculate(np.array([top_k_singleshot_run1,top_k_singleshot_run2,top_k_singleshot_run3]))

top_k_multishot_l3_run1 = np.array([0.746, 0.766, 0.746, 0.768, 0.754, 0.742, 0.76, 0.724, 0.666, 0.61, 0.438, 0.302, 0.308, 0.314, 0.182, 0.232, 0.182, 0.316, 0.316, 0.316, 0.316])
top_k_multishot_l3_run2 = np.array([0.75, 0.76, 0.766, 0.75, 0.744, 0.728, 0.718, 0.67, 0.644, 0.596, 0.482, 0.372, 0.298, 0.162, 0.268, 0.298, 0.29, 0.302, 0.156, 0.3, 0.3])
top_k_multishot_l3_run3 = np.array([0.768, 0.768, 0.768, 0.768, 0.758, 0.74, 0.726, 0.73, 0.718, 0.674, 0.47600000000000003, 0.388, 0.388, 0.374, 0.374, 0.298, 0.202, 0.318, 0.316, 0.316, 0.316])

top_k_multishot_mean, top_k_multishot_min, top_k_multishot_max = calculate(np.array([top_k_multishot_l3_run1,top_k_multishot_l3_run2,top_k_multishot_l3_run3]))

synflow_singleshot_run1 = np.array([0.77, 0.758, 0.756, 0.73, 0.66, 0.708, 0.682, 0.612, 0.55, 0.464, 0.274, 0.182, 0.256, 0.134, 0.114, 0.122, 0.114, 0.114, 0.162, 0.162, 0.162])
synflow_singleshot_run2 = np.array([0.77, 0.774, 0.766, 0.764, 0.744, 0.706, 0.662, 0.63, 0.504, 0.364, 0.316, 0.306, 0.318, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316])
synflow_singleshot_run3 = np.array([0.77, 0.768, 0.77, 0.75, 0.742, 0.688, 0.632, 0.6, 0.478, 0.28, 0.144, 0.134, 0.12, 0.136, 0.162, 0.162, 0.134, 0.134, 0.258, 0.258, 0.258])

synflow_singleshot_mean, synflow_singleshot_min, synflow_singleshot_max = calculate(np.array([synflow_singleshot_run1,synflow_singleshot_run2,synflow_singleshot_run3]))

synflow_multishot_l3_run1 = np.array([0.754, 0.77, 0.77, 0.78, 0.738, 0.764, 0.744, 0.714, 0.616, 0.466, 0.396, 0.27, 0.332, 0.39, 0.198, 0.23, 0.134, 0.134, 0.122, 0.122, 0.134])
synflow_multishot_l3_run2 = np.array([0.762, 0.788, 0.78, 0.77, 0.74, 0.732, 0.744, 0.66, 0.594, 0.436, 0.324, 0.23800000000000002, 0.374, 0.158, 0.256, 0.156, 0.312, 0.156, 0.156, 0.156, 0.156])
synflow_multishot_l3_run3 = np.array([0.746, 0.762, 0.776, 0.768, 0.758, 0.72, 0.744, 0.588, 0.664, 0.464, 0.376, 0.3, 0.18, 0.166, 0.166, 0.18, 0.122, 0.218, 0.162, 0.134, 0.218])

synflow_multishot_mean, synflow_multishot_min, synflow_multishot_max = calculate(np.array([synflow_multishot_l3_run1,synflow_multishot_l3_run2,synflow_multishot_l3_run3]))

ugs_run1 = np.array([0.794, 0.756, 0.576, 0.434, 0.376, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316])
ugs_run2 = np.array([0.79, 0.716, 0.588, 0.454, 0.318, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316])
ugs_run3 = np.array([0.79, 0.724, 0.668, 0.438, 0.364, 0.18, 0.308, 0.296, 0.298, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316])

ugs_mean, ugs_min, ugs_max = calculate(np.array([ugs_run1, ugs_run2, ugs_run3]))


random_run1 = np.array([0.806, 0.744, 0.568, 0.408, 0.318, 0.332, 0.316, 0.316, 0.316, 0.316, 0.322, 0.316, 0.316, 0.316, 0.324, 0.316, 0.316, 0.318, 0.318, 0.316, 0.316])
random_run2 = np.array([0.792, 0.702, 0.628, 0.296, 0.244, 0.192, 0.306, 0.316, 0.316, 0.316, 0.308, 0.316, 0.192, 0.244, 0.324, 0.316, 0.226, 0.308, 0.244, 0.286, 0.316])
random_run3 = np.array([0.8, 0.726, 0.614, 0.586, 0.306, 0.276, 0.316, 0.324, 0.318, 0.316, 0.316, 0.316, 0.316, 0.316, 0.322, 0.314, 0.308, 0.316, 0.316, 0.316, 0.31])

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

title = 'Weight Pruning (Cora)'
ax.set_title(title)
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('Top-1 Accuracy')
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
ax.set_xticks((0, 1, 2, 3, 4, 5))
plt.legend(loc = 1, prop={'size':12})
plt.savefig('figure.png')
plt.show()
