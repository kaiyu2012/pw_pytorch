from matplotlib import pyplot as plt
import numpy as np
def calculate(array_list):
    min_list = []
    max_list = []
    avg_list = []
    # 1000
    for i in range(len(array_list[0])):

        # 5
        list = []
        for j in range(len(array_list)):
            list.append(array_list[j][i])

        minimum = min(list)
        maximum = max(list)
        average = sum(list) / len(list)

        min_list.append(minimum)
        max_list.append(maximum)
        avg_list.append(average)
    return avg_list, min_list, max_list

ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
top_k_singleshot = {'number_of_runs': 3, 'adj_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'weight_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'run_0': {'top_k_gradient_score': [[[0.68, 0.674, 0.652, 0.664, 0.644], [0.688, 0.688, 0.66, 0.676, 0.654], [0.684, 0.686, 0.678, 0.682, 0.666], [0.634, 0.636, 0.624, 0.636, 0.624], [0.62, 0.624, 0.624, 0.616, 0.608]]]}, 'run_1': {'top_k_gradient_score': [[[0.646, 0.644, 0.646, 0.63, 0.632], [0.646, 0.652, 0.646, 0.63, 0.644], [0.62, 0.624, 0.62, 0.626, 0.63], [0.608, 0.606, 0.602, 0.592, 0.594], [0.584, 0.588, 0.578, 0.584, 0.574]]]}, 'run_2': {'top_k_gradient_score': [[[0.57, 0.6, 0.584, 0.564, 0.56], [0.594, 0.62, 0.602, 0.59, 0.58], [0.59, 0.582, 0.61, 0.57, 0.578], [0.612, 0.604, 0.604, 0.586, 0.594], [0.572, 0.572, 0.59, 0.586, 0.582]]]}}
top_k_multishot = {'number_of_runs': 3, 'run_0': {'top_k_gradient_score': [{'level_1': [0.676, 0.676, 0.686, 0.674, 0.672, 0.666, 0.67, 0.668, 0.664, 0.658, 0.666, 0.666, 0.664, 0.662, 0.66, 0.64, 0.634, 0.64, 0.638, 0.634, 0.63, 0.628, 0.636, 0.63, 0.604], 'level_2': [0.678, 0.68, 0.68, 0.678, 0.676, 0.678, 0.674, 0.676, 0.67, 0.67, 0.672, 0.674, 0.66, 0.668, 0.66, 0.644, 0.652, 0.654, 0.64, 0.64, 0.644, 0.644, 0.634, 0.644, 0.628], 'level_3': [0.674, 0.676, 0.682, 0.672, 0.678, 0.674, 0.674, 0.676, 0.666, 0.676, 0.672, 0.674, 0.666, 0.662, 0.664, 0.656, 0.65, 0.662, 0.65, 0.636, 0.642, 0.638, 0.654, 0.636, 0.642]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.662, 0.646, 0.658, 0.654, 0.646, 0.646, 0.642, 0.634, 0.652, 0.658, 0.65, 0.642, 0.658, 0.65, 0.654, 0.616, 0.634, 0.632, 0.622, 0.624, 0.628, 0.616, 0.624, 0.63, 0.618], 'level_2': [0.662, 0.658, 0.658, 0.66, 0.66, 0.654, 0.65, 0.652, 0.65, 0.67, 0.644, 0.646, 0.634, 0.644, 0.646, 0.614, 0.626, 0.63, 0.614, 0.614, 0.618, 0.638, 0.63, 0.62, 0.63], 'level_3': [0.668, 0.668, 0.668, 0.652, 0.666, 0.654, 0.654, 0.654, 0.652, 0.658, 0.636, 0.632, 0.638, 0.654, 0.644, 0.632, 0.618, 0.622, 0.63, 0.63, 0.624, 0.616, 0.632, 0.636, 0.616]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.656, 0.672, 0.672, 0.658, 0.66, 0.666, 0.656, 0.656, 0.644, 0.64, 0.666, 0.67, 0.656, 0.654, 0.646, 0.646, 0.64, 0.644, 0.624, 0.624, 0.646, 0.648, 0.642, 0.626, 0.612], 'level_2': [0.674, 0.664, 0.684, 0.66, 0.66, 0.658, 0.666, 0.66, 0.658, 0.64, 0.662, 0.66, 0.644, 0.66, 0.644, 0.632, 0.652, 0.646, 0.644, 0.628, 0.656, 0.644, 0.656, 0.652, 0.628], 'level_3': [0.68, 0.67, 0.662, 0.678, 0.676, 0.656, 0.662, 0.67, 0.652, 0.642, 0.662, 0.648, 0.634, 0.642, 0.65, 0.628, 0.64, 0.63, 0.628, 0.634, 0.634, 0.638, 0.638, 0.632, 0.626]}]}}
synflow_singleshot = {'number_of_runs': 3, 'adj_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'weight_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'run_0': {'synflow': [[[0.646, 0.644, 0.632, 0.628, 0.63], [0.654, 0.652, 0.642, 0.64, 0.638], [0.658, 0.668, 0.654, 0.652, 0.664], [0.624, 0.614, 0.626, 0.628, 0.622], [0.604, 0.614, 0.614, 0.606, 0.592]]]}, 'run_1': {'synflow': [[[0.628, 0.628, 0.616, 0.62, 0.614], [0.654, 0.644, 0.638, 0.648, 0.64], [0.636, 0.634, 0.628, 0.64, 0.652], [0.618, 0.62, 0.608, 0.624, 0.59], [0.628, 0.612, 0.614, 0.616, 0.606]]]}, 'run_2': {'synflow': [[[0.636, 0.652, 0.652, 0.63, 0.604], [0.644, 0.64, 0.638, 0.63, 0.626], [0.624, 0.63, 0.638, 0.624, 0.64], [0.618, 0.616, 0.58, 0.578, 0.576], [0.596, 0.572, 0.542, 0.534, 0.512]]]}}
synflow_multishot = {'number_of_runs': 3, 'run_0': {'synflow': [{'level_1': [0.684, 0.68, 0.666, 0.672, 0.67, 0.67, 0.67, 0.668, 0.664, 0.674, 0.668, 0.658, 0.662, 0.664, 0.656, 0.65, 0.636, 0.646, 0.63, 0.624, 0.638, 0.65, 0.622, 0.64, 0.618], 'level_2': [0.674, 0.674, 0.682, 0.686, 0.6920000000000001, 0.674, 0.674, 0.6920000000000001, 0.686, 0.682, 0.664, 0.668, 0.676, 0.67, 0.644, 0.648, 0.652, 0.65, 0.648, 0.648, 0.668, 0.642, 0.636, 0.646, 0.634], 'level_3': [0.6900000000000001, 0.686, 0.68, 0.672, 0.678, 0.682, 0.678, 0.68, 0.68, 0.666, 0.668, 0.656, 0.67, 0.65, 0.64, 0.658, 0.656, 0.664, 0.636, 0.652, 0.662, 0.648, 0.65, 0.64, 0.632]}]}, 'run_1': {'synflow': [{'level_1': [0.686, 0.676, 0.678, 0.672, 0.68, 0.686, 0.672, 0.654, 0.674, 0.668, 0.672, 0.658, 0.66, 0.664, 0.656, 0.652, 0.662, 0.66, 0.658, 0.654, 0.656, 0.634, 0.644, 0.638, 0.64], 'level_2': [0.68, 0.662, 0.678, 0.68, 0.686, 0.672, 0.672, 0.672, 0.674, 0.668, 0.668, 0.652, 0.66, 0.666, 0.664, 0.664, 0.664, 0.65, 0.658, 0.656, 0.636, 0.656, 0.636, 0.646, 0.626], 'level_3': [0.686, 0.682, 0.672, 0.684, 0.674, 0.68, 0.666, 0.676, 0.67, 0.678, 0.658, 0.662, 0.652, 0.656, 0.66, 0.644, 0.656, 0.656, 0.654, 0.638, 0.63, 0.642, 0.648, 0.63, 0.634]}]}, 'run_2': {'synflow': [{'level_1': [0.684, 0.68, 0.676, 0.682, 0.676, 0.678, 0.676, 0.674, 0.674, 0.666, 0.672, 0.676, 0.672, 0.658, 0.654, 0.67, 0.658, 0.654, 0.648, 0.646, 0.658, 0.652, 0.66, 0.652, 0.63], 'level_2': [0.686, 0.68, 0.67, 0.67, 0.662, 0.68, 0.662, 0.678, 0.662, 0.66, 0.656, 0.664, 0.654, 0.658, 0.644, 0.656, 0.664, 0.648, 0.636, 0.634, 0.658, 0.654, 0.64, 0.64, 0.634], 'level_3': [0.68, 0.67, 0.672, 0.668, 0.67, 0.674, 0.682, 0.672, 0.662, 0.652, 0.672, 0.648, 0.632, 0.648, 0.638, 0.666, 0.654, 0.644, 0.636, 0.628, 0.644, 0.646, 0.65, 0.608, 0.632]}]}}
UGS = {'number_of_runs': 3, 'adj_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'weight_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'run_0': {'ugs': [[[0.568, 0.448, 0.356, 0.298, 0.276], [0.52, 0.44, 0.36, 0.314, 0.26], [0.472, 0.422, 0.358, 0.31, 0.256], [0.488, 0.412, 0.356, 0.316, 0.266], [0.498, 0.434, 0.368, 0.31, 0.276]]]}, 'run_1': {'ugs': [[[0.618, 0.536, 0.38, 0.306, 0.254], [0.53, 0.47, 0.346, 0.28, 0.24], [0.524, 0.442, 0.292, 0.278, 0.258], [0.498, 0.406, 0.296, 0.28, 0.256], [0.48, 0.412, 0.27, 0.234, 0.248]]]}, 'run_2': {'ugs': [[[0.536, 0.442, 0.35, 0.28, 0.268], [0.48, 0.416, 0.292, 0.264, 0.254], [0.474, 0.392, 0.29, 0.256, 0.248], [0.486, 0.386, 0.294, 0.262, 0.246], [0.498, 0.38, 0.302, 0.274, 0.25]]]}}
random = {'number_of_runs': 3, 'adj_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'weight_compressions': [0.2, 0.4, 0.6, 0.8, 1.0], 'run_0': {'random': [[[0.56, 0.442, 0.324, 0.282, 0.252], [0.49, 0.426, 0.374, 0.276, 0.264], [0.462, 0.398, 0.29, 0.362, 0.228], [0.464, 0.488, 0.31, 0.29, 0.25], [0.504, 0.404, 0.362, 0.292, 0.24]]]}, 'run_1': {'random': [[[0.566, 0.51, 0.408, 0.324, 0.214], [0.47, 0.46, 0.324, 0.294, 0.222], [0.538, 0.448, 0.43, 0.306, 0.3], [0.498, 0.364, 0.366, 0.242, 0.3], [0.452, 0.434, 0.37, 0.432, 0.234]]]}, 'run_2': {'random': [[[0.64, 0.44, 0.416, 0.332, 0.314], [0.54, 0.422, 0.324, 0.266, 0.208], [0.488, 0.478, 0.318, 0.258, 0.266], [0.498, 0.398, 0.35, 0.354, 0.224], [0.494, 0.472, 0.32, 0.276, 0.266]]]}}

Adj_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
for i in range(len(Adj_ratios)):
  top_k_singleshot_data = []
  top_k_multishot_data = []
  synflow_singleshot_data = []
  synflow_multishot_data = []
  ugs_data = []
  random_data = []
  
  for run in range(3):
    top_k_singleshot_data.append(top_k_singleshot['run_' + str(run)]['top_k_gradient_score'][0][i])
    top_k_multishot_data.append(top_k_multishot['run_' + str(run)]['top_k_gradient_score'][0]['level_3'][5*i:5*(i+1)])
    synflow_singleshot_data.append(synflow_singleshot['run_' + str(run)]['synflow'][0][i])
    synflow_multishot_data.append(synflow_multishot['run_' + str(run)]['synflow'][0]['level_3'][5*i:5*(i+1)])
    ugs_data.append(UGS['run_' + str(run)]['ugs'][0][i])
    random_data.append(random['run_' + str(run)]['random'][0][i])
  

  top_k_singleshot_mean, top_k_singleshot_min, top_k_singleshot_max = calculate(top_k_singleshot_data)
  top_k_multishot_mean, top_k_multishot_min, top_k_multishot_max = calculate(top_k_multishot_data)
  synflow_singleshot_mean, synflow_singleshot_min, synflow_singleshot_max = calculate(synflow_singleshot_data)
  synflow_multishot_mean, synflow_multishot_min, synflow_multishot_max = calculate(synflow_multishot_data)
  ugs_mean, ugs_min, ugs_max = calculate(ugs_data)
  random_mean, random_min, random_max = calculate(random_data)

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
  dataset = 'Citeseer'
  title = 'Adjacency Compression Ratio = '+str(Adj_ratios[i])+' ('+dataset+')'
  ax.set_title(title)
  ax.set_xlabel('Weight Compression Ratio')
  ax.set_ylabel('Top-1 Accuracy')
  ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
  ax.set_xticks((0.2, 0.4, 0.6, 0.8, 1))
  plt.legend(loc = 3, prop={'size':10.5})
  plt.savefig('figure'+'_'+str(Adj_ratios[i])+ '.png')
  plt.show()
