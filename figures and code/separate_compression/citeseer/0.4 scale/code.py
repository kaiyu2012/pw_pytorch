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

ratios = [0.4, 0.8, 1.2, 1.6, 2.0]
top_k_singleshot = {'number_of_runs': 3, 'adj_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'weight_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'run_0': {'top_k_gradient_score': [[[0.648, 0.624, 0.594, 0.61, 0.594], [0.594, 0.592, 0.564, 0.516, 0.498], [0.584, 0.572, 0.57, 0.56, 0.532], [0.566, 0.584, 0.59, 0.576, 0.584], [0.59, 0.606, 0.586, 0.584, 0.572]]]}, 'run_1': {'top_k_gradient_score': [[[0.636, 0.638, 0.642, 0.574, 0.574], [0.616, 0.614, 0.618, 0.542, 0.53], [0.548, 0.552, 0.546, 0.514, 0.502], [0.576, 0.584, 0.588, 0.55, 0.522], [0.578, 0.614, 0.592, 0.578, 0.566]]]}, 'run_2': {'top_k_gradient_score': [[[0.644, 0.642, 0.624, 0.6, 0.594], [0.612, 0.584, 0.574, 0.558, 0.544], [0.566, 0.548, 0.524, 0.518, 0.498], [0.61, 0.604, 0.586, 0.548, 0.548], [0.578, 0.592, 0.588, 0.558, 0.572]]]}}
top_k_multishot = {'number_of_runs': 3, 'run_0': {'top_k_gradient_score': [{'level_1': [0.668, 0.664, 0.656, 0.652, 0.636, 0.642, 0.642, 0.628, 0.63, 0.648, 0.638, 0.632, 0.62, 0.614, 0.63, 0.624, 0.622, 0.606, 0.604, 0.628, 0.61, 0.628, 0.622, 0.626, 0.618], 'level_2': [0.664, 0.652, 0.652, 0.658, 0.64, 0.644, 0.626, 0.618, 0.626, 0.648, 0.632, 0.638, 0.626, 0.622, 0.638, 0.622, 0.628, 0.62, 0.61, 0.614, 0.632, 0.624, 0.63, 0.628, 0.622], 'level_3': [0.66, 0.654, 0.648, 0.65, 0.65, 0.644, 0.626, 0.616, 0.634, 0.642, 0.646, 0.642, 0.638, 0.622, 0.634, 0.626, 0.618, 0.618, 0.62, 0.622, 0.618, 0.616, 0.64, 0.612, 0.626]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.674, 0.642, 0.626, 0.616, 0.612, 0.624, 0.622, 0.62, 0.628, 0.59, 0.614, 0.63, 0.624, 0.626, 0.588, 0.592, 0.628, 0.636, 0.614, 0.61, 0.614, 0.626, 0.614, 0.62, 0.59], 'level_2': [0.5760000000000001, 0.6, 0.616, 0.606, 0.58, 0.5760000000000001, 0.598, 0.612, 0.622, 0.594, 0.582, 0.592, 0.596, 0.6, 0.578, 0.58, 0.596, 0.6, 0.6, 0.582, 0.586, 0.594, 0.592, 0.616, 0.59], 'level_3': [0.548, 0.584, 0.6, 0.594, 0.59, 0.5700000000000001, 0.58, 0.598, 0.594, 0.5720000000000001, 0.584, 0.594, 0.582, 0.602, 0.5680000000000001, 0.562, 0.588, 0.61, 0.59, 0.596, 0.588, 0.58, 0.6, 0.594, 0.588]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.666, 0.632, 0.63, 0.634, 0.62, 0.65, 0.65, 0.638, 0.646, 0.612, 0.636, 0.628, 0.644, 0.612, 0.606, 0.642, 0.63, 0.638, 0.64, 0.638, 0.618, 0.622, 0.63, 0.628, 0.644], 'level_2': [0.6880000000000001, 0.608, 0.612, 0.618, 0.606, 0.656, 0.648, 0.644, 0.64, 0.632, 0.648, 0.646, 0.656, 0.626, 0.634, 0.652, 0.642, 0.646, 0.64, 0.634, 0.644, 0.646, 0.646, 0.636, 0.636], 'level_3': [0.584, 0.602, 0.608, 0.61, 0.648, 0.652, 0.646, 0.658, 0.646, 0.628, 0.65, 0.638, 0.642, 0.636, 0.616, 0.658, 0.65, 0.65, 0.648, 0.634, 0.62, 0.63, 0.634, 0.632, 0.632]}]}}
synflow_singleshot = {'number_of_runs': 3, 'adj_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'weight_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'run_0': {'synflow': [[[0.664, 0.636, 0.652, 0.63, 0.566], [0.62, 0.596, 0.584, 0.564, 0.488], [0.592, 0.618, 0.606, 0.58, 0.51], [0.59, 0.58, 0.578, 0.578, 0.476], [0.572, 0.578, 0.576, 0.58, 0.54]]]}, 'run_1': {'synflow': [[[0.63, 0.638, 0.634, 0.61, 0.556], [0.604, 0.598, 0.59, 0.574, 0.532], [0.532, 0.57, 0.564, 0.53, 0.484], [0.574, 0.566, 0.558, 0.542, 0.468], [0.54, 0.54, 0.53, 0.54, 0.452]]]}, 'run_2': {'synflow': [[[0.664, 0.658, 0.666, 0.648, 0.622], [0.658, 0.636, 0.652, 0.636, 0.588], [0.62, 0.622, 0.61, 0.578, 0.54], [0.588, 0.604, 0.62, 0.598, 0.586], [0.582, 0.574, 0.618, 0.616, 0.618]]]}}
synflow_multishot = {'number_of_runs': 3, 'run_0': {'synflow': [{'level_1': [0.662, 0.672, 0.654, 0.658, 0.636, 0.648, 0.65, 0.65, 0.608, 0.628, 0.614, 0.648, 0.612, 0.588, 0.5640000000000001, 0.634, 0.628, 0.61, 0.586, 0.586, 0.652, 0.628, 0.634, 0.622, 0.5760000000000001], 'level_2': [0.674, 0.684, 0.664, 0.648, 0.632, 0.65, 0.602, 0.626, 0.646, 0.588, 0.628, 0.63, 0.628, 0.586, 0.58, 0.64, 0.644, 0.59, 0.632, 0.578, 0.634, 0.648, 0.612, 0.608, 0.588], 'level_3': [0.662, 0.658, 0.656, 0.674, 0.638, 0.646, 0.64, 0.622, 0.626, 0.598, 0.634, 0.606, 0.62, 0.584, 0.594, 0.64, 0.622, 0.618, 0.612, 0.602, 0.638, 0.626, 0.634, 0.602, 0.59]}]}, 'run_1': {'synflow': [{'level_1': [0.68, 0.672, 0.658, 0.62, 0.656, 0.656, 0.628, 0.646, 0.604, 0.602, 0.634, 0.638, 0.596, 0.628, 0.606, 0.622, 0.606, 0.618, 0.6, 0.538, 0.622, 0.638, 0.636, 0.616, 0.608], 'level_2': [0.674, 0.66, 0.648, 0.66, 0.63, 0.646, 0.644, 0.606, 0.626, 0.616, 0.642, 0.626, 0.612, 0.616, 0.548, 0.646, 0.636, 0.618, 0.606, 0.58, 0.632, 0.614, 0.634, 0.618, 0.5720000000000001], 'level_3': [0.664, 0.67, 0.658, 0.642, 0.62, 0.656, 0.64, 0.628, 0.62, 0.592, 0.644, 0.62, 0.636, 0.602, 0.556, 0.624, 0.636, 0.6, 0.614, 0.598, 0.608, 0.61, 0.594, 0.622, 0.5700000000000001]}]}, 'run_2': {'synflow': [{'level_1': [0.662, 0.664, 0.63, 0.63, 0.604, 0.632, 0.632, 0.616, 0.606, 0.556, 0.63, 0.598, 0.604, 0.612, 0.556, 0.626, 0.598, 0.602, 0.56, 0.5720000000000001, 0.606, 0.61, 0.594, 0.5740000000000001, 0.536], 'level_2': [0.676, 0.672, 0.656, 0.66, 0.642, 0.656, 0.642, 0.636, 0.554, 0.602, 0.642, 0.608, 0.616, 0.614, 0.5640000000000001, 0.648, 0.618, 0.64, 0.594, 0.5660000000000001, 0.612, 0.624, 0.602, 0.606, 0.554], 'level_3': [0.67, 0.672, 0.654, 0.66, 0.608, 0.63, 0.628, 0.628, 0.632, 0.5660000000000001, 0.646, 0.636, 0.604, 0.614, 0.588, 0.634, 0.632, 0.608, 0.604, 0.5720000000000001, 0.604, 0.636, 0.604, 0.602, 0.592]}]}}
UGS = {'number_of_runs': 3, 'adj_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'weight_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'run_0': {'ugs': [[[0.48, 0.326, 0.214, 0.232, 0.232], [0.408, 0.264, 0.218, 0.232, 0.232], [0.372, 0.272, 0.216, 0.232, 0.232], [0.358, 0.312, 0.226, 0.232, 0.232], [0.382, 0.274, 0.234, 0.232, 0.232]]]}, 'run_1': {'ugs': [[[0.474, 0.326, 0.258, 0.23, 0.254], [0.46, 0.318, 0.238, 0.226, 0.244], [0.428, 0.268, 0.256, 0.23, 0.246], [0.412, 0.296, 0.278, 0.232, 0.234], [0.446, 0.336, 0.322, 0.23, 0.232]]]}, 'run_2': {'ugs': [[[0.468, 0.338, 0.222, 0.214, 0.212], [0.452, 0.306, 0.216, 0.212, 0.212], [0.462, 0.304, 0.214, 0.212, 0.212], [0.464, 0.294, 0.2, 0.212, 0.212], [0.462, 0.264, 0.212, 0.212, 0.212]]]}}
random = {'number_of_runs': 3, 'adj_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'weight_compressions': [0.4, 0.8, 1.2, 1.6, 2.0], 'run_0': {'random': [[[0.434, 0.294, 0.244, 0.232, 0.232], [0.372, 0.308, 0.234, 0.232, 0.232], [0.408, 0.332, 0.24, 0.252, 0.254], [0.468, 0.406, 0.232, 0.232, 0.232], [0.408, 0.314, 0.232, 0.232, 0.232]]]}, 'run_1': {'random': [[[0.506, 0.272, 0.218, 0.184, 0.212], [0.406, 0.214, 0.194, 0.212, 0.212], [0.388, 0.29, 0.232, 0.194, 0.212], [0.368, 0.334, 0.224, 0.212, 0.212], [0.448, 0.352, 0.212, 0.266, 0.212]]]}, 'run_2': {'random': [[[0.524, 0.308, 0.176, 0.19, 0.214], [0.456, 0.306, 0.192, 0.228, 0.23], [0.442, 0.314, 0.222, 0.218, 0.212], [0.428, 0.258, 0.184, 0.202, 0.212], [0.374, 0.252, 0.228, 0.212, 0.212]]]}}

Adj_ratios = [0.4, 0.8, 1.2, 1.6, 2.0]
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
  ax.set_xticks((0.4, 0.8, 1.2, 1.6, 2))
  plt.legend(loc = 1, prop={'size':11})
  plt.savefig('figure'+'_'+str(Adj_ratios[i])+ '.png')
  plt.show()
