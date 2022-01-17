import matplotlib.pyplot as plt
import numpy as np


def calculate(array_list):
    min_list = []
    max_list = []
    avg_list = []
    # 1000
    for i in range(len(array_list)):


      minimum = min(array_list)
      maximum = max(array_list)
      average = sum(array_list) / len(array_list)

      
    return average, minimum, maximum


ratios = [1.0]

top_k_list = [5, 10, 15, 20]

# Experiment variables
prune_strategy = ['top_k_gradient_score']

singleshot_data = {'number_of_runs': 3, 'compressions': [1.0], 'k_5': {'run_0': {'top_k_gradient_score': [[0.604]]}, 'run_1': {'top_k_gradient_score': [[0.6]]}, 'run_2': {'top_k_gradient_score': [[0.616]]}}, 'k_10': {'run_0': {'top_k_gradient_score': [[0.604]]}, 'run_1': {'top_k_gradient_score': [[0.6]]}, 'run_2': {'top_k_gradient_score': [[0.618]]}}, 'k_15': {'run_0': {'top_k_gradient_score': [[0.608]]}, 'run_1': {'top_k_gradient_score': [[0.602]]}, 'run_2': {'top_k_gradient_score': [[0.62]]}}, 'k_20': {'run_0': {'top_k_gradient_score': [[0.612]]}, 'run_1': {'top_k_gradient_score': [[0.612]]}, 'run_2': {'top_k_gradient_score': [[0.612]]}}}
multishot_data = {'number_of_runs': 3, 'k_5': {'run_0': {'top_k_gradient_score': [{'level_1': [0.624], 'level_2': [0.632], 'level_3': [0.644], 'level_4': [0.648], 'level_5': [0.64]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.634], 'level_2': [0.638], 'level_3': [0.624], 'level_4': [0.634], 'level_5': [0.644]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.64], 'level_2': [0.632], 'level_3': [0.628], 'level_4': [0.63], 'level_5': [0.634]}]}}, 'k_10': {'run_0': {'top_k_gradient_score': [{'level_1': [0.628], 'level_2': [0.65], 'level_3': [0.64], 'level_4': [0.658], 'level_5': [0.644]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.626], 'level_2': [0.63], 'level_3': [0.626], 'level_4': [0.64], 'level_5': [0.614]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.626], 'level_2': [0.632], 'level_3': [0.632], 'level_4': [0.624], 'level_5': [0.622]}]}}, 'k_15': {'run_0': {'top_k_gradient_score': [{'level_1': [0.642], 'level_2': [0.642], 'level_3': [0.648], 'level_4': [0.642], 'level_5': [0.638]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.65], 'level_2': [0.652], 'level_3': [0.652], 'level_4': [0.636], 'level_5': [0.642]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.638], 'level_2': [0.64], 'level_3': [0.638], 'level_4': [0.648], 'level_5': [0.652]}]}}, 'k_20': {'run_0': {'top_k_gradient_score': [{'level_1': [0.656], 'level_2': [0.66], 'level_3': [0.656], 'level_4': [0.65], 'level_5': [0.65]}]}, 'run_1': {'top_k_gradient_score': [{'level_1': [0.638], 'level_2': [0.65], 'level_3': [0.642], 'level_4': [0.65], 'level_5': [0.644]}]}, 'run_2': {'top_k_gradient_score': [{'level_1': [0.638], 'level_2': [0.646], 'level_3': [0.64], 'level_4': [0.638], 'level_5': [0.646]}]}}}

top_k_singleshot_mean = []
top_k_multishot_mean = []
for strategy in prune_strategy:

    top_k_data = {}
    for k in top_k_list:
        top_k_data['singleshotk_'+str(k)] = []
        top_k_data['multishotk_'+str(k)] = []
    for k in top_k_list:
      if singleshot_data['number_of_runs'] == multishot_data['number_of_runs']:
        for run in range(singleshot_data['number_of_runs']):
          top_k_data['singleshotk_'+str(k)].append(singleshot_data['k_'+str(k)]['run_'+str(run)][strategy][0][0])
          top_k_data['multishotk_'+str(k)].append(multishot_data['k_'+str(k)]['run_'+str(run)][strategy][0]['level_5'][0])
        
        #print(top_k_data)
    for k in top_k_list:
        top_k_data['singleshotk_'+str(k)+'_average'], top_k_data['singleshotk_'+str(k)+'_min'], top_k_data['singleshotk_'+str(k)+'_max'] = calculate(top_k_data['singleshotk_'+str(k)])
        top_k_singleshot_mean.append(top_k_data['singleshotk_'+str(k)+'_average'])
        top_k_data['multishotk_'+str(k)+'_average'], top_k_data['multishotk_'+str(k)+'_min'], top_k_data['multishotk_'+str(k)+'_max'] = calculate(top_k_data['multishotk_'+str(k)])
        top_k_multishot_mean.append(top_k_data['multishotk_'+str(k)+'_average'])
    #print('average', top_k_singleshot_mean)
    plt.rc('font', family='serif', size=12)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(top_k_list, np.array(top_k_multishot_mean), color='purple', ls='solid', linewidth = 0.5, marker = "*", label = "IGRP (Multiple Shot)")
    ax.plot(top_k_list, np.array(top_k_singleshot_mean), color='magenta', ls='solid', linewidth = 0.5, marker = "x", label = "IGRP (Single Shot)")
    

    
    #for k in top_k_list:
        #ax.plot(ratios, top_k_data['k_'+str(k)+'_average'], color= plt.cm.tab20b(k-1), ls='solid', linewidth = 2, marker = "o", label = "top "+str(k))

    ax.set_facecolor("whitesmoke")
    plt.grid(color = 'gray', linestyle = 'solid', linewidth = 0.5)
    dataset = 'Citeseer'
    title = 'IGRP Weight pruning ('+dataset +')'
    ax.set_title(title)
    ax.set_xlabel('k')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.set_xticks((5, 10, 15, 20))
    plt.legend(loc = 4, prop={'size':12})
    plt.savefig(title+'.png')
    plt.show()

