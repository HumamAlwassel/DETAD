from __future__ import division, print_function
from action_detector_diagnosis import ActionDetectorDiagnosis

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from utils import interpolated_prec_rec
from matplotlib import gridspec, rc
import matplotlib as mpl
mpl.use('Agg')
params = {'font.family': 'serif','font.serif': 'Times',
            'text.usetex': True,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'mathtext.fontset': 'custom',
        }
mpl.rcParams.update(params)
import matplotlib.pyplot as plt

def compute_mAP_N(result,this_cls_pred,this_cls_gt):
    ap = np.zeros(len(result.tiou_thresholds))
    tp = np.zeros((len(result.tiou_thresholds), len(this_cls_pred)))
    fp = np.zeros((len(result.tiou_thresholds), len(this_cls_pred)))

    for tidx, tiou in enumerate(result.tiou_thresholds): 
        fp[tidx,pd.isnull(this_cls_pred[result.matched_gt_id_cols[tidx]]).values] = 1
        tp[tidx,~(pd.isnull(this_cls_pred[result.matched_gt_id_cols[tidx]]).values)] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / len(np.unique(this_cls_gt['gt-id']))
    precision_cumsum = recall_cumsum * result.average_num_instance_per_class / (recall_cumsum * result.average_num_instance_per_class + fp_cumsum)

    for tidx in range(len(result.tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
    
    return ap.mean()

# Initialize true positive and false positive vectors.
def compute_average_mAP_N_for_characteristic(sensitivity_analysis, characteristic_name):
    gt_by_characteristic = sensitivity_analysis.ground_truth.groupby(characteristic_name)
    average_mAP_n_by_characteristic_value = OrderedDict()
    
    for characteristic_value, this_characteristic_gt in gt_by_characteristic:
        ap = np.nan*np.zeros(len(sensitivity_analysis.activity_index))
        gt_by_cls = this_characteristic_gt.groupby('label')
        pred_by_cls = sensitivity_analysis.prediction.groupby('label')
        for cls in sensitivity_analysis.activity_index.values():
            this_cls_pred = pred_by_cls.get_group(cls).sort_values(by='score',ascending=False)
            try:
                this_cls_gt = gt_by_cls.get_group(cls)
            except:
                continue
            gt_id_to_keep = np.append(this_cls_gt['gt-id'].values, [np.nan])
        
            for tidx, tiou in enumerate(sensitivity_analysis.tiou_thresholds):
                this_cls_pred = this_cls_pred[this_cls_pred[sensitivity_analysis.matched_gt_id_cols[tidx]].isin(gt_id_to_keep)]
            
            ap[cls] = compute_mAP_N(sensitivity_analysis,this_cls_pred,this_cls_gt)

        average_mAP_n_by_characteristic_value[characteristic_value] = np.nanmean(ap)

    return average_mAP_n_by_characteristic_value

def plot_sensitivity_analysis(sensitivity_analysis, save_filename, 
                              colors=['#7fc97f','#beaed4','#fdc086','#386cb0','#f0027f','#bf5b17'],
                              characteristic_names=['context-size', 'context-distance', 'agreement','coverage', 'length', 'num-instances'],
                              characteristic_names_in_text=['Context Size', 'Context Distance', 'Agreement', 'Coverage', 'Length', '\# Instances'],
                              characteristic_names_delta_positions=[1.1,-1.4,0.25,0.5,1,-0.2],
                              buckets_order=['0','1','2','3','4','5','6','XW', 'W', 'XS','S', 'N', 'M', 'F', 'Inf', 'L', 'XL', 'H', 'XH'],
                              figsize=(25,6), fontsize=28, num_grids=4):
    average_mAP_N_by_characteristic = OrderedDict()
    average_mAP_N_by_characteristic['base'] = sensitivity_analysis.average_mAP
    for characteristic_name in characteristic_names:
        average_mAP_N_by_characteristic[characteristic_name] = compute_average_mAP_N_for_characteristic(sensitivity_analysis, 
                                                                                                 characteristic_name)
    characteristic_name_lst,bucket_lst = ['base'],['base']
    ratio_value_lst = [average_mAP_N_by_characteristic['base']]

    for characteristic_name in characteristic_names:
        characteristic_name_lst += len(average_mAP_N_by_characteristic[characteristic_name])*[characteristic_name]
        bucket_lst += average_mAP_N_by_characteristic[characteristic_name].keys()
        ratio_value_lst += average_mAP_N_by_characteristic[characteristic_name].values()

    # characteristic-name,bucket,ratio-value
    sensitivity_analysis_df = pd.DataFrame({'characteristic-name': characteristic_name_lst,
                                               'bucket': bucket_lst,
                                               'ratio-value': ratio_value_lst,
                                               })
    sensitivity_analysis_df['order'] = pd.Categorical(sensitivity_analysis_df['bucket'],
                                                                    categories=buckets_order,ordered=True)
    sensitivity_analysis_df.sort_values(by='order', inplace=True)
    sensitivity_analysis_df_by_characteristic_name = sensitivity_analysis_df.groupby('characteristic-name')
    base_average_mAP_N = sensitivity_analysis_df_by_characteristic_name.get_group('base')['ratio-value'].values[0]*100

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(1, num_grids)
    
    ax1=fig.add_subplot(grid[:-1])
    current_x_value = 0
    xticks_lst,xvalues_lst = [],[]
    for char_idx, characteristic_name in enumerate(characteristic_names):
        this_sensitivity_analysis = sensitivity_analysis_df_by_characteristic_name.get_group(characteristic_name)
        x_values = range(current_x_value, current_x_value + len(this_sensitivity_analysis))
        y_values = this_sensitivity_analysis['ratio-value'].values*100
        mybars=ax1.bar(x_values, y_values, color=colors[char_idx])
        for bari in mybars:
            height = bari.get_height()
            ax1.text(bari.get_x() + bari.get_width()/2, bari.get_height()+0.025*100, '%.1f' % height,
                         ha='center', color='black', fontsize=fontsize/1.15)
        ax1.annotate(characteristic_names_in_text[char_idx],
                    xy=(current_x_value+characteristic_names_delta_positions[char_idx],100),
                    fontsize=fontsize)

        if char_idx < len(characteristic_names) - 1:
            ax1.axvline(max(x_values)+1, linewidth=1.5, color="gray", linestyle='dotted')

        current_x_value = max(x_values) + 2
        xticks_lst.extend(this_sensitivity_analysis['bucket'].values.tolist())
        xvalues_lst.extend(x_values)


    ax1.plot([xvalues_lst[0]- 1, xvalues_lst[-1] + 1],[base_average_mAP_N, base_average_mAP_N], '--', color='k')
    ax1.annotate('%.2f' % base_average_mAP_N,xy=(xvalues_lst[-1]-0.5,base_average_mAP_N+0.025*100), fontsize=fontsize/1.15)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='dotted')
    ax1.set_axisbelow(True)
    ax1.xaxis.set_tick_params(width=0)
    ax1.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(2.5)
    plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize/1.1)
    plt.yticks(fontsize=fontsize)
    ax1.set_ylabel('Average-mAP$_{N}$ $(\%)$', fontsize=fontsize)
    ax1.set_ylim(0,100)
    ax1.set_xlim(-1.5,current_x_value-1)

    ax2=fig.add_subplot(grid[-1:])
    current_x_value = 0
    xticks_lst,xvalues_lst = [],[]
    min_y_value,max_y_value=np.infty,-np.infty
    for char_idx, characteristic_name in enumerate(characteristic_names):
        this_sensitivity_analysis = sensitivity_analysis_df_by_characteristic_name.get_group(characteristic_name)
        x_values = [current_x_value,current_x_value]
        y_values = this_sensitivity_analysis['ratio-value'].values*100
        y_values = [min(y_values)/base_average_mAP_N,max(y_values)/base_average_mAP_N]
        this_min_y_value,this_max_y_value=min(y_values),max(y_values)
        min_y_value,max_y_value=min(min_y_value,this_min_y_value),max(max_y_value,this_max_y_value)
        ax2.plot([current_x_value,current_x_value], 
                 [this_min_y_value,this_max_y_value], linestyle='-', marker='_',  mew=5, markersize=25,lw=8,color=colors[char_idx])
        for i,j in zip(x_values,y_values):
            ax2.annotate('%.1f' % j,xy=(i+0.1,j+0.05), fontsize=fontsize/1.1)
        current_x_value += 1
        xticks_lst += [characteristic_names_in_text[char_idx]]
        xvalues_lst += [x_values[0]]

    ax2.plot([xvalues_lst[0]- 1, xvalues_lst[-1] + 1],[base_average_mAP_N/base_average_mAP_N, base_average_mAP_N/base_average_mAP_N], '--', color='k',zorder=0)
    ax2.annotate('%.2f' % base_average_mAP_N,xy=(xvalues_lst[-1]+0.2,base_average_mAP_N/base_average_mAP_N+0.05), fontsize=fontsize/1.1)

    ax2.yaxis.grid(color='gray', linestyle=':',lw=1)
    ax2.xaxis.grid(color='gray', linestyle=':',lw=1)
    ax2.set_axisbelow(True)
    ax2.xaxis.set_tick_params(width=0)
    ax2.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom','left']:
        ax2.spines[axis].set_linewidth(2.5)
    plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize/1.5, rotation=90)
    plt.yticks(fontsize=fontsize)
    ax2.set_ylabel('Average-mAP$_{N}$\nRelative Change', fontsize=fontsize)
    ax2.set_ylim(min_y_value*0.8,max_y_value*1.2)

    plt.tight_layout()
    fig.savefig(save_filename,bbox_inches='tight')
    print('[Done] Output analysis is saved in %s' % save_filename)

def main(ground_truth_filename, subset, prediction_filename, output_folder, is_thumos14):
    if not is_thumos14:
        if subset == 'testing':
            # ActivityNet testing
            characteristic_names_to_bins = {'context-size': (range(-1,7), ['0','1','2','3','4','5','6']),
                                            'context-distance': (range(-1,4), ['Inf','N','M','F']),
                                            'agreement': (np.linspace(0,1.0,6), ['XW','W','M','H','XH']),
                                            'coverage': (np.linspace(0,1.0,6), ['XS','S','M','L','XL']),
                                            'length': (np.array([0,30,60,120,180,np.inf]), ['XS','S','M','L','XL']),
                                            'num-instances': (np.array([-1,1,4,8,np.inf]), ['XS','S','M','L'])}
            colors = ['#7fc97f','#beaed4','#fdc086','#386cb0','#f0027f','#bf5b17']
            characteristic_names = ['context-size', 'context-distance', 'agreement','coverage', 'length', 'num-instances']
            characteristic_names_in_text = ['Context Size', 'Context Distance', 'Agreement', 'Coverage', 'Length', '\# Instances']
            characteristic_names_delta_positions = [1.1,-1.4,0.25,0.5,1,-0.2]
            figsize = (25,6)
            num_grids = 4

        elif subset == 'validation':
            # ActivityNet validation
            characteristic_names_to_bins = {'coverage': (np.linspace(0,1.0,6), ['XS','S','M','L','XL']),
                                            'length': (np.array([0,30,60,120,180,np.inf]), ['XS','S','M','L','XL']),
                                            'num-instances': (np.array([-1,1,4,8,np.inf]), ['XS','S','M','L'])}
            colors = ['#386cb0','#f0027f','#bf5b17']
            characteristic_names = ['coverage', 'length', 'num-instances']
            characteristic_names_in_text = ['Coverage', 'Length', '\# Instances']
            characteristic_names_delta_positions = [0.5,1,-0.2]
            figsize = (17.5,6)
            num_grids = 3

        else: 
            raise RuntimeError('%s is not a valid subset' % subset)
        
        tiou_thresholds = np.linspace(0.5, 0.95, 10)
    else: 
        # THUMOS14
        characteristic_names_to_bins = {'coverage': (np.array([0,0.02,0.04,0.06,0.08,1]), ['XS','S','M','L','XL']),
                                        'length': (np.array([0,3,6,12,18,np.inf]), ['XS','S','M','L','XL']),
                                        'num-instances': (np.array([-1,1,40,80,np.inf]), ['XS','S','M','L'])}
        colors = ['#386cb0','#f0027f','#bf5b17']
        characteristic_names = ['coverage', 'length', 'num-instances']
        characteristic_names_in_text = ['Coverage', 'Length', '\# Instances']
        characteristic_names_delta_positions = [0.5,1,-0.2]
        figsize = (17.5,6)
        num_grids = 3
        tiou_thresholds = [0.5]

    sensitivity_analysis = ActionDetectorDiagnosis(ground_truth_filename=ground_truth_filename,
                                                prediction_filename=prediction_filename,
                                                tiou_thresholds=tiou_thresholds,
                                                limit_factor=None,
                                                min_tiou_thr=0.1,
                                                subset=subset, 
                                                verbose=True, 
                                                check_status=True,
                                                load_extra_annotations=True,
                                                characteristic_names_to_bins=characteristic_names_to_bins,
                                                normalize_ap=True,
                                                minimum_normalized_precision_threshold_for_detection=0.0
                                            )

    sensitivity_analysis.evaluate()

    plot_sensitivity_analysis(sensitivity_analysis=sensitivity_analysis,
                              save_filename=os.path.join(output_folder, 'sensitivity_analysis.pdf'),
                              colors=colors,
                              characteristic_names=characteristic_names,
                              characteristic_names_in_text=characteristic_names_in_text,
                              characteristic_names_delta_positions=characteristic_names_delta_positions,
                              figsize=figsize,
                              num_grids=num_grids)

if __name__ == '__main__':
    parser = ArgumentParser(description='Run the sensitivity analysis.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ground_truth_filename', required=True, type=str,
                        help='The path to the JSON file containing the ground truth annotations')
    parser.add_argument('--subset', default='validation', type=str,
                        help='The dataset subset to use for the analysis')
    parser.add_argument('--prediction_filename', required=True, type=str,
                        help='The path to the JSON file containing the method\'s predictions')
    parser.add_argument('--output_folder', required=True, type=str,
                        help='The path to the folder in which the results will be saved')
    parser.add_argument('--is_thumos14', default=False, action='store_true',
                      help='Pass this argument if the dataset used is THUMOS14 and not ActivityNet')
    args = parser.parse_args()

    main(args.ground_truth_filename, args.subset, args.prediction_filename, args.output_folder, args.is_thumos14)
