from action_detector_diagnosis import ActionDetectorDiagnosis

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from matplotlib import gridspec, rc
import matplotlib as mpl
import matplotlib.font_manager
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

def split_predictions_by_score_ranges(fp_error_analysis, groups):
    ground_truth = fp_error_analysis.ground_truth
    prediction = fp_error_analysis.prediction
    
    ground_truth_gbvn = ground_truth.groupby('label')
    prediction = prediction.sort_values(by='score', ascending=False).reset_index(drop=True)
    prediction_gbvn = prediction.groupby('label')

    filtered_prediction_df_list = {}
    for g in range(groups):
        filtered_prediction_df_list[g] = []

    for label, this_ground_truth in ground_truth_gbvn:
        try:
            # Check if there is at least one prediction for this class.
            this_prediction = prediction_gbvn.get_group(label).reset_index(drop=True)
        except Exception as e:
            print('label %s is missing from prediciton' % label)
            continue
        index = 0
        n_j = len(np.unique(this_ground_truth['gt-id']))
        max_index = len(this_prediction)
        for g in range(groups):
            # pick the top (len(this_ground_truth)*self.limit_factor) predictions
            filtered_prediction_df_list[g] += [this_prediction.iloc[index:min(index+n_j,max_index)]]
            index += n_j
            if (index >= max_index):
                continue

    filtered_prediction = {}
    fp_error_types_count = {}
    fp_error_types_count_df = {}
    fp_error_types_precentage_df = {}
    fp_error_types_legned = {'True Positive': 0,
                              'Double Detection Err': 1,
                              'Wrong Label Err': 2,
                              'Localization Err': 3,
                              'Confusion Err': 4,
                              'Background Err': 5}
    fp_error_types_inverse_legned = dict([(v, k) for k, v in fp_error_types_legned.items()])

    for g in range(groups):
        filtered_prediction[g] = pd.concat(filtered_prediction_df_list[g], ignore_index=True)
        
        for col_name, tiou in zip(fp_error_analysis.fp_error_type_cols, fp_error_analysis.tiou_thresholds):
            fp_error_types_count[tiou] = dict(zip(fp_error_types_legned.keys(), [0]*len(fp_error_types_legned)))
            error_ids, counts = np.unique(filtered_prediction[g][col_name], return_counts=True)
            for error_id,count in zip(error_ids, counts):
                fp_error_types_count[tiou][fp_error_types_inverse_legned[error_id]] = count

        fp_error_types_count_df[g] = pd.DataFrame(fp_error_types_count)
        fp_error_types_count_df[g]['avg'] = fp_error_types_count_df[g].mean(axis=1)
        fp_error_types_precentage_df[g] = fp_error_types_count_df[g]/len(filtered_prediction[g])

    return filtered_prediction, fp_error_types_count_df, fp_error_types_precentage_df

def subplot_fp_profile(fig, ax, values, labels, colors, xticks, xlabel, ylabel, title,
                       fontsize=14, bottom=0, top=100, bar_width=1, spacing=0.85,
                       grid_color='gray', grid_linestyle=':', grid_lw=1, 
                       ncol=1, legend_loc='best'):

    ax.yaxis.grid(color=grid_color, linestyle=grid_linestyle, lw=grid_lw)
    
    cumsum_values = np.cumsum(np.array(values)*100, axis=1)    
    index = np.linspace(0, spacing*bar_width*len(values),len(values))
    for i in range(cumsum_values.shape[1])[::-1]:
        rects1 = ax.bar(index, cumsum_values[:,i], bar_width,
                         capsize = i,
                         color=colors[i],
                         label=xticks[i], zorder=0)

    lgd = ax.legend(loc=legend_loc, ncol=ncol, fontsize=fontsize/1.2, edgecolor='k')
    
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize/1.2)
    plt.xticks(np.array(index), np.array(labels[:len(values)]), fontsize=fontsize/1.2, rotation=90)
    plt.yticks(np.linspace(0,1,11)*100, fontsize=fontsize/1.2 )
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlim(left=index[0]-1.25*bar_width, right=index[-1]+1.0*bar_width)
    ax.set_title(title, fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)

    return lgd

def subplot_error_type_impact(fig, ax, values, labels, colors, xlabel, ylabel, title,
                              fontsize=14, bottom=0, top=100, bar_width=1, spacing=1.1,
                              grid_color='gray', grid_linestyle=':', grid_lw=1):
    
    ax.yaxis.grid(color=grid_color, linestyle=grid_linestyle, lw=grid_lw)
    
    index = np.linspace(0, spacing*(len(values)+1),1)
    for i in range(len(values)):
        rects1 = ax.bar(index + i*spacing*bar_width, values[i]*100, bar_width,
                         capsize = i,
                         color=colors[i],
                         label=labels[i])
        for bari in rects1:
            height = bari.get_height()
            plt.gca().text(bari.get_x() + bari.get_width()/2, bari.get_height()+0.001*100, '%.1f' % height,
                         ha='center', color='black', fontsize=fontsize/1.1)
   
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    plt.xticks([])
    plt.yticks(fontsize=fontsize/1.2)
    ax.set_ylim(bottom=bottom,top=top)
    ax.set_title(title, fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)

def plot_fp_analysis(fp_error_analysis, save_filename, 
                     colors=['#33a02c','#b2df8a','#1f78b4','#fb9a99','#e31a1c','#a6cee3'],
                     error_names=['True Positive', 'Double Detection Err','Wrong Label Err', 'Localization Err', 'Confusion Err', 'Background Err'],
                     figsize=(10,4.42), fontsize=24):

    values,labels = [],[]
    _, _, fp_error_types_precentage_df = split_predictions_by_score_ranges(fp_error_analysis,fp_error_analysis.limit_factor)

    for this_limit_factor, this_fp_error_types_precentage_df  in fp_error_types_precentage_df.items():
        values+=[[this_fp_error_types_precentage_df['avg'][k] for k in error_names]]
        labels+=['$%dG$' % (this_limit_factor+1)]

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(1, 5, wspace=1.75, right=1.00)

    lgd = subplot_fp_profile(fig=fig, ax=fig.add_subplot(grid[:-2]),
                             values=values, labels=labels, colors=colors,
                             xticks=error_names,
                             xlabel='Top Predictions', ylabel='Error Breakdown ($\%$)',
                             title='False Positive Profile', fontsize=fontsize, 
                             ncol=3, legend_loc=(-0.15,1.15))

    subplot_error_type_impact(fig=fig, ax=fig.add_subplot(grid[-2:]),
                              values=list(fp_error_analysis.average_mAP_gain.values()),
                              labels=list(fp_error_analysis.average_mAP_gain.keys()),
                              colors=colors[1:],
                              xlabel='Error Type', ylabel='Average-mAP$_N$\nImprovment $(\%)$',
                              title='Removing Error Impact', fontsize=fontsize,
                              top=np.ceil(np.max(list(fp_error_analysis.average_mAP_gain.values()))*100*1.1))
    
    fig.savefig(save_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('[Done] Output analysis is saved in %s' % save_filename)


def main(ground_truth_filename, subset, prediction_filename, output_folder, is_thumos14):
    os.makedirs(output_folder, exist_ok=True)

    if not is_thumos14:
        if subset == 'testing':
            # ActivityNet testing
            characteristic_names_to_bins = {'context-size': (range(-1,7), ['0','1','2','3','4','5','6']),
                                            'context-distance': (range(-1,4), ['Inf','N','M','F']),
                                            'agreement': (np.linspace(0,1.0,6), ['XW','W','M','H','XH']),
                                            'coverage': (np.linspace(0,1.0,6), ['XS','S','M','L','XL']),
                                            'length': (np.array([0,30,60,120,180,np.inf]), ['XS','S','M','L','XL']),
                                            'num-instances': (np.array([-1,1,4,8,np.inf]), ['XS','S','M','L'])}
        elif subset == 'validation':
            # ActivityNet validation
            characteristic_names_to_bins = {'coverage': (np.linspace(0,1.0,6), ['XS','S','M','L','XL']),
                                            'length': (np.array([0,30,60,120,180,np.inf]), ['XS','S','M','L','XL']),
                                            'num-instances': (np.array([-1,1,4,8,np.inf]), ['XS','S','M','L'])}
        else: 
            raise RuntimeError('%s is not a valid subset' % subset)
        
        tiou_thresholds = np.linspace(0.5, 0.95, 10)
    else: 
        # THUMOS14
        characteristic_names_to_bins = {'coverage': (np.array([0,0.02,0.04,0.06,0.08,1]), ['XS','S','M','L','XL']),
                                        'length': (np.array([0,3,6,12,18,np.inf]), ['XS','S','M','L','XL']),
                                        'num-instances': (np.array([-1,1,40,80,np.inf]), ['XS','S','M','L'])}
        tiou_thresholds = [0.5]

    fp_error_analysis = ActionDetectorDiagnosis(ground_truth_filename=ground_truth_filename,
                                                prediction_filename=prediction_filename,
                                                tiou_thresholds=tiou_thresholds,
                                                limit_factor=10,
                                                min_tiou_thr=0.1,
                                                subset=subset, 
                                                verbose=True, 
                                                load_extra_annotations=True,
                                                characteristic_names_to_bins=characteristic_names_to_bins,
                                                normalize_ap=True,
                                                minimum_normalized_precision_threshold_for_detection=0.0
                                            )

    fp_error_analysis.evaluate()
    fp_error_analysis.diagnose()

    plot_fp_analysis(fp_error_analysis=fp_error_analysis,
                     save_filename=os.path.join(output_folder,'false_positive_analysis.pdf'))


if __name__ == '__main__':
    parser = ArgumentParser(description='Run the false positive error analysis.',
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
