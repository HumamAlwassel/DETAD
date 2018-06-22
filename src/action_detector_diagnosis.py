import json
import urllib2

import numpy as np
import pandas as pd

from utils import get_blocked_videos
from utils import interpolated_prec_rec
from utils import segment_iou

from joblib import Parallel, delayed

class ActionDetectorDiagnosis(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 limit_factor=None,
                 min_tiou_thr=0.1,
                 subset='testing', 
                 verbose=False, 
                 check_status=True,
                 load_extra_annotations=False,
                 characteristic_names_to_bins={'context-size': (range(-1,7), ['0','1','2','3','4','5','6']),
                                               'context-distance': (range(-1,4), ['Inf','N','M','F']),
                                               'agreement': (np.linspace(0,1.0,6), ['XW','W','M','H','XH']),
                                               'coverage': (np.linspace(0,1.0,6), ['XS','S','M','L','XL']),
                                               'length': (np.array([0,30,60,120,180,np.inf]), ['XS','S','M','L','XL']),
                                               'num-instances': (np.array([-1,1,4,8,np.inf]), ['XS','S','M','L'])},
                 normalize_ap=False,
                 minimum_normalized_precision_threshold_for_detection=0.00,
                 evaluate_with_multi_segments=None):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        self.load_extra_annotations = load_extra_annotations
        self.characteristic_names_to_bins = characteristic_names_to_bins
        self.characteristic_names = characteristic_names_to_bins.keys()
        self.normalize_ap = normalize_ap
        self.minimum_normalized_precision_threshold_for_detection = minimum_normalized_precision_threshold_for_detection
        self.evaluate_with_multi_segments = evaluate_with_multi_segments

        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.average_num_instance_per_class = len(self.ground_truth) / len(self.activity_index)

        self.prediction = self._import_prediction(prediction_filename)

        self.limit_factor = limit_factor
        if self.limit_factor:
            self.prediction = self._limit_prediction()

        self.matched_gt_id_cols, self.fp_error_type_cols = [], []
        for tiou in self.tiou_thresholds:
            self.matched_gt_id_cols += ['matched-gt-id-' + str(tiou)]
            self.fp_error_type_cols += ['fp-error-type-' + str(tiou)]

        self.min_tiou_thr = min_tiou_thr

        if self.verbose:
            print '[INIT] Loaded annotations from {} subset.'.format(subset)
            nr_gt = len(np.unique(self.ground_truth['gt-id']))
            print '\tNumber of ground truth instances: {}'.format(nr_gt)
            nr_pred = len(self.prediction)
            print '\tNumber of predictions: {}'.format(nr_pred)
            print '\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds)


    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        gt_id_lst, current_gt_id = [], 0
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        
        if self.load_extra_annotations:
            print '[INIT] Loading extra annotations'
            extra_annotations = dict(zip(self.characteristic_names,[[] for _ in range(len(self.characteristic_names))]))

        for videoid, v in data['database'].iteritems():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1

                if self.evaluate_with_multi_segments and self.load_extra_annotations:
                    for seg_idx in range(self.evaluate_with_multi_segments):
                        gt_id_lst.append(current_gt_id)
                        video_lst.append(videoid)
                        t_start_lst.append(float(ann['all-segments'][seg_idx][0]))
                        t_end_lst.append(float(ann['all-segments'][seg_idx][1]))
                        label_lst.append(activity_index[ann['label']])
                        
                        for characteristic_name in self.characteristic_names:
                            extra_annotations[characteristic_name].append(ann[characteristic_name]) 
                else:
                    gt_id_lst.append(current_gt_id)
                    video_lst.append(videoid)
                    t_start_lst.append(float(ann['segment'][0]))
                    t_end_lst.append(float(ann['segment'][1]))
                    label_lst.append(activity_index[ann['label']])
                    if self.load_extra_annotations:
                        for characteristic_name in self.characteristic_names:
                            extra_annotations[characteristic_name].append(ann[characteristic_name]) 
                current_gt_id +=1

        ground_truth = pd.DataFrame({'gt-id': gt_id_lst,
                                     'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     })        

        if self.load_extra_annotations:
            for characteristic_name in self.characteristic_names:
                ground_truth[characteristic_name] = extra_annotations[characteristic_name]

            for (characteristic_name, (bins, labels)) in self.characteristic_names_to_bins.iteritems():
                ground_truth[characteristic_name] = extra_annotations[characteristic_name]
                ground_truth[characteristic_name] = pd.cut(ground_truth[characteristic_name], precision=2, bins=bins, labels=labels, include_lowest=True)

            if 'coverage' in self.characteristic_names:
                # remove instances with coverage > 1
                ground_truth = ground_truth.loc[(np.array(extra_annotations['coverage'])) <= 1.0]

        # remove instances of length <=0 
        ground_truth = ground_truth.loc[ground_truth['t-start'].values < ground_truth['t-end'].values]

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].iteritems():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])

        prediction_id_lst = range(len(video_lst))

        prediction = pd.DataFrame({'prediction-id': prediction_id_lst,
                                   'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})

        return prediction


    def _limit_prediction(self):
        """
            Of each class J, limit the predictions to the top scoring (N_j * self.limit_factor) 
            predictions, where N_j is the number of ground truth instances for class J.
        """
        ground_truth_gbvn = self.ground_truth.groupby('label')
        prediction_gbvn = self.prediction.groupby('label')
        
        filtered_prediction_df_list = []
        for label, this_ground_truth in ground_truth_gbvn:
            try:
                # Check if there is at least one prediction for this class.
                this_prediction = prediction_gbvn.get_group(label)
            except Exception as e:
                continue
            
            # pick the top (len(this_ground_truth)*self.limit_factor) predictions
            filtered_prediction_df_list += [this_prediction.nlargest(n=int(len(this_ground_truth)*self.limit_factor),
                                                                     columns='score')]

        filtered_prediction = pd.concat(filtered_prediction_df_list, ignore_index=True)

        # reset prediction ids
        filtered_prediction['prediction-id'] = range(len(filtered_prediction))

        return filtered_prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        recall = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        precision = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        matched_gt_id = np.zeros((len(self.tiou_thresholds), len(self.prediction)))

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=self.ground_truth.loc[self.ground_truth['label'] == cidx].reset_index(drop=True),
                        prediction=self.prediction.loc[self.prediction['label'] == cidx].reset_index(drop=True),
                        tiou_thresholds=self.tiou_thresholds,
                        normalize_ap=self.normalize_ap, 
                        average_num_instance_per_class=self.average_num_instance_per_class,
                        minimum_normalized_precision_threshold_for_detection=self.minimum_normalized_precision_threshold_for_detection,
                    ) for cidx in self.activity_index.values())
        
        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx], matched_this_cls_gt_id, this_cls_prediction_ids, recall[:,cidx], precision[:,cidx] = results[i]
            matched_gt_id[:,this_cls_prediction_ids] = matched_this_cls_gt_id

        return ap, matched_gt_id, recall, precision

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap, self.matched_gt_id, self.recall, self.precision = self.wrapper_compute_average_precision()

        for tidx, column_name in enumerate(self.matched_gt_id_cols):
            self.prediction[column_name] = self.matched_gt_id[tidx]

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        self.mRecall = self.recall.mean(axis=1)
        self.average_mRecall = self.mRecall.mean()

        self.mPrecision = self.precision.mean(axis=1)
        self.average_mPrecision = self.mPrecision.mean()

        if self.verbose:
            print '[RESULTS] Performance on ActivityNet detection task.'
            print '[RESULTS] Using %d annotation segment(s) per instance' % self.evaluate_with_multi_segments if self.evaluate_with_multi_segments and self.load_extra_annotations else ''
            print '\tAverage-mAP{}: {}'.format('_N' if self.normalize_ap else '', self.average_mAP)
            # print '\tAverage-mRecall: {}'.format(self.average_mRecall)
            # print '\tAverage-mPrecision: {}'.format(self.average_mPrecision)

    def wrapper_analyze_fp_error_types(self):
        self.fp_error_types_legned = {'True Positive': 0,
                                      'Double Detection Err': 1,
                                      'Wrong Label Err': 2,
                                      'Localization Err': 3,
                                      'Confusion Err': 4,
                                      'Background Err': 5}

        self.fp_error_types_inverse_legned = dict([(v, k) for k, v in self.fp_error_types_legned.iteritems()])

        fp_error_types = Parallel(n_jobs=len(self.tiou_thresholds))(
                            delayed(analyze_fp_error_types)(
                                prediction=self.prediction,
                                ground_truth=self.ground_truth,
                                tiou_thr=tiou_thr,
                                matched_gt_id_col_name=matched_gt_id_col_name,
                                min_tiou_thr=self.min_tiou_thr,
                                fp_error_types_legned=self.fp_error_types_legned,
                            ) for tiou_thr, matched_gt_id_col_name in zip(self.tiou_thresholds, self.matched_gt_id_cols))
        
        return fp_error_types

    def diagnose(self):
        """Analyzes the error types and add the results to self.prediction DataFrame.
        Computes the average-mAP gain after removing each error type.

        [WARNING]: diagnose() can only be run after evaluate() has finished
        """

        # Augment the prediction DataFrame with the error types
        self.fp_error_types = self.wrapper_analyze_fp_error_types()
        self.fp_error_types_count = {}

        for tidx, column_name in enumerate(self.fp_error_type_cols):
            self.prediction[column_name] = self.fp_error_types[tidx]
            
            this_tiou = self.tiou_thresholds[tidx]
            self.fp_error_types_count[this_tiou] = dict(zip(self.fp_error_types_legned.keys(),
                                                            [0]*len(self.fp_error_types_legned)))
            error_ids, counts = np.unique(self.fp_error_types[tidx], return_counts=True)
            for error_id,count in zip(error_ids, counts):
                self.fp_error_types_count[this_tiou][self.fp_error_types_inverse_legned[error_id]] = count
        
        self.fp_error_types_count_df = pd.DataFrame(self.fp_error_types_count)
        self.fp_error_types_count_df['avg'] = self.fp_error_types_count_df.mean(axis=1)
        self.fp_error_types_precentage_df = self.fp_error_types_count_df/len(self.prediction)

        # Computes the average-mAP gain after removing each error type
        self.ap_gain, self.average_mAP_gain = {}, {}

        for err_name, err_code in self.fp_error_types_legned.iteritems():
            if err_code:
                self.ap_gain[err_name] = np.zeros((len(self.tiou_thresholds),
                                        len(self.activity_index)))
                for cidx in self.activity_index.values():

                    this_pred_df = self.prediction[self.prediction['label']==cidx].reset_index(drop=True)
                    sort_idx = this_pred_df['score'].values.argsort()[::-1]
                    this_pred_df = this_pred_df.loc[sort_idx].reset_index(drop=True)
                    this_gt_df = self.ground_truth[self.ground_truth['label']==cidx]

                    npos=len(this_gt_df)

                    for tidx in range(len(self.tiou_thresholds)):
                        this_error_types = this_pred_df[self.fp_error_type_cols[tidx]].T.values
                        tp = (~np.isnan(this_pred_df[self.matched_gt_id_cols[tidx]].T)).astype(np.int)
                        tp = tp[this_error_types!=err_code]
                        fp = np.abs(tp - 1)

                        # Computing prec-rec
                        this_tp = np.cumsum(tp).astype(np.float)
                        this_fp = np.cumsum(fp).astype(np.float)
                        rec = this_tp / npos
                        if self.normalize_ap:
                            prec = rec * self.average_num_instance_per_class / (rec * self.average_num_instance_per_class + this_fp)
                        else:
                            prec = rec * npos / (rec * npos + this_fp)
                        self.ap_gain[err_name][tidx,cidx] = interpolated_prec_rec(prec, rec)
                self.average_mAP_gain[err_name] = self.ap_gain[err_name].mean() - self.average_mAP 

        if self.verbose:
            print '[DIAGNOSIS] Analysis of false positive error types.'
            print '\tPercentage of each error type:\n{}'.format(self.fp_error_types_precentage_df)
            print '\tAverage mAP gain after removing each error type:\n{}'.format(self.average_mAP_gain)


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                        normalize_ap=False, average_num_instance_per_class=None,
                                        minimum_normalized_precision_threshold_for_detection=0.05):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    gt_id_lst = np.unique(ground_truth['gt-id'].values)
    gt_id_to_index = dict(zip(gt_id_lst, range(len(gt_id_lst))))
    lock_gt = np.ones((len(tiou_thresholds),len(gt_id_to_index))) * -1
    
    npos = float(len(gt_id_lst))

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    matched_gt_id = np.nan*np.zeros((len(tiou_thresholds), len(prediction)))

    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap, matched_gt_id, prediction['prediction-id'].values, 0, 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, gt_id_to_index[this_gt.loc[jdx]['gt-id']]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, gt_id_to_index[this_gt.loc[jdx]['gt-id']]] = idx
                matched_gt_id[tidx, idx] = this_gt.loc[jdx]['gt-id']
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    if normalize_ap:
        precision_cumsum = recall_cumsum * average_num_instance_per_class / (recall_cumsum * average_num_instance_per_class + fp_cumsum)

        discard_index = precision_cumsum <= minimum_normalized_precision_threshold_for_detection
        
        tp[discard_index] = 0
        fp[discard_index] = 1
        matched_gt_id[discard_index] = np.nan

        tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
        fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
        recall_cumsum = tp_cumsum / npos

        precision_cumsum = recall_cumsum * average_num_instance_per_class / (recall_cumsum * average_num_instance_per_class + fp_cumsum)
    else: 
        precision_cumsum = recall_cumsum * npos / (recall_cumsum * npos + fp_cumsum)
    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    recall = recall_cumsum[:,-1]
    precision = precision_cumsum[:,-1]

    return ap, matched_gt_id, prediction['prediction-id'].values, recall, precision

def analyze_fp_error_types(prediction,
                           ground_truth,
                           tiou_thr,
                           matched_gt_id_col_name,
                           min_tiou_thr=0.1,
                           fp_error_types_legned={'True Positive': 0,
                                                  'Double Detection Err': 1,
                                                  'Wrong Label Err': 2,
                                                  'Localization Err': 3,
                                                  'Confusion Err': 4,
                                                  'Background Err': 5}):
    
    """Assumes that prediction is sorted by 'prediction-id' column """
    
    fp_error_types = {}

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    fp_error_types = np.zeros(len(prediction))
    
    this_prediction = prediction[np.isnan(prediction[matched_gt_id_col_name])].reset_index(drop=True)

    this_prediction.sort_values(by='video-id',inplace=True)
    this_prediction.reset_index(drop=True,inplace=True)

    current_video_id = None

    for idx, this_pred in this_prediction.iterrows():

        if this_pred['video-id'] != current_video_id:
            try:
                this_gt = ground_truth_gbvn.get_group(this_pred['video-id']).reset_index()
            except:
                fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Background Err']
                current_video_id = this_pred['video-id']
                continue

            current_video_id = this_pred['video-id']

        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        gt_with_max_tiou_label = this_gt.loc[tiou_arr.argmax()]['label']
        top_tiou = tiou_arr.max()
        this_pred_label = this_pred['label']

        if top_tiou >= tiou_thr: 
            if gt_with_max_tiou_label == this_pred_label:
                # double detection error
                fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Double Detection Err']
            else:
                # wrong label error
                fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Wrong Label Err']
        elif top_tiou >= min_tiou_thr:
            if gt_with_max_tiou_label == this_pred_label:
                # localization error
                fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Localization Err']
            else:
                # confusion error
                fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Confusion Err']
        else:
            # background error
            fp_error_types[this_pred['prediction-id']] = fp_error_types_legned['Background Err']
    
    return fp_error_types
