import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import signal


def set_gpu():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)


def focal_loss(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.0
    alpha = 0.25

    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def weighted_focal_loss(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.0

    eps = K.epsilon()
    p = y_pred[:,1]
    q = y_pred[:,0]

    p = tf.math.maximum(p, eps)
    q = tf.math.maximum(q, eps)

    pos_loss = -(q ** gamma) * tf.math.log(p)

    pos_loss *= tf.cast(tf.reduce_sum(tf.cast(y_true[:,1] == 0, tf.float32))/(tf.reduce_sum(tf.cast(y_true[:,1], tf.float32))+K.epsilon()), 'float32')

    # Loss for the negative examples
    neg_loss = -(p ** gamma) * tf.math.log(q)

    labels = tf.dtypes.cast(y_true[:,1], dtype=tf.bool)
    loss = tf.where(labels, pos_loss, neg_loss)

    return loss


def weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)

        # weighted calc
        weight_vector = y_true[:,1] * one_weight + (1 - y_true[:,1]) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def weighted_binary_crossentropy_adapt(y_true, y_pred):

    b_ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)

    # weighted calc
    one_wt = tf.cast(tf.reduce_sum(tf.cast(y_true[:,1] == 0, tf.float32))/(tf.reduce_sum(tf.cast(y_true[:,1], tf.float32))+K.epsilon()), 'float32')
    zero_wt = tf.constant(1, 'float32')

    weight_vector = y_true[:,1] * one_wt + (1 - y_true[:,1]) * zero_wt
    weighted_b_ce = weight_vector * b_ce

    return K.mean(weighted_b_ce)


def decay_schedule(epoch, lr):
    if lr > 1e-5:
        if (epoch + 1) % 10 == 0:
            lr = lr / 2
        
    return lr


#######################################
############### metrics ###############

def aucc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def sens(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def spec(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true[:, 1]) * (1 - y_pred[:, 1]), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true[:, 1], 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def sens_ovlp(y_true, y_pred):
    TP, FN, _ = perf_measure_ovlp_tensor(y_true, y_pred)
    return tf.cast(TP, 'float64') / (tf.cast(TP+FN, 'float64')+K.epsilon())


def fah_ovlp(y_true, y_pred):
    # false alarms per duration of batch ignoring overlap between windows
    _, _, FP = perf_measure_ovlp_tensor(y_true, y_pred)
    return tf.cast(FP, 'int64')*tf.cast(tf.constant(3600), 'int64')/tf.cast(tf.shape(y_true)[0], 'int64')


def fah_epoch(y_true, y_pred):
    fa_epoch = K.sum(K.clip(K.round(y_pred[:, 1])-y_true[:, 1], 0, 1))
    return tf.cast(fa_epoch, 'float64')*tf.cast(tf.constant(3600), 'float64')/tf.cast(tf.shape(y_true)[0], 'float64')


def faRate_epoch(y_true, y_pred):
    fa_epoch = K.sum(K.clip(K.round(y_pred[:, 1])-y_true[:, 1], 0, 1))
    return tf.cast(fa_epoch, 'float64')/tf.cast(tf.shape(y_true)[0], 'float64')


def score(y_true, y_pred):
    sens = sens_ovlp(y_true, y_pred)
    fa_epoch = K.sum(K.clip(K.round(y_pred[:, 1])-y_true[:, 1], 0, 1))
    fa_rate = tf.cast(fa_epoch, 'float64')/tf.cast(tf.shape(y_true)[0], 'float64')
    return sens*tf.constant(100, dtype='float64')-tf.constant(0.4, dtype='float64')*fa_rate


def perf_measure_ovlp_tensor(y_true, y_pred):
    true_evs = tf.concat([y_true[:, 1], tf.constant([0], dtype='float32')], 0)
    true_evs = tf.concat([ tf.constant([0], dtype='float32'), true_evs], 0)
    mask = tf.equal(true_evs, 1)
    start_positions = tf.where(tf.logical_and(~mask[:-1], mask[1:])) + 1
    end_positions = tf.where(tf.logical_and(mask[:-1], ~mask[1:])) + 1
    true_ranges = tf.concat([start_positions, end_positions], axis=1)

    pred_evs = tf.concat([tf.cast(K.round(y_pred[:, 1]), 'float64'), tf.constant([0], dtype='float64')], 0)
    pred_evs = tf.concat([tf.constant([0], dtype='float64'), pred_evs], 0)
    mask = tf.equal(pred_evs, 1)
    start_positions = tf.where(tf.logical_and(~mask[:-1], mask[1:])) + 1
    end_positions = tf.where(tf.logical_and(mask[:-1], ~mask[1:])) + 1
    pred_ranges = tf.concat([start_positions, end_positions], axis=1)

    # Expand dimensions to enable broadcasting
    true_expanded = tf.expand_dims(true_ranges, axis=0)  # Shape: (1, m, 2)
    pred_expanded = tf.expand_dims(pred_ranges, axis=1)  # Shape: (n, 1, 2)

    # Calculate overlap for a and b
    overlap_start = tf.maximum(true_expanded[:, :, 0], pred_expanded[:, :, 0])
    overlap_end = tf.minimum(true_expanded[:, :, 1], pred_expanded[:, :, 1])

    overlaps = tf.maximum(tf.constant(0,dtype='int64'), overlap_end - overlap_start)
    overlaps_sum_true = tf.reduce_sum(overlaps, axis=0)
    overlaps_sum_pred = tf.reduce_sum(overlaps, axis=1)

    TP = tf.cast(tf.math.count_nonzero(overlaps_sum_true), 'int64')
    FN = tf.cast(tf.shape(overlaps_sum_true)[0], 'int64') - TP
    FP = tf.cast(tf.shape(overlaps_sum_pred)[0], 'int64') - tf.cast(tf.math.count_nonzero(overlaps_sum_pred), 'int64')

    return TP, FN, FP


#### Pre-process EEG data

def apply_preprocess_eeg(config, rec):

    idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']
    idx_cross = [i for i, c in enumerate(rec.channels) if c == 'CROSStop SD']
    if not idx_cross:
        idx_cross = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']

    ch_focal, _ = pre_process_ch(rec.data[idx_focal[0]], rec.fs[idx_focal[0]], config.fs)
    ch_cross, _ = pre_process_ch(rec.data[idx_cross[0]], rec.fs[idx_cross[0]], config.fs)
        
    # ch_focal = (ch_focal - np.mean(ch_focal))/np.std(ch_focal)
    # ch_cross = (ch_cross - np.mean(ch_cross))/np.std(ch_cross)

    return [ch_focal, ch_cross]


def pre_process_ch(ch_data, fs_data, fs_resamp):

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))
    
    b, a = signal.butter(4, 0.5/(fs_resamp/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs_resamp/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs_resamp/2), 50.5/(fs_resamp/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


#### EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen), min(int(event[1]*fs), totalLen)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0]+1)/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1]+1)/fs, (len(mask))/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([(start_i[i]+1)/fs, (end_i[i]+1)/fs])
        events += tmp
    return events


def merge_events(events, distance):
    """ Merge events.
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i-1][1] < distance:
            events[i-1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events


def get_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    events_merge = merge_events(events, 0.2*margin)
    ev_list = []
    for i in range(len(events_merge)):
        if events_merge[i][1] - events_merge[i][0] >= margin*0.8:
            ev_list.append(events_merge[i])

    return ev_list



def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)
    
    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = get_events(events, margin)
    pred = eventList2Mask(events, len(y_pred), fs)

    return pred


def getOverlap(a, b):
    ''' If > 0, the two intervals overlap.
    a = [start_a, end_a]; b = [start_b, end_b]
    '''
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def perf_measure_epoch(y_true, y_pred):
    ''' Calculate the performance metrics based on the EPOCH method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments

    Returns:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
           FP += 1
        if y_true[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure_ovlp(y_true, y_pred, fs):
    ''' Calculate the performance metrics based on the any-overlap method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments
        fs: sampling frequency of the predicted and ground-truth label arrays
            (in this challenge, fs = 1/2)

    Returns:
        TP: true positives
        FP: false positives
        FN: false negatives
    '''
    true_events = mask2eventList(y_true, fs)
    pred_events = mask2eventList(y_pred, fs)

    TP = 0
    FP = 0
    FN = 0

    for pr in pred_events:
        found = False
        for tr in true_events:
            if getOverlap(pr, tr) > 0:
                TP += 1
                found = True
        if not found:
            FP += 1
    for tr in true_events:
        found = False
        for pr in pred_events:
            if getOverlap(tr, pr) > 0:
                found = True
        if not found:
            FN += 1

    return TP, FP, FN


def get_metrics_scoring(y_pred, y_true, fs, th):
    ''' Get the score for the challenge.

    Args:
        pred_file: path to the prediction file containing the objects 'filenames',
                   'predictions' and 'labels' (as returned by 'predict_net' function)
    
    Returns:
        score: the score of the challenge
        sens_ovlp: sensitivity calculated with the any-overlap method
        FA_epoch: false alarm rate (false alarms per hour) calculated with the EPOCH method
    '''

    total_N = len(y_pred)*(1/fs)
    total_seiz = np.sum(y_true)

    # Post process predictions (merge predicted events separated by 2 second and discard events smaller than 8 seconds)
    y_pred = post_processing(y_pred, fs=fs, th=th, margin=10)

    TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_true, y_pred)

    TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_true, y_pred, fs=1/2)

    if total_seiz == 0:
        sens_ovlp = float("nan")
        prec_ovlp = float("nan")
        f1_ovlp = float("nan")
    else:
        sens_ovlp = TP_ovlp/(TP_ovlp + FN_ovlp)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_ovlp = float("nan")
            f1_ovlp = float("nan")
        else:
            prec_ovlp = TP_ovlp/(TP_ovlp + FP_ovlp)
            if prec_ovlp+sens_ovlp == 0:
                f1_ovlp = float("nan")
            else:
                f1_ovlp = (2*prec_ovlp*sens_ovlp)/(prec_ovlp+sens_ovlp)
    
    FA_ovlp = FP_ovlp*3600/total_N
    FA_epoch = FP_epoch*3600/total_N

    if total_seiz == 0:
        sens_epoch = float("nan")
        prec_epoch = float("nan")
        f1_epoch = float("nan")
    else:
        sens_epoch = TP_epoch/(TP_epoch + FN_epoch)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_epoch = float("nan")
            f1_epoch = float("nan")
        else:
            prec_epoch = TP_epoch/(TP_epoch + FP_epoch)
            if prec_epoch+sens_epoch == 0:
                f1_epoch = float("nan")
            else:
                f1_epoch = (2*prec_epoch*sens_epoch)/(prec_epoch+sens_epoch)

    spec_epoch = TN_epoch/(TN_epoch + FP_epoch)

    return sens_ovlp, prec_ovlp, FA_ovlp, f1_ovlp, sens_epoch, spec_epoch, prec_epoch, FA_epoch, f1_epoch
