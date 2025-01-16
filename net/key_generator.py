import numpy as np
import random
from tqdm import tqdm
from classes.annotation import Annotation


def generate_data_keys_sequential(config, recs_list, verbose=True):
    """Create data segment keys in a sequential time manner. The keys are 4 element lists corresponding to the file index in the 'recs_list', the start and stop in seconds of the segment and it's label.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                n_segs = int(np.floor((ev[0])/config.stride)-1)
                seg_start = np.arange(0, n_segs)*config.stride
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
            else:
                for e, ev in enumerate(annotations.events):
                    if e == 0:
                        n_segs = int(np.floor((ev[0])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs)*config.stride
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        if np.sum(seg_start<0) > 0:
                            n_segs -= np.sum(seg_start<0)
                            seg_start = seg_start[seg_start>=0]
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    elif e != len(annotations.events)-1:
                        prev_event = annotations.events[e-1]
                        n_segs = int(np.floor((ev[0] - prev_event[1])/config.stride)-1)
                        seg_start = np.arange(0, n_segs)*config.stride + prev_event[1]
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    elif e == len(annotations.events)-1:
                        prev_event = annotations.events[e-1]
                        n_segs = int(np.floor((ev[0] - prev_event[1])/config.stride)-1)
                        seg_start = np.arange(0, n_segs)*config.stride + prev_event[1]
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                        n_segs = int(np.floor((annotations.rec_duration - ev[1])/config.stride)-1)
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments



def generate_data_keys_sequential_window(config, recs_list, t_add):
    """Create data segment keys in a sequential time manner with a window of 2*t_add (where t_add is in seconds). Specific key generator for the validation data of the current framework.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
            t_add: time to add before and after the center time point of the event.
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    segments = []

    for idx, f in tqdm(enumerate(recs_list)):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if annotations.rec_duration < 600:
            print('short file: ' + f[0] + ' ' + f[1])

        if annotations.events:
            if len(annotations.events) == 1:
                ev = annotations.events[0]

                if t_add*2 < ev[1]-ev[0]:
                    print('check batches!!!')
                    to_add_ev = 30
                else:
                    to_add_ev = t_add - np.round((ev[1]-ev[0])/2)

                to_add_plus = to_add_ev
                to_add_minus = to_add_ev
                
                if ev[1] + to_add_ev > np.floor(annotations.rec_duration)-config.frame:
                    to_add_plus = np.floor(annotations.rec_duration) - ev[1] - config.frame
                    to_add_minus = to_add_ev + to_add_ev - to_add_plus

                if ev[0] - to_add_ev < 0:
                    to_add_minus = ev[0]-1
                    to_add_plus = to_add_ev + to_add_ev - to_add_minus
                
                if to_add_plus + to_add_minus + ev[1] - ev[0] > t_add*2:
                    to_add_plus = to_add_plus-(to_add_plus + to_add_minus + ev[1] - ev[0] - t_add*2)
                elif to_add_plus + to_add_minus + ev[1] - ev[0] < t_add*2:
                    if to_add_plus == np.floor(annotations.rec_duration) - ev[1] - config.frame:
                        to_add_minus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                    elif to_add_minus == ev[0]-1:
                        to_add_plus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                    else:
                        to_add_plus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))

                if to_add_plus + to_add_minus + ev[1] - ev[0] != t_add*2:
                    print('bad segmentation!!!')

                segs_nr = 0

                n_segs = int(np.floor((ev[0]-(ev[0]-to_add_minus))/config.stride)-1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[0]-to_add_minus
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                segs_nr += n_segs
                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                segs_nr += n_segs
                n_segs = int(np.floor(np.floor(ev[1] + to_add_plus - ev[1])/config.stride))
                seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                segs_nr += n_segs
                if segs_nr != 600:
                    print('wrong nr segs')
            else:
                end_rec = False
                for i, ev in enumerate(annotations.events):
                    skip = False
                    if t_add*2 < ev[1]-ev[0]:
                        print('check batches!!!')
                        to_add_ev = 30
                    else:
                        to_add_ev = t_add - np.round((ev[1]-ev[0])/2)

                    if i == 0:                        
                        to_add_plus = to_add_ev
                        to_add_minus = to_add_ev

                        if ev[0] - to_add_ev < 0:
                            to_add_minus = ev[0]-1
                            to_add_plus = to_add_ev + (to_add_ev - ev[0]) + 1

                        end_seg = to_add_plus

                    else:
                        if ev[0] > end_seg:
                            if ev[0] - to_add_ev > end_seg:
                                to_add_minus = to_add_ev
                                to_add_plus = to_add_ev
                            else:
                                to_add_minus =  ev[0] - end_seg
                                to_add_plus = 2*to_add_ev - to_add_minus
                        else:
                            if ev[1] > end_seg:
                                print('check boundary case')
                            else:
                                skip = True

                        end_seg = ev[1] + to_add_plus
                    
                    if end_seg > np.floor(annotations.rec_duration)-config.frame - t_add*2:
                        end_rec = True
                    

                    if not skip and not end_rec:
                        if to_add_plus + to_add_minus + ev[1] - ev[0] > t_add*2:
                            to_add_plus -= (to_add_plus + to_add_minus + ev[1] - ev[0] - t_add*2)
                        elif to_add_plus + to_add_minus + ev[1] - ev[0] < t_add*2:
                            if to_add_plus == annotations.rec_duration - ev[1] - config.frame:
                                to_add_minus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                            elif to_add_minus == ev[0]-1:
                                to_add_plus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))
                            else:
                                to_add_plus += (t_add*2 - (to_add_plus + to_add_minus + ev[1] - ev[0]))

                        if to_add_plus + to_add_minus + ev[1] - ev[0] != t_add*2:
                            print('bad segmentation!!!')

                        if ev[1] + to_add_plus >= np.floor(annotations.rec_duration)-config.frame:
                            to_add_plus = np.floor(annotations.rec_duration)-config.frame - ev[1]
                            to_add_minus = to_add_ev + (to_add_ev - to_add_plus)

                        segs_nr = 0

                        n_segs = int(np.floor((ev[0]-(ev[0]-to_add_minus))/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0]-to_add_minus
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                        segs_nr += n_segs

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                        segs_nr += n_segs

                        n_segs = int(np.floor(np.floor(ev[1] + to_add_plus - ev[1])/config.stride))
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                        segs_nr += n_segs

                    elif skip and not end_rec:

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        idxs_seiz = [i for i,x in enumerate(segments) if x[1] in seg_start]
                        for ii in idxs_seiz:
                            segments[ii][3] = 1


                    if segs_nr != 600:
                        print('wrong nr segs')

    return segments


def generate_data_keys_subsample(config, recs_list):
    """Create data segment keys by subsampling the data, including all seizure segments (Ns) and config.factor*Ns non-seizure segments.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments_S = []
    segments_NS = []

    for idx, f in tqdm(enumerate(recs_list)):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(annotations.events):
                n_segs = int(((ev[1]+config.frame*(1-config.boundary))-(ev[0]-config.frame*(1-config.boundary))-config.frame)/config.stride_s)
                seg_start = np.arange(0, n_segs)*config.stride_s + ev[0]-config.frame*(1-config.boundary)
                seg_stop = seg_start + config.frame
                segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                if e == 0:
                    n_segs = int(np.floor((ev[0])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    if n_segs < 0:
                        n_segs = 0
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - annotations.events[e-1][1])/config.stride)-1)
                    if n_segs < 0:
                        n_segs = 0
                    seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                if e == len(annotations.events)-1:
                    n_segs = int(np.floor((np.floor(annotations.rec_duration) - ev[1])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        
    segments_S.extend(random.sample(segments_NS, config.factor*len(segments_S)))
    random.shuffle(segments_S)

    return segments_S
