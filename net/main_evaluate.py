import os
import re
import h5py
import numpy as np
import pandas as pd
import utils
import mne
from DL_config import Config

root_save_dir = 'save_dir'
dataset = 'SZ2_test'

models = [x for x in os.listdir(os.path.join(root_save_dir, 'models')) if 'lr_01' in x]

pred_path = os.path.join(root_save_dir, 'predictions_val')

post_rmsa = True

fs = 1

thresholds = list(np.around(np.linspace(0,1,51),2))

x_plot = np.linspace(0, 200, 200)

if not os.path.exists(os.path.join(root_save_dir, 'results')):
    os.mkdir(os.path.join(root_save_dir, 'results'))


# test_recs_list = pd.read_csv(os.path.join('Datasets', 'SZ2_test.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
# test_recs_list = test_recs_list[0].to_list()
# test_recs_list = [os.path.join(p, r) for p in test_recs_list for r in os.listdir(p) if '.edf' in r]

repeat = False

for model in models:
    print(model)
    # if not os.path.exists(os.path.join(root_save_dir, 'results', model)):
    #     os.mkdir(os.path.join(root_save_dir, 'results', model))

    result_file = os.path.join(root_save_dir, 'results_val', model + '.h5')
    if post_rmsa:
        result_file = os.path.join(root_save_dir, 'results_val', model + '_RMSA.h5')

    if os.path.isfile(result_file) and not repeat:
        print('skipping')
    else:
        sens_ovlp = []
        prec_ovlp = []
        fah_ovlp = []
        sens_ovlp_plot = []
        prec_ovlp_plot = []
        f1_ovlp = []

        sens_epoch = []
        spec_epoch = []
        prec_epoch = []
        fah_epoch = []
        f1_epoch = []

        pred_files = [x for x in os.listdir(os.path.join(pred_path, model))]
        pred_files.sort()

        for file in pred_files:
            print(file)
            with h5py.File(os.path.join(pred_path, model, file), 'r') as f:
                y_pred = list(f['y_pred'])
                y_true = list(f['y_true'])

            sens_ovlp_th = []
            prec_ovlp_th = []
            fah_ovlp_th = []
            f1_ovlp_th = []

            sens_epoch_th = []
            spec_epoch_th = []
            prec_epoch_th = []
            fah_epoch_th = []
            f1_epoch_th = []

            if post_rmsa:
                print(file.split('_'))
                pat = file.split('_')[0]
                print(pat)
                if 'SUBJ-1a' in pat:
                    center = 'University_Hospital_Leuven_Adult'
                elif 'SUBJ-1b' in pat:
                    center = 'University_Hospital_Leuven_Pediatric'
                elif 'SUBJ-4' in pat:
                    center = 'Freiburg_University_Medical_Center'
                elif 'SUBJ-5' in pat:
                    center = 'University_of_Aachen'
                elif 'SUBJ-6' in pat:
                    center = 'Karolinska_Institute'
                elif 'SUBJ-7' in pat:
                    center = 'Coimbra_University_Hospital'
                rec_file = os.path.join('/esat/biomeddata/SeizeIT2/Data_clean', center, pat, file[:-9]+'.edf')
                raw = mne.io.read_raw_edf(rec_file, include=['BTEleft SD', 'BTEright SD', 'CROSStop SD'], verbose=False)
                config = Config()
                config.fs = 250
                [ch_focal, ch_cross] = utils.apply_montage_preprocess(raw, [''], config)
                rmsa_f = [np.sqrt(np.mean(ch_focal[start:start+2*config.fs]**2)) for start in range(0, len(ch_focal) - 2*config.fs + 1, 1*config.fs)]
                rmsa_c = [np.sqrt(np.mean(ch_cross[start:start+2*config.fs]**2)) for start in range(0, len(ch_focal) - 2*config.fs + 1, 1*config.fs)]
                rmsa_f = [1 if 13 < rms < 150 else 0 for rms in rmsa_f]
                rmsa_c = [1 if 13 < rms < 150 else 0 for rms in rmsa_c]
                rmsa = rmsa_f and rmsa_c
                print(len(rmsa))
                print(np.sum(rmsa))
                print(len(y_pred))
                if len(y_pred) != len(rmsa):
                    rmsa = rmsa[:len(y_pred)]
                y_pred = np.where(np.array(rmsa) == 0, 0, y_pred)

            for th in thresholds:
                print(th)
                sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec = utils.get_metrics_scoring(y_pred, y_true, fs, th)

                sens_ovlp_th.append(sens_ovlp_rec)
                prec_ovlp_th.append(prec_ovlp_rec)
                fah_ovlp_th.append(FA_ovlp_rec)
                f1_ovlp_th.append(f1_ovlp_rec)
                sens_epoch_th.append(sens_epoch_rec)
                spec_epoch_th.append(spec_epoch_rec)
                prec_epoch_th.append(prec_epoch_rec)
                fah_epoch_th.append(FA_epoch_rec)
                f1_epoch_th.append(f1_epoch_rec)

            sens_ovlp.append(sens_ovlp_th)
            prec_ovlp.append(prec_ovlp_th)
            fah_ovlp.append(fah_ovlp_th)
            f1_ovlp.append(f1_ovlp_th)

            sens_epoch.append(sens_epoch_th)
            spec_epoch.append(spec_epoch_th)
            prec_epoch.append(prec_epoch_th)
            fah_epoch.append(fah_epoch_th)
            f1_epoch.append(f1_epoch_th)

            to_cut = np.argmax(fah_ovlp_th)
            fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
            sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
            prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

            # idx_sort = sorted(range(len(fah_ovlp_plot)), key=lambda k: fah_ovlp_plot[k])
            # fah_ovlp_plot = sorted([fah_ovlp_plot[i] for i in idx_sort])
            # sens_ovlp_plot = sorted([sens_ovlp_plot[i] for i in idx_sort])

            y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
            sens_ovlp_plot.append(y_plot)
            y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
            prec_ovlp_plot.append(y_plot)


        with h5py.File(result_file, 'w') as f:
            f.create_dataset('sens_ovlp', data=sens_ovlp)
            f.create_dataset('prec_ovlp', data=prec_ovlp)
            f.create_dataset('fah_ovlp', data=fah_ovlp)
            f.create_dataset('f1_ovlp', data=f1_ovlp)
            f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
            f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
            f.create_dataset('x_plot', data=x_plot)
            f.create_dataset('sens_epoch', data=sens_epoch)
            f.create_dataset('spec_epoch', data=spec_epoch)
            f.create_dataset('prec_epoch', data=prec_epoch)
            f.create_dataset('fah_epoch', data=fah_epoch)
            f.create_dataset('f1_epoch', data=f1_epoch)

