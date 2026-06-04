import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
sys.path.append('..')
import allesfitter
import seaborn as sns

def run_plot():
    alles = allesfitter.allesclass('.')
    # fig, axes = plt.subplots(1,1,figsize=(12,6),tight_layout=True)
    fig, axes = plt.subplots(1,1,figsize=(5,3),tight_layout=True)
    colors = {'b': sns.color_palette()[0]}

    def get_infos(companion):
        infos = {}
        
        infos['transit_midtime_linear'] = np.array( alles.data[companion+'_tmid_observed_transits'] ) #linear-ephemerides midtimes that fall into the observation windows
        infos['transit_number'] = np.array( [ int(np.round( ( t - alles.posterior_params_median[companion+'_epoch'] ) / alles.posterior_params_median[companion+'_period'] ) ) for t in infos['transit_midtime_linear']] )
        infos['N_transits'] = len(infos['transit_number']) #number of linear-ephemerides midtimes that fall into the observation windows
        
        infos['ttv_median'] = np.array( [alles.posterior_params_median[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )
        infos['ttv_lerr'] = np.array( [alles.posterior_params_ll[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )
        infos['ttv_uerr'] = np.array( [alles.posterior_params_ul[companion+'_ttv_transit_'+str(i+1)] for i in range(infos['N_transits'])] )

        infos['ttv_median'][infos['ttv_median']==0] = np.nan #remove the one we fixed
        infos['ttv_lerr'][infos['ttv_lerr']==0] = np.nan #remove the one we fixed
        infos['ttv_uerr'][infos['ttv_uerr']==0] = np.nan #remove the one we fixed
        
        infos['transit_midtime_median'] = infos['transit_midtime_linear'] + infos['ttv_median']
        infos['transit_midtime_lerr'] = infos['ttv_lerr']
        infos['transit_midtime_uerr'] = infos['ttv_uerr']
        
        k = ~np.isnan(infos['ttv_median'])
        z = np.polyfit(infos['transit_number'][k], infos['transit_midtime_median'][k], 1)
        p = np.poly1d(z)
        infos['period_linear'] = z[0]
        infos['epoch_linear'] = z[1]
        infos['o_minus_c_median'] = infos['transit_midtime_median'] - p(infos['transit_number'])
        infos['o_minus_c_lerr'] = infos['ttv_lerr']
        infos['o_minus_c_uerr'] = infos['ttv_uerr']
        
        return infos


    def save_infos(companion, ax):
        infos = get_infos(companion)
        
        #::: save csv
        header = 'period_linear = '+str(infos['period_linear'])+'\n'+\
                'epoch_linear = '+str(infos['epoch_linear'])+'\n'+\
                'transit_number,transit_midtime_median,transit_midtime_lerr,transit_midtime_uerr,o_minus_c,o_minus_c_lerr,o_minus_c_uerr'
        X = np.column_stack((infos['transit_number'], infos['transit_midtime_median'], infos['transit_midtime_lerr'], infos['transit_midtime_uerr'], infos['o_minus_c_median'], infos['o_minus_c_lerr'], infos['o_minus_c_uerr']))
        np.savetxt('Target_'+companion+'_summary.csv', X, delimiter=',', header=header, fmt='%.7f')
        
        #::: save latex table
        with open('Target_'+companion+'_latex_table.txt','w') as f:
            f.write('Transit mid-time ($\mathrm{BJD_\{TDB}$) & O-C (min.)\n')
            for i in range(infos['N_transits']):
                a = allesfitter.utils.latex_printer.round_tex(infos['transit_midtime_median'][i], infos['transit_midtime_lerr'][i], infos['transit_midtime_uerr'][i])
                b = allesfitter.utils.latex_printer.round_tex(infos['o_minus_c_median'][i], infos['o_minus_c_lerr'][i], infos['o_minus_c_uerr'][i])
                f.write('$'+a+'$' + ' & ' + '$'+b+'$' + '\\\\\n')
        
        #::: save plot
        # ax.errorbar(infos['transit_number'], infos['ttv_median']*24*60, yerr=[infos['ttv_lerr']*24*60,infos['ttv_uerr']*24*60], marker='o', ls='none', alpha=0.3, color=colors[companion]) #if you want to show the fitted TTVs (dependent on the initial guess epoch and period) instead of the O-C (which was linearly fit from the posteriors)
        ax.errorbar(infos['transit_number'], infos['o_minus_c_median']*24*60, yerr=[infos['o_minus_c_lerr']*24*60,infos['o_minus_c_uerr']*24*60], marker='o', ls='none', color=colors[companion])
        ax.axhline(0,c='grey',ls='--')
        # ax.text(0.95,0.95,'Target '+companion,va='top',ha='right',transform=ax.transAxes)
        ax.set(ylabel='O-C (min)', xlabel='Transit Nr.')
        
        current_path = os.getcwd()
        toi_name = 'TOI '+current_path.split('/')[-1]
        ax.set_title(toi_name, weight='bold')
    save_infos('b',axes)
    # save_infos('c',axes[1])
    fig.savefig('o_minus_c.png', bbox_inches='tight')


def update_params_withttv():
    params_ = open(f'./params.csv').readlines()
    for line in params_:
        linestrip = line.strip()
        if 'b_period' in linestrip:
            b_period = linestrip.split(',')[1]
        if 'b_epoch' in linestrip:
            b_epoch = linestrip.split(',')[1]
    times = []
    setting_ = open(f'./settings.csv').readlines()
    fast_fit_width = 0
    for line in setting_:
        linestrip = line.strip()
        if linestrip.startswith('inst_phot'):
            flux_inst = linestrip.split(',')[1]
        if linestrip.startswith('fast_fit_width'):
            fast_fit_width = linestrip.split(',')[1]
            fast_fit_width = float(fast_fit_width)
    flux_inst = flux_inst.split(' ')
    for i in range(len(flux_inst)):
        time = np.loadtxt(f'{flux_inst[i]}.csv', usecols=(0), delimiter=',')
        times = times +  time.tolist()
    idxs = []
    start_epoch = np.round((np.sort(times)[0] - float(b_epoch)) / float(b_period)) - 1
    end_epoch = np.round((np.sort(times)[-1] - float(b_epoch)) / float(b_period)) + 1
    
    times = np.sort(np.asarray(times))
    bad_n_transits = []
    n_ttv_transits = 0
    print('start_epoch:', start_epoch)
    print('end_epoch:', end_epoch)
    for epoch in np.arange(start_epoch, end_epoch):
        epoch = int(epoch)
        idx = np.where((times >= float(b_epoch) + epoch * float(b_period) - fast_fit_width/2) & (times <= float(b_epoch) + epoch * float(b_period) + fast_fit_width/2))[0]
        if len(idx) > 0:
            n_ttv_transits += 1
            exptime = np.mean(np.diff(times[idx]))
            nn = fast_fit_width/exptime
            midtime = np.mean(times[idx])
            #print('epoch:', n_ttv_transits, 'exptime:', exptime, 'nn:', nn, 'len(idx):', len(idx), 'midtime:', midtime)
            if len(idx) < nn*0.75:
                bad_n_transits.append(n_ttv_transits)
            idxs.append(idx)
    print('bad_n_transits:', bad_n_transits)
    newlines = []
    for line in params_:
        linestrip = line.strip()
        if 'b_ttv_transit_' in linestrip:
            continue
        newlines.append(linestrip)
    path = '.'
    ttv_initial_guess_params = open(f'./ttv_preparation/ttv_initial_guess_params.csv').readlines()
    for line in ttv_initial_guess_params:
        linestrip = line.strip()+','
        if '_ttv_transit_' in linestrip:
            linestripsplited = linestrip.split(',')
            namettv = linestripsplited[0]
            num_ttv = int(namettv.split('_')[-1])
            
            if num_ttv in bad_n_transits:
                value = 0
                lvalue = 0 - 0.1
                uvalue = 0 + 0.1
                
                linestripsplited[1] = str(value)
                linestripsplited[2] = '0'
                # linestripsplited[3] = 'trunc_normal '+str(value-2/24)+' '+str(value+2/24)+str(value)+' 0.1'
                linestripsplited[3] = 'trunc_normal '+str(lvalue)+' '+str(uvalue)+' '+str(value)+' 0.1'
            else:
                value = float(linestripsplited[1])
                lvalue = value - 0.1
                uvalue = value + 0.1
                linestripsplited[3] = 'trunc_normal '+str(lvalue)+' '+str(uvalue)+' '+str(value)+' 0.1'
            linestrip = ','.join(linestripsplited)
        newlines.append(linestrip)
    with open(f'./params.csv', 'w') as f:
        for line in newlines:
            print(line, file=f)
def update_params_withgp():
    params_ = open(f'./params.csv').readlines()
    newlines = []
    for line in params_:
        linestrip = line.strip()
        if 'baseline_gp_matern32' in linestrip:
            continue
        newlines.append(linestrip)
    
    gp_priors = pd.read_csv('./priors/summary_phot.csv')
    names = gp_priors['#name']
    gp_lines = []
    for name in names:
        exptime = float(name.split('.')[-2])/24/3600
        lnrho_lower = np.log(1/24)
        gp_ln_sigma_median = gp_priors[gp_priors['#name'] == name]['gp_ln_sigma_median'].values[0]
        gp_ln_sigma_ll = gp_priors[gp_priors['#name'] == name]['gp_ln_sigma_ll'].values[0]
        gp_ln_sigma_ul = gp_priors[gp_priors['#name'] == name]['gp_ln_sigma_ul'].values[0]
        gp_ln_sigma_err = np.max([gp_ln_sigma_ll, gp_ln_sigma_ul])
        gp_ln_rho_median = gp_priors[gp_priors['#name'] == name]['gp_ln_rho_median'].values[0]
        gp_ln_rho_ll = gp_priors[gp_priors['#name'] == name]['gp_ln_rho_ll'].values[0]
        gp_ln_rho_ul = gp_priors[gp_priors['#name'] == name]['gp_ln_rho_ul'].values[0]
        gp_ln_rho_err = np.max([gp_ln_rho_ll, gp_ln_rho_ul])
        
        if gp_ln_rho_median <= lnrho_lower:
            gp_ln_rho_median = np.log(2/24)
        
        newlines.append('baseline_gp_matern32_lnsigma_flux_'+name.strip('_flux_gp_decor')+','+str(gp_ln_sigma_median)+',1,normal '+str(gp_ln_sigma_median)+' '+str(gp_ln_sigma_err)+',matern32lnsigma;'+name.strip('_flux_gp_decor')+',,')
        newlines.append('baseline_gp_matern32_lnrho_flux_'+name.strip('_flux_gp_decor')+','+str(gp_ln_rho_median)+',1,trunc_normal '+str(lnrho_lower)+' 10 '+str(gp_ln_rho_median)+' '+str(gp_ln_rho_err)+',matern32lnrho;'+name.strip('_flux_gp_decor')+',,')
    with open(f'./params.csv', 'w') as f:
        for line in newlines:
            print(line, file=f)
    
            
def run_allesfitter(path):
#    allesfitter.prepare_ttv_fit(path)
#    if os.path.exists('./priors/summary_phot.csv'):
#        update_params_withgp()
#    else:
#        allesfitter.estimate_noise_out_of_transit(path)
#        update_params_withgp()
       
#    update_params_withttv()
   allesfitter.show_initial_guess(path,do_logprint=False)
   allesfitter.mcmc_fit(path)
   allesfitter.mcmc_output(path)
   run_plot()


path = os.getcwd()
run_allesfitter(path)


