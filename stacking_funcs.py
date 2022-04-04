##Note on Feb-21-2022, 6:20pm:
##For now, to be able to use this script, you will need to have both the TS distributions for the sources and control fields.
##I will soon create tabulated information for the Control Fields. 
##Currently, to use the this package, import all the functions to your python script or notebook. 
##Call deltaTScum() function twice for the sources and CFs respectively. 
##You'll have to put the first part of the names of the files in the first argument of the function. 
##The 2nd arg would be the array for the TS values;
##The 3rd arg would be the total number in stack;
##The 4th arg would be the bootstrap resampling (which is for the cumTS() func)
##The 5th arg would be the overall amount to repeat the whole process (the final number in the final distribution)


from psrqpy import QueryATNF
import os 
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import QTable, Table, Column, MaskedColumn, join

from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import fnmatch
from os import path, listdir, chdir, mkdir
from astropy.io import fits
from subprocess import call

from scipy.integrate import quad

import glob
from astropy.wcs import WCS
import sys

from scipy.stats import chi2,norm

##Calculate the cumulative TS vs. stack.
def cumTS(ts1, nnn1, ntrials=20): 
    if any(x<0 for x in ts1):
        tsfinal=ts1[ts1>0]
        
    else:
        tsfinal=ts1
    #if len(tsfinal)>nnn1:
    #    nnn1=len(tsfinal)
    overall_TS_psr=np.zeros([nnn1,ntrials])
    
    for k in range(ntrials):
        indices_psr=np.random.randint(0,len(tsfinal),nnn1)
        
        TS300sub=np.array([])
        for i in range(nnn1):
            TS300sub=np.append(TS300sub,(tsfinal[indices_psr[i]]))
        
        TS_acc = np.array([])
        
        for i in range(nnn1):
            TS_acc=np.append(TS_acc, np.sum(TS300sub[:i]))
        
        overall_TS_psr[:,k]=TS_acc

    todraw_psr=np.zeros(nnn1)
    todraw_psr_err=np.zeros(nnn1)
    for i in range(nnn1):
        todraw_psr[i]=np.mean(overall_TS_psr[i,:])
        todraw_psr_err[i]=np.std(overall_TS_psr[i,:])
    return(todraw_psr, todraw_psr_err)
    

##Calculate the cumulative TS distribution. 
def deltaTScum(name,TS_source, n_source, n_bs=100, n_samples=1000):
    TSv2=TS_source[TS_source>0]
    final_dTS_fs=np.array([])
    final_dis_fs=np.array([])
    for i in range(n_samples):
        cf2,cf2err=cumTS(TSv2,n_source,n_bs)
        final_dTS_fs=np.append(final_dTS_fs, cf2[-1])
        final_dis_fs=np.append(final_dis_fs,cf2err[-1])
    np.save(name+'_dTS.npy',final_dTS_fs,allow_pickle=True)
    np.save(name+'_dis.npy',final_dis_fs,allow_pickle=True)
    return(final_dTS_fs,final_dis_fs)
        
##This is to resample the TS distribution for control fields since there's excess for them;
def resampling_CFTS(TScf,ncfs=412,ntrials=100):
    overall_tscf=np.zeros([30,ntrials])
    TScf_final=TScf[TScf>0]
    histbins=np.arange(0,31)
    for i in range(ntrials):
        ind=np.random.randint(0,len(TScf_final),ncfs)
        tspart=TScf_final[ind]
        overall_tscf[:,i]=np.histogram(tspart,histbins)[0]

    bs_tscf=np.zeros(30)
    bs_tscf_std=np.zeros(30)

    for i in range(30):
        bs_tscf[i]=np.mean(overall_tscf[i,:])
        bs_tscf_std[i]=np.std(overall_tscf[i:])
        
    return(bs_tscf,bs_tscf_std)

##The final plot. 
def draw_the_plot(TS_300, TScf_300, name1,name2, dof=2):
    fig1 = plt.figure(constrained_layout=True,figsize=(11,7))
    gs1 = fig1.add_gridspec(2,2)
    
    ###TS distribution of the sources
    f1_ax1=fig1.add_subplot(gs1[0,0])

    pos_ts=TS_300[TS_300>0]
    where_are_NaNs = np.isnan(pos_ts)
    pos_ts[where_are_NaNs] = 0
    print(pos_ts.max())
    pos_ts[pos_ts<0]=0
    #axs10.set_title('TS Dist, PSRs')
    f1_ax1.axvline(25,c='r',ls='--',label="TS = 25")
    f1_ax1.set_yscale('log')
    #f1_ax1.set_xlabel('Test Statistic (TS)',fontsize=14)
    f1_ax1.set_ylabel('# PSRs',fontsize=14)
    f1_ax1.set_xlim(0,np.amax(pos_ts)+2)
    f1_ax1.set_ylim(0.9,len(pos_ts)/2)
    f1_ax1.set_ylim(0.9,400)
    #axs10.set_ylim(1./len(TS),1)
    f1_ax1.set_yticklabels(['','',r'10$^{0}$',r'10$^{1}$',r'10$^{2}$'],fontsize=14)
    f1_ax1.set_xticklabels([])#[0,5,10,15,20,25,30],fontsize=14)
    #axs3.set_ylim(1e-3, 1)
    bins=np.arange(0, int(np.amax(pos_ts))+2)
    extrabins=np.linspace(0,1,20)
    #(nn, nnbins, nnpatch)=axs10.hist([pos_ts],histtype='step',density=True, bins=bins,label=['TS, PLEC'])
    (nn, nnbins, nnpatch)=f1_ax1.hist([pos_ts],histtype='step', bins=bins,label=[name1])

    #newnbins=np.append(extrabins,nnbins[1:])
    newnbins=np.append(extrabins,np.linspace(1,(nnbins.max()-1),(nnbins.max()-1)*4))
    chi2int=np.zeros(len(nnbins))
    for i in range(len(nnbins)-1):
        I=quad(chi2.pdf,nnbins[i],nnbins[i+1],args=(dof,0,1))
        chi2int[i]=len(pos_ts)*I[0]/2
    f1_ax1.plot(nnbins, chi2int*2,drawstyle='steps-post',label=r"${\chi}^2/2$, dof=3",c='orange')
    #axs10.plot(newnbins,nnchi2)
    f1_ax1.legend(loc='upper right',fontsize=14)

    ###Bootstrapped TS distribution of the controls 
    f1_ax2=fig1.add_subplot(gs1[1,0])
    pos_ts2=TScf_300[TScf_300>0]
    #axs10.set_title('TS Dist, CFs')
    f1_ax2.axvline(25,c='r',ls='--',label="TS = 25")
    f1_ax2.set_yscale('log')
    f1_ax2.set_xlabel('Test Statistic (TS)',fontsize=14)
    f1_ax2.set_ylabel('# test sources',fontsize=14)
    f1_ax2.set_xlim(0,np.amax(pos_ts)+2)
    f1_ax2.set_ylim(0.9,len(TScf_300)/2)
    f1_ax2.set_ylim(0.9,400)
    f1_ax2.set_yticklabels(['','',r'10$^{0}$',r'10$^{1}$',r'10$^{2}$'],fontsize=14)
    f1_ax2.set_xticklabels([0,5,10,15,20,25,30],fontsize=14)
    #axs3.set_ylim(1e-3, 1)
    bins=np.arange(0, int(np.amax(pos_ts2))+2)
    extrabins=np.linspace(0,1,20)
    bs_tscf, bs_tscf_std=resampling_CFTS(pos_ts2,len(pos_ts),100)
    f1_ax2.plot(np.arange(-1,30),np.append([0],bs_tscf),drawstyle='steps-post',label='TS of test sources',c='g',lw=0.8)
    f1_ax2.errorbar(np.arange(0,30)+0.5,bs_tscf,yerr=bs_tscf_std,c='g',ls='none',lw=0.8)
    
    newnbins=np.append(extrabins,np.linspace(1,(nnbins.max()-1),(nnbins.max()-1)*4))
    chi2int=np.zeros(len(nnbins))
    for i in range(len(nnbins)-1):
        I=quad(chi2.pdf,nnbins[i],nnbins[i+1],args=(dof,0,1))
        chi2int[i]=len(pos_ts)*I[0]/2
    f1_ax2.plot(nnbins, chi2int*2,drawstyle='steps-post',label=r"${\chi}^2/2$, dof=3",c='orange')
    
    f1_ax2.legend(loc='upper right',fontsize=14)

    final_dTS_fs3=np.load(name1+'_dTS.npy',allow_pickle=True)
    final_cTS=np.load(name2+'_dTS.npy',allow_pickle=True)
    final_cTS_dis=np.load(name2+'_dis.npy',allow_pickle=True)
    final_dis_fs3=np.load(name1+'_dis.npy',allow_pickle=True)
    
    ###Cumulative TS distribution
    f1_ax3=fig1.add_subplot(gs1[0,1])
    nnn1=len(pos_ts)
    todraw_psr, todraw_psr_err=cumTS(pos_ts, nnn1, 100)
    print(todraw_psr[-1],norm.fit(final_dTS_fs3)[0])
    while (abs(todraw_psr[-1]-norm.fit(final_dTS_fs3)[0])>5) or abs((todraw_psr_err[-1]-norm.fit(final_dis_fs3)[0])>50):
        todraw_psr, todraw_psr_err=cumTS(pos_ts, nnn1, 100)
    
    todraw_cfs, todraw_cfs_err=cumTS(pos_ts2, nnn1, 100)
    print(todraw_cfs[-1],norm.fit(final_cTS)[0])

    while (abs(todraw_cfs[-1]-norm.fit(final_cTS)[0])>5) or abs((todraw_cfs_err[-1]-norm.fit(final_cTS_dis)[0])>50):
        todraw_cfs, todraw_cfs_err=cumTS(pos_ts2, nnn1, 100)
    
    f1_ax3.plot(np.arange(nnn1), todraw_psr, c='b', ls='-', label='Sources')
    f1_ax3.fill_between(np.arange(nnn1), todraw_psr+todraw_psr_err,todraw_psr-todraw_psr_err,alpha=0.3, interpolate=False, step=None, color='deepskyblue')
    f1_ax3.plot(np.arange(nnn1), todraw_cfs, c='g', ls='-',  ms=2, label='Background')
    f1_ax3.fill_between(np.arange(nnn1), todraw_cfs+todraw_cfs_err, todraw_cfs-todraw_cfs_err, alpha=0.3,interpolate=False, step=None, color='lawngreen')
    print(todraw_psr[-1]-todraw_cfs[-1])
    f1_ax3.set_xlim(0,nnn1+1)
    f1_ax3.set_ylim(0,1.1*(todraw_psr[-1]+todraw_psr_err[-1]))
    f1_ax3.set_xlabel('Number in Stack',fontsize=14)
    f1_ax3.set_ylabel(ylabel='TS',fontsize=14)
    f1_ax3.legend(loc='upper left')

    ###delta(TS)_cum distributions 
    f1_ax4=fig1.add_subplot(gs1[1,1])


    f1_ax4.hist([final_dTS_fs3,final_cTS],\
             bins=50,color=['b','g'],histtype='step',label=[name1,'Control Fields'])
    #ax.hist([final_cTS2,norm.fit(final_dTS_fs3)[0]-final_cTS2+norm.fit(final_cTS)[0]],\
    #         bins=30,color=['gray','r'],density=True,histtype='step')
    #plt.legend(loc='upper left',fontsize=8)
    f1_ax4.axvline(norm.fit(final_cTS)[0]+norm.fit(final_cTS_dis)[0], c='g',alpha=0.5,ls='dotted')#,label=r'Control Field $\sigma$')
    f1_ax4.axvline(norm.fit(final_cTS)[0]-norm.fit(final_cTS_dis)[0], c='g',alpha=0.5,ls='dotted')
    f1_ax4.axvspan(norm.fit(final_cTS)[0]-norm.fit(final_cTS_dis)[0], norm.fit(final_cTS)[0]+norm.fit(final_cTS_dis)[0],\
               alpha=0.3, color='lawngreen')


    f1_ax4.axvline(norm.fit(final_dTS_fs3)[0]+norm.fit(final_dis_fs3)[0], c='b',alpha=0.5,ls='dotted')#,label=r'PSRs $\sigma$')
    f1_ax4.axvline(norm.fit(final_dTS_fs3)[0]-norm.fit(final_dis_fs3)[0], c='b',alpha=0.5,ls='dotted')
    f1_ax4.axvspan(norm.fit(final_dTS_fs3)[0]-norm.fit(final_dis_fs3)[0], norm.fit(final_dTS_fs3)[0]+norm.fit(final_dis_fs3)[0],\
               alpha=0.3, color='deepskyblue')

    f1_ax4.axvline(norm.fit(final_cTS)[0],ls='--',c='g')#,label='Control Field Mean')
    f1_ax4.axvline(norm.fit(final_dTS_fs3)[0],ls='--',c='b')#,label='PSRs Mean')
    handles, labels = f1_ax4.get_legend_handles_labels()
    f1_ax4.set_xlabel('TS',fontsize=14)
    f1_ax4.legend(handles[::-1], labels[::-1], loc='upper center')
    
    fig1.savefig(name1+'overall.png')