from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import json


'''Pew Replotter --------------------------------------------------------'''
with open("parameters_sim=0.986842262918.json") as data_file:    
    P = json.load(data_file)[0]
dataframe=pd.read_pickle("data_sim=0.986842262918.pkl")
pew_dict=eval(open('pew_data.txt').read())

info={}
C={}
empirical={}
expressed={}
t_plot={}
for date in pew_dict.iterkeys():
	empirical[date]=[]
	for opin in pew_dict[date].iterkeys():
		for count in range(int(100*pew_dict[date][opin])):
			empirical[date].append(int(opin))
	C[date]=np.histogram(empirical[date],bins=np.arange(-10,11),density=True)[0]
	info[date]={'sim':0,'time':0,'expressed':None,'empirical':None}
for t in np.arange(0,P['t_sim'],P['t_measure']):
	expressed[t]=dataframe.query("time==%s"%t)['expressed']/5.0-10

#measure the similarity to the first date (1994) in the pew dataset, keeping times on the
#end for comparisons below
for t in np.arange(P['t_measure'],P['t_sim']-4*P['t_measure'],P['t_measure']):
	B=np.asfarray(np.histogram(expressed[t],bins=np.arange(-10,11))[0])
	B/=np.sum(np.histogram(expressed[t],bins=np.arange(-10,11))[0])
	if P['loss_metric']=='JSD':
		M_e = 0.5 * (B + C['1994'])
		jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
		sim=1.0-jsd_e
	if P['loss_metric']=='RMSE':
		sim=1-np.sqrt(np.average((B-C['1994'])**2))
	if sim > info['1994']['sim']:
		info['1994']={'sim':sim,'time':t,'expressed':expressed[t],'empirical':empirical['1994']}

#measure the similarity to the last date (2014), with 3 measures in between first and last
for t in np.arange(info['1994']['time']+4*P['t_measure'],P['t_sim'],P['t_measure']):
	B=np.asfarray(np.histogram(expressed[t],bins=np.arange(-10,11))[0])
	B/=np.sum(np.histogram(expressed[t],bins=np.arange(-10,11))[0])
	if P['loss_metric']=='JSD':
		M_e = 0.5 * (B + C['2014'])
		jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
		sim=1.0-jsd_e
	if P['loss_metric']=='RMSE':
		sim=1-np.sqrt(np.average((B-C['2014'])**2))
	if sim > info['2014']['sim']:
		info['2014']={'sim':sim,'time':t,'expressed':expressed[t],'empirical':empirical['2014']}

final_sim=np.average([info['1994']['sim'],info['2014']['sim']])

if final_sim>P['sim_threshold']:
	print 'final_sim=%s' %final_sim
	dataframe.to_pickle('data_sim=%s.pkl' %final_sim)
	param_df=pd.DataFrame([P])
	param_df.reset_index().to_json('parameters_sim=%s.json'%final_sim,orient='records')
	t_plot['1994']=info['1994']['time']
	t_plot['2014']=info['2014']['time']
	delta=(t_plot['2014']-t_plot['1994'])/(4*P['t_measure'])
	t_plot['1999']=t_plot['1994']+P['t_measure']*np.floor(delta)
	t_plot['2004']=t_plot['1994']+P['t_measure']*np.floor(2*delta)
	t_plot['2011']=t_plot['1994']+P['t_measure']*np.floor(3*delta)
	info['1999']={'sim':None,'time':t_plot['1999'],'expressed':expressed[t_plot['1999']],'empirical':empirical['1999']}
	info['2004']={'sim':None,'time':t_plot['2004'],'expressed':expressed[t_plot['2004']],'empirical':empirical['2004']}
	info['2011']={'sim':None,'time':t_plot['2011'],'expressed':expressed[t_plot['2011']],'empirical':empirical['2011']}
	for date in t_plot.iterkeys():
		sns.set(context='poster')
		figure1, ax1 = plt.subplots(1, 1)
		sns.distplot(info[date]['empirical'],bins=np.arange(-10,11),
						norm_hist=True,kde=True,ax=ax1,label='empirical')
		sns.distplot(info[date]['expressed'],bins=np.arange(-10,11),
						norm_hist=True,kde=True,ax=ax1,label='model'),
		plt.legend()
		ax1.set(title='%s'%date, xlim=(-10,10),ylim=(0,0.1))
		figure1.savefig('new_sim=%s_date=%s_t=%s.png' %(final_sim,date,t_plot[date]))
		plt.close(figure1)	