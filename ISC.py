# Peter Duggins
# July-August 2015, Updated August 2016
# Influence, Susceptibility, and Conformity Model

def import_params(filename):
	the_params=eval(open(filename).read())
	return the_params

def init_dataframe(P,agentdict):
	import pandas as pd
	import numpy as np
	columns=('time','agent','opinion','expressed',) 
	dataframe = pd.DataFrame(columns=columns,
							index=np.arange(0,P['t_sim']/P['t_measure']*P['popsize']))
	for i in agentdict.iterkeys():
		time=0
		iden=agentdict[i].iden
		o=agentdict[i].O
		e=agentdict[i].E
		dataframe.loc[i]=[time,iden,o,e]
	return dataframe
	
def update_dataframe(time,t_measure,agentdict,dataframe):
	import pandas as pd
	start=int(time/t_measure)*len(agentdict)
	for i in agentdict.iterkeys():
		iden=agentdict[i].iden
		o=agentdict[i].O
		e=agentdict[i].E
		dataframe.loc[start+i]=[time,iden,o,e]
	return dataframe

# def init_JSD(P):
# 	import pandas as pd
# 	import numpy as np
# 	columns=('time','agent','opinion','expressed',) 
# 	jsd_dataframe = pd.DataFrame(columns=columns,index=np.arange(0,P['t_sim']/P['t_measure']))
# 	return jsd_dataframe
	
# def update_JSD(time,t_measure,dataframe):
# 	import pandas as pd
# 	import numpy as np
#	import scipy
# 	start=int(time/t_measure)

# 	histP=np.histogram(P,bins=10,density=True)[0]
# 	histQ=np.histogram(Q,bins=10,density=True)[0]
# 	M = 0.5 * (histP + histQ)
# 	return 0.5 * (stats.entropy(histP, M) + stats.entropy(histQ, M))
# 	return dataframe

def id_generator(size=6):
	import string
	import random
	return ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(size))

def create_agents(P,rng):
	from agent import agent
	agentdict={}
	for i in range(P['popsize']):
		a=rng.uniform(0,P['gridsize'])
		b=rng.uniform(0,P['gridsize'])
		if int(P['std_init_opinion']) != 0:
			c=rng.normal(P['mean_init_opinion'],P['std_init_opinion'])
			if c<0: c=0
			if c>100: c=100
		else: c=P['mean_init_opinion']
		if int(P['std_intolerance']) != 0:
			d=rng.normal(P['mean_intolerance'],P['std_intolerance'])
			if d<0: d=0
		else: d=P['mean_intolerance']
		if int(P['std_intolerance']) != 0:
			e=rng.normal(P['mean_susceptibility'],P['std_susceptibility'])
			if e<0: e=0
		else: e=P['mean_susceptibility']
		if int(P['std_conformity'] )!= 0:
			f=rng.normal(P['mean_conformity'],P['std_conformity'])
			if f<0: f=0
			#if f<0: f=0 #negative implies anticonformity / distinctiveness
			#if f>1: f=1 #over 1 implies overshooting the group norm in the effort to conform
		else: f=P['mean_conformity']
		if int(P['std_social_reach']) != 0:
			g=rng.normal(P['mean_social_reach'],P['std_social_reach'])
			if g<0: g=0
		else: g=P['mean_social_reach']
		agentdict[i]=agent(i,a,b,c,d,e,f,g)
	return agentdict	

def network_agents(agentdict):
	for i in agentdict.itervalues():
		for j in agentdict.itervalues():
			if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < min(i.radius,j.radius):
				i.addtonetwork(j)

def plot_opinion_trajectory(dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context=P['plot_context'])
	figure1, ax1 = plt.subplots(1, 1)
	figure2, ax2 = plt.subplots(1, 1)
	sns.tsplot(time="time",value="opinion",data=dataframe,unit="agent",ax=ax1, err_style="unit_traces")
	sns.tsplot(time="time",value="expressed",data=dataframe,unit="agent",ax=ax2, err_style="unit_traces")
	# for a in range(P['popsize']):
	# 	sns.tsplot(time="time",value="opinion",data=dataframe.query("agent==%s"%a).reset_index(),ax=ax1)
	# 	sns.tsplot(time="time",value="expressed",data=dataframe.query("agent==%s"%a).reset_index(),ax=ax2)
	ax1.set(ylim=(0,100))
	ax2.set(ylim=(0,100))
	figure1.savefig('opinion_trajectory.png')
	figure2.savefig('expressed_trajectory.png')

def plot_histograms(dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	sns.set(context=P['plot_context'])
	for t in P['t_plot']:
		opinions=dataframe.query("time==%s"%t)['opinion']
		expressed=dataframe.query("time==%s"%t)['expressed']
		figure1, ax1 = plt.subplots(1, 1)
		figure2, ax2 = plt.subplots(1, 1)
		sns.distplot(opinions,ax=ax1,label='t=%s' %t) #bins=P['bins'],
		sns.distplot(expressed,ax=ax2,label='t=%s' %t)
		ax1.set(xlim=(0,100))
		ax2.set(xlim=(0,100))
		figure1.savefig('opinion_histogram_t=%s.png' %t)
		figure2.savefig('expressed_histogram_t=%s.png' %t)
		plt.close(figure1)
		plt.close(figure2)

def plot_maps(agentdict,dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from matplotlib import colors
	import numpy as np
	import ipdb
	sns.set(context=P['plot_context'],style='white')
	for t in P['t_plot']:
		cm = plt.cm.get_cmap('seismic')
		df_t=dataframe.query("time==%s"%t).reset_index()
		opinions=np.array(df_t['opinion'])/100
		expressed=np.array(df_t['expressed'])/100
		agentorder=np.array(df_t['agent']).astype(int)
		X=[agentdict[i].x for i in agentorder]
		Y=[agentdict[i].y for i in agentorder]
		figure1, ax1 = plt.subplots(1, 1)
		figure2, ax2 = plt.subplots(1, 1)
		one=ax1.scatter(X,Y,P['gridsize']/3,c=opinions,vmin=0,vmax=1,cmap=cm)
		two=ax2.scatter(X,Y,P['gridsize']/3,c=expressed,vmin=0,vmax=1,cmap=cm)
		ax1.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		ax2.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		figure1.colorbar(one)
		figure2.colorbar(two)
		figure1.savefig('opinion_map_t=%s.png' %t)
		figure2.savefig('expressed_map_t=%s.png' %t)
		plt.close(figure1)
		plt.close(figure2)

def main():
	import pandas as pd
	import os
	import numpy as np
	import sys
	# import ipdb

	'''Import Parameters from File'''
	P=import_params('parameters.txt')
	rng=np.random.RandomState(seed=P['seed'])
	agentdict=create_agents(P,rng)
	network_agents(agentdict)
	dataframe=init_dataframe(P,agentdict)

	print 'Running Simulation...'
	for t in np.arange(1,P['t_sim']+1):
		sys.stdout.write("\r%d%%" %(100*t/P['t_sim']))
		sys.stdout.flush()
		order=np.array(agentdict.keys())
		rng.shuffle(order)
		for i in order: #randomize order of dialogue initiation
			agentdict[i].hold_dialogue(rng)
		if t % P['t_measure'] == 0:
			update_dataframe(t,P['t_measure'],agentdict,dataframe)
			# update_JSD(t,P['t_measure'],agentdict,dataframe)

	print '\nExporting Data...'
	root=os.getcwd()
	addon=str(id_generator(9))
	os.makedirs(root+'\\data\\'+addon)
	os.chdir(root+'\\data\\'+addon)
	dataframe.to_pickle('data.pkl')
	param_df=pd.DataFrame([P])
	param_df.reset_index().to_json('parameters.json',orient='records')

	print 'Plotting...'
	plot_context='poster'
	plot_opinion_trajectory(dataframe,P)
	plot_histograms(dataframe,P)
	plot_maps(agentdict,dataframe,P)
	os.chdir(root)

if __name__=='__main__':
	main()