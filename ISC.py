# Peter Duggins
# July-August 2015, Updated August 2016
# Influence, Susceptibility, and Conformity Model

'''Initialization ###########################################'''

def ch_dir():
	#change directory for data and plot outputs
	import os
	import sys
	import string
	import random
	root=os.getcwd()
	addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(9)))
	datadir=''
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		datadir=root+'/data/'+addon #linux or mac
	elif sys.platform == "win32":
		datadir=root+'\\data\\'+addon #windows
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def import_params(filename):
	the_params=eval(open(filename).read())
	return the_params

def init_dataframe(P,agentdict):
	#The internal data file where information about agents' opinions over time is stored
	#Requires the package pandas
	import pandas as pd
	import numpy as np
	columns=('time','agent','opinion','expressed') 
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
	#update the dataframe every t_measure timesteps by adding a new row for every agent
	import pandas as pd
	start=int(time/t_measure)*len(agentdict)
	for i in agentdict.iterkeys():
		iden=agentdict[i].iden
		o=agentdict[i].O
		e=agentdict[i].E
		dataframe.loc[start+i]=[time,iden,o,e] #add the new row onto the end, at index 'start+i'
	return dataframe

def init_JSD(P):
	#The internal data file where information about the Jensen-Shannon Divergence is stored
	#Requires Pandas
	import pandas as pd
	import numpy as np
	columns=('time','JSD') 
	jsd_dataframe = pd.DataFrame(columns=columns)
	jsd_dataframe.loc[0]=[0,0.0]
	return jsd_dataframe
	
def update_JSD(time,params,dataframe,jsd_dataframe):
	#update the dataframe every t_measure timesteps by adding one new row
	import pandas as pd
	import numpy as np
	from scipy import stats
	df_t=dataframe.query("time==%s"%time).reset_index()
	opinions=np.array(df_t['opinion'])
	expressed=np.array(df_t['expressed'])
	A=np.histogram(opinions,density=True)[0]
	B=np.histogram(expressed,density=True)[0]
	M = 0.5 * (A + B)
	start=int(time/params['t_measure'])
	jsd=0.5*(stats.entropy(A,M)+stats.entropy(B,M))
	jsd_dataframe.loc[start]=[time,jsd] #add the new row onto the end, at index 'start'
	return jsd_dataframe

def create_agents(P,rng):
	#initialize agents' internal parameters: initial opinion, intolerance, susceptibility,
	#conformity, social reach. Truncate parameters below zero (or over 100 for O_i)
	from agent import agent
	agentdict={}
	for i in range(P['popsize']):
		x=rng.uniform(0,P['gridsize'])
		y=rng.uniform(0,P['gridsize'])
		if P['std_init_opinion'] != 0: #opinion
			o_i=rng.normal(P['mean_init_opinion'],P['std_init_opinion'])
			if o_i<0: o_i=0
			if o_i>100: o_i=100
		else: o_i=P['mean_init_opinion']
		if P['std_intolerance'] != 0: #intolerance
			t_i=rng.normal(P['mean_intolerance'],P['std_intolerance'])
			if t_i<0: t_i=0
		else: t_i=P['mean_intolerance']
		if P['std_susceptibility'] != 0: #susceptibility
			s_i=rng.normal(P['mean_susceptibility'],P['std_susceptibility'])
			if s_i<0: s_i=0
		else: s_i=P['mean_susceptibility']
		if P['std_conformity'] != 0: #conformity
			s_i=rng.normal(P['mean_conformity'],P['std_conformity'])
			#if s_i<0: s_i=0 #negative implies anticonformity / distinctiveness
			#if s_i>1: s_i=1 #over 1 implies overshooting the group norm in the effort to conform
		else: s_i=P['mean_conformity']
		if P['std_social_reach'] != 0: #social reach
			r_i=rng.normal(P['mean_social_reach'],P['std_social_reach'])
			if r_i<0: r_i=0
		else: r_i=P['mean_social_reach']
		#create an agent with these parameters and add it to the agent dictionary
		agentdict[i]=agent(i,x,y,o_i,t_i,s_i,s_i,r_i)
	return agentdict	

def network_agents(agentdict):
	#create social networks: if euclidian distance sqrt(dx^2+dy^2)<r_i, add to network
	for i in agentdict.itervalues():
		for j in agentdict.itervalues():
			if i != j and ((j.x - i.x)**2 + (j.y - i.y)**2)**(0.5) < min(i.radius,j.radius):
				i.addtonetwork(j)


'''Plotting ###########################################'''

#Plotting funcitons require the packages Matplotlib and Seaborn
def plot_opinion_trajectory(dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context=P['plot_context'])
	figure1, ax1 = plt.subplots(1, 1)
	figure2, ax2 = plt.subplots(1, 1)
	sns.tsplot(time="time",value="opinion",data=dataframe,unit="agent",ax=ax1,err_style="unit_traces")
	sns.tsplot(time="time",value="expressed",data=dataframe,unit="agent",ax=ax2,err_style="unit_traces")
	ax1.set(ylim=(0,100))
	ax2.set(ylim=(0,100))
	figure1.savefig('opinion_trajectory.png')
	figure2.savefig('expressed_trajectory.png')

def plot_histograms(dataframe,P,datadir):
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	import os
	import sys
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		opiniondir=datadir+'/opinion_histograms/'
		expresseddir=datadir+'/expressed_histograms/'
	elif sys.platform == "win32":
		opiniondir=datadir+'\\opinion_histograms\\'
		expresseddir=datadir+'\\expressed_histograms\\'
	sns.set(context=P['plot_context'])
	os.makedirs(opiniondir)
	os.chdir(opiniondir)
	for t in P['t_plot']:
		opinions=dataframe.query("time==%s"%t)['opinion']
		figure1, ax1 = plt.subplots(1, 1)
		sns.distplot(opinions,kde=True,ax=ax1,label='t=%s' %t)
		ax1.set(xlim=(0,100))
		figure1.savefig('opinion_histogram_t=%s.png' %t)
		plt.close(figure1)
	os.makedirs(expresseddir)
	os.chdir(expresseddir)
	sns.set(context=P['plot_context'])
	for t in P['t_plot']:
		expressed=dataframe.query("time==%s"%t)['expressed']
		figure2, ax2 = plt.subplots(1, 1)
		sns.distplot(expressed,kde=True,ax=ax2,label='t=%s' %t)
		ax2.set(xlim=(0,100))
		figure2.savefig('expressed_histogram_t=%s.png' %t)
		plt.close(figure2)
	os.chdir(datadir)

def plot_maps(agentdict,dataframe,P,datadir):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from matplotlib import colors
	import numpy as np
	import os
	import sys
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		opiniondir=datadir+'/opinion_maps/'
		expresseddir=datadir+'/expressed_maps/'
	elif sys.platform == "win32":
		opiniondir=datadir+'\\opinion_maps\\'
		expresseddir=datadir+'\\expressed_maps\\'
	sns.set(context=P['plot_context'],style='white')
	cm = plt.cm.get_cmap('seismic') #color map for matplotlib, defines color bar
	os.makedirs(opiniondir)
	os.chdir(opiniondir)
	for t in P['t_plot']:
		df_t=dataframe.query("time==%s"%t).reset_index()
		opinions=np.array(df_t['opinion'])/100
		agentorder=np.array(df_t['agent']).astype(int)
		X=[agentdict[i].x for i in agentorder]
		Y=[agentdict[i].y for i in agentorder]
		figure1, ax1 = plt.subplots(1, 1)
		figure2, ax2 = plt.subplots(1, 1)
		one=ax1.scatter(X,Y,P['gridsize']/3,c=opinions,vmin=0,vmax=1,cmap=cm)
		ax1.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		figure1.colorbar(one)
		figure1.savefig('opinion_map_t=%s.png' %t)
		plt.close(figure1)
	os.makedirs(expresseddir)
	os.chdir(expresseddir)
	for t in P['t_plot']:
		df_t=dataframe.query("time==%s"%t).reset_index()
		expressed=np.array(df_t['expressed'])/100
		agentorder=np.array(df_t['agent']).astype(int)
		X=[agentdict[i].x for i in agentorder]
		Y=[agentdict[i].y for i in agentorder]
		figure2, ax2 = plt.subplots(1, 1)
		two=ax2.scatter(X,Y,P['gridsize']/3,c=expressed,vmin=0,vmax=1,cmap=cm)
		ax2.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		figure2.colorbar(two)
		figure2.savefig('expressed_map_t=%s.png' %t)
		plt.close(figure2)
	os.chdir(datadir)	

def plot_JSD(dataframe,jsd_dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context=P['plot_context'])
	figure1, ax1 = plt.subplots(1, 1)
	# sns.tsplot(time="time",value="JSD",data=jsd_dataframe,ax=ax1) #inexplicably broken
	ax1.plot(jsd_dataframe['time'],jsd_dataframe['JSD']) #so use normal matplotlib command here
	ax1.set(ylim=(0,1),xlabel='time',ylabel='JSD')
	figure1.savefig('JSD.png')



'''Main ###########################################'''

def main():
	import pandas as pd
	import os
	import sys
	import numpy as np
	import sys

	'''Importing Parameters, Initialize Agents and Dataframes, Change Directories'''
	P=import_params('parameters.txt')
	rng=np.random.RandomState(seed=P['seed']) #set the simulation seed
	agentdict=create_agents(P,rng)
	network_agents(agentdict)
	dataframe=init_dataframe(P,agentdict)
	jsd_dataframe=init_JSD(P)
	datadir=ch_dir()

	print 'Running Simulation...'
	for t in np.arange(1,P['t_sim']+1):
		sys.stdout.write("\r%d%%" %(100*t/P['t_sim'])) #a simulation progress counter, can be omitted
		sys.stdout.flush()
		order=np.array(agentdict.keys())
		rng.shuffle(order) #randomize order of dialogue initiation
		for i in order: 
			agentdict[i].hold_dialogue(rng)
		if t % P['t_measure'] == 0: #update dataframes every t_measure timesteps
			update_dataframe(t,P['t_measure'],agentdict,dataframe)
			update_JSD(t,P,dataframe,jsd_dataframe)

	print '\nExporting Data...'
	dataframe.to_pickle('data.pkl') #export in the pickle data format
	jsd_dataframe.to_pickle('jsd.pkl')
	param_df=pd.DataFrame([P])
	param_df.reset_index().to_json('parameters.json',orient='records')

	print 'Plotting...'
	plot_context='poster'
	plot_JSD(dataframe,jsd_dataframe,P)
	plot_opinion_trajectory(dataframe,P)
	plot_histograms(dataframe,P,datadir)
	plot_maps(agentdict,dataframe,P,datadir)
	os.chdir(os.getcwd())

if __name__=='__main__':
	main()