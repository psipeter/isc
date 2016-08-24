# Peter Duggins
# July-August 2015, Updated August 2016
# Generates data from the ISC model and compares it to empirical data on American Politics

def run(exp_params):
	from ISC import create_agents,network_agents,init_dataframe,update_dataframe,id_generator
	import numpy as np
	from scipy import stats
	import pandas as pd
	import os 
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	import numpy as np
	import sys

	'''Randomize Parameters'''
	rng=np.random.RandomState(seed=exp_params[0])
	P={
		'seed':rng.randint(1,9000),
		'gridsize':316,
		'popsize':1000,
		't_sim':1000,
		't_measure':10,
		'mean_init_opinion':50,
		'std_init_opinion':20,
		'mean_intolerance':np.round(rng.uniform(0.7,1.0),decimals=1),
		'mean_susceptibility':np.round(rng.uniform(1,10),decimals=0),
		'mean_conformity':np.round(rng.uniform(0,1),decimals=1),
		'std_intolerance':np.round(rng.uniform(0,0.5),decimals=1),
		'std_susceptibility':np.round(rng.uniform(0,0.5),decimals=1),
		'std_conformity':np.round(rng.uniform(0,0.5),decimals=1),
		'mean_social_reach':22.0,
		'std_social_reach':4,
		'plot_context':'poster',
	}

	print '''Initializing Simulation'''
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

	print '\nCalculating Similarity...'
	brookman_dict=eval(open('brookman_data.txt').read())
	pew_dict=eval(open('pew_data.txt').read())
	max_similarity={}
	for issue in brookman_dict.iterkeys():
		max_similarity[issue]={'sim':0,'time':0,'expressed':None,'empirical':None}
		empirical=[]
		for opin in brookman_dict[issue].iterkeys():
			for count in range(brookman_dict[issue][opin]):
				empirical.append(int(opin))
		C=np.histogram(empirical,bins=7,density=True)[0]
		for t in np.arange(0,P['t_sim']/P['t_measure']):
			expressed=dataframe.query("time==%s"%(t*P['t_measure']))['expressed']*6.0/100+1
			B=np.histogram(expressed,bins=7,density=True)[0]
			M_e = 0.5 * (B + C)
			jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
			if jsd_e > max_similarity[issue]['sim']:
				max_similarity[issue]={'sim':jsd_e,'time':t*P['t_measure'],
										'expressed':expressed,'empirical':empirical}

	root=os.getcwd()
	addon='emp_'+str(id_generator(9))
	os.makedirs(root+'/data/'+addon) #linux
	os.chdir(root+'/data/'+addon) 
	# os.makedirs(root+'\\data\\'+addon) #pc
	# os.chdir(root+'\\data\\'+addon)
	dataframe.to_pickle('data.pkl')
	param_df=pd.DataFrame([P])
	param_df.reset_index().to_json('parameters.json',orient='records')
	for key in max_similarity.iterkeys():
		if max_similarity[key]['sim']>exp_params[1]:
			print 'issue=%s, JSD=%s, t=%s' %(key,max_similarity[key]['sim'],max_similarity[key]['time'])
			sns.set(context=P['plot_context'])
			figure1, ax1 = plt.subplots(1, 1)
			sns.distplot(max_similarity[key]['empirical'],bins=range(1,8,1),
							norm_hist=True,kde=False,ax=ax1,label='empirical')
			sns.distplot(max_similarity[key]['expressed'],bins=range(1,8,1),
							norm_hist=True,kde=False,ax=ax1,label='model'),
			plt.legend()
			ax1.set(xlim=(1,7))
			figure1.savefig('issue=%s_JSD=%s_t=%s.png' %(key,max_similarity[key]['sim'],max_similarity[key]['time']))
			plt.close(figure1)

def main():
	from pathos.multiprocessing import ProcessingPool as Pool
	from pathos.helpers import freeze_support #for Windows
	import numpy as np

	trials=10
	sim_threshold=0.5
	freeze_support()
	pool = Pool(nodes=trials)
	exp_params=[[np.random.randint(1,9000),sim_threshold] for _ in range(trials)]
	pool.map(run, exp_params)

if __name__=='__main__':
	main()
