# Peter Duggins
# July-August 2015, Updated August 2016
# Generates data from the ISC model and compares it to empirical data on American Politics

def search_space(seed_rng):
	from hyperopt import hp
	space={
		'seed': seed_rng.randint(1,1000000000),
		'mean_intolerance': hp.quniform('mean_intolerance',0.7,1.0,0.1),
		'mean_susceptibility':hp.quniform('mean_susceptibility',1,10,1),
		'mean_conformity': hp.quniform('mean_conformity',0,1,0.1),
		'std_intolerance': hp.quniform('std_intolerance',0,0.5,0.1),
		'std_susceptibility': hp.quniform('std_susceptibility',0,0.5,0.1),
		'std_conformity': hp.quniform('std_conformity',0,0.5,0.1),
		'gridsize':316,
		'popsize':1000,
		't_sim':1000,
		't_measure':50,
		'mean_init_opinion':50,
		'std_init_opinion':20,
		'mean_social_reach':22.0,
		'std_social_reach':4,
		'sim_threshold':0.99,
		'issue':'abortion',
	}
	return space

def objective(P):
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
	from hyperopt import STATUS_OK

	rng=np.random.RandomState(seed=P['seed']) #the rng is the same every generation
	agentdict=create_agents(P,rng)
	network_agents(agentdict)
	dataframe=init_dataframe(P,agentdict)

	# print 'Running Simulation...'
	for t in np.arange(1,P['t_sim']+1):
		sys.stdout.write("\r%d%%" %(100*t/P['t_sim']))
		sys.stdout.flush()
		order=np.array(agentdict.keys())
		rng.shuffle(order)
		for i in order: #randomize order of dialogue initiation
			agentdict[i].hold_dialogue(rng)
		if t % P['t_measure'] == 0:
			update_dataframe(t,P['t_measure'],agentdict,dataframe)

	# print '\nCalculating Similarity...'
	brookman_dict=eval(open('brookman_data.txt').read())
	max_similarity={'sim':0,'time':0,'expressed':None,'empirical':None}
	empirical=[]
	for opin in brookman_dict[P['issue']].iterkeys():
		for count in range(brookman_dict[P['issue']][opin]):
			empirical.append(int(opin))
	C=np.histogram(empirical,bins=7,density=True)[0]
	for t in np.arange(0,P['t_sim']/P['t_measure']):
		expressed=dataframe.query("time==%s"%(t*P['t_measure']))['expressed']*6.0/100+1
		B=np.histogram(expressed,bins=7,density=True)[0]
		M_e = 0.5 * (B + C)
		jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
		sim=1.0-jsd_e
		if sim > max_similarity['sim']:
			max_similarity={'sim':sim,'time':t*P['t_measure'],
								'expressed':expressed,'empirical':empirical}

	if max_similarity['sim']>P['sim_threshold']:
		print 'sim=%s, t=%s' %(max_similarity['sim'],max_similarity['time'])
		root=os.getcwd()
		addon='emp_'+str(id_generator(9))
		os.makedirs(root+'/data/'+addon) #linux
		os.chdir(root+'/data/'+addon) 
		# os.makedirs(root+'\\data\\'+addon) #pc
		# os.chdir(root+'\\data\\'+addon)
		dataframe.to_pickle('data.pkl')
		param_df=pd.DataFrame([P])
		param_df.reset_index().to_json('parameters.json',orient='records')
		sns.set(context='poster')
		figure1, ax1 = plt.subplots(1, 1)
		sns.distplot(max_similarity['empirical'],bins=range(1,8,1),
						norm_hist=True,kde=False,ax=ax1,label='empirical')
		sns.distplot(max_similarity['expressed'],bins=range(1,8,1),
						norm_hist=True,kde=False,ax=ax1,label='model'),
		plt.legend()
		ax1.set(xlim=(1,7))
		figure1.savefig('issue=%s_sim=%s_t=%s.png' \
			%(P['issue'],max_similarity['sim'],max_similarity['time']))
		plt.close(figure1)
		os.chdir(root)

	return {'loss': 1.0-max_similarity['sim'], 'status': STATUS_OK}

def main():
	# from pathos.multiprocessing import ProcessingPool as Pool
	# from pathos.helpers import freeze_support #for Windows
	# freeze_support()
	from hyperopt import fmin,tpe,hp,Trials
	from hyperopt.mongoexp import MongoTrials
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns

	master_seed=3
	seed_rng=np.random.RandomState(seed=master_seed)
	space=search_space(seed_rng)
	trials=Trials()
	# trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
	best=fmin(objective,space=space,algo=tpe.suggest,max_evals=4,trials=trials)

	# print()
	# for t in range(len(trials.trials)):
	# 	print 'values: %s' %trials.trials[t]['misc']['vals']
	# 	print 'loss: %s' %trials.trials[t]['result']['loss']

	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials.trials]
	Y=[t['result']['loss'] for t in trials.trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='loss')
	plt.show()

if __name__=='__main__':
	main()
