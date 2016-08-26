# Peter Duggins
# July-August 2015, Updated August 2016
# Generates data from the ISC model and compares it to empirical data on American Politics

'''initialize-----------------------------------------------------------------------'''
def init_directory(fit_params):
	import os
	import string
	import random
	root=os.getcwd()
	addon='emp_'+fit_params['issue']+'_'+fit_params['optimization']+'_'+\
		str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(6)))
	os.makedirs(root+'/data/'+addon) #linux
	os.chdir(root+'/data/'+addon) 
	# os.makedirs(root+'\\data\\'+addon) #pc
	# os.chdir(root+'\\data\\'+addon)

'''evolutionary methods-----------------------------------------------------------------------'''
def init_evo_pop(fit_params):
	import numpy as np
	evo_pop={}
	pop_seeds=np.random.randint(1,1000000000,size=fit_params['evo_popsize'])
	for i in range(fit_params['evo_popsize']):
		rng=np.random.RandomState(seed=pop_seeds[i])
		P={
			'seed':pop_seeds[i],
			'mean_intolerance':np.round(rng.uniform(0.7,1.0),decimals=1),
			'mean_susceptibility':np.round(rng.uniform(1,10),decimals=0),
			'mean_conformity':np.round(rng.uniform(0,1),decimals=1),
			'std_intolerance':np.round(rng.uniform(0,0.5),decimals=1),
			'std_susceptibility':np.round(rng.uniform(0,0.5),decimals=1),
			'std_conformity':np.round(rng.uniform(0,0.5),decimals=1),
			'gridsize':fit_params['gridsize'],
			'popsize':fit_params['popsize'],
			't_sim':fit_params['t_sim'],
			't_measure':fit_params['t_measure'],
			'mean_init_opinion':fit_params['mean_init_opinion'],
			'std_init_opinion':fit_params['std_init_opinion'],
			'mean_social_reach':fit_params['mean_social_reach'],
			'std_social_reach':fit_params['std_social_reach'],
			'sim_threshold':fit_params['sim_threshold'],
			'issue':fit_params['issue'],
			'optimization':fit_params['optimization'],
			'loss_metric':fit_params['loss_metric'],
		}
		evo_pop[i]={'P':P,'F':0.0} #P=parameters, F=fitness
	return evo_pop

def tournament_selection(fitness_list,fit_params):
	from random import shuffle
	def pullFitness(ind): return ind[1]
	fittest_pop=[]
	for n in range(len(fitness_list)):
		tournament=[]
		shuffle(fitness_list) #shuffle population to randomize tournament players
		for i in range(fit_params['size_tournaments']):
			tournament.append(fitness_list[i])
		winner=sorted(tournament,key=pullFitness,reverse=True)[0]
		fittest_pop.append(winner) #each tournament places one winner into the new population
	return fittest_pop

def rank_proportional_selection(fitness_list):
	import numpy as np
	def pullFitness(ind): return ind[1]
	fitness_list=sorted(fitness_list,key=pullFitness,reverse=False) #sort lowest to highest
	fitnesses=[ind[1] for ind in fitness_list] #lowest to highest
	rank_array=[]
	rank_sum=0
	#turns rank=[1,2,3,4], 4=highest fitness, into rank_array=[1,3,6,10]
	for i in range(len(fitness_list)): 
		rank_array.append((i+1)+rank_sum)
		rank_sum+=(i+1)
	#draw pop_size random integers between 0 and rank_array[last]
	randoms=[np.random.uniform(0,rank_array[-1]) for _ in range(len(rank_array))]
	fittest_pop=[]
	for n in randoms:#for each random number,
		index=-1 #catches the case when n cannot be greater than rank_array[last]
		for i in range(len(rank_array)):#find the next largest number in rank_array
			if rank_array[i]-n>=0: 
				index=i
				break
		#add the evo_individual with that fitness to the new population
		fittest_pop.append(fitness_list[index])
	return fittest_pop

def remake(evo_pop,new_gen_list):
	remade_evo_pop={}
	for i in range(len(evo_pop)):
		remade_evo_pop[i]={'P':new_gen_list[i][0], 'F':new_gen_list[i][1]}
	return remade_evo_pop

def mutate(remade_evo_pop,old_evo_pop,fit_params):
	import numpy as np
	mutated_pop=old_evo_pop
	i=0
	for ind in remade_evo_pop.iterkeys():
		for param in remade_evo_pop[ind]['P'].iterkeys():
			if param == 'mean_intolerance' or param == 'mean_susceptibility' \
			or param == 'mean_conformity' or param == 'std_intolerance' \
			or param =='std_susceptibility' or param =='std_conformity':
				if np.random.uniform(0,1)<fit_params['mutation_rate']:
					change=fit_params['mutation_amount']*(1-2*np.random.randint(0,1))
					mutated_pop[i]['P'][param]=remade_evo_pop[ind]['P'][param]+change
				else:
					mutated_pop[i]['P'][param]=remade_evo_pop[ind]['P'][param]
		mutated_pop[i]['F']=remade_evo_pop[ind]['F']
		i+=1
	return mutated_pop

def crossover(evo_pop): #uniform crossover method
	import numpy as np
	crossed_pop={}
	keys=[key for key in evo_pop.iterkeys()]
	for i in range(len(keys)/2):
		parent1=evo_pop[keys[i]] #front of the list forward
		parent2=evo_pop[keys[-i]] #back of the list backward
		child1=parent1
		child2=parent2
		for param in parent1['P'].iterkeys():
			if param == 'mean_intolerance' or param == 'mean_susceptibility' \
			or param == 'mean_conformity' or param == 'std_intolerance' \
			or param =='std_susceptibility' or param =='std_conformity':
				if np.random.uniform(0,1)<0.5:
					child1['P'][param]=parent1['P'][param]
					child2['P'][param]=parent2['P'][param]
				else:
					child1['P'][param]=parent2['P'][param]
					child2['P'][param]=parent1['P'][param]
		crossed_pop[i]=child1
		crossed_pop[len(keys)-(i+1)]=child2
	return crossed_pop

'''hyperopt methods-----------------------------------------------------------------------'''
def search_space(fit_params):
	from hyperopt import hp
	import numpy as np
	space={
		'seed': np.random.randint(1,1000000000),
		'mean_intolerance': hp.quniform('mean_intolerance',0.7,1.0,0.1),
		'mean_susceptibility':hp.quniform('mean_susceptibility',1,10,1),
		'mean_conformity': hp.quniform('mean_conformity',0,1,0.1),
		'std_intolerance': hp.quniform('std_intolerance',0,0.5,0.1),
		'std_susceptibility': hp.quniform('std_susceptibility',0,0.5,0.1),
		'std_conformity': hp.quniform('std_conformity',0,0.5,0.1),
		'gridsize':fit_params['gridsize'],
		'popsize':fit_params['popsize'],
		't_sim':fit_params['t_sim'],
		't_measure':fit_params['t_measure'],
		'mean_init_opinion':fit_params['mean_init_opinion'],
		'std_init_opinion':fit_params['std_init_opinion'],
		'mean_social_reach':fit_params['mean_social_reach'],
		'std_social_reach':fit_params['std_social_reach'],
		'sim_threshold':fit_params['sim_threshold'],
		'issue':fit_params['issue'],
		'optimization':fit_params['optimization'],
		'loss_metric':fit_params['loss_metric'],
	}
	return space

def plot_results(trials):
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials]
	Y=[1.0-t['result']['loss'] for t in trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='similarity')
	plt.show()

'''shared methods-----------------------------------------------------------------------'''
def calculate_similarity(P,dataframe):
	from scipy import stats
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	import numpy as np
	import os

	brookman_dir='/home/pduggins/influence_susceptibility_conformity/brookman_data.txt'
	brookman_dict=eval(open(brookman_dir).read())
	info={'sim':0,'time':0,'expressed':None,'empirical':None}
	empirical=[]
	for opin in brookman_dict[P['issue']].iterkeys():
		for count in range(brookman_dict[P['issue']][opin]):
			empirical.append(int(opin))
	C=np.histogram(empirical,bins=7,density=True)[0]

	for t in np.arange(P['t_measure'],P['t_sim'],P['t_measure']):
		expressed=dataframe.query("time==%s"%t)['expressed']*6.0/100+1
		B=np.asfarray(np.histogram(expressed,bins=7)[0])
		B/=np.sum(np.histogram(expressed,bins=7)[0])
		if P['loss_metric']=='JSD':
			M_e = 0.5 * (B + C)
			jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
			sim=1.0-jsd_e
		if P['loss_metric']=='RMSE':
			sim=1-np.sqrt(np.average((B-C)**2))
		if sim > info['sim']: info={'sim':sim,'time':t,'expressed':expressed,'empirical':empirical}

	if info['sim']>P['sim_threshold']:
		print 'sim=%s, t=%s' %(info['sim'],info['time'])
		dataframe.to_pickle('data_sim=%s.pkl' %info['sim'])
		param_df=pd.DataFrame([P])
		param_df.reset_index().to_json('parameters_sim=%s.json'%info['sim'],orient='records')
		sns.set(context='poster')
		figure1, ax1 = plt.subplots(1, 1)
		sns.distplot(info['empirical'],bins=range(1,8,1),
						norm_hist=True,kde=False,ax=ax1,label='empirical')
		sns.distplot(info['expressed'],bins=range(1,8,1),
						norm_hist=True,kde=False,ax=ax1,label='model'),
		plt.legend()
		ax1.set(xlim=(1,7))
		figure1.savefig('sim=%s_t=%s.png' %(info['sim'],info['time']))
		plt.close(figure1)

	return 1.0-info['sim']

'''main-----------------------------------------------------------------------'''
def run(P):
	from ISC import create_agents,network_agents,init_dataframe,update_dataframe
	from hyperopt import STATUS_OK
	import numpy as np
	import sys

	if P['optimization']=='hyperopt':
		rng=np.random.RandomState(seed=np.random.randint(1,1000000000))
	if P['optimization']=='evolve':
		rng=np.random.RandomState(seed=P['seed'])
	agentdict=create_agents(P,rng)
	network_agents(agentdict)
	dataframe=init_dataframe(P,agentdict)

	for t in np.arange(1,P['t_sim']+1):
		sys.stdout.write("\r%d%%" %(100*t/P['t_sim']))
		sys.stdout.flush()
		order=np.array(agentdict.keys())
		rng.shuffle(order)
		for i in order: agentdict[i].hold_dialogue(rng)
		if t % P['t_measure'] == 0: update_dataframe(t,P['t_measure'],agentdict,dataframe)

	loss=calculate_similarity(P,dataframe)
	if P['optimization']=='hyperopt':
		return {'loss': loss, 'status': STATUS_OK}
	if P['optimization']=='evolve':
		return [P,1-loss]

def main():
	from hyperopt import fmin,tpe,hp,Trials
	from hyperopt.mongoexp import MongoTrials
	from ISC import id_generator
	import os 

	fit_params=eval(open('fitting_parameters.txt').read())
	init_directory(fit_params)

	if fit_params['optimization']=='hyperopt':
		space=search_space(fit_params)
		trials=Trials()
		# trials=MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
		best=fmin(run,space=space,algo=tpe.suggest,max_evals=fit_params['max_evals'],trials=trials)
		plot_results(trials.trials)

	if fit_params['optimization']=='evolve':
		from pathos.multiprocessing import ProcessingPool as Pool
		from pathos.helpers import freeze_support #for Windows
		import numpy as np
		import pandas as pd
		freeze_support()
		evo_pop=init_evo_pop(fit_params)
		pool = Pool(nodes=fit_params['threads'])

		for g in range(fit_params['generations']):
			exp_params=[value['P'] for value in evo_pop.itervalues()]
			fitness_list=pool.map(run, exp_params)
			new_gen_list=tournament_selection(fitness_list,fit_params)
			# new_gen_list=rank_proportional_selection(fitness_list,evo_rng)
			remade_pop=remake(evo_pop,new_gen_list)
			mutated_pop=mutate(remade_pop,evo_pop,fit_params)
			crossed_pop=crossover(mutated_pop)
			evo_pop=crossed_pop
			print '\ngen %s mean F = %s' \
				%(g,np.average([evo_pop[ind]['F'] for ind in evo_pop.iterkeys()]))

		out_pop=pd.DataFrame([evo_pop])
		out_pop.reset_index().to_json('evo_pop.json',orient='records')

if __name__=='__main__':
	main()
