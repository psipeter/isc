# Peter Duggins
# July-August 2015, Updated August 2016
# Generates data from the ISC model and compares it to empirical data on American Politics

def init_evo_pop(popsize,evo_rng,sim_threshold,issue):
	import numpy as np
	evo_pop={}
	# seed=evo_rng.randint(1,1000000000)
	pop_seeds=evo_rng.randint(1,1000000000,size=popsize)
	for i in range(popsize):
		# rng=evo_rng
		# rng=np.random.RandomState(seed=seed)
		rng=np.random.RandomState(seed=pop_seeds[i])
		P={
			'seed':pop_seeds[i],
			'rng':rng,
			'gridsize':316,
			'popsize':1000,
			't_sim':1000,
			't_measure':50,
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
			'sim_threshold':sim_threshold,
			'issue':issue,
		}
		evo_pop[i]={'P':P,'F':0.0} #P=parameters, F=fitness
	return evo_pop

def tournament_selection(fitness_list,size_tournaments):
	from random import shuffle
	def pullFitness(ind): return ind[1]
	fittest_pop=[]
	for n in range(len(fitness_list)):
		tournament=[]
		shuffle(fitness_list) #shuffle population to randomize tournament players
		for i in range(size_tournaments):
			tournament.append(fitness_list[i])
		winner=sorted(tournament,key=pullFitness,reverse=True)[0]
		fittest_pop.append(winner) #each tournament places one winner into the new population
	return fittest_pop

def rank_proportional_selection(fitness_list,evo_rng):
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
	randoms=[evo_rng.uniform(0,rank_array[-1]) for _ in range(len(rank_array))]
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

def mutate(remade_evo_pop,old_evo_pop,mutation_rate,mutation_amount,evo_rng):
	import numpy as np
	mutated_pop=old_evo_pop
	i=0
	for ind in remade_evo_pop.iterkeys():
		for param in remade_evo_pop[ind]['P'].iterkeys():
			if param == 'mean_intolerance' or param == 'mean_susceptibility' \
			or param == 'mean_conformity' or param == 'std_intolerance' \
			or param =='std_susceptibility' or param =='std_conformity':
				if evo_rng.uniform(0,1)<mutation_rate:
					change=mutation_amount*(1-2*evo_rng.randint(0,1))
					mutated_pop[i]['P'][param]=remade_evo_pop[ind]['P'][param]+change
				else:
					mutated_pop[i]['P'][param]=remade_evo_pop[ind]['P'][param]
		mutated_pop[i]['F']=remade_evo_pop[ind]['F']
		i+=1
	return mutated_pop

def crossover(evo_pop,evo_rng): #uniform crossover method
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
				if evo_rng.uniform(0,1)<0.5:
					child1['P'][param]=parent1['P'][param]
					child2['P'][param]=parent2['P'][param]
				else:
					child1['P'][param]=parent2['P'][param]
					child2['P'][param]=parent1['P'][param]
		crossed_pop[i]=child1
		crossed_pop[len(keys)-(i+1)]=child2
	return crossed_pop




def run(P):
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

	# print 'Realization %s/%s' %(a+1,P['averages'])
	# print '''Initializing Simulation'''
	rng=P['rng'] #global rng
	# rng=np.random.RandomState(seed=P['seed']) #the rng is the same every generation
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
	for t in np.arange(P['t_measure'],P['t_sim'],P['t_measure']):
		expressed=dataframe.query("time==%s"%t)['expressed']*6.0/100+1
		B=np.asfarray(np.histogram(expressed,bins=7)[0])
		B/=np.sum(np.histogram(expressed,bins=7)[0])
		M_e = 0.5 * (B + C)
		jsd_e=0.5*(stats.entropy(B,M_e)+stats.entropy(B,M_e))
		sim=1.0-jsd_e
		if sim > max_similarity['sim']:
			max_similarity={'sim':sim,'time':t,'expressed':expressed,'empirical':empirical}

	if max_similarity['sim']>P['sim_threshold']:
		print 'sim=%s, t=%s' %(max_similarity['sim'],max_similarity['time'])
		root=os.getcwd()
		addon='emp_sim_%s' %max_similarity['sim'] + str(id_generator(9))
		os.makedirs(root+'/data/'+addon) #linux
		os.chdir(root+'/data/'+addon) 
		# os.makedirs(root+'\\data\\'+addon) #pc
		# os.chdir(root+'\\data\\'+addon)
		dataframe.to_pickle('data.pkl')
		param_df=pd.DataFrame([P])
		param_df.reset_index().to_json('parameters.json',orient='records')
		sns.set(context=P['plot_context'])
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

	return [P,max_similarity['sim']]

def export_pop(evo_pop):
	from ISC import id_generator
	import os
	import pandas as pd

	root=os.getcwd()
	addon='evo_pop_'+str(id_generator(9))
	os.makedirs(root+'/data/'+addon) #linux
	os.chdir(root+'/data/'+addon) 
	# os.makedirs(root+'\\data\\'+addon) #pc
	# os.chdir(root+'\\data\\'+addon)
	out_pop=pd.DataFrame([evo_pop])
	out_pop.reset_index().to_json('evo_pop.json',orient='records')

def main():
	from pathos.multiprocessing import ProcessingPool as Pool
	from pathos.helpers import freeze_support #for Windows
	freeze_support()
	import numpy as np


	issue='abortion'
	evo_seed=3
	generations=10
	popsize=100 #even
	threads=10
	size_tournaments=3
	mutation_rate=0.2
	mutation_amount=0.1
	sim_threshold=0.99

	evo_rng=np.random.RandomState(seed=evo_seed)
	evo_pop=init_evo_pop(popsize,evo_rng,sim_threshold,issue)
	pool = Pool(nodes=threads)

	for g in range(generations):
		exp_params=[value['P'] for value in evo_pop.itervalues()]
		fitness_list=pool.map(run, exp_params)
		new_gen_list=tournament_selection(fitness_list,size_tournaments)
		# new_gen_list=rank_proportional_selection(fitness_list,evo_rng)
		remade_pop=remake(evo_pop,new_gen_list)
		mutated_pop=mutate(remade_pop,evo_pop,mutation_rate,mutation_amount,evo_rng)
		crossed_pop=crossover(mutated_pop,evo_rng)
		evo_pop=crossed_pop
		print '\ngen %s mean F = %s' \
			%(g,np.average([evo_pop[ind]['F'] for ind in evo_pop.iterkeys()]))

	export_pop(evo_pop)

if __name__=='__main__':
	main()
