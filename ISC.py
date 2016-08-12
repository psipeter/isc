# Peter Duggins
# July-August 2015, Updated August 2016
# Influence, Susceptibility, and Conformity Model

def import_params(filename):
	the_params=eval(open(filename).read())
	return the_params

def init_dataframe(P,agentlist):
	import pandas as pd
	import numpy as np
	columns=('time','agent','opinion','expressed',) 
	dataframe = pd.DataFrame(columns=columns,
							index=np.arange(0,P['t_sim']/P['t_measure']*P['popsize']))
	for a in range(len(agentlist)):
		time=0
		o=agentlist[a].O
		e=agentlist[a].E
		dataframe.loc[a]=[time,a,o,e]
	return dataframe
	
def update_dataframe(time,t_measure,agentlist,dataframe):
	import pandas as pd
	i=int(time/t_measure)*len(agentlist)
	for a in range(len(agentlist)):
		o=agentlist[a].O
		e=agentlist[a].E
		dataframe.loc[i+a]=[time,a,o,e]
	return dataframe

def id_generator(size=6):
	import string
	import random
	return ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(size))

def create_agents(P,rng):
	from agent import agent
	agentlist=[]
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
		agentlist.append(agent(i,a,b,c,d,e,f,g))
	# print mean([len(ind.getnetwork()) for ind in agentlist])
	return agentlist	

def network_agents(agentlist):
	for i in agentlist:
		for j in agentlist:
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

def plot_maps(agentlist,dataframe,P):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from matplotlib import colors
	import numpy as np
	import ipdb
	colorvals=[]
	sns.set(context=P['plot_context'],style='white')
	for t in P['t_plot']:
		cm = plt.cm.get_cmap('RdYlBu')
		opinions=dataframe.query("time==%s"%t)['opinion']
		expressed=dataframe.query("time==%s"%t)['expressed']
		X=[i.x for i in agentlist]
		Y=[i.y for i in agentlist]
		figure1, ax1 = plt.subplots(1, 1)
		figure2, ax2 = plt.subplots(1, 1)
		o_colors=np.asfarray(opinions/100)
		e_colors=np.asfarray(expressed/100)
		one=ax1.scatter(X,Y,P['gridsize']/5,c=o_colors,vmin=0,vmax=1,cmap=cm)
		two=ax2.scatter(X,Y,P['gridsize']/5,c=e_colors,vmin=0,vmax=1,cmap=cm)
		ax1.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		ax2.set(xlim=(0,P['gridsize']),ylim=(0,P['gridsize']),xticklabels=[],yticklabels=[])
		figure1.colorbar(one)
		figure2.colorbar(two)
		figure1.savefig('opinion_map_t=%s.png' %t)
		figure2.savefig('expressed_map_t=%s.png' %t)
		plt.close(figure1)
		plt.close(figure2)

# def JSD(P, Q):
# 	histP=np.histogram(P,bins=10,density=True)[0]
# 	histQ=np.histogram(Q,bins=10,density=True)[0]
# 	M = 0.5 * (histP + histQ)
# 	return 0.5 * (stats.entropy(histP, M) + stats.entropy(histQ, M))

def main():
	import pandas as pd
	import os
	import numpy as np
	import sys
	# import ipdb

	'''Import Parameters from File'''
	P=import_params('parameters.txt')
	rng=np.random.RandomState(seed=P['seed'])
	agentlist=create_agents(P,rng)
	network_agents(agentlist)
	dataframe=init_dataframe(P,agentlist)

	print 'Running Simulation...'
	for t in np.arange(1,P['t_sim']+1):
		sys.stdout.write("\r%d%%" %(100*t/P['t_sim']))
		sys.stdout.flush()
		for i in agentlist: i.hold_dialogue()
		if t % P['t_measure'] == 0: update_dataframe(t,P['t_measure'],agentlist,dataframe)
		rng.shuffle(agentlist)

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
	plot_maps(agentlist,dataframe,P)
	os.chdir(root)

if __name__=='__main__':
	main()


# def plot_dynamic_maps(agentlist,o_history,e_history,circlesize,gridsize,t_sim,tmeasure,rootdir,dirname,filename):	
# 	os.makedirs(rootdir+dirname+filename)
# 	os.chdir(rootdir+dirname+filename)
# 	xvals=[]
# 	yvals=[]	
# 	for i in range(len(agentlist)):
# 		xvals.append(agentlist[i].getxpos())
# 		yvals.append(agentlist[i].getypos())
# 	for i in range(t_sim/tmeasure):
# 		o_dynm_maps=plt.figure()
# 		a=o_dynm_maps.add_subplot(111)
# 		colorvals=[]
# 		for j in range(len(agentlist)):
# 			colorvals.append(agentlist[j].getohist()[i] / 100)
# 		plt.scatter(xvals,yvals,circlesize,colorvals)
# 		plt.colorbar(ticks=[0,.25,.50,.75,1.00])
# 		plt.xlabel("x")
# 		plt.ylabel("y")
# 		plt.xlim([0,gridsize])
# 		plt.ylim([0,gridsize])
# 		plt.clim([0,1])
# 		plt.title("Opinion Map at t="+str(i*tmeasure))
# 		o_dynm_maps.savefig("Opinion Map at t="+str(i*tmeasure)+".png")
# 		plt.close()
# 	for i in range(t_sim/tmeasure):
# 		e_dynm_maps=plt.figure()
# 		a=e_dynm_maps.add_subplot(111)
# 		colorvals=[]
# 		for j in range(len(agentlist)):
# 			colorvals.append(agentlist[j].getehist()[i] / 100)
# 		plt.scatter(xvals,yvals,circlesize,colorvals)
# 		plt.xlabel("x")
# 		plt.colorbar(ticks=[0,.25,.50,.75,1.00])
# 		plt.ylabel("y")
# 		plt.xlim([0,gridsize])
# 		plt.ylim([0,gridsize])
# 		plt.clim([0,1])
# 		plt.title("Expression Map at t="+str(i*tmeasure))
# 		e_dynm_maps.savefig("Expression Map at t="+str(i*tmeasure)+".png")
# 		plt.close()

# def plot_dynamic_opinions(agentlist,t_sim,tmeasure):
# 	odynmplot=plt.figure()
# 	a=odynmplot.add_subplot(111)
# 	xvals=[]
# 	yvals=[]
# 	for i in range(len(agentlist)):
# 		xvals=np.arange(0,t_sim+1,tmeasure)
# 		yvals=agentlist[i].getohist()
# 		plt.plot(xvals,yvals)
# 	plt.xlabel("t")
# 	plt.ylabel("o")
# 	plt.title("Opinion Dynamics")
# 	odynmplot.savefig("Dynamic Opinion.png")

# 	edynmplot=plt.figure()
# 	a=edynmplot.add_subplot(111)
# 	xvals=[]
# 	yvals=[]
# 	for i in range(len(agentlist)):
# 		xvals=np.arange(0,t_sim+1,tmeasure)
# 		yvals=agentlist[i].getehist()
# 		plt.plot(xvals,yvals)
# 	plt.xlabel("t")
# 	plt.ylabel("e")
# 	plt.title("Expressed Dynamics")
# 	edynmplot.savefig("Dynamic Expressions.png")

# def plot_statistics(mean_o_list,std_o_list,skew_o_list,kurt_o_list,bimod_coeff_o_list,mean_e_list,std_e_list,skew_e_list,kurt_e_list,bimod_coeff_e_list,JSD_list, t_sim, tmeasure):
# 	ostatsplot=plt.figure()
# 	tvals=np.arange(0,t_sim+1,tmeasure)
# 	plt.subplots_adjust(hspace=0.3,wspace=0.2)
# 	plt.title("Opinion and Expression Statistics")
# 	plt.tick_params(axis="both",which="both",bottom="off",top="off",left="off",right="off",labelbottom="off",labelleft="off")
# 	a=ostatsplot.add_subplot(211)
# 	plt.plot(tvals,mean_o_list,"b-",label="mean o")
# 	plt.plot(tvals,std_o_list,"r-",label="standard deviation o")
# 	plt.plot(tvals,mean_e_list,"b--",label="mean e")
# 	plt.plot(tvals,std_e_list,"r--",label="standard deviation e")
# 	plt.xlabel("time")
# 	plt.ylabel("value")
# 	plt.legend(loc="lower right", prop={"size":7})
# 	b=ostatsplot.add_subplot(212)
# 	#plt.plot(tvals,skew_o_list,"b-",label="skew o")
# 	# plt.plot(tvals,kurt_o_list,"r--",label="kurtosis o")
# 	plt.plot(tvals,bimod_coeff_o_list,"g-",label="bimodal coefficient o")
# 	#plt.plot(tvals,skew_e_list,"b--",label="skew e")
# 	# plt.plot(tvals,kurt_o_list,"r--",label="kurtosis o")
# 	plt.plot(tvals,bimod_coeff_e_list,"g--",label="bimodal coefficient e")
# 	plt.plot(tvals,JSD_list,"r-",label="Jensen Shannon Divergence")
# 	plt.xlabel("time")
# 	plt.ylabel("value")
# 	plt.legend(loc="lower right", prop={"size":7})
# 	ostatsplot.savefig("Opinion and Expression Statistics.png")

# def plot_ideologies(agentlist,o_ideology_list,e_ideology_list,t_sim,tmeasure):
# 	o_centrists_list=[]
# 	o_moderates_list=[]
# 	o_extremists_list=[]
# 	e_centrists_list=[]
# 	e_moderates_list=[]
# 	e_extremists_list=[]
# 	for i in range(t_sim/tmeasure +1):
# 		o_centrists=0
# 		o_moderates=0
# 		o_extremists=0
# 		e_centrists=0
# 		e_moderates=0
# 		e_extremists=0
# 		for j in range(len(agentlist)):
# 			myopinion=abs(agentlist[j].getohist()[i])
# 			myexpression=abs(agentlist[j].getehist()[i])
# 			if myopinion > 33 and myopinion < 66:
# 				o_centrists += 1
# 			elif (myopinion > 16 and myopinion <= 33) or (myopinion >= 66 and myopinion < 82):
# 				o_moderates += 1
# 			else:
# 				o_extremists += 1
# 			if myexpression >= 33 and myexpression < 66:
# 				e_centrists += 1
# 			elif (myexpression > 16 and myexpression <= 33) or (myexpression >= 66 and myexpression < 82):
# 				e_moderates += 1
# 			else:
# 				e_extremists += 1
# 		o_ideology_list.append([o_centrists,o_moderates,o_extremists])
# 		e_ideology_list.append([e_centrists,e_moderates,e_extremists])
# 		o_centrists_list.append(o_centrists)
# 		o_moderates_list.append(o_moderates)
# 		o_extremists_list.append(o_extremists)
# 		e_centrists_list.append(e_centrists)
# 		e_moderates_list.append(e_moderates)
# 		e_extremists_list.append(e_extremists)
# 	ideology_plot=plt.figure()
# 	a=ideology_plot.add_subplot(111)
# 	plt.title("Dynamic Ideologies")
# 	plt.xlabel("time")
# 	plt.ylabel("Count")
# 	tvals=np.arange(0,t_sim+1,tmeasure)
# 	plt.plot(tvals,o_centrists_list,"g-",label="true centrists")
# 	plt.plot(tvals,o_moderates_list,"b-",label="true moderates")
# 	plt.plot(tvals,o_extremists_list,"r-",label="true extremists")
# 	plt.plot(tvals,e_centrists_list,"g--",label="expressed centrists")
# 	plt.plot(tvals,e_moderates_list,"b--",label="expressed moderates")
# 	plt.plot(tvals,e_extremists_list,"r--",label="expressed extremists")
# 	plt.legend(loc="lower right", prop={"size":7})
# 	ideology_plot.savefig("Dynamic Ideologies.png")


# def plot_final(agentlist,gridsize,o_history,e_history,bins,circlesize,t_sim,tmeasure,
# 	mean_e_list,std_e_list,skew_e_list,kurt_e_list,bimod_coeff_e_list):

# 	ohistplot=plt.figure()
# 	a=ohistplot.add_subplot(111)
# 	weights=100*np.ones_like(o_history[-1])/len(o_history[-1]) #normalizes properly, convert to %
# 	plt.hist(o_history[-1],bins=bins,weights=weights,normed=False)
# 	plt.xlabel("Opinion")
# 	plt.ylabel("Percent")
# 	plt.xlim([0,100])
# 	# plt.ylim([0,100])
# 	plt.title("Histogram of Opinions at t=" + str(t_sim))
# 	ohistplot.savefig("Final Histogram of Opinions.png")

# 	ehistplot=plt.figure()
# 	a=ehistplot.add_subplot(111)
# 	weights=100*np.ones_like(e_history[-1])/len(e_history[-1]) #normalizes properly	
# 	plt.hist(e_history[-1],bins=bins,weights=weights,normed=False) #list, bins
# 	plt.xlabel("Expressed Opinion")
# 	plt.ylabel("Percent")
# 	plt.xlim([0,100])
# 	# plt.ylim([0,100])
# 	plt.title("Histogram of Expressed at t=" + str(t_sim))
# 	ehistplot.savefig("Final Histogram of Expressions.png")

# 	omap=plt.figure()
# 	a=omap.add_subplot(111)	
# 	colorvals=[]
# 	xvals=[]
# 	yvals=[]
# 	for i in range(len(agentlist)):
# 		xvals.append(agentlist[i].getxpos())
# 		yvals.append(agentlist[i].getypos())
# 	for i in range(len(agentlist)):
# 		colorvals.append(agentlist[i].geto() / 100)
# 	plt.scatter(xvals,yvals,circlesize,colorvals)
# 	plt.colorbar(ticks=[0,.25,.50,.75,1.00])
# 	plt.xlabel("x")
# 	plt.ylabel("y")
# 	plt.xlim([0,gridsize])
# 	plt.ylim([0,gridsize])
# 	plt.clim([0,1])
# 	plt.title("Opinion Map at t=" + str(t_sim))
# 	omap.savefig("Final Opinion Map.png")

# 	emap=plt.figure()
# 	a=emap.add_subplot(111)	
# 	colorvals=[]
# 	xvals=[]
# 	yvals=[]
# 	for i in range(len(agentlist)):
# 		xvals.append(agentlist[i].getxpos())
# 		yvals.append(agentlist[i].getypos())
# 	for i in range(len(agentlist)):
# 		colorvals.append(agentlist[i].gete() / 100)
# 	plt.scatter(xvals,yvals,circlesize,colorvals)
# 	plt.colorbar(ticks=[0,.25,.50,.75,1.00])
# 	plt.xlabel("x")
# 	plt.ylabel("y")
# 	plt.xlim([0,gridsize])
# 	plt.ylim([0,gridsize])
# 	plt.clim([0,1])
# 	plt.title("Expression Map at t=" + str(t_sim))
# 	emap.savefig("Final Expression Map.png")

# main()


	# o_list=[]
	# e_list=[]
	# popsize=len(agentlist)
	# for k in range(len(agentlist)):
	# 	agentlist[k].updatehistory()
	# 	o_list.append(round(agentlist[k].geto()))
	# 	e_list.append(round(agentlist[k].gete()))
	# o_history.append(o_list) #add the entire distribution
	# e_history.append(e_list)
	# mean_o_list.append(mean(o_list))
	# mean_e_list.append(mean(e_list))
	# std_o_list.append(std(o_list))
	# std_e_list.append(std(e_list))
	# if std_o_list[-1] != 0:
	# 	kurt_o_list.append(stats.kurtosis(o_list, None, True, True))
	# 	skew_o_list.append(stats.skew(o_list, None, True))
	# 	bimod_coeff_o_list.append((skew_o_list[-1]**2 + 1) / (kurt_o_list[-1] + 3*(popsize-1)**2 / ( (popsize-2)*(popsize-3) ) ) )
	# else:
	# 	kurt_o_list.append(-3.0)
	# 	skew_o_list.append(0.0)
	# 	bimod_coeff_o_list.append(0.0) #need to fix
	# if std_e_list[-1] != 0:
	# 	kurt_e_list.append(stats.kurtosis(e_list, None, True, True))
	# 	skew_e_list.append(stats.skew(e_list, None, True))
	# 	bimod_coeff_e_list.append((skew_e_list[-1]**2 + 1) / (kurt_e_list[-1] + 3*(popsize-1)**2 / ( (popsize-2)*(popsize-3) ) ) )
	# else:
	# 	kurt_e_list.append(-3.0)
	# 	skew_e_list.append(0.0)
	# 	bimod_coeff_e_list.append(0.0)
	# JSD_list.append(JSD(o_list,e_list))

	# o_history=[]
	# e_history=[]
	# mean_o_list=[]
	# mean_e_list=[]
	# std_o_list=[]
	# std_e_list=[]
	# kurt_o_list=[]
	# kurt_e_list=[]
	# skew_o_list=[]
	# skew_e_list=[]
	# bimod_coeff_o_list=[]
	# bimod_coeff_e_list=[]
	# JSD_list=[]
	# o_ideology_list=[]
	# e_ideology_list=[]