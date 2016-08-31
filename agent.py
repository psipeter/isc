# Peter Duggins
# August 11, 2016
# ISC Model

import numpy as np
class agent:

	def __init__(self,iden,xpos,ypos,o,intol,sus,con,radius):
		self.iden = iden
		self.x = xpos
		self.y = ypos
		self.O = o
		self.E = o
		self.int = intol
		self.sus = sus
		self.con = con
		self.commit = 1.0
		self.radius=radius
		self.network=[]

	def set_E(self,elist):
		if len(elist) == 0.0: posturing = 0.0 #if nobody has yet spoken in dialogue, no falsification
		else: posturing = np.mean(elist) - self.O #O_j is a constant, so pull out of sum
		self.E = self.O + self.con/self.commit * posturing '''EQUATION 1'''
		if self.E < 0.0: self.E = 0.0 #truncate below 0 or above 100
		elif self.E > 100.0: self.E = 100.0

	def set_commit(self):
		self.commit = 1.0 + self.sus * (abs(50.0 - self.O) / 50.0) '''EQUATION 2'''

	def set_influence(self,elist,wlist):
		if len(self.network) == 0.0: return 0.0 #stop if nobody in network
		sum_influence,sum_weight = 0.0,0.0
		for j in range(len(elist)):
			sum_influence += wlist[j] * (elist[j] - self.O) '''EQUATION 3'''
			sum_weight += abs(wlist[j])
		if sum_weight != 0.0:
			influence = sum_influence / sum_weight
		else: influence = 0.0
		return influence

	def set_w(self,E_j):
		return 1 - self.int * abs(E_j - self.O)/50.0 '''EQUATION 4'''

	def set_O(self,influence):
		self.O = self.O + 0.1 * influence / self.commit '''EQUATION 5'''
		#the 0.1 above slows the rate of opinion change but can be omitted
		if self.O < 0.0: self.O = 0.0 #truncate opinions below 0 or over 100
		elif self.O > 100.0: self.O = 100.0

	def addtonetwork(self,other):
		self.network.append(other)

	def hold_dialogue(self,rng):
		elist=[]
		wlist=[]
		elist.append(self.E) #i initiates by speaking his true opinion
		wlist.append(1.0) #placekeeper
		self.set_commit() #calculate i's susceptibility
		rng.shuffle(self.network) #randomize order of speaking each dialogue
		for j in self.network:
			j.set_E(elist) #each member of the dialogue calculates expressed
			elist.append(j.E) #expressed is spoken to the dialogue
			wlist.append(self.set_w(j.E)) #i calculates interagent weight
		influence=self.set_influence(elist[1:],wlist[1:]) #calculate dialogue's influence
		self.set_O(influence) #update opinion after dialogue