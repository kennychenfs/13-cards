import numpy as np
from player import *
from card import *
from playerinstances import *
import random
from copy import copy,deepcopy
import statistics
from time import sleep
import net
import asyncio
import gc

WEIGHT=0.8
class result:
	def __init__(self,win=None,lose=None,front=None,mid=None,back=None):
		if win==None:
			self.win=0
			self.lose=0
			self.policy=np.zeros((3,52),dtype='float32')
			return
		self.win=win
		self.lose=lose
		self.policy=np.zeros((3,52),dtype='float32')
		for c in front.cards:
			self.policy[0,c.toindex()]=1.
		for c in mid.cards:
			self.policy[1,c.toindex()]=1.
		for c in back.cards:
			self.policy[2,c.toindex()]=1.
	def __lt__(self,other):
		return self.win-self.lose < other.win-other.lose
def givecard(players,stack=None):
	ret=0
	if stack==None:
		stack=cardstack()
		ret=1
	else:
		stack.shuffle()
	if type(players)==nnplayer or type(players)==randomplayer:
		players.getcards(stack)
	else:
		for i in players:
			i.getcards(stack)
	if ret:
		return stack
def letdecide(players,nnm=None):#players must be a list of player(randomplayer or nnplayer)
	ret=[]
	for i in players:
		if type(i)==nnplayer:
			if not nnm:
				print('ERROR: get nnplayers but no nnmanager!!!')
				return
			future=nnm.loop.create_future()
			nnm.add_coroutine_list(i.decidelayout(f=future))#only add to list
			ret.append(future)
		else:
			i.decidelayout()
	return ret
def compare(players):
	l=len(players)
	skip=[i for i in range(l) if not players[i].valid()]
	wins=[0]*l
	loses=[0]*l
	for i in skip:
		loses[i]=3
	for i,p1 in enumerate(players):
		if i in skip:
			continue
		for j,p2 in enumerate(players):
			if j in skip:
				continue
			if i==j:
				continue
			if p1<p2:
				wins[j]+=1
				loses[i]+=1
	return wins,loses
A=10
B=5
def prepare_generate_data(nnm=None,ptype='nn',config=0):
	others=range(1,4)
	global A,B
	players=[[],[],[],[]]
	if ptype=='random':
		func=lambda:randomplayer()#故意寫得一致，不然func=randomplayer也可以，如果影響效率再改掉
	elif ptype=='nn':
		if nnm==None:
			print('ERROR: in generate_data, ptype=nn, but no nnm')
			return
		func=lambda:nnplayer(nnm)
		for i in range(A):
			players[0].append(func())
			
		for i in others:
			for _ in range(B):
				players[i].append(func())#total=A+3*B players
	else:
		print('ERROR: in generate_data, ptype=',ptype,sep='')
		return
	
	s=givecard(players[0][0])
	for i in range(1,A):
		players[0][i].cards=players[0][i-1].cards#I don't change the cards, so it's ok!
		
	tmps=s.stack[:]
	for i in range(B):
		s.stack=tmps[:]
		for p in others:
			givecard(players[p][i],s)
	#give cards
	if ptype=='random':
		for i in range(4):
			letdecide(players[i])
		return players
	else:
		waits=letdecide(players[0],nnm)
		for i in others:
			waits.extend(letdecide(players[i],nnm))
		return players,waits
t_stats=[0]*5
async def nn_generate_data(players,wait,config=0):
	start=time()
	for i in wait:
		await i
	t_stats[0]+=time()-start
	start=time()
	results=[]
	global A,B
	for i in range(A):#A players
		win=0
		lose=0
		for j in range(B):#A*B*3 other players
			wins,loses=compare((players[0][i],players[1][j],players[2][j],players[3][j]))
			win+=wins[0]
			lose+=loses[0]
		results.append(result(win/B,lose/B,players[0][i].front,players[0][i].mid,players[0][i].back))
	t_stats[1]+=time()-start
	start=time()
	results.sort()
	ans=result()
	t_stats[2]+=time()-start
	start=time()
	for i in results:
		ans.policy*=(1-WEIGHT)
		ans.policy+=i.policy*WEIGHT
		ans.win*=(1-WEIGHT)
		ans.win+=i.win*WEIGHT
		ans.lose*=(1-WEIGHT)
		ans.lose+=i.lose*WEIGHT
	t_stats[3]+=time()-start
	start=time()
	return players[0][0].getinput(config),(ans.policy.reshape((1,3,52)),np.array(ans.win).reshape((1,1)),np.array(ans.lose).reshape((1,1)))
def append_data(f,times,ptype='nn',nnm=None,config=0):
	t_stats=[0]*5
	indata=[]#[input(7 arrays) array(size,13,52)... ]
	outdata=[]#[output(3 arrays) array(size,3,52),array(size,1),array(size,1)]
	playerss=[]
	waitss=[]
	if ptype=='nn':
		g=lambda:prepare_generate_data(nnm=nnm,config=config)
		for _ in range(times):
			if(_%50==49):print(_+1,',','%d'%((_+1)/times*100),'%',sep='')
			p,w=g()
			nnm.add_coroutine_list(nn_generate_data(p,w))
		data=[i for i in nnm.run_coroutine_list() if i]
	elif ptype=='random':
		g=lambda:generate_data(config=config)
		data=[]
		for _ in range(times):
			data.append(g())
	inp,outp=data[0]
	for i in range(7):
		indata.append(np.array(inp[i]))
	for i in range(3):
		outdata.append(np.array(outp[i]))
	
	for _ in range(1,times):
		#print(indata)
		inp,outp=data[_]
		for i in range(7):
			#print(i,inp[i].shape,indata[i].shape)
			indata[i]=np.append(inp[i],indata[i],axis=0)
		for i in range(3):
			#print(i,outp[i].shape,outdata[i].shape)
			outdata[i]=np.append(outp[i],outdata[i],axis=0)
		#sleep(0.01)
	findata=[]
	foutdata=[]
	try:
		with open(f,'rb') as F:
			for i in range(7):
				findata.append(np.load(F))
				findata[i]=np.append(indata[i],findata[i],axis=0)
			for i in range(3):
				foutdata.append(np.load(F))
				foutdata[i]=np.append(outdata[i],foutdata[i],axis=0)
	except:
		foutdata=outdata
		findata=indata
	print('data:',len(findata[0]))
	with open(f,'wb') as F:
		for i in range(7):
			np.save(F,findata[i])
		for i in range(3):
			np.save(F,foutdata[i])
	print(t_stats)
	gc.collect()
	'''
def randomdata(batch=1000,times=5):
	for i in range(times):
		start=time()
		append_data('test.npy',batch)
		sec=time()-start
		print(batch,' times spent ',sec,' seconds',sep='')
'''
model=net.load_model('test.h5')
nnm=nnmanager(model)
def nndata(batch=100,times=100):
	global nnm
	for i in range(times):
		start=time()
		append_data('test.npy',batch,'nn',nnm)
		sec=time()-start
		print(batch,' times spent ',sec,' seconds',sep='')
		sleep(1)
nndata(100,90)
if __name__=='__main__':
	'''
	model=net.load_model('test.h5')
	nnm=nnmanager(model)
	p1=nnplayer(nnm)
	p2=nnplayer(nnm)
	p=play([p1,p2,None,None],nnm=nnm)
	re=0
	for i in range(1):
		if i%50==0:
			print(i)
		p.givecard()
		p.letdecide()
		for pl in p.players:
			print(pl)
		p.compare()
		re+=p.wins[0]+p.wins[1]-p.loses[0]-p.loses[1]-(p.wins[2]+p.wins[3]-p.loses[2]-p.loses[3])
	print(re)
	
	for i,pl in enumerate(p.players):
		print(pl,p.wins[i],p.loses[i])'''
