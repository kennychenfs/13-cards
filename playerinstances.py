from player import *
from card import *
import asyncio
import net
from collections import namedtuple
import tensorflow as tf
import statistics
from time import time,sleep

class randomplayer(player):
	def decidelayout(self):
		while 1:
			self.cards.sort(key=lambda x:random.random())
			self.front.cards=self.cards[:3]
			self.front.determine()
			self.mid.cards=self.cards[3:8]
			self.mid.determine()
			self.back.cards=self.cards[8:13]
			self.back.determine()
			if self.valid() and self.back.ctype==1:
				break
		self.decided=True
class nnplayer(player):
	def __init__(self,manager,noise=True):
		super().__init__()
		self.push_queue=manager.push_queue_func
		manager.add_coroutine_list(self.decidelayout(noise))
	async def decidelayout(self,noise=True,f=None):
		self.clearsets()
		inputs=self.getinput()
		cards={}
		exist=[False]*52
		for i,card in enumerate(self.cards):
			cards[card.toindex()]=i
			exist[card.toindex()]=True
		future=await self.push_queue(inputs)  # type: Future
		await future
		policyf,policym,policyb,_1,_2=future.result()#policys should be in shape (52)
		
		def addifnotfull(cs,toadd,limit=5):#cs is self.front or self.mid or self.back
			if len(cs)<limit:
				cs.cards.append(toadd)
				return True
			return False
		if noise:#softmax
			policyf=tf.nn.softmax(policyf).numpy().tolist()
			policym=tf.nn.softmax(policym).numpy().tolist()
			policyb=tf.nn.softmax(policyb).numpy().tolist()
			policys=policyf+policym+policyb
			#This way is slightly faster than np.concatenate
			rest=13
			while True:
				if len(self.front)==3 and len(self.mid)==5 and len(self.back)==5:
					break
				a=random.choices(range(156),weights=policys,k=rest)
				for ind in a:
					i=ind//52
					j=ind%52
					if not exist[j]:
						policys[ind]=0
						continue
					suc=False#success
					if i==0:
						suc=addifnotfull(self.front,self.cards[cards[j]],3)
					if i==1:
						suc=addifnotfull(self.mid,self.cards[cards[j]])
					if i==2:
						suc=addifnotfull(self.back,self.cards[cards[j]])
					if suc:
						exist[j]=False
						rest-=1
		else:#argmax
			policys=np.concatenate((policyf,policym,policyb),axis=None)
			rest=13
			for i,e in enumerate(exist):
				if not e:
					policys[i]=policys[52+i]=policys[104+i]=-1000000
			while True:
				if len(self.front)==3 and len(self.mid)==5 and len(self.back)==5:
					break
				a=np.argpartition(policys,-rest)[-rest:]
				a=a[np.argsort(-policys[a])]
				for ind in a:
					i=ind//52
					j=ind%52
					if policys[ind]==-1000000:
						continue
					if not exist[j]:
						print(ind,exist[j],policys[ind])
						exit(2)
						continue
					suc=False#success
					if i==0:
						suc=addifnotfull(self.front,self.cards[cards[j]],3)
					if i==1:
						suc=addifnotfull(self.mid,self.cards[cards[j]])
					if i==2:
						suc=addifnotfull(self.back,self.cards[cards[j]])
					policys[ind]=-1000000
					if suc:
						policys[j]=policys[52+j]=policys[104+j]=-1000000
						rest-=1
		if f!=None:
			f.set_result(1)
		self.decided=True
QueueItem = namedtuple("QueueItem", "feature future")
class nnmanager:
	def __init__(self,forward_func,max_threads=3000):
		self.loop=asyncio.get_event_loop()
		self.forward=forward_func
		self.queue=asyncio.queues.Queue(max_threads)
		self.coroutine_list=[self.prediction_worker()]
		async def push_queue(features):
			future=self.loop.create_future()
			item=QueueItem(features,future)
			await self.queue.put(item)
			return future
		self.push_queue_func=push_queue
	def add_coroutine_list(self,toadd):
		if toadd not in self.coroutine_list:
			self.coroutine_list.append(toadd)
	def run_coroutine_list(self):
		ret=self.loop.run_until_complete(asyncio.gather(*(self.coroutine_list)))
		self.coroutine_list=[self.prediction_worker()]
		return ret
	async def prediction_worker(self):
		"""For better performance, queueing prediction requests and predict together in this worker.
		speed up about 45sec -> 15sec for example.
		"""
		q = self.queue
		margin = 10  # avoid finishing before other searches starting.
		while margin > 0:
			if q.empty():
				await asyncio.sleep(1e-3)
				if q.empty() and margin > 0:
					margin -= 1
				continue
			item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
			features=[]
			for i in range(7):
				features.append(np.concatenate([j.feature[i] for j in item_list]))
			print('nn',len(item_list))
			with tf.device('/device:GPU:0'):
				#start=time()
				results = self.forward(features)
				#print('inference:',time()-start)
			for a,b,c,d,e,item in zip(*results,item_list):
				item.future.set_result((a,b,c,d,e))
		#print('nn ends')

if __name__=='__main__':
	'''
	for ___ in range(100):
		f='test.npy'
		indata=[]
		outdata=[]
		s=cardstack()
		tmps=s.stack[:]
		p=randomplayer()
		
		p.getcards(s)
		p.decidelayout()
		a=p.getinput()
		for i in a:
			indata.append(i)
		outdata=[np.zeros((1,3,52),dtype='int')]
		for i,cs in enumerate((p.front,p.mid,p.back)):
			for c in cs.cards:
				outdata[0][0,i,c.toindex()]=1
		start=time()
		A=1000
		for i in range(A):
			s.stack=tmps[:]
			for j in range(4):
				p.getcards(s)
				p.decidelayout()
				a=p.getinput()
				for k,l in enumerate(a):
					indata[k]=np.append(l,indata[k],axis=0)
				out=np.zeros((1,3,52),dtype='int')
				for k,cs in enumerate((p.front,p.mid,p.back)):
					for c in cs.cards:
						out[0,k,c.toindex()]=1
				outdata[0]=np.append(out,outdata[0],axis=0)
			if i%50==0:
				print(i)
		outdata.append(np.ones((4*A+1,1),dtype='int'))
		outdata.append(np.ones((4*A+1,1),dtype='int'))
		print(time()-start)
		
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
		print('data:',findata[0].shape)
		with open(f,'wb') as F:
			for i in range(7):
				np.save(F,findata[i])
			for i in range(3):
				np.save(F,foutdata[i])
	exit(0)
	'''
	model=net.load_model('test.h5')
	nnm=nnmanager(model)
	players=[]
	for i in range(1000):
		s=cardstack()
		for _ in range(4):
			tmp=nnplayer(nnm,True)
			tmp.getcards(s)
			#tmp.decidelayout()
			players.append(tmp)
	start=time()
	nnm.run_coroutine_list()
	#print(statistics.mean(stats),statistics.median(stats),statistics.stdev(stats),)
	print(time()-start)
	v=0
	iv=0
	for i in players:
		if i.valid():
			v+=1
		else:
			iv+=1
	print(v)
	print(v/(v+iv))
	print(players[0],players[1])
