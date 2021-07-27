import numpy as np
from card import *
import random
class player:
	def __init__(self):#still slow:6*10^5 times a sec
		self.cards=[]
		self.front=cardset()
		self.mid=cardset()
		self.back=cardset()
		self.p=[0]*13
		self.t=[0]*13
		self.straight=[0]*10
		#0:A2345,1:23456,...,9:10JQKA
		self.samesuit=[0]*4
		self.four=[0]*13
		self.ss=[0]*40
		self.decided=False
		#suit(1~4 to 0~3)*10+start(A~10 to 0~9)
	def clearsets(self):
		self.front.clear()
		self.mid.clear()
		self.back.clear()
	def getcards(self,stack):
		if len(stack)<13:
			print('Error!!! There are only '+str(len(stack))+' cards in the stack')
			exit(1)
		self.cards=stack.get13()
	def decidelayout(self):
		pass
	def valid(self):
		if len(self.front)!=3 or len(self.mid)!=5 or len(self.back)!=5:
			return False
		for i in [self.front,self.mid,self.back]:
			if not i.valid():
				return False
		return self.front<self.mid and self.mid<self.back
	def counttypes(self):#4*10^4 times a sec
		self.p=[0]*13
		self.t=[0]*13
		self.straight=[0]*10
		self.samesuit=[0]*4
		self.four=[0]*13
		self.ss=[0]*40
		self.cards.sort(key=lambda x:x.num)
		last3=-1
		last2=-1
		last=-1
		num=[0]*15
		for i in self.cards:
			n=i.num
			if n==last:
				self.p[n-2]=1
				num[n]=2
			if n==last2:
				self.t[n-2]=1
				self.p[n-2]=0
				num[n]=3
			if n==last3:
				self.four[n-2]=1
				self.t[n-2]=0
				num[n]=4
			if num[n]==0:
				num[n]=1
			last3=last2
			last2=last
			last=n
			
			self.samesuit[i.suit-1]+=1
			
			for j in range(5):
				if n-j>=1 and n-j<=10:
					self.ss[(i.suit-1)*10+n-j-1]+=1
			if n==14:
				self.ss[(i.suit-1)*10]+=1
		for i in range(40):
			self.ss[i]=self.ss[i]//5
		num[1]=num[14]
		for i in range(1,11):#straight starts from A,2,3,4,...,10
			self.straight[i-1]=num[i]
			for j in range(1,5):
				self.straight[i-1]=min(self.straight[i-1],num[i+j])
	def getinput(self,config=0):#1:52*13, 2:17*13
		self.counttypes()
		if config==0:
			out=np.zeros((1,52),dtype='int')
			for c in self.cards:
				out[0,(c.suit-1)*13+c.num-2]=1
		elif config==1:#2.5*10^5 times a sec(without counttype)
			out=np.zeros((1,13,52),dtype='int')
			for i in range(len(self.cards)):
				c=self.cards[i]
				out[0,i,(c.suit-1)*13+c.num-2]=1
		else:#3*10^5 times a sec(without counttype)
			out=np.zeros((1,13,17),dtype='int')#num:0~12 suit:13~16
			for i in range(len(self.cards)):
				c=self.cards[i]
				out[0,i,c.num-2]=1
				out[0,i,c.suit+12]=1#-1+13=+12
		return [np.array([i]) if type(i)==type([]) else i for i in [out,self.p,self.t,self.straight,self.samesuit,self.four,self.ss]]
	def __lt__(self,other):
		if self.front<other.front and self.mid<other.mid and self.back<other.back:
			return True
		return False
	def __str__(self):
		ans='This player has cards:\n'
		for i in sorted(self.cards,key=lambda x:x.num*13+x.suit):
			ans+=str(i)+', '
		ans+='\n\n'
		if self.decided:
			ans+='This player has decided the layout:'
			ans+=str(self.front)+'\n'
			ans+=str(self.mid)+'\n'
			ans+=str(self.back)
		else:
			ans+='This player hasn\'t decided the layout'
			ans+=str(self.front)+'\n'
			ans+=str(self.mid)+'\n'
			ans+=str(self.back)
		return ans
from time import time
if __name__ == '__main__':
	exit(0)
	p=player()
	print(p.getinput(0))
	ss=0
	f=0
	p=0
	t=0
	straight=0
	samesuit=0
	s=cardstack()
	p1=player()
	p1.cards=s.get13()
	start=time()
	for _ in range(300000):
		p1.getinput(2)
		'''p+=sum(p1.p)
		t+=sum(p1.t)
		ss+=sum(p1.ss)
		straight+=sum(p1.straight)
		for i in range(4):
			if p1.samesuit[i]>=5:
				samesuit+=1
		f+=sum(p1.four)'''
	print(time()-start)
	'''
		p+=sum(p1.p)
		t+=sum(p1.t)
		ss+=sum(p1.ss)
		straight+=sum(p1.straight)
		for i in range(4):
			if p1.samesuit[i]>=5:
				samesuit+=1
		f+=sum(p1.four)
	print(ss,f,straight,samesuit,p,t)
	'''
	'''
		p2=player()
		for i in [p1,p2]:
			i.getcards(s)
			i.randomlayout()
			print(i)
		print(times)
		times+=1
		if p1.valid() and p2.valid():
			break
		else:
			for i in p1.cards:
				print(i)
			print()
			for i in p2.cards:
				print(i)
	print(p1<p2)'''
