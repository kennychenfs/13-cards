from card import *
class player:
	def __init__(self):
		self.cards=[]
		self.front=cardset()
		self.mid=cardset()
		self.back=cardset()
	def getcards(self,stack):
		if len(stack)<13:
			print('Error!!! There are only '+str(len(stack))+' cards in the stack')
		for i in range(13):
			self.cards.append(stack.get1())
	def randomlayout(self):
		#self.cards.sort(key=lambda x:random.random())#random
		self.cards.sort(key=lambda x:x.num*4+x.suit)#in order
		self.front.cards=self.cards[:3]
		self.front.determine()
		self.mid.cards=self.cards[3:8]
		self.mid.determine()
		self.back.cards=self.cards[8:13]
		self.back.determine()
	def valid(self):
		if len(self.front)!=3 or len(self.mid)!=5 or len(self.back)!=5:
			return False
		for i in [self.front,self.mid,self.back]:
			if not i.valid():
				return False
		if not (self.front<self.mid) or not (self.mid<self.back):
			return False
		return True
	def __lt__(self,other):
		if self.front<other.front and self.mid<other.mid and self.back<other.back:
			return True
		return False
	def __str__(self):
		return str(self.front)+'\n'+str(self.mid)+'\n'+str(self.back)
if __name__ == '__main__':
	s=cardstack()
	p1=player()
	p2=player()
	for i in [p1,p2]:
		i.getcards(s)
		i.randomlayout()
		print(i)
	print(p1.valid(),p2.valid())
	print(p1<p2)
