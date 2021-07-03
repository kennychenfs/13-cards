import random
import sys
suitout={1:'黑桃',2:'紅心',3:'方塊',4:'梅花'}
numout={11:'J',12:'Q',13:'K',14:'A'}
class card:
	suit=1
	num=2
	def __init__(self,suit=1,num=2):#14 means A
		self.suit=suit
		self.num=num
	def __eq__(self,other):
		if self.suit==other.suit and self.num==other.num:return True
		return False
	def __str__(self):
		return suitout[self.suit]+numout.get(self.num,str(self.num))
	def valid(self):
		if self.suit not in range(1,5):
			return False
		if self.num not in range(2,15):
			return False
		return True
class cardstack:
	stack=[]
	def __init__(self):
		for suit in range(1,5):
			for num in range(2,15):
				self.stack.append(card(suit,num))
		seed=random.randrange(sys.maxsize)
		print("Seed was:", seed)
		random.seed()
	def get1(self):
		if len(self)==0:
			print('Error!!! No card to do get1')
			return
		i=random.randrange(0,len(self.stack))
		return self.stack.pop(i)
	def __len__(self):
		return len(self.stack)
cardsetout={0:'單張',1:'胚',2:'吐胚',3:'三條',4:'順',5:'同花',6:'葫蘆',7:'鐵支',8:'同花順'}
class cardset:
	def __init__(self,cards=None):
		if cards==None:
			cards=[]
		self.cards=cards
		self.ctype=None
		self.hash=None#一律把2~A變成0~12再做
		self.determined=False
	def determine(self):
		self.cards.sort(key=lambda x:x.num*4+x.suit)
		if len(self)!=3 and len(self)!=5:
			self.determined=False
			self.ctype=None
			return
		self.determined=True
		p=[]
		t=0
		for i in range(len(self.cards)-1):
			if self.cards[i].num==self.cards[i+1].num:
				if not self.cards[i].num in p:
					p.append(self.cards[i].num)
				else:
					for j in range(len(p)-1,-1,-1):
						if p[j]==self.cards[i].num:
							p.pop(j)
						else:
							break
		for i in range(len(self.cards)-2):
			if self.cards[i].num==self.cards[i+1].num and self.cards[i].num==self.cards[i+2].num:
				t=self.cards[i].num
		if len(self)==5:#判斷鐵支
			for i in range(2):
				if self.cards[i].num==self.cards[i+1].num and self.cards[i].num==self.cards[i+2].num and self.cards[i].num==self.cards[i+3].num:
					self.ctype=7
					self.hash=self.cards[i].num-2
					return
		single=[]
		for i in self.cards:
			if i.num not in p:
				single.append(i.num)
		if t>0 and len(p)==1:
			self.ctype=6
			self.hash=(t-2)*13+p[0]-2
			return
		if t>0:
			self.ctype=3
			self.hash=t-2
			return
		if len(p)==2:
			self.ctype=2
			self.hash=(p[1]-2)*169+(p[0]-2)*13+single[0]
			return
		if len(p)==1:
			self.ctype=1
			self.hash=(p[0]-2)
			for i in reversed(single):
				self.hash=self.hash*13+i-2
			if len(self)==3:
				self.hash*=169
			return
		if len(self)==3:
			self.ctype=0
			self.hash=0
			for i in reversed(self.cards):
				self.hash=self.hash*13+i.num-2
			self.hash*=169
			return
		straight=True
		samesuit=True
		for i in range(len(self.cards)-1):
			if self.cards[i].num+1!=self.cards[i+1].num:
				straight=False
				break
		for i in range(len(self.cards)-1):
			if self.cards[i].suit!=self.cards[i+1].suit:
				samesuit=False
				break
		if straight and samesuit:
			self.ctype=8
			self.hash=self.cards[0].num-2
			return
		if straight:
			self.ctype=4
			self.hash=self.cards[0].num-2
			return
		if samesuit:
			self.ctype=5
			self.hash=0
			for i in reversed(self.cards):
				self.hash=self.hash*13+i.num-2
			return
		self.ctype=0
		self.hash=0
		for i in reversed(self.cards):
			self.hash=self.hash*13+i.num-2
	def valid(self):
		v=True
		for i in range(len(self.cards)):
			if not self.cards[i].valid():
				print('ERRER!!! ('+str(self.cards[i].suit)+','+str(self.cards[i].num)+') is not a valid card')
				v=False
			for j in range(len(self.cards)):
				if i>=j:continue
				if self.cards[i]==self.cards[j]:
					print('ERRER!!! Some cards in cards are the same')
					v=False
		return v
	def __lt__(self,other):
		for i in [self,other]:
			if not i.determined:
				i.determine()
		if self.ctype!=other.ctype:
			return self.ctype<other.ctype
		return self.hash<other.hash
	def __len__(self):
		return len(self.cards)
	def __str__(self):
		self.determine()
		ans='有'+str(len(self))+'張牌\n'
		for i in self.cards:
			ans+=i.__str__()+'\n'
		if self.ctype!=None:
			ans+='是'+cardsetout[self.ctype]+'\n'
		return ans
if __name__ == '__main__':
	nums=[8,8,9,9,7]
	suits=[4,1,2,3,4]
	cards=[]
	for i in range(5):
		cards.append(card(suits[i],nums[i]))
	a=cardset(cards)
	
	nums=[8,8,9,9,6]
	suits=[4,1,2,3,4]
	cards=[]
	for i in range(5):
		cards.append(card(suits[i],nums[i]))
	b=cardset(cards)
	print(a)
	print(b)
	
	print(a<b)
