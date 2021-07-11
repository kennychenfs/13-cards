from card import *
class player:
	def __init__(self):
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
		#suit(1~4 to 0~3)*10+start(A~10 to 0~9)
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
	def counttypes(self):
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
	def __lt__(self,other):
		if self.front<other.front and self.mid<other.mid and self.back<other.back:
			return True
		return False
	def __str__(self):
		return str(self.front)+'\n'+str(self.mid)+'\n'+str(self.back)
from time import time
if __name__ == '__main__':
	ss=0
	f=0
	p=0
	t=0
	straight=0
	samesuit=0
	start=time()
	for _ in range(100000):
		s=cardstack()
		p1=player()
		p1.cards=s.get13()
		p1.counttypes()
		p+=sum(p1.p)
		t+=sum(p1.t)
		ss+=sum(p1.ss)
		straight+=sum(p1.straight)
		for i in range(4):
			if p1.samesuit[i]>=5:
				samesuit+=1
		f+=sum(p1.four)
	print(time()-start)
'''
輸入：
牌的原始輸入：
1.52x13=676，對於13張牌，表示每一張牌。
2.17x13=221，對於13張牌，表示每張牌的數字、花色。
加上坯、三條、順、同花的特殊處理：
坯：十三個數字分別有沒有坯：0或1，13個
三條：十三個數字分別有沒有三條：0或1，13個
順：照大小12345~10JQKA：個數，10個
同花：四種花色的牌數，10個
鐵支、同花順不知道要不要加，可能要。

輸入應該是全連接層，分兩個或多個input：原始一個、坯和三條可能要和起來，判斷葫蘆會比較方便。

輸出：
1.3堆分別選52張牌的機率：3x52=156
2.被打的機率
3.打人的機率
資料產生方法：
發牌給四個人，每個人產生一些(100)排法，對於每個人的每一種牌，隨機抽取一些(100)對手來比較，並取用其中的方法來作為資料，越好的組合權重越高。打人機率和被打機率照真實數據算。

可以考慮每次針對一個人，其他三人的牌可能可以重新發牌多次，可能能處理的牌型會比較多元，感覺這個方法比較可行。

'''
