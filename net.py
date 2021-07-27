import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model,layers,optimizers,losses
from card import *
from player import *
from time import time,sleep
BATCH_SIZE=20
noise_dim=5

SAMPLES=3
def getmodel(_type=0):
	if _type==0:
		rawinput=keras.Input(shape=(52),name='raw')
	elif _type==1:
		rawinput=keras.Input(shape=(13,52),name='raw')
	else:
		rawinput=keras.Input(shape=(13,17),name='raw')
	p_input=keras.Input(shape=(13),name='pair')
	t_input=keras.Input(shape=(13),name='three')
	str_input=keras.Input(shape=(10),name='straight')
	suit_input=keras.Input(shape=(4),name='samesuit')
	four_input=keras.Input(shape=(13),name='four')
	ss_input=keras.Input(shape=(40),name='ss')
	noise_input=keras.Input(shape=(noise_dim),name='noise')
	if _type!=0:
		raw_feature=layers.Flatten()(rawinput)
		raw_feature=layers.Dense(256,activation='relu')(raw_feature)
	else:
		raw_feature=layers.Dense(52,activation='relu')(rawinput)
	#raw_feature=layers.Dropout(0.2)(raw_feature)
	
	p_feature=layers.Dense(16,activation='relu')(p_input)
	
	t_feature=layers.Dense(16,activation='relu')(t_input)
	
	str_feature=layers.Dense(16,activation='relu')(str_input)
	
	suit_feature=layers.Dense(16,activation='relu')(suit_input)
	
	four_feature=layers.Dense(16,activation='relu')(four_input)
	
	ss_feature=layers.Dense(16,activation='relu')(ss_input)
	
	x=layers.concatenate([raw_feature,p_feature,t_feature,str_feature,suit_feature,four_feature,ss_feature,noise_input])#shape=(None,352)
	#x=layers.Dropout(0.2)(x)
	x=layers.Dense(128,activation='relu')(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(128,activation='relu')(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(128,activation='relu')(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(128,activation='relu')(x)
	'''
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(512,activation='relu')(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(256,activation='relu')(x)
	x=layers.Dropout(0.7)(x)'''
	
	policy_head=layers.Dense(52,activation='relu')(x)
	policy_head=layers.Dropout(0.5)(policy_head)
	#policy_head=layers.Dense(52,activation='relu')(policy_head)
	policy_head=layers.add([rawinput,policy_head])
	policy_head=layers.Dense(52,activation='relu')(policy_head)
	policy_head_front=layers.Dense(52,name='policy_head_front')(policy_head)
	policy_head_mid=layers.Dense(52,name='policy_head_mid')(policy_head)
	policy_head_back=layers.Dense(52,name='policy_head_back')(policy_head)
	
	hit_head=layers.Dense(16,activation='relu')(x)
	hit_head=layers.Dropout(0.2)(hit_head)
	'''hit_head=layers.Dense(16,activation='relu')(hit_head)
	hit_head=layers.Dropout(0.2)(hit_head)'''
	hit_head=layers.Dense(1,activation='sigmoid')(hit_head)
	hit_head=layers.Lambda(lambda x:x*3,name='hit_head')(hit_head)
	
	behit_head=layers.Dense(16,activation='relu')(x)
	behit_head=layers.Dropout(0.2)(behit_head)
	'''behit_head=layers.Dense(16,activation='relu')(behit_head)
	behit_head=layers.Dropout(0.2)(behit_head)'''
	behit_head=layers.Dense(1,activation='sigmoid')(behit_head)
	behit_head=layers.Lambda(lambda x:x*3,name='behit_head')(behit_head)
	
	return Model(
		inputs=[rawinput,p_input,t_input,str_input,suit_input,four_input,ss_input,noise_input],
		outputs=[policy_head_front,policy_head_mid,policy_head_back,hit_head,behit_head]
	)
def load_numpy(f):
	indata=[]
	outdata=[]
	with open(f,'rb') as F:
		for i in range(7):
			indata.append(np.load(F).astype('float32'))
			#print(indata[-1][3125])
		for i in range(3):
			outdata.append(np.load(F).astype('float32'))
	print(outdata[0].shape)
	outdata[0]=np.transpose(outdata[0],axes=(1,0,2))
	print(outdata[0].shape)
	outdata[0][0]/=3.
	outdata[0][1]/=5.
	outdata[0][2]/=5.
	return indata,(outdata[0][0],outdata[0][1],outdata[0][2],outdata[1],outdata[2])
def init_model(to_compile=False):
	model=getmodel(0)
	print(model.summary())
	if to_compile:
		model.compile(
			optimizer=optimizers.Adam(1e-2),
			loss=[
				losses.CategoricalCrossentropy(from_logits=True),
				losses.CategoricalCrossentropy(from_logits=True),
				losses.CategoricalCrossentropy(from_logits=True),
				losses.MeanSquaredError(),
				losses.MeanSquaredError()
			],
			loss_weights=[1.0,1.0,1.0,1.0,1.0],
		)
	return model
def load_model(f):
	return keras.models.load_model(f)
def save_model(f,model):
	model.save(f)
'''
def train(model,dataf,modelf):
	with tf.device('/device:GPU:0'):
		xdata,ydata=load_numpy(dataf)
		trainx=xdata
		trainy=ydata
		for i in range(1):
			model.fit(
				trainx,
				trainy,
				epochs=5,
				#batch_size=64,
				#steps_per_epoch=200,
				verbose=1,
				validation_split=0.1
			)
			save_model(modelf,model)
		return model
'''
t_stats=[0]*5
def valid_loss(xdata,ypredict):
	start=time()
	def addifnotfull(cs,toadd,limit=5):#cs is self.front or self.mid or self.back
		if len(cs)<limit:
			cs.cards.append(toadd)
			return True
		return False
	p=player()
	ypredict=ypredict[:3]
	for i in range(3):
		ypredict[i]=tf.nn.softmax(ypredict[i],axis=-1).numpy()
	#xdata:array, shape=(batch_size,52)
	#ypredict:three arrays, shape=(batch_size,52)
	batchsize=BATCH_SIZE
	valid_rate=[0]*batchsize#np.zeros((batchsize),dtype='float32')
	t_stats[0]+=time()-start
	for num in range(batchsize):
		start=time()
		x=xdata[num]
		'''
		for i in range(4):
			for j in range(13):
				print(int(x[i*13+j]),end=' ')
			print()'''
		policyf,policym,policyb=[ypredict[i][num] for i in range(3)]
		policyf=policyf.tolist()
		policym=policym.tolist()
		policyb=policyb.tolist()
		po=policyf+policym+policyb
		valid=0
		t_stats[1]+=time()-start
		start=time()
		for _ in range(SAMPLES):
			exist=[i==1 for i in x]
			p=player()
			policys=po[:]
			#This way is slightly faster than np.concatenate
			rest=13
			while True:
				if len(p.front)==3 and len(p.mid)==5 and len(p.back)==5:
					break
				a=random.choices(range(156),weights=policys,k=rest)
				for ind in a:
					i=ind//52
					j=ind%52
					if not exist[j]:
						policys[j]=policys[j+52]=policys[j+104]=0
						continue
					suc=False#success
					if i==0:
						suc=addifnotfull(p.front,card(index=j),3)
					if i==1:
						suc=addifnotfull(p.mid,card(index=j))
					if i==2:
						suc=addifnotfull(p.back,card(index=j))
					if suc:
						exist[j]=False
						rest-=1
			if p.valid():
				valid_rate[num]+=1
		t_stats[2]+=time()-start
		start=time()
	#print(valid_rate)
	valid_rate=tf.convert_to_tensor(valid_rate,dtype=tf.float32)
	valid_rate/=SAMPLES
	ans=sum(valid_rate)/batchsize
	print(ans)
	return (1-ans)*(1-ans)*100
optimizer=optimizers.Adam(1e-2)
def train_step(model,xdata,ydata):
	with tf.GradientTape() as tape:
		outputs=model(xdata,training=True)#[three (52) shaped policy, two (1) shaped value]
		crossentropy=losses.CategoricalCrossentropy(from_logits=True)
		
		floss=crossentropy(ydata[0],outputs[0])
		mloss=crossentropy(ydata[1],outputs[1])
		bloss=crossentropy(ydata[2],outputs[2])
		
		hitloss=crossentropy(ydata[3],outputs[3])
		behitloss=crossentropy(ydata[4],outputs[4])
		
		vloss=valid_loss(xdata[0],outputs)
		loss=floss+mloss+bloss+hitloss+behitloss+vloss
	print(vloss)
	gradients=tape.gradient(loss,model.trainable_variables)

	optimizer.apply_gradients(zip(gradients,model.trainable_variables))
	return floss,mloss,bloss,hitloss,behitloss,vloss
def train(model,dataf,modelf,epochs=5):
	with tf.device('/device:GPU:0'):
		xdata,ydata=load_numpy(dataf)
		length=len(xdata[0])
		print(length)
		noise=np.random.random((length, noise_dim))
		xdata.append(noise)
		for i in xdata:
			print(i.shape)
		dataset = tf.data.Dataset.from_tensor_slices((*xdata, *ydata))
		test_dataset = dataset.take(int(length*0.1))
		train_dataset = dataset.take(100)
		batched_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
		for epoch in range(epochs):
			for data in batched_dataset:
				
				floss,mloss,bloss,hitloss,behitloss,vloss=train_step(model,data[:8],data[8:])
				#print(t_stats)
		vloss_sum=0
		num=0
		print('training ends')
		for data in batched_dataset:
			outputs=model(data[:8],training=True)#[three (52) shaped policy, two (1) shaped value]
			vloss=valid_loss(data[0],outputs)
			vloss_sum+=vloss
			print(vloss)
		print(vloss_sum/num)
if __name__=='__main__':
	model=init_model()
	train(model,'test.npy','test.h5')
	'''
	TRAINING=1
	if TRAINING:
		model=init_model()
		#sleep(1)
		#train(model,'test.npy','test.h5')
		
	else:
		model=load_model('test.h5')
	'''
	
	
	
	
