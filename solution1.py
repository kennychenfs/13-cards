import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model,layers,optimizers,losses
from card import *
from player import *
from time import time,sleep
BATCH_SIZE=20
noise_dim=5
LEARNING_RATE=1e-3
SAMPLES=3
def playermodel(_type=0):
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
def discriminatormodel():
	#input is [policy_head_front,policy_head_mid,policy_head_back], all of which are in shape (52)
	input_front=keras.Input(shape=(52),name='front')
	input_mid=keras.Input(shape=(52),name='mid')
	input_back=keras.Input(shape=(52),name='back')
	x=layers.concatenate([input_front,input_mid,input_back]);
	x=layers.Dense(128)(x)
	x=layers.LeakyReLU()(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(128)(x)
	x=layers.LeakyReLU()(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(128)(x)
	x=layers.LeakyReLU()(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(16)(x)
	x=layers.LeakyReLU()(x)
	x=layers.Dropout(0.5)(x)
	x=layers.Dense(1,activation='sigmoid')(x)
	return Model(
		inputs=[input_front,input_mid,input_back],
		outputs=x
	)
def init_model(to_compile=False):
	pmodel=playermodel(0)
	print(pmodel.summary())
	if to_compile:
		pmodel.compile(
			optimizer=optimizers.Adam(LEARNING_RATE),
			loss=[
				losses.CategoricalCrossentropy(from_logits=True),
				losses.CategoricalCrossentropy(from_logits=True),
				losses.CategoricalCrossentropy(from_logits=True),
				losses.MeanSquaredError(),
				losses.MeanSquaredError()
			],
			loss_weights=[1.0,1.0,1.0,1.0,1.0],
		)
	dmodel=discriminatormodel()
	print(dmodel.summary())
	return pmodel,dmodel
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
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(valid,invalid):
    valid_loss=cross_entropy(tf.ones_like(valid),valid)
	invalid_loss=cross_entropy(tf.zeros_like(invalid),invalid)
	total_loss=valid_loss+invalid_loss
    return total_loss
def generator_loss(output):
    return cross_entropy(tf.ones_like(output),output)

t_stats=[0]*5
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
if __name__=='__main__':
	pmodel,dmodel=init_model()
