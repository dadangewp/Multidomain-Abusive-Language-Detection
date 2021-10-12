from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import numpy as np
import codecs
import tensorflow_hub as hub
import keras
from keras.engine import Layer
from keras.models import Model
from keras.layers import LSTM, GRU, Activation, Dense, Dropout, Input, Embedding, Bidirectional, Flatten, Lambda, concatenate, GlobalAveragePooling1D, dot
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import optimizers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from keras.optimizers import Adam
from keras import initializers, regularizers
from keras import optimizers
from keras.engine.topology import Layer
from keras import constraints

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(1)
# for later versions: tf.random.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)

def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def dot_product(x, kernel):
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights
	Returns:
	"""
	if K.backend() == 'tensorflow':
		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
		return K.dot(x, kernel)

class AttentionWithContext(Layer):
	"""
	Attention operation, with a context/query vector, for temporal data.
	Supports Masking.
	follows these equations:
	
	(1) u_t = tanh(W h_t + b)
	(2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
	(3) v_t = \alpha_t * h_t, v in time t
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		3D tensor with shape: `(samples, steps, features)`.
	"""

	def __init__(self,
				 W_regularizer=None, u_regularizer=None, b_regularizer=None,
				 W_constraint=None, u_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.u_regularizer = regularizers.get(u_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.u_constraint = constraints.get(u_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(AttentionWithContext, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1], input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		if self.bias:
			self.b = self.add_weight((input_shape[-1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)

		self.u = self.add_weight((input_shape[-1],),
								 initializer=self.init,
								 name='{}_u'.format(self.name),
								 regularizer=self.u_regularizer,
								 constraint=self.u_constraint)

		super(AttentionWithContext, self).build(input_shape)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		uit = dot_product(x, self.W)

		if self.bias:
			uit += self.b

		uit = K.tanh(uit)
		ait = dot_product(uit, self.u)

		a = K.exp(ait)

		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			a *= K.cast(mask, K.floatx())

		# in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's. 
		# Should add a small epsilon as the workaround
		# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a
		
		return weighted_input

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[1], input_shape[2]
	
class Addition(Layer):
	"""
	This layer is supposed to add of all activation weight.
	We split this from AttentionWithContext to help us getting the activation weights
	follows this equation:
	(1) v = \sum_t(\alpha_t * h_t)
	
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		2D tensor with shape: `(samples, features)`.
	"""

	def __init__(self, **kwargs):
		super(Addition, self).__init__(**kwargs)

	def build(self, input_shape):
		self.output_dim = input_shape[-1]
		super(Addition, self).build(input_shape)

	def call(self, x):
		return K.sum(x, axis=1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    abusives = []
    targets = []
    corpus = []
    multis = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                abusive = int(line.split("\t")[2])
                target = int(line.split("\t")[3])
                multi = int(line.split("\t")[6])
                abusives.append(abusive)
                targets.append(target)
                corpus.append(tweet)
                multis.append(multi)

    return corpus, abusives, targets, multis

def parse_testing(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    abusives = []
    targets = []
    corpus = []
    multis = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                abusive = int(line.split("\t")[2])
                target = int(line.split("\t")[3])
                multi = int(line.split("\t")[5])
                abusives.append(abusive)
                targets.append(target)
                corpus.append(tweet)
                multis.append(multi)

    return corpus, abusives, targets, multis

def parse_training_cast(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    count_pos = 0
    count_neg = 0
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[0]
                label = line.split("\t")[1]
                if "hateful" in label :
                    #print(label)
                    misogyny = 1
                else :
                    misogyny = 0
                #misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)
    return corpus, y

def parse_testing_cast(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[2]
                label = line.split("\t")[1]
                if "sexism" in label :
                    misogyny = 1
                else :
                    misogyny = 0
                #misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def combine_label(y_hate, y_target):
  multis = []
  for hate, target in zip (y_hate,y_target):
    if (hate == 0 and target == 0):
      multi = 0
    elif (hate == 1 and target == 0):
      multi = 1
    elif (hate == 0 and target == 1):
      multi = 2
    elif (hate == 1 and target == 1):
      multi = 3
    elif (hate == 0 and target == 2):
      multi = 4
    elif (hate == 1 and target == 2):
      multi = 5
    elif (hate == 0 and target == 3):
      multi = 6
    else :
      multi = 7
    multis.append(multi)
  return multis

def RNN(X):
    input1 = Input(shape=(1,), dtype=tf.string)
    layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input1)
    dense1 = Dense(256,name='FC1')(layer)
    dense1 = Activation('relu')(dense1)
    #dense1 = Dropout(0.1)(dense1)
    dense2 = Dense(256,name='FC2')(layer)
    dense2 = Activation('relu')(dense2)
    #dense2 = Dropout(0.1)(dense2)
    #concat = concatenate([dense1, dense2], axis=-1)
    #dense3 = Dense(128,name='FC3')(concat)
    out1 = Dense(2,name='out1')(dense1)
    out1 = Activation('sigmoid')(out1)
    out2 = Dense(4,name='out2')(dense2)
    out2 = Activation('sigmoid')(out2)
    model = Model(inputs=input1,outputs=[out1,out2])
    return model

dataTrain, hateTrain, targetTrain, multiTrain = parse_training("data-training.tsv")
dataTest1, hateTest1, targetTest1, multiTest1 = parse_testing("racism-testing.tsv")
dataTest2, hateTest2, targetTest2, multiTest2 = parse_testing("sexism-testing.tsv")
dataTest3, hateTest3, targetTest3, multiTest3 = parse_testing("misogyny-evalita-testing.tsv")
dataTest4, hateTest4, targetTest4, multiTest4 = parse_testing("misogyny-ibereval-testing.tsv")
dataTest5, hateTest5, targetTest5, multiTest5 = parse_testing("misogyny-hateval-testing.tsv")
dataTest6, hateTest6, targetTest6, multiTest6 = parse_testing("misogyny-testing.tsv")
dataTest7, hateTest7, targetTest7, multiTest7 = parse_testing("immigrant-testing.tsv")
dataTest8, hateTest8, targetTest8, multiTest8 = parse_testing("racism-immigrant-testing.tsv")
dataTest9, hateTest9, targetTest9, multiTest9 = parse_testing("sexism-misogyny-testing.tsv")

hateTrain = pd.get_dummies(hateTrain).values
targetTrain = pd.get_dummies(targetTrain).values

#word_representation
max_len = 128
max_words = 15000
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain)
vocab = len(tok.word_index)

train_text = np.array(dataTrain, dtype=object)[:, np.newaxis]
test_text1 = np.array(dataTest1, dtype=object)[:, np.newaxis]
test_text2 = np.array(dataTest2, dtype=object)[:, np.newaxis]
test_text3 = np.array(dataTest3, dtype=object)[:, np.newaxis]
test_text4 = np.array(dataTest4, dtype=object)[:, np.newaxis]
test_text5 = np.array(dataTest5, dtype=object)[:, np.newaxis]
test_text6 = np.array(dataTest6, dtype=object)[:, np.newaxis]
test_text7 = np.array(dataTest7, dtype=object)[:, np.newaxis]
test_text8 = np.array(dataTest8, dtype=object)[:, np.newaxis]
test_text9 = np.array(dataTest9, dtype=object)[:, np.newaxis]

model = RNN(dataTrain)
model.summary()


myadam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(), metrics=['acc'])
model.fit(train_text,[hateTrain,targetTrain],batch_size=32,epochs=3)

#accr = model.evaluate(test_text, [hateTest,targetTest,aggTest])
#print('Test set\n  Loss: {:0.4f}\n  Precision: {:0.4f}'.format(accr[0],accr[1]))
    
y_prob1 = model.predict(test_text1) 
y_prob2 = model.predict(test_text2) 
y_prob3 = model.predict(test_text3) 
y_prob4 = model.predict(test_text4) 
y_prob5 = model.predict(test_text5) 
y_prob6 = model.predict(test_text6) 
y_prob7 = model.predict(test_text7)
y_prob8 = model.predict(test_text8) 
y_prob9 = model.predict(test_text9)  

y_pred_hate1 = np.argmax(y_prob1[0], axis=1)
y_pred_target1 = np.argmax(y_prob1[1], axis=1)
y_pred_hate2 = np.argmax(y_prob2[0], axis=1)
y_pred_target2 = np.argmax(y_prob2[1], axis=1)
y_pred_hate3 = np.argmax(y_prob3[0], axis=1)
y_pred_target3 = np.argmax(y_prob3[1], axis=1)
y_pred_hate4 = np.argmax(y_prob4[0], axis=1)
y_pred_target4 = np.argmax(y_prob4[1], axis=1)
y_pred_hate5 = np.argmax(y_prob5[0], axis=1)
y_pred_target5 = np.argmax(y_prob5[1], axis=1)
y_pred_hate6 = np.argmax(y_prob6[0], axis=1)
y_pred_target6 = np.argmax(y_prob6[1], axis=1)
y_pred_hate7 = np.argmax(y_prob7[0], axis=1)
y_pred_target7 = np.argmax(y_prob7[1], axis=1)
y_pred_hate8 = np.argmax(y_prob8[0], axis=1)
y_pred_target8 = np.argmax(y_prob8[1], axis=1)
y_pred_hate9 = np.argmax(y_prob9[0], axis=1)
y_pred_target9 = np.argmax(y_prob9[1], axis=1)

target1=[]
hate1=[]
for hate, target in zip(y_pred_hate1, y_pred_target1):
  if (hate == 0 and target == 1):
    hate1.append(0)
    target1.append(0)
  else:
    hate1.append(hate)
    target1.append(target)

target2=[]
hate2=[]
for hate, target in zip(y_pred_hate2, y_pred_target2):
  if (hate == 0 and target == 0):
    hate2.append(0)
    target2.append(1)
  else:
    hate2.append(hate)
    target2.append(target)

target8=[]
hate8=[]
for hate, target in zip(y_pred_hate8, y_pred_target8):
  if (hate == 0 and target == 1):
    hate8.append(0)
    target8.append(0)
  else:
    hate8.append(hate)
    target8.append(target)

target9=[]
hate9=[]
for hate, target in zip(y_pred_hate9, y_pred_target9):
  if (hate == 0 and target == 0):
    hate9.append(0)
    target9.append(1)
  else:
    hate9.append(hate)
    target9.append(target)


y_multi1 = combine_label(hate1, target1)
y_multi2 = combine_label(hate2, target2)
y_multi3 = combine_label(y_pred_hate3, y_pred_target3)
y_multi4 = combine_label(y_pred_hate4, y_pred_target4)
y_multi5 = combine_label(y_pred_hate5, y_pred_target5)
y_multi6 = combine_label(y_pred_hate6, y_pred_target6)
y_multi7 = combine_label(y_pred_hate7, y_pred_target7)
y_multi8 = combine_label(hate8, target8)
y_multi9 = combine_label(hate9, target9)


print("Racism")
print("-------------------------------------------")
print(classification_report(y_multi1, multiTest1, digits = 4))
print(confusion_matrix(y_multi1, multiTest1))
print("-------------------------------------------")

print("Sexism")
print("-------------------------------------------")
print(classification_report(y_multi2, multiTest2, digits = 4))
print(confusion_matrix(y_multi2, multiTest2))
print("-------------------------------------------")

print("Misogyny Evalita")
print("-------------------------------------------")
print(classification_report(y_multi3, multiTest3, digits = 4))
print(confusion_matrix(y_multi3, multiTest3))
print("-------------------------------------------")

print("Misogyny IberEval")
print("-------------------------------------------")
print(classification_report(y_multi4, multiTest4, digits = 4))
print(confusion_matrix(y_multi4, multiTest4))
print("-------------------------------------------")

print("Misogyny HatEval")
print("-------------------------------------------")
print(classification_report(y_multi5, multiTest5, digits = 4))
print(confusion_matrix(y_multi5, multiTest5))
print("-------------------------------------------")

print("Misogyny All")
print("-------------------------------------------")
print(classification_report(y_multi6, multiTest6, digits = 4))
print(confusion_matrix(y_multi6, multiTest6))
print("-------------------------------------------")

print("Immigrant")
print("-------------------------------------------")
print(classification_report(y_multi7, multiTest7, digits = 4))
print(confusion_matrix(y_multi7, multiTest7))
print("-------------------------------------------")

print("Racism Immigrant")
print("-------------------------------------------")
print(classification_report(y_multi8, multiTest8, digits = 4))
print(confusion_matrix(y_multi8, multiTest8))
print("-------------------------------------------")

print("Sexism Misogyny")
print("-------------------------------------------")
print(classification_report(y_multi9, multiTest9, digits = 4))
print(confusion_matrix(y_multi9, multiTest9))
print("-------------------------------------------")
