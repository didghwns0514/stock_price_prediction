#-*-coding: utf-8-*-
from konlpy.tag import Komoran
from konlpy.tag import Kkma
#from konlpy.tag import Okt

# https://somjang.tistory.com/entry/Keras%EA%B8%B0%EC%82%AC-%EC%A0%9C%EB%AA%A9%EC%9D%84-%EA%B0%80%EC%A7%80%EA%B3%A0-%EA%B8%8D%EC%A0%95-%EB%B6%80%EC%A0%95-%EC%A4%91%EB%A6%BD-%EB%B6%84%EB%A5%98%ED%95%98%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0
import tensorflow_hub as hub
import tensorflow as tf
from konlpy.tag import Komoran
import CRAWLER__ML_LSTM_lang as lstmz

import pandas as pd
import numpy as np
import os
import time

import copy
# https://wikidocs.net/16038


def padding(sentence):
	fixed_len = 100
	wv_len = 200

	if len(sentence) >= 100 :
		sentence = sentence[:100]


	array_len = len(sentence)

	for i in range(0,fixed_len-array_len,1):
		sentence.insert(0,np.zeros(wv_len).tolist())

	return sentence

class Meanvector_w2v(object):
	def __init__(self, word2vec):
		self.w2v = word2vec
		self.dim = 200


	def transform(self, sentence):
		result = []
		for words in sentence:
			if words in self.w2v:
				result.append(self.w2v[words].tolist())
			else:
				result.append(np.zeros(self.dim).tolist())

		return result

def sentence_to_token(sentence_pos_list):
	# kmr pos 된 것 넣어서... 다시 합치기!!
	result_list = []
	skipped_list = ['한국경제TV', '기자', '라이온봇', '씽크풀', '한국경제신문', '이데일리TV', '서울경제', '연합뉴스','사진','로이터' ,'머니투데이', '이데일리', '흥극증권', '키움증권', '한경닷컴','마켓인사이트', '헤럴드경제','아시아경제' ,'인천공항','뉴시스']

	if isinstance(sentence_pos_list, list) and len(sentence_pos_list) != 0 :
		for tokens  in sentence_pos_list :
			if tokens[0] in ['-', '+', '%', '그러나', '?', '!', '하지만']:
				result_list.append(tokens[0])
			elif tokens[0] in ['코스피지수']:
				result_list.append('코스피')
			elif tokens[0] in ['코스닥지수']:
				result_list.append('코스닥')
			elif tokens[0] in ['코스닥시장', '코스피 시장']:
				result_list.append('코스피')
			elif tokens[0] in ['코스피 지수']:
				result_list.append('코스피')
			elif tokens[0] in ['코스닥 지수']:
				result_list.append('코스닥')
			elif tokens[0] in ['코스닥시장','코스닥 시장']:
				result_list.append('코스닥')
			elif tokens[0] in skipped_list :
				pass

			elif tokens[1] in ['SN']:
				#result_list.append('숫자숫자숫자')
				num_len = len(str(tokens[0]))
				if num_len == 1 :
					result_list.append('숫자1개')
				elif num_len == 2 :
					result_list.append('숫자2개')
				elif num_len == 3:
					result_list.append('숫자3개')
				elif num_len == 4:
					result_list.append('숫자4개')
				elif num_len == 5:
					result_list.append('숫자5개')
				elif num_len == 6:
					result_list.append(str(tokens[0])) # 종목코드임 거의
				elif num_len >= 7:
					result_list.append('숫자7개이상')
				else:
					pass

			elif tokens[1] in ['NNG', 'NNP', 'NNB', 'VV', 'VX', 'VA', 'XSV', 'EC', 'JKB', 'MAG', 'JX', 'VCN', 'VCP', 'XR']:
				result_list.append(tokens[0])

			else:
				pass
		
		print('length of tokenized sentence... : ',  len(result_list))
		print('^'*40)

		return result_list

	else:
		return None

def main():
	import os
	folder_path = os.getcwd()
	save_path = str(folder_path + '\\CRAWLER__train_data\\train_data.xlsx').replace('/', '\\')
	df = pd.read_excel(save_path, sheet_name='Sheet1', encoding='utf-8')
	#df = df.sample(frac=1).reset_index(drop=True) # shuffle data
	#df.to_excel(folder_path + "\\train_data" + "\\train_data_rand.xlsx", encoding='utf-8')
	#input('?')

	# %matplotlib inline
	# import matplotlib.pyplot as plt
	# df['Score'].value_counts().plot(kind='bar')

	dic_path = str(folder_path + '\\CRAWLER__necessary_data\\user_dic.txt').replace('/', '\\')
	kmr = Komoran(userdic=dic_path)
	data_length_check = []

	X_train = []
	y_train = []
	X_test = []
	y_test = []
	X_test_data = []
	X_train_data = []

	split_num = int(len(df)*0.8)
	#df = df.sample(frac=1).reset_index(drop=True) # shuffle data

	w2v_list = []
	for i in range(len(df)):
		original_sentence = df['Sentence'].iloc[i]
		tokened = sentence_to_token(kmr.pos(original_sentence))
		label = str(df['Topic'].iloc[i])
		tokened.insert(0, label)
		answer = int(df['Score'].iloc[i])
		data_length_check.append( tokened )

		if i <= split_num:
			X_train_data.append(original_sentence)
			X_train.append(tokened)
			y_train.append(answer)
		else:
			X_test_data.append(original_sentence)
			X_test.append(tokened)
			y_test.append(answer)

		w2v_list.append(tokened)

	print('length of data : ', len(data_length_check))

	print('max data len : ', max(len(l) for l in data_length_check))
	print('mean data len : ', sum(map(len,data_length_check))/len(data_length_check))

	#%matplotlib inline
	import matplotlib.pyplot as plt
	plt.hist([len(s) for s in data_length_check], bins=50)


	# w2v 모델 학습
	print('begin loading w2v...')
	print('check the list file for training...')
	#print(w2v_list[:10])
	from gensim.models import Word2Vec
	from gensim.models import KeyedVectors
	# modelz = Word2Vec(sentences=w2v_list, size=200, window=10, min_count=1, workers=4, sg=1, iter=10)
	# https://radimrehurek.com/gensim/models/word2vec.html

	#vector_path = str(folder_path + "\\vector" +"\\word_vector.bin")
	vector_path = str(folder_path + "\\CRAWLER__vector" +"\\word_vector_1.1.txt")

	vector_folder_path = str(folder_path + "\\CRAWLER__vector")
	if not os.path.isdir(vector_folder_path):
		os.mkdir(vector_folder_path)
	else:
		pass

	loaded_model = KeyedVectors.load(vector_path)



	w2v_obj = Meanvector_w2v(loaded_model)

	for i in range(len(X_train)):
		X_train[i] = w2v_obj.transform(X_train[i])
	for j in range(len(X_test)):
		X_test[j] = w2v_obj.transform(X_test[j])


	import numpy as np
	for i in range(len(y_train)):
		if y_train[i] == 0 : # 부정적
			y_train[i] = [1,0,0]
		elif y_train[i] == 1 :
			y_train[i] = [0,1,0]
		elif y_train[i] == 2 :
			y_train[i] = [0,0,1]

	for i in range(len(y_test)):
		if y_test[i] == 0 : # 부정적
			y_test[i] = [1,0,0]
		elif y_test[i] == 1 :
			y_test[i] = [0,1,0]
		elif y_test[i] == 2 :
			y_test[i] = [0,0,1]

	y_train = np.array(y_train)
	y_test = np.array(y_test)


	from keras.layers import Embedding, Dense, LSTM, Dropout
	from keras.models import Sequential

	max_len = 100 # 전체 데이터의 길이를 20로 맞춘다

	for i in range(len(X_train)):
		X_train[i] = padding(X_train[i])
	for j in range(len(X_test)):
		X_test[j] = padding(X_test[j])
	max_words = 35000

	#lstm_lang_score_AI
	#########################################################################
	lstmz.session(X_train, y_train, X_test, y_test, X_train_data, X_test_data)

	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_train = np.array(X_train).reshape(X_train.shape[0],max_len,200)
	X_test = np.array(X_test).reshape(X_test.shape[0],max_len,200)





	model = Sequential()
	#model.add(loaded_model.wv.get_keras_embedding(train_embeddings=False))
	#model.add(Embedding(max_words, 100))
	#model.add(LSTM(128))
	#model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(128, input_shape = (max_len,200)))
	model.add(Dropout(0.45))
	model.add(Dense(3, activation='softmax'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print('begin testing model _ 1...')
	history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.1)

	print("\n 테스트 정확도1 : {:.2f}%".format(model.evaluate(X_test, y_test)[1]*100))


	# model2 = Sequential()
	# model2.add(Dropout(0.3))
	# #model2.add(Embedding(max_words, 100))
	# model2.add(LSTM(128, input_shape = (max_len,200)))
	# model2.add(Dropout(0.45))
	# #model2.add(LSTM(128))
	# #model2.add(LSTM(128, input_shape = (max_len,1)))
	# model2.add(Dense(3, activation='softmax'))

	# model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# print('begin testing model _ 2...')
	# history2 = model2.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.1)

	# print("\n 테스트 정확도2 : {:.2f}%".format(model2.evaluate(X_test, y_test)[1]*100))

	# ----------------------------------------------------------------------------------------------------------------
	from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, BatchNormalization
	from tensorflow.keras import Input, Model
	from tensorflow.keras import optimizers
	import os


	class BahdanauAttention(tf.keras.Model):
		def __init__(self, units):
			super(BahdanauAttention, self).__init__()
			self.W1 = Dense(units)
			self.W2 = Dense(units)
			self.V = Dense(1)

		def call(self, values, query): # 단, key와 value는 같음
			# hidden shape == (batch_size, hidden size)
			# hidden_with_time_axis shape == (batch_size, 1, hidden size)
			# we are doing this to perform addition to calculate the score
			hidden_with_time_axis = tf.expand_dims(query, 1)

			# score shape == (batch_size, max_length, 1)
			# we get 1 at the last axis because we are applying score to self.V
			# the shape of the tensor before applying self.V is (batch_size, max_length, units)
			score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

			# attention_weights shape == (batch_size, max_length, 1)
			attention_weights = tf.nn.softmax(score, axis=1)

			# context_vector shape after sum == (batch_size, hidden_size)
			context_vector = attention_weights * values
			context_vector = tf.reduce_sum(context_vector, axis=1)

			return context_vector, attention_weights


	max_len = 100
	sequence_input = Input(shape=(max_len, 200), dtype='float32')
	#embedded_sequences = Embedding(vocab_size, 128, input_length=max_len)(sequence_input)
	lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
		(LSTM
		 (128*2,
		  dropout=0.45,
		  return_sequences=True,
		  return_state=True,
		  recurrent_activation='relu',
		  recurrent_initializer='glorot_uniform'))(sequence_input)
	state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
	state_c = Concatenate()([forward_c, backward_c]) # 셀 상태
	attention = BahdanauAttention(128*2) # 가중치 크기 정의
	context_vector, attention_weights = attention(lstm, state_h)
	hidden = BatchNormalization()(context_vector)
	output = Dense(3, activation='softmax')(hidden)
	model_3 = Model(inputs=sequence_input, outputs=output)
	# model.add(Dense(3, activation='softmax'))
	# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	# history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.1)
	Adam = optimizers.Adam(lr=0.0001, clipnorm=1.)
	model_3.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
	history = model_3.fit(X_train, y_train, epochs=2000, batch_size=128, validation_data=(X_test, y_test), verbose=1)
	# ----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	main()