

 #-*-coding: utf-8-*-
# mnist_autoencoder.py
#
# Autoencoder class stolen from Tensorflow's official models repo:
# http://github.com/tensorflow/models
#
import time
import numpy as np
# import random
import tensorflow as tf
import os
import copy
import pickle
import datetime
# import math
#
# from matplotlib import pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt

class Options :
	def __init__(self, env):
		self.INPUT_DATA_DIM = env[0] # 입력 데이터 variable 갯수
		self.HIDDEN_CELL_1_DIM = env[1] # 1차 히든 레이어 -> 1차 디코더
		self.HIDDEN_CELL_2_DIM = env[2]	# 2차 히든 레이어
		self.FINAL_CODING_LAYER = env[3]  # 마지막 encoding 레이어

		self.INPUT_VERTICAL_DIM = env[4]

		self.MAX_EPISODE = env[5]  # max number of episodes iteration
		self.LR = env[6]  # learning rate # 학습률
		self.L2_REGULARIZATION = env[7] # L2 규제를 위한 값
		self.DROP_OUT = env[8]


class Autoencoder:
	#           0,   1,   2,   3,   4,     5,      6,       7,    8
	envs = [  100,  88,  44,  20,   2, 30000,  0.001,  0.0005,  0.6] # input 100 -> 200개 으로 잡아야 할 듯? Out은 20개로?
	# 100 여개 기사까지 커버치기
	NAME = 'Autoencoder'

	def __init__(self, module = True, simple=True):

		if not simple: # simple 하면 계산만 해서 돌려주기
			self.options = Options(self.envs)

			# @ 초기 세팅
			############################################################
			############################################################
			# 완전 초기화 용
			from tensorflow.python.client import device_lib
			print(device_lib.list_local_devices())

			self.config = tf.ConfigProto(
				device_count={'GPU': 0}
			)
			self.test_query = None
			# @ Initial environment setting
			if module == False:
				print('trainable... applied')
				self.test_query = int(input('want to save test NN : 0 ... else : 1 '))
				if int(self.test_query) != 0 and int(self.test_query) != 1:
					raise ValueError('Wrong test_query input..! ')
			else:
				print('test only... applied')
				self.test_query = 1 #테스트용!
				self.options.DROP_OUT = 1

			if tf.test.gpu_device_name():
				print('GPU found')
			else:
				print("No GPU found")

			self.module = module


			# https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
			# https://excelsior-cjh.tistory.com/151
			self.GRAPH = tf.Graph()
			# print('self.GRAPH 가 기본 그래프인가? : ', self.GRAPH is tf.get_default_graph())
			# self.GRAPH.as_default()
			# print('self.GRAPH 가 기본 그래프인가? : ', self.GRAPH is tf.get_default_graph())

			self.score_dir = str(os.getcwd() + "\\ENCODER__FFNN_ENCODER")
			self.score_dir_txt = str(self.score_dir) + "\\fnn_encoder_score.txt"

			self.curr_dir_folder = str(os.getcwd() + "\\ENCODER__checkpoints-FNN_Encoder")
			self.curr_dir = os.path.join(self.curr_dir_folder, "fnn_encoder-DQN")

			with self.GRAPH.as_default() as g:
				if module == False:
					#self.sess = tf.Session(config=self.config)
					self.sess = tf.Session(graph=g)
				else:
					self.sess = tf.Session(config=self.config, graph=g)
					#self.sess = tf.Session()

			with self.GRAPH.as_default() as g:
				with self.sess.as_default() as sess:
					### for cells
					############################################################
					############################################################
					self.L2_REG = tf.contrib.layers.l2_regularizer(scale = self.options.L2_REGULARIZATION)


					# @ W / B 를 stack 별로 설정
					self.W1_E = tf.Variable(self.xavier_initializer([self.options.INPUT_DATA_DIM * self.options.INPUT_VERTICAL_DIM, self.options.HIDDEN_CELL_1_DIM]), name=self.NAME+'W1_E')
					self.B1_E = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_1_DIM]), name=self.NAME+'B1_E')
					self.W2_E = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_1_DIM, self.options.HIDDEN_CELL_2_DIM]), name=self.NAME+'W2_E')
					self.B2_E = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_2_DIM]), name=self.NAME+'B2_E')
					self.W3_F = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_2_DIM, self.options.FINAL_CODING_LAYER]), name=self.NAME+'W3_F')
					self.B3_F = tf.Variable(self.xavier_initializer([self.options.FINAL_CODING_LAYER]), name=self.NAME+'B3_F')

					self.W3_D = tf.transpose(self.W3_F)
					self.B3_D = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_2_DIM]), name=self.NAME+'B3_D')
					self.W2_D = tf.transpose(self.W2_E)
					self.B2_D = tf.Variable(self.xavier_initializer([self.options.HIDDEN_CELL_1_DIM]), name=self.NAME+'B2_D')
					self.W1_D = tf.transpose(self.W1_E)
					self.B1_D = tf.Variable(self.xavier_initializer([self.options.INPUT_DATA_DIM * self.options.INPUT_VERTICAL_DIM]), name=self.NAME+'B1_D')


					# fetch the layers
					self.observation, self.encoding, self.hypothesis = self.build_value_net()

					# @ LOSS
					# self.reconstruction_loss = tf.reduce_sum(tf.pow(tf.subtract( self.observation, self.hypothesis ),tf.subtract( self.observation, self.hypothesis )))

					self.reconstruction_loss = tf.reduce_mean(tf.square( self.observation - self.hypothesis))
					#self.reconstruction_loss = tf.reduce_mean(self.observation - self.hypothesis)
					self.reg_loss = self.L2_REG(self.W1_E) + self.L2_REG(self.W2_E) + self.L2_REG(self.W3_F)
					self.loss_tot = self.reconstruction_loss + self.reg_loss
					self.optimizer = tf.train.AdamOptimizer(self.options.LR)
					self.train = self.optimizer.minimize(self.loss_tot)
					#self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.answer, self.predictions)))

					self.saver = tf.train.Saver(save_relative_paths=True)

					# @ Initialize
					sess.run(tf.global_variables_initializer())
					self.load_model()  # if necessary

			# @ Used variables
			self.input_queue = None
			self.output_queue = None

			self.global_step = 0

			# @ model score
			self.content = None
			# self.curr_dir_f = None
			self.lv_score_board_fix = None
			self.score_board_value = None
			self.score_now_max = None

			self.score = None
			self.score_list = [] # 이것을 계산해서 self.score에 기록
			self.err_list = []



			self.init_model() # 추가했는데... 오류나올수도..

			self.return_answer = None
			self.err = None


		else: # simple model for calculation
			pass

	def FUNC_SIMPLE__read_article(self, article_loc, specific_time, stock_name):

		with open(article_loc, 'wb') as file:

			pickle_file = pickle.load(file)

			target=None
			for stock_names in pickle_file:
				if stock_names == stock_name:
					target = stock_names
					break
			else:
				return None # if none is matched

			rtn = parse_four_days(specific_time=specific_time,
								  stock_name=stock_name,
								  pickle_data=pickle_file)

			return rtn


	def build_value_net(self): # W/B 생성 후 묶어주는 부분
		# observation and prediction already defined above
		if self.test_query == 1:
			keep_prob = 1
		else:
			keep_prob = self.options.DROP_OUT

		observation = tf.placeholder(tf.float32, [None, self.options.INPUT_DATA_DIM * self.options.INPUT_VERTICAL_DIM], name=self.NAME + 'observation')
		# predictions = tf.placeholder(tf.float32, [None, self.options.INPUT_DATA_DIM], name='same_as_observ') # 이것은 관찰 결과와 같아야 한다!!! -> hypothesis로 옮김
		e1 = tf.nn.relu(tf.matmul(observation, self.W1_E) + self.B1_E)
		e1 = tf.nn.dropout(e1, keep_prob = keep_prob)
		e2 = tf.nn.relu(tf.matmul(e1, self.W2_E) + self.B2_E)
		e2 = tf.nn.dropout(e2, keep_prob=keep_prob)
		encoding_layer = tf.nn.relu(tf.matmul(e2, self.W3_F) + self.B3_F)
		d2 = tf.nn.relu(tf.matmul(encoding_layer, self.W3_D) + self.B3_D)
		d2 = tf.nn.dropout(d2, keep_prob=keep_prob)
		d1 = tf.nn.relu(tf.matmul(d2, self.W2_D) + self.B2_D)
		d1 = tf.nn.dropout(d1, keep_prob=keep_prob)
		H_ = tf.nn.relu(tf.matmul(d1, self.W1_D) + self.B1_D)

		return observation, encoding_layer, H_


	def xavier_initializer(self, shape):
		dim_sum = np.sum(shape)
		if len(shape) == 1 :
			dim_sum = dim_sum + 1
		bound = np.sqrt(6.0/ dim_sum)
		return tf.random_uniform(shape, minval = -bound, maxval=bound, dtype=tf.float32)

	def observation_to_train(self, inputz):
		self.global_step = self.global_step + 1
		
		with self.GRAPH.as_default() as g:
			with self.sess.as_default() as sess:

				# @ 자료 shape 변환
				self.input_queue = np.asarray(inputz, dtype=np.float32)
				#self.input_queue = copy.deepcopy(np.array(inputz))
				self.input_queue = np.asarray(self.input_queue).reshape(-1, self.options.INPUT_VERTICAL_DIM * self.options.INPUT_DATA_DIM) # 1직선으로 편다
				#self.input_queue = copy.deepcopy(self.input_queue.reshape(-1, self.options.INPUT_VERTICAL_DIM * self.options.INPUT_DATA_DIM))

				# @Training 부분
				if self.test_query == 0: # 테스트 아닌 학습하는 것일 때
					_  = sess.run(self.train , feed_dict={self.observation : self.input_queue})
				err = sess.run(self.loss_tot, feed_dict={self.observation : self.input_queue})
				# print('ERR : ', err)
				# input('?')
				self.err_list.append(err)
		

	def observation_to_test(self, inputz):
		with self.GRAPH.as_default() as g:
			with self.sess.as_default() as sess:
				# @ 자료 shape 변환
				self.input_queue = np.array(inputz, dtype=np.float32)
				self.input_queue = np.array(self.input_queue).reshape(-1, self.options.INPUT_VERTICAL_DIM * self.options.INPUT_DATA_DIM) # 1직선으로 편다

				# @Test 부분
				err = sess.run(self.loss_tot, feed_dict={self.observation : self.input_queue})

				self.err_list.append(err)

	def observation_to_predict(self, inputz):
		with self.GRAPH.as_default() as g:
			with self.sess.as_default() as sess:
				# @ 자료 shape 변환
				self.input_queue = np.array(inputz, dtype=np.float32)
				self.input_queue = np.array(self.input_queue).reshape(-1, self.options.INPUT_VERTICAL_DIM * self.options.INPUT_DATA_DIM) # 1직선으로 편다

				#@ encoding 출력 부분
				encoded = sess.run(self.encoding, feed_dict={self.observation : self.input_queue})
				hypothesis = sess.run(self.hypothesis, feed_dict={self.observation : self.input_queue})

				return encoded, self.input_queue, hypothesis

	def load_model(self):
		# global self.test_query
		# global self.sess
		# curr_dir1 = str(os.getcwd() + "\\checkpoints-LSTM_lang")
		# self.curr_dir = os.path.join(curr_dir1, "lstm_lang-DQN")
		print(self.curr_dir)
		# question = input('directory check')
		# tensor save되는 폴더
		# saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(self.curr_dir_folder)
		if not os.path.isdir(self.curr_dir_folder):
			# 존재하지 않는 폴더라면
			os.mkdir(self.curr_dir_folder)
			# 만들어줌
			print("▼" * 50)
			print('Could not find old network weights and added a new folder')
			print("▼" * 50)

		else:
			# 존재한다면
			if checkpoint and checkpoint.model_checkpoint_path:
				# 둘다 체크 들어가는데 폴더는 있어야 할듯
				with self.GRAPH.as_default() as g:
					with self.sess.as_default() as sess:
						self.saver.restore(sess, checkpoint.model_checkpoint_path)
						# restore 하기 전에 init 하지 마라는데 무슨 뜻??
						print("★" * 50)
						print('Successfully loaded : ', checkpoint.model_checkpoint_path)
						print("★" * 50)

			else:
				print("※" * 50)
				print('Just proceeding with no loading but existing directory')
				print("※" * 50)

	def save_model(self):
		if not self.err_list : # 빈 리스트
			return ValueError('Empty score list')

		self.score =  1 / (sum(self.err_list)/len(self.err_list) )
		# self.score_now_max = self.score
		self.score_board()
		if self.score_now_max <= self.score:
			self.score_now_max = self.score
			if self.score_now_max > self.score_board_value :
				if self.test_query == 0: #테스트 아닐 시
					with self.GRAPH.as_default() as g:
						with self.sess.as_default() as sess:
							self.saver.save(sess, self.curr_dir, self.global_step)
							print('model has been saved....')
		self.score_board()

	def print_result(self):

		os.system('cls')  # window
		os.system('cls' if os.name == 'nt' else 'clear')
		#print('accuracy : ', 1 - sum(self.err_list) / len(self.err_list))
		print('rmse true value : ', sum(self.err_list) / len(self.err_list))
		print('rmse err score : ', 1/(sum(self.err_list) / len(self.err_list)) )
		print('total ', self.global_step, 'th number of steps iterated while training...')

		return self.score_list

	def init_model(self):
		#####
		# @reset
		self.score = 0
		self.score_now_max = 0
		self.score_board_value = 0
		self.lv_score_board_fix = 0
		self.score_list = []
		self.err_list = []

	def score_board(self):
		# global self.content, self.curr_dir, self.curr_dir_f, self.lv_score_board_fix, self.score_now_max, self.score
	# 점수를 파일로 저장해서, 지금까지 돌린거 중에 가장 좋은 score 남도록 한다
	# 그리고 가장 높았던 점수를 리턴해준다
		self.content = None
		# 점수 마커
		# curr_dir = str(os.getcwd() + "\\LSTM_LANG")
		# # 지금 작업중인 경로
		# self.curr_dir_f = str(curr_dir) + "\\lstm_lang_score.txt"
		# 파일까지의 경로
		file_exist = os.path.isfile(self.score_dir_txt)
		# 논리값 return, 파일이 있는가 없는가

		def write():
			if file_exist : # 파일 존재하면
				f = open(self.score_dir_txt, 'r')
				f.seek(0)
				self.content = f.read()
				f.close()

				if not(self.content == ""): # null이 아니면
					if self.lv_score_board_fix == 0 :
						self.score_board_value = float(self.content) # 에피 끝날 떄 까지 고정값
						self.lv_score_board_fix = 1
					if float(self.content) >= self.score_now_max: #저장 값이 크다
						pass
					else:
						with open(self.score_dir_txt, 'w+') as out_file :
						# data is over-written with w+ parameter so use r+
							out_file.seek(0) # goint to top of the file just in case
							out_file.write(str(self.score_now_max)) # 값을 쓴다
								#G.score_board_value = G.score
				else: # 파일있는데 null인 경우
					with open(self.score_dir_txt, 'w+') as out_file:
						out_file.write(str(self.score))
						self.score_board_value = self.score
			else: # 파일 생성만 하고 끝난다
				with open(self.score_dir_txt, 'w+') as out_file :
					out_file.write(str(self.score))
					self.score_board_value = self.score

		if not os.path.isdir(self.score_dir):
		# 존재하지 않는다면
			os.mkdir(self.score_dir)
			write()
			# 경로를 만들어준다
		else:
		# 폴더가 존재하면
		# 바로 파일 작업으로 넘어감
			write()
			
def get_pickle():

	folder_path = os.getcwd()
	print('folder path : ', folder_path)
	pickle_path = str(folder_path + "\\ENCODER__test\\pickle.p").replace('/', '\\')
	pickle_path_2 = str(folder_path + "\\ENCODER__test\\pickle_test_only.p").replace('/', '\\')
	print('pickle path : ', pickle_path)
	print('pickle path : ', pickle_path_2)

	copy_obj = None
	copy_obj_2 = None

	with open(pickle_path, 'rb') as file:
		#copy_obj = copy.deepcopy(file)
		#global copy_obj
		copy_obj = copy.deepcopy(pickle.load(file))

	with open(pickle_path_2, 'rb') as file_2:
		#copy_obj = copy.deepcopy(file)
		#global copy_obj
		copy_obj_2 = copy.deepcopy(pickle.load(file_2))

	if copy_obj == None or copy_obj_2 == None :
		raise ValueError('wrong get pickle...')
	else:
		print('successfully loaded pickle file...')
		return copy_obj, copy_obj_2

def list_sorter(list_obj):
	"""
	삭제 이전 소팅해서 먼거, 오래된거 삭제하는 방식
	"""
	pass
	
def zero_padding(list_obj):
	
	"""
	없는애 time : 20, score : 0 로 해서 완전 옛날 자료  + 점수 다르다는걸 보여주기
	"""
	tmp_list_original = copy.deepcopy(list_obj)
	tmp_list_len = len(list_obj)
	if tmp_list_len < 100:
		for i in range(100-tmp_list_len):
			tmp_list_original.append([20,0])
	
	elif tmp_list_len == 100:
		pass # skip, it is already fulfilled

	elif tmp_list_len > 100:
		# sorting 필요, 그 후에 자른다
		tmp_list_original = sorted(tmp_list_original, key=lambda x: x[0])[:100]

	return tmp_list_original

def parse_four_days(specific_time, stock_name, pickle_data, judge_zero_article_bool = False):
	"""
	@ specific_time
	datetime.datetime.now() 
	 - ex) 2020-06-02 12:15:02.901542 의 형태로 넣어주어야 함!
	
	@ 설명
	article 가장 최근 ~ 가장 늦은 list 받고,
	해당 stock이름
	작업할 데이터 가져옴
	return :: 4일치 가져오고, padding 해준다
	"""
	"""
	
	@ dictionary 형태
	{
	    종목이름1 : [ 종목코드 , { 기사주소 : [ ['2020-04-29 15:25', 0.23208051174879074, 기사내용], ... , [,,] ] } ] 
	  , 종목이름2 : ...
	  , ....
	}
	
	autoencoder input은 300개 -> 150 개 기사
	"""
	time_window_now__obj = specific_time # 지금 시점
	time_window_start__obj = (specific_time - datetime.timedelta(days=4)).replace(hour=0, minute=0) # 4일 전 정각
	
	tmp_result_list = [] # 임시 해당 부분 담을 list
	article_hash = pickle_data[stock_name][1] #기사 hash list
	
	for article in article_hash:
		tmp_article_list = article_hash[article] # hash value 값 list 전체 
		for i in range(len(tmp_article_list)):
			tmp_article_list_date__obj = datetime.datetime.strptime(tmp_article_list[i][0], "%Y-%m-%d %H:%M")
			if tmp_article_list_date__obj >= time_window_start__obj and tmp_article_list_date__obj <= time_window_now__obj : # 4일치 window에 포함된다면
				tmp_time_delta = time_window_now__obj - tmp_article_list_date__obj
				tmp_duration_in_sec = tmp_time_delta.total_seconds()
				#tmp_duration_in_hour = tmp_duration_in_sec / (60*60)
				tmp_duration_in_hour = return_exp( tmp_duration_in_sec / (60*60) )
				tmp_result_list.append( [tmp_duration_in_hour, (tmp_article_list[i][1] + 1)*10 ] ) # 점수 10~20사이로 세팅, 10이 lowest, 20가 maximum socre

	# print(f'tmp_result_list : {tmp_result_list}')
	# print(f'zero_padding( tmp_result_list ) : {zero_padding( tmp_result_list )}')
	# print(f'reshaper(zero_padding( tmp_result_list )) : {reshaper(zero_padding( tmp_result_list ))}')
	if judge_zero_article_bool == True:
		if len(tmp_result_list) == 0:
			raise RuntimeError('parse_four_days - error!') # ValueError
		else:
			return reshaper(zero_padding(tmp_result_list))
	else:
		return reshaper(zero_padding(tmp_result_list))

	
	# 100개 넘기면 짜르기
	# padding
	# return

def reshaper(list_obj):
	tmp_list_np__obj = np.asarray(list_obj)
	#print(f'tmp_list_np__obj :: {list_obj}')
	tmp_list_np__obj.tolist()

	return tmp_list_np__obj.tolist()


def return_exp(value):
	"""

	:param value:
	:return: set the return value to flat out in 20
	"""
	if value <= 120:
		return value/6
	if value > 120:
		return 20

			
def parse_date_min_max_article_hash(stock_name, pickle_data):
	"""
	전체 날짜별 min ~ max 값 가져오기
	
	-> dictionary 형태
	{
	    종목이름1 : [ 종목코드 , { 기사주소, [ ['2020-04-29 15:25', 0.23208051174879074, 기사내용], ... , [,,] ] } ] 
	  , 종목이름2 : /...
	  , ....
	}
	"""
	
	# @ 전체 min max 날짜 가져옴
	date_list = []
	tmp_list_1 =  pickle_data[stock_name] # 종목이름 하위 list
	for key_2 in tmp_list_1[1]: # 기사 주소가 key값
		tmp_list_2 = tmp_list_1[1][key_2] # -> [ ['2020-04-29 15:25', 0.23208051174879074, 기사내용], ... , [,,] ]
		for item in (tmp_list_2):
			date_list.append(item[0])
	
	# @ sorting 함
	"""
	list 0 가 가장 latest 시간
	"""
	for i in range(0, len(date_list)-1, 1):
		datetime_obj_1 = datetime.datetime.strptime(date_list[i], "%Y-%m-%d %H:%M")
		datetime_obj_2 = datetime.datetime.strptime(date_list[i+1], "%Y-%m-%d %H:%M")
		
		if datetime_obj_1 < datetime_obj_2: # 이전값이 이후 값 보다 빠른 시간일 때
			tmp_list_item = copy.deepcopy(date_list[i+1])
			date_list[i+1] = date_list[i]
			date_list[i] = tmp_list_item
	
	return date_list
			
			
def Session_2(get_saved_list = False): # real - parsed from minute-base
	"""
	값은 0 / 1~2로 양분해야됨 0 : padding용, 1~2는 점수, 1은 낮음 2는 높음
	"""
	encoder = Autoencoder(module=False)

	#input('?')
	
	"""
	10.8 넘기면 됨
	0) K 종목에대해
	1) 데이터 전체 timeframe 구함 - A
	2) A 시간을 시간순으로 4일치를 구하여, 특정 시점으로부터 4일간의 데이터 모두의 분 delta  + 점수를 가져온다
	3) data feed 시켜 학습
	4) 1~3이 A가 끝날 때 까지 반복
	5) 4)를 전체 종목에 대해 반복
	----------------------------------
	ex)) 삼성전자 등에 대해 4일치 데이터 뽑았을 때 가장 많은 or 적절한 크기의 input 크기 ex) 40? 결정해야 한다
	ex2) 한 날짜에 대해 4일치 list 돌려주는 - padding 및 점수 변환까지 다해서 - 를 구현하면 될듯 : ***********************
	"""

	if get_saved_list == False:
		pickle_data, test_only_data = copy.deepcopy(get_pickle())
		tmp_list_4days_article_number = []
		num_count = 1
		start_time = time.time()
		for key_1 in pickle_data: # pickle_data[0] : 주식 종목코드
			tmp_start_time = time.time()
			print(num_count,'- th loop of the for-loop ... of ', str(key_1) )
			if num_count == 50:
				break
			# @ 전체 기사 timeframe 가져옴
			article_date_list = parse_date_min_max_article_hash(key_1, pickle_data)

			# @ timeframe 시작 / 끝
			# datetime.datetime.strptime(tmp_article_list[i][0], "%Y-%m-%d %H:%M")
			finish_date = datetime.datetime.strptime(article_date_list[0], "%Y-%m-%d %H:%M").replace(hour=15, minute=30)
			start_date =  datetime.datetime.strptime(article_date_list[len(article_date_list) - 1], "%Y-%m-%d %H:%M").replace(hour=9, minute=0)
			#print(tmp_return_list)
			#input('???')

			"""
			4일치 가져오는 함수 : ***********************
			"""
			time_now = start_date
			i = 0
			while time_now <= finish_date : # 끝까지
				if i % 10000 == 0:
					print('    +++time_now    :: ', time_now)
					print('    +++finish_date :: ', finish_date)
				tmp_list_4days_article_number.append( parse_four_days(time_now, key_1, pickle_data) )
				time_now = copy.deepcopy(time_now + datetime.timedelta(minutes=60))


				i = i + 1

			num_count = num_count + 1

		tmp_save_list_dir = os.getcwd().replace('/', '\\') + '\\ENCODER__test_train_list\\pickle_list.p'
		with open(tmp_save_list_dir , 'w+b') as file:
			pickle.dump(tmp_list_4days_article_number, file)
			print('pickle successfully saved...!')

	else:
		tmp_save_list_dir = os.getcwd().replace('/', '\\') + '\\ENCODER__test_train_list\\pickle_list.p'
		with open(tmp_save_list_dir, 'rb') as file:
			tmp_list_4days_article_number = copy.deepcopy(pickle.load(file))

	print(f'first 2 list of file : {tmp_list_4days_article_number[:2]}')
	#for items in tmp_list_4days_article_number:
	X_train = tmp_list_4days_article_number[:int(len(tmp_list_4days_article_number)*0.7)]
	X_test = tmp_list_4days_article_number[int(len(tmp_list_4days_article_number)*0.7):]
	for i in range(encoder.options.MAX_EPISODE):
		print(f' i th episode ... {i+1}')
		start_time = time.time()
		encoder.observation_to_train(X_train)
		encoder.print_result()
		encoder.init_model()

		print("--- %s seconds ---" % (time.time() - start_time))

		print("\n" * 1)
		print('begin testing model...')
		start_time = time.time()
		encoder.observation_to_test(X_test)
		tmp_ans_list = encoder.print_result()
		encoded, flattened_input, hypothesis = encoder.observation_to_predict(X_test[1])
		print('original : ',flattened_input)
		print('predict : ', hypothesis)
		print('encoded : ', encoded)

		encoder.save_model()
		encoder.init_model()
		print("--- %s seconds ---" % (time.time() - start_time))
		print('+' * 60)
		print('+' * 60)
		print("\n" * 3)




###########################################################################################
###########################################################################################
###########################################################################################

def Session(): # training 하는 부분
	encoder = Autoencoder(module=False)
	pickle_data , test_only_data = copy.deepcopy(get_pickle())
	# print(test_only_data)
	# input('^^')
	
	tmp_in = None
	while tmp_in != 'x' or tmp_in != 'X':
		tmp_in = str(input('put in the company you like to search :: '))
		try:
			result= pickle_data[tmp_in]
			print(result)
			print('length of article data :: ',tmp_in,' is :: ', len(list(result[1].keys() )))
		except:
			print('wrong input... try again..!!!')

	tmp_data_len = []
	tmp_count = 0
	tmp_article_count = []
	for key_1 in pickle_data :
		tmp_article_count.append(int(len(pickle_data[key_1][1].keys())))
		tmp_count = 0
		for key_2 in pickle_data[key_1][1]:
			tmp_count = tmp_count + len(pickle_data[key_1][1][key_2])

		tmp_data_len.append(int(tmp_count))

		if tmp_count == 0:
			print('pickle_data 0 : ', pickle_data[key_1])

	tmp_data_len.sort()

	print('1 : ', tmp_data_len[:10])
	print('2 : ', tmp_data_len[-10:-1])

	tmp_article_count.sort()
	print('1 article : ', tmp_article_count[:10])
	print('2 article : ', tmp_article_count[-10:-1])
	
	print(pickle_data['삼성전자'])

	print('begin creating test data .....!!!!!')

	# real_data = copy.deepcopy(parse_data(pickle_data))
	# minute_conversion = copy.deepcopy(str_date_to_min(real_data))
	# padded_data = copy.deepcopy(padding(minute_conversion))
	#
	# test_only_data_TO = copy.deepcopy(parse_data(test_only_data))
	# minute_conversion_TO = copy.deepcopy(str_date_to_min_test(test_only_data_TO))
	# padded_data_TO = copy.deepcopy(padding(minute_conversion_TO))

	# check_list = []
	# for article_conv in padded_data:
	# 	check_list.append(len(article_conv))

	# print(check_list)

	# a3d = np.array(padded_data)
	# print('a3d : ', a3d[:2])
	# print('a3d type : ', type(a3d))
	# print(a3d.shape)
	# print('a3d len : ', len(a3d))
	# print('a3d[0] len : ', len(a3d[0]))
	# print('a3d[0][0] len : ', len(a3d[0][0]))
	# # input('?')
	#
	#
	# # padded_data shape analysis
	#
	#
	#
	# # Training begin
	# #=====================================================================
	# # =====================================================================
	# # =====================================================================
	# print('check for max time in the data...')
	# max_len = []
	# for item in padded_data:
	# 	for item_2 in item:
	# 		max_len.append(item_2[0])
	#
	# print('max time... : ', max(max_len))
	#
	#
	# data_length = int(len(padded_data) * 0.8 )
	# X_train = padded_data[ : data_length]
	# X_test = padded_data[data_length : ]

	print('\n'*10)

	print('check if it works..!')
	# x_test_single = X_test[:4]
	# for i in range(len(x_test_single)):
	# 	encoded, flattened_input,  hypothesis = encoder.observation_to_predict(x_test_single[i])
	# 	print(i+1,'th test...')
	# 	print('original : ',flattened_input)
	# 	print('predict : ', hypothesis)
	# 	print('encoded : ', encoded)
	# 	print('+' * 70)
	# 	print('+' * 70)
	# print('\n'*3)
	#
	#
	# print('total data for training : ', len(X_train))
	# print('total data for testing : ', len(X_test))
	# print('\n' * 3)
	#
	# print('execute only test data for analysis!!!')
	# print('@'*50)
	# print('@' * 50)
	# encoder.observation_to_test(padded_data_TO)
	# encoder.print_result()
	# encoder.init_model()
	# print('@' * 50)
	# print('@' * 50)
	#
	# input('press any to continue ...')
	#
	#
	# for i in range(encoder.options.MAX_EPISODE):
	# 	print(i + 1, 'th epoch has started ...')
	# 	start_time = time.time()
	# 	encoder.observation_to_train(X_train)
	# 	encoder.print_result()
	# 	# lstm_lang.save_model()
	# 	encoder.init_model()
	# 	print("--- %s seconds ---" % (time.time() - start_time))
	#
	# 	print("\n" * 1)
	# 	print('begin testing model...')
	# 	start_time = time.time()
	# 	encoder.observation_to_test(X_test)
	# 	tmp_ans_list = encoder.print_result()
	#
	#
	# 	encoder.save_model()
	# 	encoder.init_model()
	# 	print("--- %s seconds ---" % (time.time() - start_time))
	# 	print('+' * 60)
	# 	print('+' * 60)
	# 	print("\n" * 3)





if __name__ == '__main__':
	#Session_2(get_saved_list = False)
	Session_2(get_saved_list = True)