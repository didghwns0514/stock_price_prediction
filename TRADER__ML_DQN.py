#-*-coding: utf-8-*-
#######################sublime text 2용 hotkey##############
#ctrl + / => block comment hotkey
#shift + tab or tab => indent hotkey

########################import py field####################
import tensorflow as tf
#import pandas
import random
import numpy as np
#import time
import os
from itertools import chain
import math
import copy
import datetime

########################import field#######################
#import state
########################global value#######################
#import Global as G



###########################################################

class Options :
	def __init__(self, env):
		# 인자들 전부 전달해주는 부분
		self.OBSERVATION_DIM = env[0] # input 디멘션
		self.H1_SIZE = env[1] # size of hidden layer 1
		self.H2_SIZE = env[2] # size of hidden layer 2
		self.H3_SIZE = env[3] # size of hidden layer 3
		self.ACTION_DIM = env[4] # number of actions to take
		
		self.MAX_EPISODE = env[5] # max number of episodes iteration
		self.GAMMA = env[6] # discount factor of Q learning
		self.INIT_EPS = env[7] # initial probability for randomly sampled action
		self.FINAL_EPS = env[8] # final probability for randomly sampled action
		self.EPS_DECAY = env[9] # epsilon decay rate
		self.EPS_ANNEAL_STEPS = env[10] #steps of intervals to decay epsilon
		self.LR = env[11] # learning rate
		self.MAX_EXPERIENCE = env[12] # size of experience replay memory
		self.BATCH_SIZE = env[13] # mini batch size

		self.DROPOUT_KEEP_RATE = env[14] # keep 할 dropout rate 지정

def FUNC_TF__environment_check():
	
	print('tensorflow version : ', tf.VERSION)

	
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	### tensorflow에서 사용가능한 device list 출력
	
	if tf.test.is_gpu_available() == False :
	#사용가능한 gpu있으면 True
		print('._No Available GPU_.' * 5)
	else :
	#참일 떄
		print('GPU is available' * 3)
		
	#proceed = input('hit anything to proceed')
	print('\n'*5)


class Trade_agent :
	#MAX_SCORE_QUEUE_SIZE = 3000
	MAX_SCORE_QUEUE_SIZE = 8000
	MAX_OBSERVE_STOCK_SIZE = 60 * 1
	MAX_OBSERVE_FOR_ALL = MAX_OBSERVE_STOCK_SIZE + 2 + 2
	# -> 0: 60분 * 2(2시간) * 2 (price & volume) + 주식 보유개수 + 현재 budget / 보유주식 평단가, 시작 budget
	STOCK_FEE = 0.34792/100

	#                        0,   1,   2,   3,  4,   5,    6,  7,    8,    9,   10,      11,    12,   13  14
	env = [MAX_OBSERVE_FOR_ALL, 400, 200, 200,  3, 100, 0.95,  1, 1e-5, 0.95,  700, 0.00003,  8000, 2000,  1]
	#env = [MAX_OBSERVE_FOR_ALL, 600, 300, 150,  3, 200, 0.95,  1,  1e-5, 0.95,  700, 1e-4, 7000, 1000,  1]
	# -> random 일부러 엄청 안주고 하면 바로 찾았던거로 기억... why? IDK

	def __init__(self, module = True):

		self.options = Options(self.env)

		# option으로 뭔가를 해주는 듯, 일단 input 차원만큼 넣어준다, 매 layer 옵션 다 들어있는 듯
		FUNC_TF__environment_check()

		self.config = tf.ConfigProto(
			device_count={'GPU': 0}
		)

		if module == True:
			self.options.DROPOUT_KEEP_RATE = 1
			self.test_query = 2
		else:
			asked = input('0 : train model \n1 : profit graph \n2 : transaction graph + no neuron drop')
			# to run test or not when it is loaded
			try :
				if int(asked) == 1:
					self.options.DROPOUT_KEEP_RATE = 0.75
					self.test_query = 1
				elif int(asked) == 2:
					self.options.DROPOUT_KEEP_RATE = 1
					self.test_query = 2
				elif int(asked) == 0 :
					self.options.DROPOUT_KEEP_RATE = 0.80 # 0.75
					#self.options.DROPOUT_KEEP_RATE = 0.75
					self.test_query = 0
				else:
					raise ValueError(' wrong number input, of neuron keep prob')
			except ValueError :
				print(' worng number input, proceeding w/o TEST')
				self.options.DROPOUT_KEEP_RATE = 1

		self.GRAPH = tf.Graph()
		#self.sess = tf.compat.v1.InteractiveSession()
		if module == True: # 다른곳에서 쓰일 때
			self.sess = tf.compat.v1.InteractiveSession(config=self.config, graph=self.GRAPH)
			#self.sess = tf.Session(config=self.config) # gpu - off for memory safe calculation
		else:
			self.sess = tf.compat.v1.InteractiveSession(config=self.config, graph=self.GRAPH)
			#self.sess = tf.compat.v1.InteractiveSession()



		with self.GRAPH.as_default():
			self.w1 = self.FUNC_BUILD__weight([self.options.OBSERVATION_DIM, self.options.H1_SIZE])
			self.b1 = self.FUNC_BUILD__bias([self.options.H1_SIZE])
			self.w2 = self.FUNC_BUILD__weight([self.options.H1_SIZE, self.options.H2_SIZE])
			self.b2 = self.FUNC_BUILD__bias([self.options.H2_SIZE])
			self.w3 = self.FUNC_BUILD__weight([self.options.H2_SIZE, self.options.H3_SIZE])
			self.b3 = self.FUNC_BUILD__bias([self.options.H3_SIZE])
			self.w4 = self.FUNC_BUILD__weight([self.options.H3_SIZE, self.options.ACTION_DIM])
			self.b4 = self.FUNC_BUILD__bias([self.options.ACTION_DIM])

			self.obs, self.Q1 = self.FUNC_BUILD__make_value_net(self.options)
			self.next_obs, self.Q2 = self.FUNC_BUILD__make_value_net(self.options)
			self.act = tf.compat.v1.placeholder(tf.float32, [None, self.options.ACTION_DIM])
			self.rwd = tf.compat.v1.placeholder(tf.float32, [None, ])
			self.values1 = tf.reduce_sum(tf.multiply(self.Q1, self.act), reduction_indices=1)
			self.values2 = self.rwd + (self.options.GAMMA * tf.reduce_max(self.Q2, reduction_indices=1))
			self.loss = tf.reduce_mean(self.FUNC_BUILD_ML__clipped_error(tf.square(self.values1 - self.values2)))
			self.train_step = tf.train.AdamOptimizer(self.options.LR).minimize(self.loss)


			self.obs_queue = np.empty([self.options.MAX_EXPERIENCE, self.options.OBSERVATION_DIM])
			self.act_queue = np.empty([self.options.MAX_EXPERIENCE, self.options.ACTION_DIM])
			self.rwd_queue = np.empty([self.options.MAX_EXPERIENCE])
			self.next_obs_queue = np.empty([self.options.MAX_EXPERIENCE, self.options.OBSERVATION_DIM])

			self.saver = tf.train.Saver()

			self.sess.run(tf.global_variables_initializer())
			self.load_model()  # load the model in the class


		#__________________________________________________________________________________
		#################################################
		self.STOCK_VARIABLE__current_price = 0
		self.STOCK_VARIABLE__inventory = []

		self.STOCK_VARIABLE__current_budget = 0
		self.STOCK_VARIABLE__started_budget = 0 # sell_all 이후 사용할 변수
		self.STOCK_VARIABLE__action_dictionary = {'buy_success':0, 'sell_success':0, 'hold_success':0, 'buy_failure':0, 'sell_failure':0} # 시행했던 dict
		self.STOCK_VARIABLE__state_profit = 0 # profit 계산하는 부분

		#################################################
		#----------------------------------------------------------------------------------
		self.AT_OBS__input = []
		self.AT_OBS__additional = []
		self.AT_OBS__total = []
		self.AT_REWARD = None
		self.AT_ACTION = None
		
		self.AT_EPS__initial = self.options.INIT_EPS
		self.GLOBAL = 0
		
		self.TRAINING_DONE_COUNTER = 0
		self.TRAINING_DO_FLAG = False
		
		self.AT_EXP__pointer = 0
		self.AT_DICTIONARY__feeder = {}
		self.AT_STEP__loss = None
		self.test_query = 0
		self.AT_DICTIONARY__action_result = {}
		#################################################
		"""
		test query = 0 : 학습
		test query = 1 : profit 만 기록 -> 실제 agent 내부에서 하는 건 없음
		test query = 2 : profit 기록 + 학습 없이 trading graph만 기록
		"""
		#################################################

		
		# @ socre 저장시 쓰이는 변수들
		self.AT_SAVE__content = None
		self.AT_SAVE_PATH__folder = None
		self.AT_SAVE_PATH__file_to_txt = None
		self.AT_FLAG__score_board_fixed = 0
		self.AT_SAVE__current_max_score = 0
		self.AT_SAVE__score = 0
		self.AT_SAVE__txt_score_value = 0
		self.AT_PRINT__error_list = []


	def FUNC_BUILD_ML__clipped_error(self, error):
	# 후버 로스를 사용하여 error clip
		return tf.where(tf.abs(error) < 1.0, 0.5*tf.square(error), tf.abs(error) - 0.5)

	def FUNC_BUILD_ML__xavier_init(self, shape):
	# weight용 초기화 함수 - 자비어 방식
		dim_sum = np.sum(shape)
		if len(shape) == 1 : 
		# 1차원 짜리 행렬 이라면
			dim_sum = dim_sum + 1
			# 2 이상의 차원으로 생각???
		bound = np.sqrt(6.0/ dim_sum) 
		return tf.random_uniform(shape, minval = -bound, maxval=bound, dtype=tf.float32)
		# -3 ~ 3 아마 검증 된 부분인 듯
	
	
	def FUNC_BUILD__weight(self, Shape):
	# weight Variable 만들기 위함 - variable인 텐서는 값이 바뀔 수 있다
		return tf.Variable(self.FUNC_BUILD_ML__xavier_init(Shape))

	
	def FUNC_BUILD__bias(self, Shape):
	# bias Variable 만들기 위함
		return tf.Variable(self.FUNC_BUILD_ML__xavier_init(Shape))
	
	
	def FUNC_BUILD__make_value_net(self, options): #그래프들 def 들을 연결한다
	# adding options to graph - 차후 사용성을 위해서 input 용인 듯
		observation = tf.compat.v1.placeholder(tf.float32, [None, options.OBSERVATION_DIM])
		#tf.compat.v1.disable_eager_execution()
		h1 = tf.compat.v1.nn.relu(tf.matmul(observation, self.w1) + self.b1)
		h1 = tf.nn.dropout(h1, keep_prob = options.DROPOUT_KEEP_RATE)
		h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
		h2 = tf.nn.dropout(h2, keep_prob = options.DROPOUT_KEEP_RATE)
		h3 = tf.nn.relu(tf.matmul(h2, self.w3) + self.b3)
		#h3 = tf.nn.dropout(h3, keep_prob = options.DROPOUT_KEEP_RATE)
		Q = tf.squeeze(tf.matmul(h3, self.w4) + self.b4)
		#요소 갯수 재한 없도록 squeeze 사용, 마지막 output layer

		return observation, Q

	def FUNC_AT__observation_refresh(self, observation_hash):
		"""
		budget, 보유 개수랑 합산
		이걸 돌리고  FUNC_AT__observation_to_train 수행하면 됨
		"""

		# initalized before use
		self.AT_OBS__additional = []  # 쓰기 전 초기화
		self.AT_OBS__total = [] # 쓰기 전 초기화

		# @ observation update
		self.AT_OBS__input = self.FUNC_AT__reshape_obs_and_cur_price(observation_hash) # maybe shape change is needed.
		# print(f'observation_hash : \n{observation_hash}')
		# print(f'AT_OBS__input : \n{self.AT_OBS__input}')
		# input('&')

		# @ 추가 정보 기입
		self.AT_OBS__additional.append(self.STOCK_VARIABLE__current_budget)
		self.AT_OBS__additional.append(self.STOCK_VARIABLE__started_budget)
		if len(self.STOCK_VARIABLE__inventory) != 0:
			self.AT_OBS__additional.append(sum(self.STOCK_VARIABLE__inventory)/len(self.STOCK_VARIABLE__inventory))
		else:
			self.AT_OBS__additional.append(0)
		self.AT_OBS__additional.append(len(self.STOCK_VARIABLE__inventory))
		
		# joined_list = [*l1, *l2]
		#observation_tmp =  list(chain.from_iterable(self.AT_OBS__input)) # one dimention vector
		#self.AT_OBS__total = [*observation_tmp, *self.AT_OBS__additional] # concatenate two lists
		self.AT_OBS__total = copy.deepcopy([*self.AT_OBS__input, *self.AT_OBS__additional]) # concatenate two lists
		# print(f'len of self.AT_OBS__total : {len(self.AT_OBS__total)}')
		#print(f'self.AT_OBS__total : {self.AT_OBS__total}')
		#input('^')
		# print(f'current budget : {self.STOCK_VARIABLE__current_budget}')
		# print(f'current inventory : {self.STOCK_VARIABLE__inventory}')
		# input('#\n'*2)

	def FUNC_AT__reshape_obs_and_cur_price(self, observation_hash):
		"""
		observation_hash : 그냥 가져온거 전체 다 넣어도 됨. missing point만 없으면 됨
		observation np reshape 작업 및, 현재 price 기록하는 부분 : self.STOCK_VARIABLE__current_price

		as part of hash ->
		stock_code : {  {date_stamp : {price : AAA, volume : BBB}}  , ...}
		"""
		# 정렬
		tmp_list_date_stamp = list(observation_hash.keys())
		tmp_list_date_stamp.sort()
		#print(f'^^^^ len of tmp_list_date_stamp : {len(tmp_list_date_stamp)}')

		# 가장 최신, 늦는 데이터 구별
		tmp_most_recent_date = tmp_list_date_stamp[-1]
		tmp_most_past_date = tmp_list_date_stamp[0]
		# 지금 price update
		self.STOCK_VARIABLE__current_price = observation_hash[tmp_most_recent_date]['price']
		
		tmp_list_for_return = []

		for i in range(len(tmp_list_date_stamp)-1,-1,-1):
			tmp_price = observation_hash[tmp_list_date_stamp[i]]['price']
			#tmp_volume = observation_hash[tmp_list_date_stamp[i]]['volume']
			#tmp_list_for_return.insert(0,tmp_volume)
			tmp_list_for_return.insert(0,tmp_price)
			
			if len(tmp_list_for_return) >= self.MAX_OBSERVE_STOCK_SIZE:
				break
		
		#print(f'$$$$ len of tmp_list_for_return : {len(tmp_list_for_return)}')
		
		return tmp_list_for_return
		

	def FUNC_AT__update_inner_condiiton(self, action):
		"""
		action 수행 이후마다, 내부 변수 업데이트 함

		self.STOCK_VARIABLE__inventory = []
		self.STOCK_VARIABLE__current_budget = 0

		action 0 : buy,  action 1 : sell,  action 2 : hold
		"""
		if action == 0: # buy
			if self.STOCK_VARIABLE__current_budget - self.STOCK_VARIABLE__current_price*(self.STOCK_FEE+1) < 0:
				return False
			else:
				self.STOCK_VARIABLE__current_budget = self.STOCK_VARIABLE__current_budget - self.STOCK_VARIABLE__current_price*(self.STOCK_FEE+1)
				self.STOCK_VARIABLE__inventory.append(self.STOCK_VARIABLE__current_price)
				return True
		elif action == 1 : # sell
			if len(self.STOCK_VARIABLE__inventory) == 0:
				return False
			else:
				self.STOCK_VARIABLE__inventory.pop(0)
				self.STOCK_VARIABLE__current_budget = self.STOCK_VARIABLE__current_budget + self.STOCK_VARIABLE__current_price*(1-self.STOCK_FEE)
				return True
		elif action == 2 : # hold
			return True
	
	def FUNC_AT__observation_to_prediction(self, observation_hash):
		# @ initialize
		action = None
		tmp_action_before = None
		tmp_action_counter = 0

		tmp_next_action_bool = True
		tmp_fist_enter_bool = True
		self.STOCK_VARIABLE__action_dictionary = {'buy_success': 0, 'sell_success': 0, 'hold_success': 0,
												  'buy_failure': 0, 'sell_failure': 0}

		# 1)
		#### observation들 합친다
		# self.FUNC_AT__observation_refresh(observation_hash)

		# @ for at learning!
		# self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total

		"""
		test query = 0 : 학습
		test query = 1 : profit 만 기록 -> 실제 agent 내부에서 하는 건 없음
		test query = 2 : profit 기록 + 학습 없이 trading graph만 기록
		"""
		# 2)
		# 여기서 반복해서 무슨 action을 몇번하는지 ...
		while tmp_next_action_bool == True:

			# @ counter update
			tmp_action_counter = tmp_action_counter + 1

			# a) refresh and get the action
			# --------------------------------------
			self.FUNC_AT__observation_refresh(observation_hash)
			action = self.FUNC_AT__action_return(self.Q1, {self.obs: np.reshape(self.AT_OBS__total, (1, -1))},
												 self.AT_EPS__initial, self.options)


			# 전단계와 비교
			if tmp_fist_enter_bool == True:  # tmp_action_before 값 들어오기전에는 비교 안함
				tmp_fist_enter_bool = False
			else:
				if tmp_action_before != action:
					tmp_next_action_bool = False

					break
			# --------------------------------------
			tmp_inner_update = self.FUNC_AT__update_inner_condiiton(action)

			if tmp_inner_update == True:
				if action == 0 or action == 1:  # buy / sell
					if action == 0:
						self.STOCK_VARIABLE__action_dictionary['buy_success'] = self.STOCK_VARIABLE__action_dictionary[
																					'buy_success'] + 1
					elif action == 1:
						self.STOCK_VARIABLE__action_dictionary['sell_success'] = self.STOCK_VARIABLE__action_dictionary[
																					 'sell_success'] + 1
					# save the before action
					tmp_action_before = action

				elif action == 2:  # hold
					self.STOCK_VARIABLE__action_dictionary['hold_success'] = self.STOCK_VARIABLE__action_dictionary[
																				 'hold_success'] + 1
					break
			else:
				if action == 0:  # buy
					self.STOCK_VARIABLE__action_dictionary['buy_failure'] = self.STOCK_VARIABLE__action_dictionary[
																				'buy_failure'] + 1
				elif action == 1:  # sell
					self.STOCK_VARIABLE__action_dictionary['sell_failure'] = self.STOCK_VARIABLE__action_dictionary[
																				 'sell_failure'] + 1
				break

		else:  # reached w/o break -> unreachable
			pass

		# @ 저장된 hash 돌려줌
		# 		self.STOCK_VARIABLE__action_dictionary = {'buy_success': 0, 'sell_success': 0, 'hold_success': 0,
		# 												  'buy_failure': 0, 'sell_failure': 0}
		if self.STOCK_VARIABLE__action_dictionary['buy_success'] != 0:
			return ("BUY", self.STOCK_VARIABLE__action_dictionary['buy_success'] )

		elif self.STOCK_VARIABLE__action_dictionary['sell_success'] != 0:
			return ("SELL", self.STOCK_VARIABLE__action_dictionary['sell_success'] )

		elif self.STOCK_VARIABLE__action_dictionary['hold_success'] != 0:
			return ("HOLD", 1 )

		else:
			return ("HOLD", 1 )



		
	def FUNC_AT__observation_to_train(self, observation_hash):
		
		# @ initialize
		action = None
		tmp_action_before = None
		tmp_action_counter = 0

		tmp_next_action_bool = True
		tmp_fist_enter_bool = True
		self.STOCK_VARIABLE__action_dictionary = {'buy_success':0, 'sell_success':0, 'hold_success':0, 'buy_failure':0, 'sell_failure':0}

		self.STOCK_VARIABLE__state_profit = 0

		# 1) 
		#### observation들 합친다
		#self.FUNC_AT__observation_refresh(observation_hash)
		
		# @ for at learning!
		#self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total

		"""
		test query = 0 : 학습
		test query = 1 : profit 만 기록 -> 실제 agent 내부에서 하는 건 없음
		test query = 2 : profit 기록 + 학습 없이 trading graph만 기록
		"""
		# 2)
		#여기서 반복해서 무슨 action을 몇번하는지 ...
		while tmp_next_action_bool == True :
			
			# @ counter update
			tmp_action_counter = tmp_action_counter + 1
			
			# @ step increase
			self.GLOBAL = self.GLOBAL + 1
			
			# epsilon value decay
			if self.GLOBAL % self.options.EPS_ANNEAL_STEPS == 0 and self.AT_EPS__initial > self.options.FINAL_EPS :
				self.AT_EPS__initial = self.AT_EPS__initial * self.options.EPS_DECAY
			
			if self.GLOBAL % self.options.MAX_EXPERIENCE == 0:
				self.TRAINING_DO_FLAG = True
			
			# a) refresh and get the action
			#--------------------------------------
			self.FUNC_AT__observation_refresh(observation_hash)
			action = self.FUNC_AT__action_return(self.Q1, {self.obs : np.reshape(self.AT_OBS__total, (1,-1))}, self.AT_EPS__initial, self.options)
			
			# 전단계와 비교
			#print(f'action : {action} , tmp_action_before : {tmp_action_before}')
			#print(f'tmp_next_action_bool is a stop : {tmp_next_action_bool}, action counter : {tmp_action_counter}')
			if tmp_fist_enter_bool == True: # tmp_action_before 값 들어오기전에는 비교 안함
				tmp_fist_enter_bool = False
			else:
				if tmp_action_before != action:
					tmp_next_action_bool = False
					#print(f'tmp_next_action_bool is a stop : {tmp_next_action_bool}, action counter : {tmp_action_counter}')
					#input('^^')
					break
			#--------------------------------------
			tmp_inner_update = self.FUNC_AT__update_inner_condiiton(action)

			if tmp_inner_update == True:
				if action == 0 or action == 1 : # buy / sell
					if action == 0:
						self.STOCK_VARIABLE__action_dictionary['buy_success'] = self.STOCK_VARIABLE__action_dictionary['buy_success'] + 1
					elif action == 1:
						self.STOCK_VARIABLE__action_dictionary['sell_success'] = self.STOCK_VARIABLE__action_dictionary['sell_success'] + 1
					#print(f'#### action taken : {action}, bool : {tmp_inner_update}')

					# b) save previous at_obs_total - if it is a pass
					self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total
					self.act_queue[self.AT_EXP__pointer] = action
					self.AT_REWARD = self.FUNC_AT__action_to_reward((action, tmp_inner_update))
					self.rwd_queue[self.AT_EXP__pointer] = self.AT_REWARD
					self.FUNC_AT__observation_refresh(observation_hash) # update inner condition 덕분
					self.next_obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total
					
					# c)
					# exp_pointer 올린다
					self.AT_EXP__pointer = self.AT_EXP__pointer + 1
					if self.AT_EXP__pointer == self.MAX_SCORE_QUEUE_SIZE:
						self.AT_EXP__pointer = 0
					

					
					# save the before action
					tmp_action_before = action

				elif action == 2: # hold
					self.STOCK_VARIABLE__action_dictionary['hold_success'] = self.STOCK_VARIABLE__action_dictionary['hold_success'] + 1
					#print(f'#### action taken : {action}, bool : {tmp_inner_update}')

					# b) save previous at_obs_total - if it is a pass
					self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total
					self.act_queue[self.AT_EXP__pointer] = action
					self.AT_REWARD = self.FUNC_AT__action_to_reward((action, tmp_inner_update))
					self.rwd_queue[self.AT_EXP__pointer] = self.AT_REWARD
					self.FUNC_AT__observation_refresh(observation_hash)
					self.next_obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total # next_obs_queue
					
					# c)
					# exp_pointer 올린다
					self.AT_EXP__pointer = self.AT_EXP__pointer + 1
					if self.AT_EXP__pointer == self.MAX_SCORE_QUEUE_SIZE:
						self.AT_EXP__pointer = 0
						
					break
			else:
				if action == 0: # buy
					self.STOCK_VARIABLE__action_dictionary['buy_failure'] = self.STOCK_VARIABLE__action_dictionary['buy_failure'] + 1
				elif action == 1: # sell
					self.STOCK_VARIABLE__action_dictionary['sell_failure'] = self.STOCK_VARIABLE__action_dictionary['sell_failure'] + 1
				#print(f'#### action taken : {action}, bool : {tmp_inner_update}')
				# b) save previous at_obs_total - if it is a pass
				self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total
				self.act_queue[self.AT_EXP__pointer] = action
				self.AT_REWARD = self.FUNC_AT__action_to_reward((action, tmp_inner_update))
				self.rwd_queue[self.AT_EXP__pointer] = self.AT_REWARD
				self.FUNC_AT__observation_refresh(observation_hash) # action 취하고 나서 agent 환경 조금 변하니깐
				self.next_obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total # next_obs_queue
				
				# c)
				# exp_pointer 올린다
				self.AT_EXP__pointer = self.AT_EXP__pointer + 1
				if self.AT_EXP__pointer == self.MAX_SCORE_QUEUE_SIZE:
					self.AT_EXP__pointer = 0
					
				break

		else: # reached w/o break -> unreachable
			# b) save previous at_obs_total - if it is a fail
			self.obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total
			self.act_queue[self.AT_EXP__pointer] = action
			self.AT_REWARD = self.FUNC_AT__action_to_reward((action, tmp_inner_update))
			self.rwd_queue[self.AT_EXP__pointer] = self.AT_REWARD
			self.FUNC_AT__observation_refresh(observation_hash) # action 취하고 나서 agent 환경 조금 변하니깐
			self.next_obs_queue[self.AT_EXP__pointer] = self.AT_OBS__total # next_obs_queue
			
			# c)
			# exp_pointer 올린다
			self.AT_EXP__pointer = self.AT_EXP__pointer + 1
			if self.AT_EXP__pointer == self.MAX_SCORE_QUEUE_SIZE:
				self.AT_EXP__pointer = 0

		# profit 계산부분
		#self.STOCK_VARIABLE__state_profit = ( (self.STOCK_VARIABLE__current_budget + len(self.STOCK_VARIABLE__inventory)*self.STOCK_VARIABLE__current_price) - (self.STOCK_VARIABLE__started_budget) ) / ( self.STOCK_VARIABLE__started_budget )
		#self.STOCK_VARIABLE__state_profit = (self.STOCK_VARIABLE__current_budget + len(self.STOCK_VARIABLE__inventory) * self.STOCK_VARIABLE__current_price) - (self.STOCK_VARIABLE__started_budget)
		self.STOCK_VARIABLE__state_profit = (self.STOCK_VARIABLE__current_budget  - self.STOCK_VARIABLE__started_budget)


		# 3)
		#### training 부분
		if  self.TRAINING_DO_FLAG == True: # 이전에 큐에 담았던 부분들 모았다가(잠시 고정값)
			self.TRAINING_DO_FLAG = False
			self.TRAINING_DONE_COUNTER = self.TRAINING_DONE_COUNTER + 1
			rand_indexes = np.random.choice(self.options.MAX_EXPERIENCE, self.options.BATCH_SIZE)
			self.AT_DICTIONARY__feeder.update({self.obs : self.obs_queue[rand_indexes]})
			self.AT_DICTIONARY__feeder.update({self.act : self.act_queue[rand_indexes]})
			self.AT_DICTIONARY__feeder.update({self.rwd : self.rwd_queue[rand_indexes]})
			self.AT_DICTIONARY__feeder.update({self.next_obs : self.next_obs_queue[rand_indexes]})
			self.AT_STEP__loss, _ = self.sess.run([self.loss, self.train_step], feed_dict = self.AT_DICTIONARY__feeder)
			self.AT_PRINT__error_list.append(self.AT_STEP__loss)
			self.AT_DICTIONARY__feeder = {} # reset feed just in case

# 		# 2)
# 		# epsilon value decay
# 		if self.GLOBAL % self.options.EPS_ANNEAL_STEPS == 0 and self.AT_EPS__initial > self.options.FINAL_EPS :
# 			self.AT_EPS__initial = self.AT_EPS__initial * self.options.EPS_DECAY

		self.AT_OBS__total = [] # 리셋 시켜줌, 다시 쓰기 전에

	def FUNC_AT__action_return(self, Q, feed, eps, options):
	# Q는 전체 그래프, feed :?
	# eps : epsilon-greedy 위한 것
	# options : NN 개수 파라미터
		
		act_values = Q.eval(feed_dict=feed)
		# input으로 기억함, 이게 들어갔을 때 q의 output
		# action dimension 전체 
		
		if random.random() <= eps :
		# random action / Q value는 어차피 max로 업데이트 됨
			action_index = random.randrange(options.ACTION_DIM) # action_dim 바로 전 숫자까지
			
		else: 
			action_index = np.argmax(act_values)
			#max Q 따르는 action 고를 때
		
# 		action = np.zeros(options.ACTION_DIM)
# 		action[action_index] = 1
		# 고른 action을 1로 하자...
		
		# 0: buy, 1: sell, 2:hold
		return action_index


	def FUNC_AT__action_to_reward(self, tuple_action_return):
		"""
		self.AT_REWARD = self.FUNC_AT__action_to_reward((action, tmp_inner_update))
		"""
		reward = 0
		tmp_tuple_1, tmp_tuple_2 = tuple_action_return
		if tmp_tuple_2 == False : # wrong action under condition
			reward = - 1
			return  reward
		else:
			return math.log(self.STOCK_VARIABLE__current_budget) - math.log(self.STOCK_VARIABLE__started_budget)
			# if len(self.STOCK_VARIABLE__inventory) == 0:
			# 	return math.log(self.STOCK_VARIABLE__current_budget) - math.log(self.STOCK_VARIABLE__started_budget)
			# else:
			# 	return (math.log(self.STOCK_VARIABLE__current_budget) - math.log(self.STOCK_VARIABLE__started_budget))*len(self.STOCK_VARIABLE__inventory)

		# 	if tmp_tuple_1 != 2: # hold 가 아니면
		# 		tmp_reward = self.FUNC_AT__profit_to_reward() * len(self.STOCK_VARIABLE__inventory)
		# 		if tmp_reward < 0:
		# 			reward = reward + tmp_reward
		# 			#return reward*5
		# 			#return 0
		# 		else:
		# 			reward = reward + tmp_reward
		# 			#return reward*5
		# 			#return reward
		# 	else:
		# 		tmp_reward = math.log(self.STOCK_VARIABLE__current_price) - math.log(self.AT_OBS__input[-1 -1])
		# 		if tmp_reward > 0 : #missed oppertunity
		# 			reward = reward - (tmp_reward)
		# 		else : #well skipped, tmp_reward < 0
		# 			reward = reward + (tmp_reward)
		# return reward*5000

	def FUNC_AT__profit_to_reward(self):
		"""
		로그 수익률을 돌려준다. 순간 계산하면 됨
		self.STOCK_VARIABLE__current_price = 0
		self.STOCK_VARIABLE__inventory = []

		self.STOCK_VARIABLE__current_budget = 0
		self.STOCK_VARIABLE__started_budget = 0 # sell_all 이후 사용할 변수
		self.STOCK_VARIABLE__action_list = [] # 시행했던 list
		0.029558 -> 3프로 수익

		"""
		reward = 0
		for stock_bought_price in self.STOCK_VARIABLE__inventory:
			tmp_calc = math.log(self.STOCK_VARIABLE__current_price) - math.log(stock_bought_price)
			reward = reward + tmp_calc

		return reward



		
	def score_board(self):
		# global self.AT_SAVE__content, self.AT_SAVE_PATH__folder, self.AT_SAVE_PATH__file_to_txt, self.AT_FLAG__score_board_fixed, self.AT_SAVE__current_max_score, self.AT_SAVE__score
		# 점수를 파일로 저장해서, 지금까지 돌린거 중에 가장 좋은 score 남도록 한다
		# 그리고 가장 높았던 점수를 리턴해준다
		self.AT_SAVE__content = None
		# 점수 마커
		self.AT_SAVE_PATH__folder = str(os.getcwd() + "\\TRADER__profit").replace('/', '\\')
		# 지금 작업중인 경로
		self.AT_SAVE_PATH__file_to_txt = str(self.AT_SAVE_PATH__folder) + "\\profit_board.txt"
		# 파일까지의 경로
		file_exist = os.path.isfile(self.AT_SAVE_PATH__file_to_txt)
		# 논리값 return, 파일이 있는가 없는가

		def write():
			if file_exist : # 파일 존재하면
				f = open(self.AT_SAVE_PATH__file_to_txt, 'r')
				f.seek(0)
				self.AT_SAVE__content = f.read()
				f.close()

				if not(self.AT_SAVE__content == ""): # null이 아니면
					if self.AT_FLAG__score_board_fixed == 0 :
						self.AT_SAVE__txt_score_value = float(self.AT_SAVE__content) # 에피 끝날 떄 까지 고정값
						self.AT_FLAG__score_board_fixed = 1
					if float(self.AT_SAVE__content) >= self.AT_SAVE__current_max_score: #저장 값이 크다
						pass
					else:
						with open(self.AT_SAVE_PATH__file_to_txt, 'w+') as out_file :
						# data is over-written with w+ parameter so use r+
							out_file.seek(0) # goint to top of the file just in case
							out_file.write(str(self.AT_SAVE__current_max_score)) # 값을 쓴다
								#G.score_board_value = G.score
				else: # 파일있는데 null인 경우
					with open(self.AT_SAVE_PATH__file_to_txt, 'w+') as out_file:
						out_file.write(str(self.AT_SAVE__score))
						self.AT_SAVE__txt_score_value = self.AT_SAVE__score
			else: # 파일 생성만 하고 끝난다
				with open(self.AT_SAVE_PATH__file_to_txt, 'w+') as out_file :
					out_file.write(str(self.AT_SAVE__score))
					self.AT_SAVE__txt_score_value = self.AT_SAVE__score
		if not os.path.isdir(self.AT_SAVE_PATH__folder):
		# 존재하지 않는다면
			os.mkdir(self.AT_SAVE_PATH__folder)
			write()
			# 경로를 만들어준다
		else:
		# 폴더가 존재하면
		# 바로 파일 작업으로 넘어감
			write()
	

	def load_model(self):
		# global self.test_query
		# global self.sess
		curr_dir1 = str(os.getcwd() + "\\TRADER__checkpoints-stock_trading")
		self.curr_dir = os.path.join(curr_dir1, "stock_trading-DQN")
		print(self.curr_dir)
		#question = input('directory check')
		# tensor save되는 폴더
		#saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(curr_dir1)
		if not os.path.isdir(curr_dir1):
		# 존재하지 않는 폴더라면
			os.mkdir(curr_dir1)
			#만들어줌
			print("▼"*50)
			print('Could not find old network weights and added a new folder')
			print("▼"*50)

		else:
		# 존재한다면
			if checkpoint and checkpoint.model_checkpoint_path : 
			# 둘다 체크 들어가는데 폴더는 있어야 할듯
				self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
				#restore 하기 전에 init 하지 마라는데 무슨 뜻??
				print("★"*50)
				print('Successfully loaded : ', checkpoint.model_checkpoint_path)
				print("★"*50)
				
			else:
				print("※"*50)
				print('Just proceeding with no loading but existing directory')
				print("※"*50)
				
	def save_model(self):
		# self.STOCK_VARIABLE__started_budget = 0
		tmp_log_end_money = math.log( self.STOCK_VARIABLE__current_budget + self.STOCK_VARIABLE__current_price * len(self.STOCK_VARIABLE__inventory) )
		tmp_log_start_money = math.log( self.STOCK_VARIABLE__started_budget )
		self.AT_SAVE__score = tmp_log_end_money - tmp_log_start_money
		self.score_board()
		if self.AT_SAVE__current_max_score <= self.AT_SAVE__score:
			self.AT_SAVE__current_max_score = self.AT_SAVE__score
			if self.AT_SAVE__current_max_score > self.AT_SAVE__txt_score_value :
				# if self.test_query == 0:
				# 	self.saver.save(self.sess, self.curr_dir, self.GLOBAL)
				self.saver.save(self.sess, self.curr_dir, self.GLOBAL)
				print(f'model has been saved...')
		self.score_board()
		#####
	
	def print_result(self):
		print(f'global step... : {self.GLOBAL}')
		print(f'agent random : {self.AT_EPS__initial}')
		#print(f'actions taken : {self.STOCK_VARIABLE__action_dictionary}')
		print(f'trainings done in the episode : {self.TRAINING_DONE_COUNTER}')
		print(f'current price : {self.STOCK_VARIABLE__current_price}')
		print(f'budget started with : {formatPrice(self.STOCK_VARIABLE__started_budget)}')
		#tmp_end_budget = self.STOCK_VARIABLE__current_budget + self.STOCK_VARIABLE__current_price * len(self.STOCK_VARIABLE__inventory)
		tmp_end_budget = self.STOCK_VARIABLE__current_budget
		print(f'budget ended with : {formatPrice(tmp_end_budget)} - 중간 정산해서 출력')
		print(f'profit in cash : {formatPrice(tmp_end_budget - self.STOCK_VARIABLE__started_budget)}')
		print('-'*50)
		if len(self.STOCK_VARIABLE__inventory) == 0:
			print(f'not holding any stocks..')
		else:
			print(f'owning stock list price average of -> {formatPrice(sum(self.STOCK_VARIABLE__inventory)/len(self.STOCK_VARIABLE__inventory))} of numbers -> {len(self.STOCK_VARIABLE__inventory)}, total of {formatPrice(sum(self.STOCK_VARIABLE__inventory))}')
		print('-' * 50)
		print(f'self.AT_SAVE__score of profit in log scale : {self.AT_SAVE__score}')
		if len(self.AT_PRINT__error_list) == 0:
			print(f'rms error mean value of self.AT_PRINT__error_list not ready yet')
		else:
			print(f'rms error mean value : {sum(self.AT_PRINT__error_list)/len(self.AT_PRINT__error_list)}')
		print('\n'*2)


	
	def init_model(self):
		# @reset
		print(f'model has been initialized...')
		self.AT_SAVE__content = None
		self.AT_SAVE_PATH__folder = None
		self.AT_SAVE_PATH__file_to_txt = None
		self.AT_FLAG__score_board_fixed = 0
		self.AT_SAVE__current_max_score = 0
		self.AT_SAVE__score = 0
		self.AT_SAVE__txt_score_value = 0
		self.AT_PRINT__error_list = []
		
		self.TRAINING_DONE_COUNTER = 0

		self.STOCK_VARIABLE__current_price = 0
		self.STOCK_VARIABLE__inventory = []
		self.STOCK_VARIABLE__current_budget = 0
		self.STOCK_VARIABLE__started_budget = 0 # sell_all 이후 사용할 변수
		#self.STOCK_VARIABLE__action_dictionary = {'buy_success':0, 'sell_success':0, 'hold_success':0, 'buy_failure':0, 'sell_failure':0} # 시행했던 dict

		#tmp_budget = random.randint(2000*10, 5000*10* (1000) ) # 20만원 ~ 5천
		tmp_choice = int(random.randint(0,1))
		print(f'tmp_choice : {tmp_choice}')
		if tmp_choice == 0:
			tmp_budget = int(np.random.normal(6000000, 1600000, 1)[0])
			if tmp_budget < 0:
				tmp_budget = random.randint(2000 * 10, 500 * 10 * (1000))  # 20만원 ~ 500만
		else:
			tmp_budget = random.randint(2000 * 10, 500 * 10 * (1000))  # 20만원 ~ 500만
		# print(f'tmp_budget : {tmp_budget}')
		# print(f'type tmp_budget : {type(tmp_budget)}')
		# print(f'want... {tmp_budget[0]}')
		# input('*')
		self.STOCK_VARIABLE__started_budget = copy.deepcopy(tmp_budget)
		self.STOCK_VARIABLE__current_budget = copy.deepcopy(tmp_budget)
		print(f'model with start budget with : {self.STOCK_VARIABLE__started_budget}')

	def init_for_episode(self):
		print(f'model has been initialized for an episode...')
		
		self.TRAINING_DONE_COUNTER = 0
		
		self.STOCK_VARIABLE__current_price = 0
		self.STOCK_VARIABLE__inventory = []
		tmp_choice = int(random.randint(0,1))
		print(f'tmp_choice : {tmp_choice}')
		if tmp_choice == 0:
			tmp_budget = int(np.random.normal(6000000, 1600000, 1)[0])
			if tmp_budget < 0:
				tmp_budget = random.randint(2000 * 10, 500 * 10 * (1000))  # 20만원 ~ 500만
		else:
			tmp_budget = random.randint(2000 * 10, 500 * 10 * (1000))  # 20만원 ~ 500만
		self.STOCK_VARIABLE__started_budget = copy.deepcopy(tmp_budget)
		self.STOCK_VARIABLE__current_budget = copy.deepcopy(tmp_budget)
		#self.STOCK_VARIABLE__current_budget = copy.deepcopy(self.STOCK_VARIABLE__started_budget)
		#self.STOCK_VARIABLE__action_dictionary = {'buy_success':0, 'sell_success':0, 'hold_success':0, 'buy_failure':0, 'sell_failure':0} # 시행했던 dict
		print(f'STOCK_VARIABLE__current_budget : {self.STOCK_VARIABLE__current_budget}, STOCK_VARIABLE__started_budget : {self.STOCK_VARIABLE__started_budget}')

def formatPrice(n):
	#return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n) )
	# print 'This: ${:0,.0f} and this: ${:0,.2f}'.format(num1, num2).replace('$-','-$')
	return '₩ {:0,.0f}'.format(n).replace('₩ -','- ₩ ')


def Session(db_list_parsed = False):
	# @ agent load
	agent = Trade_agent(module=False)

	# @ db location
	tmp_db_folder_location = str(os.getcwd() + "\\TRADER__DATABASE_single").replace('/', '\\')
	tmp_db_file_location = tmp_db_folder_location + '\\SINGLE_DB.db'
	tmp_pickle_file_location = tmp_db_folder_location + '\\parsed_list_pickle.p'

	# @ import for db
	import sqlite3
	import pandas as pd
	import pickle

	sqlite_con_top = sqlite3.connect(tmp_db_file_location)
	tmp_parsed_stock_code_list_from_db = []  # 최종 원하는 code_list

	if db_list_parsed == False:  # 가져온 유가 없으면
		sqlite_cur_top = sqlite_con_top.cursor()
		tmp_list_db_codes_obj = sqlite_cur_top.execute("SELECT name FROM sqlite_master WHERE type='table';")
		tmp_tmp_list_db_codes = copy.deepcopy(tmp_list_db_codes_obj.fetchall())

		tmp_list_of_codes = []  # 후보군
		for code_item in tmp_tmp_list_db_codes:
			tmp_list_of_codes.append(code_item[0])

		for stock_code in tmp_list_of_codes:
			print(f'parsing {stock_code} from sqlite database...!')
			head_string = 'SELECT * FROM '
			tmp_table_name_sql = "'" + str(stock_code) + "'"
			tmp_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top, index_col=None)

			bool_1 = (len(tmp_df) >= 900 * 0.95) and (len(tmp_df) != 0)  # 비지 않고 900*0.9 이상

			bool_2 = not (tmp_df.empty)
			# @ 3) 가격
			bool_3 = True
			if not tmp_df.empty:
				for row_tuple in tmp_df.itertuples():
					if row_tuple.open < 5000:
						bool_3 = False
						break

			bool_4 = True
			tmp_counter = 0
			tmp_volume_sum = 0
			if not tmp_df.empty:  # 빈 데이터프레임 아니면
				for row_tuple in tmp_df.itertuples():
					tmp_counter = tmp_counter + 1
					tmp_volume_sum = row_tuple.volume + tmp_volume_sum

				if tmp_counter == 0:  # avoid division by zero
					bool_4 = False
				else:
					if tmp_volume_sum / tmp_counter < 500:
						bool_4 = False
			else:
				bool_4 = False

			if bool(bool_1 * bool_2 * bool_3 * bool_4):
				tmp_parsed_stock_code_list_from_db.append(stock_code)

		# save list in a location
		with open(tmp_pickle_file_location, 'wb') as file:
			pickle.dump(tmp_parsed_stock_code_list_from_db, file)

	else:
		with open(tmp_pickle_file_location, 'rb') as file:
			tmp_parsed_stock_code_list_from_db = copy.deepcopy(pickle.load(file))

	print(f'stock list length to work with : {len(tmp_parsed_stock_code_list_from_db)}')
	print(f'stock list to work with : {tmp_parsed_stock_code_list_from_db}')
	input('&&&&&&&&&&')

	# @ parisng 시작
	# stock_code : {  {date_stamp : {price : AAA, volume : BBB}}  , ...} 형태로 넣어주어야 함
	tmp_initial_learning_counter = 0
	for stock_code in tmp_parsed_stock_code_list_from_db:
		try:
			if stock_code in ['265520', '152100', '253160', '161510', '122090', '328370', '301440', '005830', '000990', '114090', '083450', '006360', '293180', '322410', '012630', '294870', '095340', '001060', '035900', '285000', '148020', '252420', '270800', '272560', '196230', '270810', '334690', '105560', '035600', '001390', '060720', '105190', '205720', '226980', '278530', '069500', '252650', '252670', '325010', '292190', '278540', '315930', '271050', '279530', '132030', '122630', '304940', '261250', '261260', '325020', '114800', '229200', '251340', '226490', '069660', '253230', '130730', '122260', '033780', '030200', '003550', '034220', '001120', '032640', '011070', '066570', '051910', '023150', '035420', '060250', '005940', '030190', '005490', '218410', '010950', '011790', '178920', '034730', '006120', '052260', '096770', '285130', '28513K', '017670', '000660', '139260', '102110', '252000', '252710', '300610', '310970', '292150', '157450', '123320', '225030', '329750', '157490', '123310', '192090', '204480', '217780', '232080', '250780', '277650', '570022']:
				continue
			# if stock_code in ["265520", "152100", "253160"]:
			# 	continue
			print(f'stock code that was selected... : {stock_code}')

			head_string = 'SELECT * FROM '
			tmp_table_name_sql = "'" + str(stock_code) + "'"
			tmp_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top, index_col=None)
			print(f'dataframe - initial : \n{tmp_df}')

			# @ change date to datetime obj
			tmp_df['date'] = pd.to_datetime(tmp_df['date'], format="%Y%m%d%H%M%S")
			# print(f'dataframe - changed to datetime object : \n{tmp_df}')

			# @ 전체 시작~ 끝 데이터
			tmp_datetime_start__obj = tmp_df.date.min()
			tmp_datetime_start__obj = tmp_datetime_start__obj + datetime.timedelta(days=1)
			tmp_datetime_end__obj = tmp_df.date.max()
			tmp_dictionary_hashed = {}

			# @ skip for error bool
			tmp_skip_for_error_bool = False

			while tmp_datetime_start__obj < tmp_datetime_end__obj:  # 전체 시작 / 끝

				# @ 해당 day에 사용 될 obj 고정해둠
				tmp_datetime_start__obj_now = tmp_datetime_start__obj.replace(hour=9, minute=0, microsecond=0)
				tmp_datetime_start__obj__fix = tmp_datetime_start__obj.replace(hour=9, minute=0, microsecond=0)
				tmp_datetime_start__obj_end = tmp_datetime_start__obj.replace(hour=15, minute=30, microsecond=0)
				tmp_datetime_start__obj_start = None

				# @ 주말이면 거르기 위해
				tmp_df_for_whole_day = tmp_df.loc[(tmp_df.date >= tmp_datetime_start__obj_now) & (tmp_df.date < tmp_datetime_start__obj_end)]

				if tmp_df_for_whole_day.empty == True: # 자른게 weekend면 empty라서 skip
					print(f'weekend, skipping!')
				else: # 주중이라면 pass, 이 path를 탈것임

					# profit graph
					tmp_dictionary_transaction_record = {}
					tmp_dictionary_profit_record = {}

					# @ 너무 폭락장은 pass 하기위함
					tmp_whole_day_open_end = tmp_df_for_whole_day.open.iloc[0]
					tmp_whole_day_open_start = tmp_df_for_whole_day.open.iloc[-1]
					#if tmp_whole_day_open_end < tmp_whole_day_open_start*(1- 0.04) : # 폭락장 아닌 경우
					if tmp_whole_day_open_end < tmp_whole_day_open_start * (1 + 0.01):  # 폭락장 아닌 경우
						print(f'too much falling skipping!')
					else:

						agent.init_model()
						if not agent.AT_EPS__initial == 1:
							agent.AT_EPS__initial = 0.1 # 자유도 for next iter
						else:
							pass
							agent.AT_EPS__initial = 0  # 자유도 for next iter
							#agent.AT_EPS__initial = 0.1
						tmp_initial_learning_counter = tmp_initial_learning_counter + 1
						print(f'tmp_initial_learning_counter : {tmp_initial_learning_counter}')

						tmp_switch = 0
						if tmp_initial_learning_counter <= 2:
							tmp_switch = agent.options.MAX_EPISODE
						else:
							tmp_switch = 40

						for i in range(tmp_switch + 1):
							# @ episode 마다 쓸 환경(?) 변수 reset
							print(f'tmp_switch value : {tmp_switch}')
							tmp_datetime_start__obj_now = tmp_datetime_start__obj.replace(hour=9, minute=0, microsecond=0) + datetime.timedelta(hours=1)
							tmp_datetime_start__obj_end = tmp_datetime_start__obj.replace(hour=15, minute=30, microsecond=0)
							tmp_datetime_start__obj_start = None
							tmp_dictionary_transaction_record = {}
							tmp_dictionary_profit_record = {}
							tmp_dictionary_hashed = {}
							print(f'stock_code : {stock_code}, episode num : {i + 1}')
							print(f'tmp_datetime_start__obj_now : {tmp_datetime_start__obj_now}')


							# @ 그냥 들고있을 때 수익 표기
							returned_list = list(np.where(tmp_df_for_whole_day['date'] == tmp_datetime_start__obj_now))
							try:
								y_value_found = tmp_df_for_whole_day.open.iloc[returned_list[0][0]]
								tmp_log_profit = math.log(tmp_whole_day_open_end) - math.log(y_value_found)
								print(f'tmp_log_profit in the day if held : {tmp_log_profit}')
							except:
								print(f'tmp_log_profit not able to calculate')

							while tmp_datetime_start__obj_now < tmp_datetime_start__obj_end: # 같은거는 다음 iter에서 처리
								# @ 전날것 까지 찾아야 되는 경우
								if tmp_datetime_start__obj_now - tmp_datetime_start__obj__fix < datetime.timedelta(hours=1):
									# @ 목표 시간! -> 토/일을 위에서 skip 되어서 안들어옴
									tmp_days = 0
									if tmp_datetime_start__obj_now.weekday() == 0: # 월요일
										tmp_days = 3
									else:
										tmp_days = 1
									target_time_delta = (tmp_datetime_start__obj__fix + datetime.timedelta(hours=1) ) - tmp_datetime_start__obj_now
									tmp_datetime_start__obj_start = tmp_datetime_start__obj_now - datetime.timedelta(days=tmp_days)
									tmp_datetime_start__obj_start = tmp_datetime_start__obj_start.replace(hour=15, minute=30, microsecond=0)
									tmp_datetime_start__obj_start = tmp_datetime_start__obj_start - target_time_delta
									#print(f' --- tmp_datetime_start__obj_start : {tmp_datetime_start__obj_start}')

									# @ dataframe 2개로 자르기
									tmp_df_sliced_1 = tmp_df.loc[(tmp_df.date >= tmp_datetime_start__obj_start) & (tmp_df.date <= tmp_datetime_start__obj_start.replace(hour=15, minute=30, microsecond=0))]
									#print(f'tmp_df_sliced_1 : {tmp_df_sliced_1}')
									tmp_tmp_dictionary_hashed_1 = copy.deepcopy(SESS__convert_dataframe_to_dic(tmp_df_sliced_1))
									tmp_dictionary_hashed_1 = copy.deepcopy(SESS__fill_missing_data_in_dict(tmp_tmp_dictionary_hashed_1,tmp_datetime_start__obj_start,tmp_datetime_start__obj_start.replace(hour=15, minute=30, microsecond=0)))

									tmp_df_sliced_2 = tmp_df.loc[(tmp_df.date >= tmp_datetime_start__obj__fix) & (tmp_df.date < tmp_datetime_start__obj_now)]
									#print(f'tmp_df_sliced_2 : {tmp_df_sliced_2}')
									tmp_tmp_dictionary_hashed_2 = copy.deepcopy(SESS__convert_dataframe_to_dic(tmp_df_sliced_2))
									tmp_dictionary_hashed_2 = copy.deepcopy(SESS__fill_missing_data_in_dict(tmp_tmp_dictionary_hashed_2,tmp_datetime_start__obj__fix,tmp_datetime_start__obj_now))
									#print(f'tmp_dictionary_hashed_1 : {tmp_dictionary_hashed_1} , tmp_dictionary_hashed_2 : {tmp_dictionary_hashed_2}')
									tmp_dictionary_hashed_1.update(tmp_dictionary_hashed_2)
									tmp_dictionary_hashed = copy.deepcopy(tmp_dictionary_hashed_1)

								else:
									tmp_datetime_start__obj_start = tmp_datetime_start__obj_now - datetime.timedelta(hours=1)
									tmp_df_sliced = tmp_df.loc[(tmp_df.date >= tmp_datetime_start__obj_start) & (
												tmp_df.date < tmp_datetime_start__obj_now)]
									#print(f'----- tmp_df_sliced : {tmp_df_sliced}')
									#print(f'tmp_df_sliced : {tmp_df_sliced}')
									tmp_tmp_dictionary_hashed = copy.deepcopy(SESS__convert_dataframe_to_dic(tmp_df_sliced))
									#print(f'tmp_tmp_dictionary_hashed : {tmp_tmp_dictionary_hashed}')
									tmp_dictionary_hashed = copy.deepcopy(
										SESS__fill_missing_data_in_dict(tmp_tmp_dictionary_hashed,
																		tmp_datetime_start__obj_start,
																		tmp_datetime_start__obj_now))
									#print(f'tmp_dictionary_hashed : {tmp_dictionary_hashed}')
									#print(f'tmp_dictionary_hashed : {tmp_dictionary_hashed}')


								tmp_df_for_whole_day = tmp_df.loc[(tmp_df.date >= tmp_datetime_start__obj__fix) & (tmp_df.date < tmp_datetime_start__obj_end)]

								# @ 완료된 시점으로 자르기!

								#print(f'tmp_df_sliced : {tmp_df_sliced}')
								# @ 이전날 데이터 자른게 full 공란이면 skip할거임
								# tmp_df_check_for_previous_slice = tmp_df.loc[ (tmp_df.date >= tmp_datetime_start__obj_start) & (tmp_df.date < tmp_datetime_start__obj__fix)]
								# if tmp_df_check_for_previous_slice.empty == True:
								# 	print(f'skipp...empty previous dataframe!')
								# else:
								try:

									#print(f'tmp_dictionary_hashed : {tmp_dictionary_hashed}')
									#input('PP')
									########################################################################
									# 여기서 실제 학습용 데이터 input
									########################################################################
									# @ 학습
									agent.FUNC_AT__observation_to_train(tmp_dictionary_hashed)
									#input('PPpp')

									# @ action 기록
									tmp_agent_action_dict_return = agent.STOCK_VARIABLE__action_dictionary
									tmp_dictionary_transaction_record[tmp_datetime_start__obj_now] = SESS__agent_action_dict_to_tuple(tmp_agent_action_dict_return)

									# @ profit 기록
									# STOCK_VARIABLE__state_profit
									tmp_dictionary_profit_record[tmp_datetime_start__obj_now] = agent.STOCK_VARIABLE__state_profit
								except Exception as e:
									import traceback
									print(f'failure due to {e}')
									traceback.print_exc()

									tmp_skip_for_error_bool = True

									#print(f'tmp_dictionary_hashed : {tmp_dictionary_hashed}')

									#input('PPpp-----')
									break


								# @ now 시점 1분 추가
								tmp_datetime_start__obj_now = tmp_datetime_start__obj_now + datetime.timedelta(minutes=1)

							# @ break for loop
							if tmp_skip_for_error_bool == True:
								tmp_skip_for_error_bool = False
								break

							# at the end of for loop / episode done
							if i % 10 == 0:
								tmp_start_budget = agent.STOCK_VARIABLE__started_budget
								tmp_end_budget = agent.STOCK_VARIABLE__current_budget + agent.STOCK_VARIABLE__current_price * len(
									agent.STOCK_VARIABLE__inventory)
								tmp_return = formatPrice(tmp_end_budget - tmp_start_budget)
								SESS__save_image(tmp_datetime_start__obj__fix.strftime('%Y-%m-%d') , agent.AT_SAVE__score, tmp_return, formatPrice( agent.STOCK_VARIABLE__started_budget ),tmp_df_for_whole_day, tmp_dictionary_transaction_record, tmp_dictionary_profit_record, stock_code, i, tmp_datetime_start__obj__fix, tmp_datetime_start__obj_end )
							agent.save_model()
							agent.print_result()
							agent.init_for_episode()

				# @ move to next
				tmp_datetime_start__obj = tmp_datetime_start__obj + datetime.timedelta(days=1)
				tmp_datetime_start__obj.replace(hour=9, minute=0, microsecond=0)

		except Exception as e:
			print(f'skipping for loop of stock_code for following error : {e}')
			traceback.print_exc()

		# input('???')

	# @ close sqlite connection
	print(f'finished every learning...')
	if db_list_parsed == False:
		sqlite_cur_top.close()
	sqlite_con_top.close()


def SESS__save_image(start_day_str, log_profit, money_profit, agent_start_budget ,dataframe, transaction_tuple_dictionary, profit_dictionary, stock_code, episode_num, start_datetime_obj, end_datetime_obj):
	import matplotlib.pyplot as plt
	folder_location = (os.getcwd() + '\\TRADER__Image_result').replace('/','\\')#
	file_location = folder_location + '\\' + str(stock_code) + '_' + str(start_day_str) + '__' + 'episode_num_' + str(episode_num) + '.png'


	fig = plt.figure(figsize=(100, 50))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212, sharex=ax1)
	# ax1 = fig.add_subplot(211)
	# ax2 = fig.subpot(212)

	plt.title('Log budget profit : ' + str(log_profit) + ' in money scale : ' + str(money_profit) + ' start budget : ' + str(agent_start_budget))

	#ax1 = dataframe.plot(x='date', y='open', figsize = (100, 50), grid=True, Linewidth=1, fontsize=5)
	dataframe.plot(x='date', y='open', figsize=(80, 50), grid=True, Linewidth=1, fontsize=5, ax=ax1)

	# https://datascienceschool.net/view-notebook/372443a5d90a46429c6459bba8b4342c/
	#plt.title('hi')
	for date in transaction_tuple_dictionary:
		x = date
		y = None
		returned_list = list(np.where(dataframe['date'] == date))
		# print(f'returned_list : {returned_list}')
		# print(f'returned_list[0] : {returned_list[0]}')
		# print(f'type of returned_list[0] : {type(returned_list[0])}')
		try:
			y = dataframe.open.iloc[returned_list[0][0]]
			tmp_tuple = transaction_tuple_dictionary[date]
			tmp_str_1 = ''
			tmp_str_2 = ''
			if tmp_tuple[0] == 'BUY':
				tmp_str_1 = 'B'
				tmp_str_2 = str(tmp_tuple[1])
			elif tmp_tuple[0] == "SELL":
				tmp_str_1 = 'S'
				tmp_str_2 = str(tmp_tuple[1])
			elif tmp_tuple[0] == "HOLD":
				tmp_str_1 = 'H'
				tmp_str_2 = str(tmp_tuple[1])
			else:
				tmp_str_1 = 'F'
			ax1.text(x, y, str(tmp_str_1 + tmp_str_2), fontsize=5)
		except:
			continue

	for date in profit_dictionary:
		ax2.plot_date(date, profit_dictionary[date], '.r-')
		# if date in profit_dictionary:
		# 	ax2.plot_date(date, profit_dictionary[date], '.r-')
		# else:
		# 	ax2.plot_date(date, 0, '.b-')

	#plts.figure.savefig(file_location, dpi = 300)
	#ax.xaxis.grid(True, which="minor")
	#plts.figure.xaxis.grid(True, which="minor")
	try:
		# plts.figure.savefig(file_location, dpi=150)
		# plt.close(plts.figure)
		fig.savefig(file_location, dpi=150)
		plt.close(fig)
		plt.close(ax1)
		plt.close(ax2)
		print(f'plotting successfully saved!')
	except Exception as e:
		import traceback
		print(f'failed to save the plotting...: {e}')
		traceback.print_exc()


	# @ try deleting them
	try:
		fig = None
		ax1 = None
		ax2 = None

		del fig
		del ax1
		del ax2
		print(f'successful deleting them in SESS__save_image')
	except Exception as e:
		print(f'error in  deleting them in SESS__save_image... {e}')





def SESS__agent_action_dict_to_tuple(dictionary):
	# self.STOCK_VARIABLE__action_dictionary = {'buy_success':0, 'sell_success':0, 'hold_success':0, 'buy_failure':0, 'sell_failure':0}
	if dictionary['buy_success'] != 0:
		return ('BUY', dictionary['buy_success'])
	elif dictionary['sell_success'] != 0:
		return ('SELL', dictionary['sell_success'])
	elif dictionary['hold_success'] != 0:
		return ('HOLD', dictionary['hold_success'])
	elif dictionary['buy_failure'] != 0 or dictionary['sell_failure'] != 0 :
		return ('FAIL', 1)


def SESS__convert_dataframe_to_dic(dataframe):
	# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
	#print(f'★★★ convert_dataframe_to_dic len of dataframe : {len(dataframe)}')
	tmp_dictionary_return = {}
	
	for row_tuple in dataframe.itertuples(): 
		tmp_dictionary_return[row_tuple.date.strftime('%Y%m%d%H%M%S')] = {'price': row_tuple.open, 'volume':row_tuple.volume}
	
	#print(f'☆☆☆ convert_dataframe_to_dic len of tmp_dictionary_return : {len(list(tmp_dictionary_return.keys()))}')
	#print(f'dataframe, tmp_dictionary_return : {dataframe, tmp_dictionary_return}')
	return tmp_dictionary_return

def SESS__fill_missing_data_in_dict(dictionary, start_time_obj, end_time_obj):
	
	####여기서 missing 나온다
	try:
		tmp_return_dictionary = copy.deepcopy(dictionary)

		tmp_list_of_missing_datastamp = []

		tmp_datetime_stamp_list = list(dictionary.keys())
		tmp_datetime_stamp_list.sort()

		#print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')

		tmp_start_datetime_stamp = tmp_datetime_stamp_list[0] #첫 데이터
		tmp_end_datetime_stamp = tmp_datetime_stamp_list[-1] #마지막 데이터
		tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)
		tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)

		if tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj:
			before_price = None
			before_volume = None
			while tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj : # datetime obj끼리 비교 while 문이라 위험??
				# @ 처음은 list에서 뽑아왔으므로 있다
				tmp_start_datetime_stamp_obj_convert = tmp_start_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
				if tmp_start_datetime_stamp_obj_convert in dictionary:
					before_price = dictionary[tmp_start_datetime_stamp_obj_convert]['price']
					before_volume = dictionary[tmp_start_datetime_stamp_obj_convert]['volume']
				else:
					tmp_list_of_missing_datastamp.append(tmp_start_datetime_stamp_obj_convert)
					tmp_return_dictionary[tmp_start_datetime_stamp_obj_convert] = {'price':before_price, 'volume':before_volume}

				tmp_start_datetime_stamp_obj = tmp_start_datetime_stamp_obj + datetime.timedelta(minutes=1)

		# 1) 뒤쪽에서 값이 missing된 경우
		tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)
		tmp_end_stub_price = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['price']
		tmp_end_stub_volume = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['volume']
		while tmp_end_datetime_stamp_obj < end_time_obj :
			tmp_end_datetime_stamp_obj_convert = tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
			if tmp_end_datetime_stamp_obj_convert in tmp_return_dictionary:
				pass
			else:
				tmp_return_dictionary[tmp_end_datetime_stamp_obj_convert] = {'price':tmp_end_stub_price, 'volume':tmp_end_stub_volume}
			tmp_end_datetime_stamp_obj = tmp_end_datetime_stamp_obj + datetime.timedelta(minutes=1)


		# 2) 앞쪽에서 값이 missing된 경우
		tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0, microsecond=0)
		tmp_start_stub_price = dictionary[tmp_start_datetime_stamp]['price']
		tmp_start_stub_volume = dictionary[tmp_start_datetime_stamp]['volume']
		tmp_end_time_obj = start_time_obj
		while tmp_end_time_obj <= tmp_start_datetime_stamp_obj:
			tmp_end_time_obj_convert = tmp_end_time_obj.strftime('%Y%m%d%H%M%S')
			if tmp_end_time_obj_convert in tmp_return_dictionary:
				pass
			else:
				tmp_return_dictionary[tmp_end_time_obj_convert] = {'price':tmp_start_stub_price, 'volume':tmp_start_stub_volume }

			tmp_end_time_obj = tmp_end_time_obj + datetime.timedelta(minutes=1)

		# 앞에서 시작값이 missing된 경우
		#tmp_check_count = len(list(tmp_return_dictionary.keys()))
		# if tmp_check_count < 60*2: # 2시간 120개 있어야 함, 자를 때는 latest 값 포함 안함
		# 	tmp_stub_price = tmp_return_dictionary[tmp_start_datetime_stamp]['price']
		# 	tmp_stub_volume = tmp_return_dictionary[tmp_start_datetime_stamp]['volume']
		# 	tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)
		#
		# 	for i in range(0, (60*2)-tmp_check_count, 1 ):
		# 		tmp_start_datetime_stamp_obj__missing = tmp_start_datetime_stamp_obj - datetime.timedelta(minutes=(i+1))
		# 		tmp_start_datetime_stamp_obj__missing_convert = tmp_start_datetime_stamp_obj__missing.strftime('%Y%m%d%H%M%S')
		# 		tmp_return_dictionary[tmp_start_datetime_stamp_obj__missing_convert] = {'price': tmp_stub_price,'volume':tmp_stub_volume }

		#print(f'★★★ fill_missing_data_in_dict len of tmp_return_dictionary : {len(list(tmp_return_dictionary.keys()))}')
		#print(f'☆☆☆ fill_missing_data_in_dict tmp_list_of_missing_datastamp  : {tmp_list_of_missing_datastamp}')
		return tmp_return_dictionary

	except:
		return dictionary
	

if __name__ == '__main__':
	Session(db_list_parsed = True)
	#Session(db_list_parsed = False)0