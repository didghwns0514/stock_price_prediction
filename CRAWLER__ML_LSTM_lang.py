#-*-coding: utf-8-*-
import time
import numpy as np
import os
# IMPIRTANT DATAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMPIRTANT DATAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMPIRTANT DATAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://financedata.github.io/posts/python_news_cloud_text.html
import tensorflow as tf

class Options :
	def __init__(self, env):

		self.INPUT_DATA_DIM = env[0] # 입력 데이터 variable 갯수
		self.RNN_HIDDEN_CELL_DIM = env[1] # 각 셀의 (hidden)출력 크기
		self.RESULT_DATA_DIM = env[2] # 결과데이터의 컬럼 개수 : many to one
		self.N_EMBEDDING = env[3] # stacked LSTM layers 개수
		self.SEQ_LENGTH = env[4] # window, for time series length

		self.FORGET_BIAS = env[5] # 망각편향(기본값 1.0)
		self.MAX_EPISODE = env[6]  # max number of episodes iteration
		self.LR = env[7]  # learning rate # 학습률
		self.DROP_OUT = env[8]

class LSTM :

	""""
	https://teddylee777.github.io/machine-learning/sklearn%EC%99%80-pandas%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%84%EB%8B%A8-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D
	https://m.blog.naver.com/wideeyed/221160038616
	https://github.com/aqibsaeed/Multilabel-timeseries-classification-with-LSTM/blob/master/RNN%20-%20Multilabel.ipynb
	"""
	# 일단위 학습 모델에 분을 추가해도 되려나...????
	# many to one
	# 다음날 종가 / open가
	#         0,    1,   2,   3,     4,    5,     6,      7,     8
	envs = [200,  180,   3,   3,   100,  1.0,  6000,  0.001,   0.6] # 0.6

	def __init__(self, module = True):

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

		def lstm_cell():  # single lstm celss
			cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.options.RNN_HIDDEN_CELL_DIM,
												forget_bias=self.options.FORGET_BIAS, state_is_tuple=True,
												activation=tf.nn.softsign)

			if self.options.DROP_OUT < 1.0:
				if int(self.test_query) == 1 :
					self.options.DROP_OUT == 1 # keep prob
				cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.options.DROP_OUT)
			else:
				pass
			return cell


		if tf.test.gpu_device_name():
			print('GPU found')
		else:
			print("No GPU found")


		self.global_step = 0

		tf.set_random_seed(777)
		#self.sess = tf.compat.v1.InteractiveSession()

		#@ CPU setting
		#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		self.sess = tf.Session(config = self.config)
		self.options = Options(self.envs)

		# @ Plcaeholders
		self.observation = tf.placeholder(tf.float32, [None, self.options.SEQ_LENGTH, self.options.INPUT_DATA_DIM], name='observation')
		self.predictions = tf.placeholder(tf.float32, [None, self.options.RESULT_DATA_DIM], name='predictions')
		self.answer = tf.placeholder(tf.float32, [None, self.options.RESULT_DATA_DIM], name='answer')

		# @ make connections
		# ===================================================================================
		self.stackedRNNs = [lstm_cell() for _ in range(self.options.N_EMBEDDING)]
		self.multi_cells = tf.contrib.rnn.MultiRNNCell(self.stackedRNNs, state_is_tuple=True) if self.options.N_EMBEDDING > 1 else lstm_cell()
		self.H_, self._states = tf.nn.dynamic_rnn(self.multi_cells, self.observation, dtype=tf.float32)
		self.LSTM_out = tf.contrib.layers.fully_connected(self.H_[:,-1], self.options.RESULT_DATA_DIM, activation_fn=tf.identity)


		self.W = tf.Variable(self.xavier_initializer([self.options.RESULT_DATA_DIM,self.options.RESULT_DATA_DIM]))
		self.B = tf.Variable(self.xavier_initializer([self.options.RESULT_DATA_DIM]))

		self.Y_mod = tf.squeeze(tf.matmul(self.LSTM_out, self.W) + self.B)
		self.hypothesis = tf.nn.softmax(self.Y_mod)
		# ===================================================================================


		# @ 트레이닝 부분
		# ===================================================================================
		self.softmax_cost = tf.reduce_mean(- tf.reduce_sum(self.answer * tf.log( tf.clip_by_value(self.hypothesis, 1e-10, 1.0) ),axis=1))
		# https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
		self.optimizer = tf.train.AdamOptimizer(self.options.LR)
		self.train = self.optimizer.minimize(self.softmax_cost)
		self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.answer, self.predictions)))
		# ===================================================================================


		self.sess.run(tf.global_variables_initializer())
		print('initialization of LSTM done ... ')
		
		# 사용해야하는 변수들
		#-----------------
		# @ model train
		self.feed = {}
		self.input_queue = None
		self.output_queue = None
		self.saver = tf.train.Saver()
		
		# @ model score
		self.content = None
		# self.curr_dir_f = None
		self.lv_score_board_fix = None
		self.score_board_value = None
		self.score_now_max = None
		self.score = None
		self.score_list = [] # 이것을 계산해서 self.score에 기록
		self.err_list = []

		self.score_dir = str(os.getcwd() + "\\CRAWLER__LSTM_lang")
		self.score_dir_txt = str(self.score_dir) + "\\lstm_lang_score.txt"

		self.curr_dir_folder = str(os.getcwd() + "\\CRAWLER__checkpoints-LSTM_lang")
		self.curr_dir = os.path.join(self.curr_dir_folder, "lstm_lang-DQN")

		self.print_inner_connection()
		self.load_model() # if necessary
		self.init_model() # 추가했는데... 오류나올수도..

		self.return_answer = None
		self.err = None
		
	def print_inner_connection(self):
		print("observation: ", self.observation)
		print("predictions: ", self.predictions)
		print("answer: ", self.answer)
		print("H_: ", self.H_)
		print("LSTM_out: ", self.LSTM_out)
		print("Y_mod: ", self.Y_mod)
		print("hypothesis: ", self.hypothesis)

	def xavier_initializer(self, shape):
		dim_sum = np.sum(shape)
		if len(shape) == 1 :
			dim_sum = dim_sum + 1
		bound = np.sqrt(6.0/ dim_sum)
		return tf.random_uniform(shape, minval = -bound, maxval=bound, dtype=tf.float32)


	def observation_to_train(self, inputz, outputz):
		self.global_step = self.global_step + 1

		self.input_queue = np.array(inputz, dtype=np.float32)
		self.output_queue = np.array(outputz, dtype=np.float32)
		# http://mjgim.me/2018/03/26/multiclass_mlp.html
		self.input_queue = np.array(self.input_queue).reshape(-1 ,self.options.SEQ_LENGTH,
self.options.INPUT_DATA_DIM)
		self.output_queue = np.array(self.output_queue).reshape(-1 , self.options.RESULT_DATA_DIM)

		#@ Training 부분
		train_predict = self.sess.run(self.hypothesis, feed_dict={self.observation : self.input_queue})
		if self.test_query == 0: #테스트 아닐 시
			_ = self.sess.run(self.train, feed_dict={self.observation:self.input_queue, self.answer : self.output_queue})
		err = self.sess.run(self.rmse, feed_dict={self.answer : self.output_queue, self.predictions : train_predict.reshape(-1, self.options.RESULT_DATA_DIM)})

		try:
			self.score_list.extend(self.answer_check(train_predict, outputz))
			self.err = err
			self.err_list.append(err)
		except Exception as e:
			print(e)

	
	def observation_to_test(self, inputz, outputz):

		self.input_queue = np.array(inputz)
		self.output_queue = np.array(outputz)

		self.input_queue = np.array(self.input_queue).reshape(-1 ,self.options.SEQ_LENGTH,self.options.INPUT_DATA_DIM)
		self.output_queue = np.array(self.output_queue).reshape(-1 ,self.options.RESULT_DATA_DIM)
		
		#@ Testing 부분
		test_predict = self.sess.run(self.hypothesis, feed_dict={self.observation : self.input_queue})
		err = self.sess.run(self.rmse, feed_dict={self.answer : self.output_queue, self.predictions : test_predict.reshape(-1, self.options.RESULT_DATA_DIM)})
		try:
			self.score_list.extend(self.answer_check(test_predict, outputz))
			self.err = err
			self.err_list.append(err)
		except Exception as e:
			print(e)

	def observation_to_predict(self, inputz):

		def score_mapping(prediction):
			try:
				#ans_index = np.argmax(prediction)
				#answer = prediction[ans_index]
				ans_1 = prediction[0] # negative
				ans_2 = prediction[1] # good
				ans_3 = prediction[2] # neutral

				return ( -(ans_1*0.5) + (ans_2*0.5) + 0.5) # 0 ~ 1


			except Exception as e:
				print('LSTM prediction err', e)
				return 0


		self.input_queue = np.array(inputz)
		self.input_queue = np.array(self.input_queue).reshape(-1, self.options.SEQ_LENGTH, self.options.INPUT_DATA_DIM)
		test_predict = self.sess.run(self.hypothesis, feed_dict={self.observation: self.input_queue})


		return score_mapping(test_predict)



	def answer_check(self, model_answer, true_answer): #@ arg max로 답안지 체크
		tmp_ans = []
		for i in range(len(model_answer)):
			if np.argmax(model_answer[i]) == np.argmax(true_answer[i]):
				#return 1
				tmp_ans.append(1)
			else:
				#return 0
				tmp_ans.append(0)
		
		return tmp_ans
	
	def print_result(self):
		print('accuracy : ', sum(self.score_list)/len(self.score_list))
		print('rmse err : ', sum(self.err_list)/len(self.err_list))
		print('total ',self.global_step,'th number of steps iterated while training...')
		
		return self.score_list

	def clear_nan(self, dataset):
		dataset.dropna(axis=0, how='any', inplace=True)
		return dataset

	def load_model(self):
		# global self.test_query
		# global self.sess
		# curr_dir1 = str(os.getcwd() + "\\checkpoints-LSTM_lang")
		# self.curr_dir = os.path.join(curr_dir1, "lstm_lang-DQN")
		print(self.curr_dir)
		#question = input('directory check')
		# tensor save되는 폴더
		#saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(self.curr_dir_folder)
		if not os.path.isdir(self.curr_dir_folder):
		# 존재하지 않는 폴더라면
			os.mkdir(self.curr_dir_folder)
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
		if not self.score_list : # 빈 리스트
			return ValueError('Empty score list')
		self.score =  sum(self.score_list)/len(self.score_list)
		self.score_board()
		if self.score_now_max <= self.score:
			self.score_now_max = self.score
			if self.score_now_max > self.score_board_value :
				if self.test_query == 0: #테스트 아닐 시
					self.saver.save(self.sess, self.curr_dir, self.global_step)
					print('model has been saved....')
		self.score_board()
	
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

			
def session_(X_train, y_train, X_test, y_test):
	lstm_lang = LSTM(module=False)
	tmp_ans = None
	tmp_err = None
	list_ans = []
	list_err = []
	# check train test data consistency
	# bool_1 = empty_item_list_check(X_train)
	# bool_2 = empty_item_list_check(y_train)
	# bool_3 = empty_item_list_check(X_test)
	# bool_4 = empty_item_list_check(y_test)
	# bool_all = bool_1 and bool_2 and bool_3 and bool_4
	if len(X_train) == len(y_train) and len(X_test) == len(y_test) :
		pass
	else:
		raise ValueError(' wrong data consistancy')
	
	for i in range(lstm_lang.options.MAX_EPISODE):
		print( i + 1, 'th epoch has started ...')
		start_time = time.time()
		for i in range(len(X_train)):
			lstm_lang.observation_to_train(X_train[i], y_train[i])
		lstm_lang.print_result()
		#lstm_lang.save_model()
		lstm_lang.init_model()
		print("--- %s seconds ---" % (time.time() - start_time))
		
		print("\n"*1)
		print('begin testing model...')
		start_time = time.time()
		for j in range(len(X_test)):
			lstm_lang.observation_to_test(X_test[j], y_test[j])
		lstm_lang.print_result()
		lstm_lang.save_model()
		lstm_lang.init_model()
		print("--- %s seconds ---" % (time.time() - start_time))
		print('+'*60)
		print('+'*60)
		print("\n"*3)
		
def session(X_train, y_train, X_test, y_test,  X_train_data, X_test_data):
	lstm_lang = LSTM(module=False)
	tmp_ans = None
	tmp_err = None
	list_ans = []
	list_err = []
	# check train test data consistency
	# bool_1 = empty_item_list_check(X_train)
	# bool_2 = empty_item_list_check(y_train)
	# bool_3 = empty_item_list_check(X_test)
	# bool_4 = empty_item_list_check(y_test)
	# bool_all = bool_1 and bool_2 and bool_3 and bool_4
	if len(X_train) == len(y_train) and len(X_test) == len(y_test) :
		pass
	else:
		raise ValueError(' wrong data consistancy')
	
	for i in range(lstm_lang.options.MAX_EPISODE):
		print( i + 1, 'th epoch has started ...')
		start_time = time.time()
		lstm_lang.observation_to_train(X_train, y_train)
		lstm_lang.print_result()
		#lstm_lang.save_model()
		lstm_lang.init_model()
		print("--- %s seconds ---" % (time.time() - start_time))
		
		print("\n"*1)
		print('begin testing model...')
		start_time = time.time()
		lstm_lang.observation_to_test(X_test, y_test)
		tmp_ans_list = lstm_lang.print_result()
		try:
			for i in range(len(tmp_ans_list)):
				if tmp_ans_list[i] == 0 :
					#print('wrong answer : ', X_test_data[i])
					pass
		except:
			pass
		
		lstm_lang.save_model()
		lstm_lang.init_model()
		print("--- %s seconds ---" % (time.time() - start_time))
		print('+'*60)
		print('+'*60)
		print("\n"*3)

def empty_item_list_check(lister):
	if len(lister) > 0 :
		for i in range(len(lister)):
			if lister[i]:
				pass
			else:
				return False

		return True

	else: # 0 칸짜리 list
		return False
if __name__ == '__main__':
	pass