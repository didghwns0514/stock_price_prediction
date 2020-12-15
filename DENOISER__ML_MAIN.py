# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, UpSampling1D
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras import backend as K
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import datetime
import copy
import keras
print(keras.__version__)




class Option:

	def __init__(self, envs):
		self.INPUT_SHAPE = envs[0] # 60 * 4 = 240
		self.SEQUENCE_SHAPE = envs[1] # 1
		self.KERNEL_SIZE = envs[2] # 몇분간 쓸건지?
		self.DROPOUT_RATE = envs[3] # actual dropout / not keeping rate
		self.LAYER_1_FILER_NUM = envs[4]
		self.LAYER_2_FILER_NUM = envs[5]
		self.POOLING_SIZE = envs[6]
		self.EPOCH_NUM = envs[7]
		self.BATCH_NUM = envs[8]


class Denoiser:
	NAME = 'denoiser_'
	HOUR_LENGTH = 4
	INPUT_LENGTH = int( 60 * HOUR_LENGTH )
	TRAIN_DATA_SPLIT_PER = 0.2


	#         0, 1, 2,   3,  4,  5, 6      7    8
	envs = [240, 1, 4, 0.3, 45, 20, 2, 40000, 400]

	def __init__(self, module = False):
		self.options = Option(self.envs)
		self.module = module

		self.config=tf.ConfigProto(
			device_count={'GPU': 0}
		)


		# @ 로컬 그래프 / 세션 할당해서 밑에서 build 할것!
		# https://medium.com/@vovaprivalov/ml-tips-working-with-multiple-models-in-keras-tensorflow-graphs-24b9745bc29c
		self.MAIN_GRAPH = tf.Graph()
		
		with self.MAIN_GRAPH.as_default() as g:
			if self.module == True:
				self.MAIN_SESS = tf.Session(config=self.config, graph=g)
				self.options.DROPOUT_RATE = 0
			else:
				self.MAIN_SESS = tf.Session(config=self.config, graph=g)
				#self.MAIN_SESS = tf.Session(graph=self.MAIN_GRAPH)
		
		# @ 모델들 class
		self.MAIN_MODEL = None
		self.MAIN_MODEL_ENCODED = None
		self.MAIN_ENCODED = None
		self.MAIN_MODEL__history = None
		self.MAIN_ENCODED__history = None
		self.MAIN_MODEL__early_stop = None
		self.MAIN_MODEL__checkpoint = None

		self.STOCK__max_value_now = None
		self.STOCK_input_list = []

		# @ checkpoint
		self.python_checkpoint_folder_path = str(os.getcwd() + '\\DENOISER__checkpoints').replace('/', '\\')
		if os.path.isdir(self.python_checkpoint_folder_path):
			pass
		else:
			os.mkdir(self.python_checkpoint_folder_path)
		self.python_checkpoint_file_path = self.python_checkpoint_folder_path + '\\denoiser_model.h5'

		if self.FUNC_LOAD__model():
			pass
		else:
			self.FUNC_BUILD__layers()


	def FUNC_BUILD__layers(self):
		"""
		https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
		https://keraskorea.github.io/posts/2018-10-23-keras_autoencoder/

		:return:
		"""
		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:

				self.main_inputs = Input(shape= (self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE ), name=self.NAME + 'Inputs_1')
				#self.main_reshape__1 = Reshape( (self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE), input_shape=self.options.INPUT_SHAPE, name=self.NAME + 'reshape_1')(self.main_inputs)
				self.main_reshape__1 = Reshape((self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE), name=self.NAME + 'reshape_1')(self.main_inputs)

				# @ 1차 conv layer
				self.main_conv1d__1 = Conv1D( activation='relu', filters= self.options.LAYER_1_FILER_NUM, kernel_size=self.options.KERNEL_SIZE, strides=1, padding='same', name=self.NAME + 'Conv1D_1')(self.main_reshape__1)
				self.main_maxpooling__1 = MaxPooling1D( pool_size=self.options.POOLING_SIZE, padding='same', name=self.NAME + 'MaxPooling1D_1' )(self.main_conv1d__1)
				self.main_dropout__1 = Dropout(self.options.DROPOUT_RATE, name=self.NAME + 'Dropout_1')(self.main_maxpooling__1)

				# @ 2ck conv layer
				self.main_conv1d__2 = Conv1D( activation='relu', filters= self.options.LAYER_2_FILER_NUM, kernel_size=self.options.KERNEL_SIZE, strides=1, padding='same', name=self.NAME + 'Conv1D_2')(self.main_dropout__1)
				self.main_maxpooling__2 = MaxPooling1D( pool_size=self.options.POOLING_SIZE, padding='same', name=self.NAME + 'MaxPooling1D_2' )(self.main_conv1d__2)
				#self.main_dropout__2 = Dropout(self.options.DROPOUT_RATE, name=self.NAME + 'Dropout_2')(self.main_maxpooling__2)
				self.main_encoded = Dropout(self.options.DROPOUT_RATE, name=self.NAME + 'Dropout_2')(self.main_maxpooling__2)

				# @ encoding layer
				#self.main_encoded = Flatten(name=self.NAME + 'Flatten')(self.main_dropout__2)

				self.MAIN_ENCODED = Model(self.main_inputs, self.main_encoded)


				# @ 1차 decoding layer
				self.main_dropout__3 = Dropout(self.options.DROPOUT_RATE, name=self.NAME + 'Dropout_3')(self.main_encoded)
				self.main_decode__1 = Conv1D(activation='relu', filters= self.options.LAYER_2_FILER_NUM, kernel_size=self.options.KERNEL_SIZE, strides=1, padding='same', name=self.NAME + 'Conv1D_3')(self.main_dropout__3)
				self.main_upsampling__1 = UpSampling1D(size=self.options.POOLING_SIZE, name=self.NAME + 'UpSampling1D_1')(self.main_decode__1)

				# @ 2차 decoding layer
				self.main_dropout__4 = Dropout(self.options.DROPOUT_RATE, name=self.NAME + 'Dropout_4')(self.main_upsampling__1)
				self.main_decode__2 = Conv1D(activation='relu', filters=self.options.LAYER_1_FILER_NUM, kernel_size=self.options.KERNEL_SIZE, strides=1, padding='same', name=self.NAME + 'Conv1D_4')(self.main_dropout__4)
				self.main_upsampling__2 = UpSampling1D(size=self.options.POOLING_SIZE, name=self.NAME + 'UpSampling1D_2')(self.main_decode__2)

				# @ decoded layer
				self.main_decoded = Conv1D(activation='sigmoid', filters=1, kernel_size=self.options.KERNEL_SIZE,padding='same', name=self.NAME + 'Conv1D_FINAL')(self.main_upsampling__2)


				self.MAIN_MODEL = Model(self.main_inputs, self.main_decoded)
				self.MAIN_MODEL.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='mean_squared_error' )
				#self.MAIN_MODEL.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='binary_crossentropy')

				self.MAIN_MODEL.summary()
				#tf.keras.utils.plot_model(self.MAIN_MODEL, 'model.png', show_shapes=True)


		# self.main_model = Sequential()
		# self.main_model.add(Reshape( (self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE), input_shape=self.options.INPUT_SHAPE, name=self.NAME + 'reshape_1') )
		# self.main_model.add(Conv1D( activation='relu', filters= 45, kernel_size=5, strides=1, padding='same', name=self.NAME + 'Conv1D_1'))
		# self.main_model.add(MaxPooling1D( pool_size=2, padding='valid', name=self.NAME + 'MaxPooling1D_1' ))

	def func_plot_history(self, history):
		plt.figure(figsize=(15, 5))
		ax = plt.subplot(1, 2, 1)
		plt.plot(history.history["loss"])
		plt.title("Train loss")
		ax = plt.subplot(1, 2, 2)
		plt.plot(history.history["val_loss"])
		plt.title("Test loss")
		plt.show()

	def func_plot_examples(self, stock_input, stock_decoded):
		n = 10
		test_samples = 2000
		plt.figure(figsize=(20, 4))
		for i, idx in enumerate(list(np.arange(0, test_samples, 200))):
			# display original
			ax = plt.subplot(2, n, i + 1)
			if i == 0:
				ax.set_ylabel("Input", fontweight=600)
			else:
				ax.get_yaxis().set_visible(False)
			plt.plot(stock_input[idx])
			ax.get_xaxis().set_visible(False)

			# display reconstruction
			ax = plt.subplot(2, n, i + 1 + n)
			if i == 0:
				ax.set_ylabel("Output", fontweight=600)
			else:
				ax.get_yaxis().set_visible(False)
			plt.plot(stock_decoded[idx])
			ax.get_xaxis().set_visible(False)

	def FUNC_SAVE__model(self):
		pass

	def FUNC_LOAD__model(self):

		if os.path.isfile(self.python_checkpoint_file_path):
			print(f'checkpoint {self.python_checkpoint_file_path} exists !')

			with self.MAIN_GRAPH.as_default() as g:
				with self.MAIN_SESS.as_default() as sess:
					K.set_session(sess)
					self.MAIN_MODEL = load_model(self.python_checkpoint_file_path)
					print(f'self.MAIN_MODEL.summary()')
					self.MAIN_MODEL.summary()
					
					self.MAIN_MODEL_ENCODED = Model(inputs=self.MAIN_MODEL.input,
												   outputs=self.MAIN_MODEL.get_layer(self.NAME + 'Dropout_2').output)
					print(f'self.MAIN_MODEL_ENCODED.summary()')
					self.MAIN_MODEL_ENCODED.summary()
					
					# input('denoiser....! 1)')

			return True
		else:
			return False

	def FUNC_TRAIN_MAIN__from_sess(self, _x_data):
		"""
		학습용
		:param x_train: 
		:param x_test: 
		:return: 
		"""
		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:

				#x_data = np.array(_x_data).reshape(-1, self.options.SEQUENCE_SHAPE, self.options.INPUT_SHAPE )
				x_data = np.array(_x_data).reshape( -1, self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE )

				self.MAIN_MODEL__early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100) # baseline !!!
				self.MAIN_MODEL__checkpoint = ModelCheckpoint(monitor='val_loss', mode='min',verbose=1, save_best_only=True, filepath=self.python_checkpoint_file_path)

				self.MAIN_MODEL__history = self.MAIN_MODEL.fit(x_data, x_data,
											 epochs=self.options.EPOCH_NUM,
											 batch_size=self.options.BATCH_NUM,
											 shuffle=True,
											 verbose = 1,
											 validation_split=self.TRAIN_DATA_SPLIT_PER,
											 callbacks=[self.MAIN_MODEL__early_stop, self.MAIN_MODEL__checkpoint]) # validation_data=(x_test, x_test),


				self.func_plot_history(self.MAIN_MODEL__history)

	def FUNC_PREDICT_MAIN__sess(self, x_input_hash):

		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				K.set_session(sess)
				_tmp_stock_list, tmp_stock_max = self.func_normalize_data(x_input_hash)
				tmp_stock_list = np.array(_tmp_stock_list).reshape(-1, self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE )
				#print(f'tmp_stock_list.shape : {tmp_stock_list.shape()}')
				#tmp_stock_list = np.array(_tmp_stock_list).reshape(self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE)

				_tmp_return = self.MAIN_MODEL.predict(tmp_stock_list)
				#print(f'_tmp_return.shape : {_tmp_return.shape()}')
				tmp_return = _tmp_return.reshape(self.options.INPUT_SHAPE ).tolist()
				tmp_real_return_descaled = [x * tmp_stock_max for x in tmp_return]

				# print(f'tmp_return : {tmp_return}')
				# print(f'tmp_real_return_descaled : {tmp_real_return_descaled}')
				# input('****__0')

				return tmp_real_return_descaled


	def FUNC_PREDICT_MAIN__ontherun(self, x_input_list):

		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				K.set_session(sess)
				_tmp_stock_list, tmp_stock_max = self.func_normalize_data__ontherun(x_input_list)
				tmp_stock_list = np.array(_tmp_stock_list).reshape(-1, self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE )
				#print(f'tmp_stock_list.shape : {tmp_stock_list.shape()}')
				#tmp_stock_list = np.array(_tmp_stock_list).reshape(self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE)

				_tmp_return = self.MAIN_MODEL_ENCODED.predict(tmp_stock_list)
				# print(f'type(_tmp_return) : {type(_tmp_return)}')
				# print(f'_tmp_return.shape : {_tmp_return.shape}')
				# print(f'_tmp_return : {_tmp_return}')
				#tmp_return = _tmp_return.reshape( -1, 60, self.options.SEQUENCE_SHAPE ).tolist()
				#tmp_return = _tmp_return.reshape(-1, self.options.SEQUENCE_SHAPE).tolist()

				_tmp_return = _tmp_return.reshape(-1, 1200)
				# print(f'type(_tmp_return) : {type(_tmp_return)}')
				# print(f'_tmp_return.shape : {_tmp_return.shape}')
				# print(f'_tmp_return : {_tmp_return}')

				tmp_return = _tmp_return.tolist()[0]

				# print(f'denoiser....! 2)')
				# print(f' -> tmp_return :: {tmp_return}')
				# input(f'FUNC_PREDICT_MAIN__ontherun --- enter to skip !')

				#tmp_real_return_descaled = [x * tmp_stock_max for x in tmp_return]

				# print(f'tmp_return : {tmp_return}')
				# print(f'tmp_real_return_descaled : {tmp_real_return_descaled}')
				# input('****__0')

				#return tmp_real_return_descaled
				return tmp_return
			
	def FUNC_PREDICT_MAIN__ontherun_FULL_LAYER(self, x_input_list):

		with self.MAIN_GRAPH.as_default() as g:
			with self.MAIN_SESS.as_default() as sess:
				K.set_session(sess)
				_tmp_stock_list, tmp_stock_max = self.func_normalize_data__ontherun(x_input_list)
				tmp_stock_list = np.array(_tmp_stock_list).reshape(-1, self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE )
				#print(f'tmp_stock_list.shape : {tmp_stock_list.shape()}')
				#tmp_stock_list = np.array(_tmp_stock_list).reshape(self.options.INPUT_SHAPE, self.options.SEQUENCE_SHAPE)

				_tmp_return = self.MAIN_MODEL.predict(tmp_stock_list)
				#print(f'_tmp_return.shape : {_tmp_return.shape()}')
				tmp_return = _tmp_return.reshape(self.options.INPUT_SHAPE ).tolist()
				tmp_real_return_descaled = [x * tmp_stock_max for x in tmp_return]

				# print(f'tmp_return : {tmp_return}')
				# print(f'tmp_real_return_descaled : {tmp_real_return_descaled}')
				# input('****__0')

				return tmp_real_return_descaled

	def func_normalize_data__ontherun(self, input_data_list):
		"""
		input을 normalize 해준다면, output 복구는 어떻게?
		:return:
		"""
		# @ initialize
		self.STOCK_input_list = []
		self.STOCK__max_value_now = None


		self.STOCK__max_value_now = max(input_data_list)
		self.STOCK_input_list = [ x / self.STOCK__max_value_now for x in input_data_list]

		# print(f'self.STOCK_input_list : {self.STOCK_input_list}')
		# input('**')

		return  self.STOCK_input_list, self.STOCK__max_value_now

	def func_normalize_data(self, input_data_hash):
		"""
		input을 normalize 해준다면, output 복구는 어떻게?
		:return:
		"""
		# @ initialize
		self.STOCK_input_list = []
		self.STOCK__max_value_now = None

		tmp_list_date_stamp = list(input_data_hash.keys())
		tmp_list_date_stamp.sort()

		tmp_price_obj = []
		for i in range(len(input_data_hash) - 1, -1, -1):
			tmp_price = input_data_hash[tmp_list_date_stamp[i]]['price']

			tmp_price_obj.insert(0, tmp_price)

			if len(tmp_price_obj) >= self.INPUT_LENGTH:
				break
		# print(f'input_data_hash : {input_data_hash}')
		# print(f'tmp_price_obj : {tmp_price_obj}')
		# print(f'len(tmp_price_obj) : {len(tmp_price_obj)}')

		self.STOCK__max_value_now = max(tmp_price_obj)
		self.STOCK_input_list = [ x / self.STOCK__max_value_now for x in tmp_price_obj]

		# print(f'self.STOCK_input_list : {self.STOCK_input_list}')
		# input('**')

		return  self.STOCK_input_list, self.STOCK__max_value_now




def Session(db_list_parsed=False):
	denoiser = Denoiser(module=False)


	# @ db location
	tmp_db_folder_location = str(os.getcwd() + "\\DENOISER__DATABASE_single").replace('/', '\\')
	tmp_train_db_folder_loaction = str(os.getcwd() + "\\DENOISER__TRAIN_saved").replace('/', '\\')

	tmp_db_file_location = tmp_db_folder_location + '\\SINGLE_DB.db'
	tmp_pickle_file_location = tmp_db_folder_location + '\\parsed_list_pickle.p'
	tmp_article_file_location = tmp_db_folder_location + '\\pickle.p'
	tmp_skipped_file_location = tmp_db_folder_location + '\\stock_list_pickle__skipped.p'

	MUST_WATCH_LIST = ["226490", "261250", "252670"]
	# ㅋㅋ# KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X
	SKIP_LIST = ["229200", "069500", "122630", "102110", "114800", "251340", "278540", "315930", "278530", "310970", "271050", "252710", "334690", "292150", "196230", "570045", "204480", "148020", "192090", "161510", "252420", "325010", "105190", "217780", "328370", "139260", "570044", "293180", "570027", "253160", "122260", "253230", "132030", "570019", "279530", "325020", "152100", "157490", "123310", "123320", "304940", "285000", "292190", "122090", "301440", "225030", "250780", "205720", "272560", "270800", "232080", "270810", "226980", "300610", "322410", "261260", "570023"]

	# @ import for db
	import sqlite3
	import pandas as pd
	import pickle
	import copy
	import datetime
	import time
	import traceback
	import joblib

	# @ open pickle file

	SKIP_LIST_2 = None
	with open(tmp_skipped_file_location, 'rb') as file:
		SKIP_LIST_2 = copy.deepcopy(pickle.load(file))
		print(f'SKIP_LIST_2 : {SKIP_LIST_2}')
		print(f'skip list loaded !')

	# @ make sqlite connection
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

			bool_1 = (len(tmp_df) >= 900 * 0.99) and (len(tmp_df) != 0)  # 비지 않고 900*0.9 이상

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
	# @ count work / skip list
	tmp_list_to_work = []
	tmp_list_to_skip = []
	for stock_code in tmp_parsed_stock_code_list_from_db:
		if stock_code in MUST_WATCH_LIST or stock_code in SKIP_LIST or stock_code in SKIP_LIST_2:
			tmp_list_to_skip.append(stock_code)
		else:
			tmp_list_to_work.append(stock_code)
	print(f'len(tmp_list_to_work) : {len(tmp_list_to_work)}')
	print(f'len(tmp_list_to_skip) : {len(tmp_list_to_skip)}')
	input('&&&&&&&&&&')

	tmp_bool_logic = int(input('for training : 0, \nfor testing : 1, \nfor creating data & train: 2, \nfor training with created data : 3'))
	LIST_TRAIN__data = [[], []]

	# @ parisng 시작
	# stock_code : {  {date_stamp : {price : AAA, volume : BBB}}  , ...} 형태로 넣어주어야 함
	if tmp_bool_logic != 3:

		tmp_initial_learning_counter = 0
		for stock_code in tmp_parsed_stock_code_list_from_db:
			tmp_initial_learning_counter = tmp_initial_learning_counter + 1

			if tmp_initial_learning_counter >= int(len(tmp_parsed_stock_code_list_from_db) * 0.8):
				continue

			# try:
			if stock_code in MUST_WATCH_LIST or stock_code in SKIP_LIST or stock_code in SKIP_LIST_2:
				continue
			if stock_code in []:
				continue

			# if stock_code in ['265520', '152100', '253160', '161510', '122090', '328370', '301440', '005830', '000990', '114090', '083450', '006360', '293180', '322410', '012630', '294870', '095340', '001060', '035900', '285000', '148020', '252420', '270800', '272560', '196230', '270810', '334690', '105560', '035600', '001390', '060720', '105190', '205720', '226980', '278530', '069500', '252650', '252670', '325010', '292190', '278540', '315930', '271050', '279530', '132030', '122630', '304940', '261250', '261260', '325020', '114800', '229200', '251340', '226490', '069660', '253230', '130730', '122260', '033780', '030200', '003550', '034220', '001120', '032640', '011070', '066570', '051910', '023150', '035420', '060250', '005940', '030190', '005490', '218410', '010950', '011790', '178920', '034730', '006120', '052260', '096770', '285130', '28513K', '017670', '000660', '139260', '102110', '252000', '252710', '300610', '310970', '292150', '157450', '123320', '225030', '329750', '157490', '123310', '192090', '204480', '217780', '232080', '250780', '277650', '570022', '570043', '570045', '570023', '570044', '570027', '570019', '079940', '035250', '002100', '014570', '000270', '024110', '308100', '286750', '138610', '001260', '111710', '091590', '095660', '007390', '033640', '225570', '251270', '278650', '144510', '031390', '234690', '214870', '068240']:
			# 	continue

			print(f'stock code that was selected... : {stock_code}')
			head_string = 'SELECT * FROM '
			tmp_table_name_sql = "'" + str(stock_code) + "'"
			tmp_whole_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top, index_col=None)
			print(f'dataframe - initial : \n{tmp_whole_df}')

			tmp_whole_df['date'] = pd.to_datetime(tmp_whole_df['date'], format="%Y%m%d%H%M%S")

			datetime_whole_start__obj = tmp_whole_df.date.min() + datetime.timedelta(days=1)
			datetime_whole_end__obj = tmp_whole_df.date.max()

			tmp_bool_break_while_loop_1 = False



			while datetime_whole_start__obj < datetime_whole_end__obj:
				try:
					# @ single day
					datetime_single_start__now_obj = datetime_whole_start__obj.replace(hour=9, minute=0, microsecond=0)
					datetime_single_start__fix_obj = datetime_whole_start__obj.replace(hour=9, minute=0, microsecond=0)
					datetime_single_start__end_obj = datetime_whole_start__obj.replace(hour=15, minute=30,
																					   microsecond=0)

					tmp_single_day_df = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_single_start__fix_obj) & (
							tmp_whole_df.date < datetime_single_start__end_obj)]

					if tmp_single_day_df.empty == True:
						print(f'weekend, skipping!')
					elif len(tmp_single_day_df) < 300:
						print(f'too little data!')
					else:
						#agent.FUNC_MODLE__init()

						# for i in range(agent.options.MAX_EPISODE):
						for i in range(1):
							print(f'date of {datetime_single_start__fix_obj}')
							tmp_return_dictionary_for_drawing = {}  # initialize
							tmp_return = None

							# @ initialize for every episode
							datetime_single_start__now_obj = datetime_whole_start__obj.replace(hour=9, minute=0,
																							   microsecond=0)
							datetime_single_start__end_obj = datetime_whole_start__obj.replace(hour=15, minute=30,
																							   microsecond=0)

							# @ start while loop
							while datetime_single_start__now_obj <= datetime_single_start__end_obj:
								try:

									tmp_dict__stock_min = SESS_parse_data_from_sqlite(tmp_whole_df,
																					  datetime_single_start__now_obj,
																					  denoiser.HOUR_LENGTH)
									#input('****__1')
									if tmp_bool_logic == 0 or tmp_bool_logic == 2:  # training / training + create data
										tmp_input_list, tmp_max_value  = denoiser.func_normalize_data(tmp_dict__stock_min)

										if tmp_bool_logic == 0:
											pass
										elif tmp_bool_logic == 2:  # create data
											print(f'adding data to LIST_TRAIN__data')
											LIST_TRAIN__data[0].append(tmp_input_list)
											LIST_TRAIN__data[1].append(tmp_max_value)
											try:
												#print(f'LIST_TRAIN__data[0][0] : {LIST_TRAIN__data[0][0]}')
												pass
											except Exception as e:
												pass
									elif tmp_bool_logic == 1:  # just testing
										pass
										#input('****__2')
										tmp_input_list, tmp_max_value = denoiser.func_normalize_data(tmp_dict__stock_min)

										start_time = time.monotonic()
										tmp_return = denoiser.FUNC_PREDICT_MAIN__sess(tmp_dict__stock_min)
										end_time = time.monotonic()
										total_seconds = (end_time - start_time)
										print(f'estimated elapsed time for initial training!!! : {datetime.timedelta(seconds=total_seconds)}')
										#input('****__3')
										#print(f'tmp_return : {tmp_return}')
										#print(f'type(tmp_return) : {type(tmp_return)}')
										tmp_return_dictionary_for_drawing[datetime_single_start__now_obj] = [tmp_input_list, tmp_max_value, tmp_return]


									# @ move to next iter -> avoid too many noisy data
									if tmp_bool_logic != 1:  # just testing
										#datetime_single_start__now_obj = datetime_single_start__now_obj + datetime.timedelta(minutes=30)
										datetime_single_start__now_obj = datetime_single_start__now_obj + datetime.timedelta(hours=2)
									else:
										datetime_single_start__now_obj = datetime_single_start__now_obj + datetime.timedelta(hours=2)

								except ValueError:
									# @ value Error from article
									print(f'no article exists...')
									datetime_single_start__now_obj = datetime_single_start__now_obj + datetime.timedelta(minutes=1)
									traceback.print_exc()
									continue

								except Exception as e:
									# handle possible missing values from dataframe
									print(f'error in somewhere...?! {e}')
									# traceback.print_exc()

									# @ move to next minute
									datetime_single_start__now_obj = datetime_single_start__now_obj + datetime.timedelta(minutes=1)
									continue

							# # @ move to next day
							# break
							if tmp_return_dictionary_for_drawing and tmp_bool_logic == 1:  # hash not empty!!
								#input('****__4')
								SESS__save_image(datetime_single_start__fix_obj, tmp_return_dictionary_for_drawing, stock_code, i + 1)
								#input('****__5')

					# @ move to next day
					datetime_whole_start__obj = datetime_whole_start__obj + datetime.timedelta(days=1)

				except Exception as e:
					print(f'error in single day... {e}')
					traceback.print_exc()

					# move to next day
					datetime_whole_start__obj = datetime_whole_start__obj + datetime.timedelta(days=1)
					continue

			tmp_train_db_location = None
			try:
				# tmp_initial_learning_counter
				tmp_train_db_location = tmp_train_db_folder_loaction + '\\LIST_TRAIN__data' + '__' + str(
					tmp_initial_learning_counter) + '.sav'
				joblib.dump(LIST_TRAIN__data, tmp_train_db_location)
			except Exception as e:
				print(f'error in saving ?! ')

			# break automatically
			tmp_size = os.path.getsize(tmp_train_db_location)
			print(f'tmp size of the .sav file : {tmp_size}')
			if os.path.getsize(tmp_train_db_location) > 883718400:  # 900 mb
				print(f'automatically breaking!')
			else:
				print(f'successfully proceed to next saving!')

		# except Exception as e:
		# 	print(f'error in most outter for loop : {e}')
		# 	traceback.print_exc()

		# @ close sqlite connection
		print(f'finished every learning...')
		if db_list_parsed == False:
			sqlite_cur_top.close()
		sqlite_con_top.close()

	else:  # train right away with created data

		tmp_train_db_location = tmp_train_db_folder_loaction + '\\LIST_TRAIN__data' + '__' + str(415) + '.sav'
		LIST_TRAIN__data = joblib.load(tmp_train_db_location)
		print(f'successfuly loaded sav file!!!!!')

	# tmp_batch_training_bool = int(input('do temp batch training : 0, do whole batch training : 1'))


	print(f'entire stock train data : {len(LIST_TRAIN__data[0])}')
	denoiser.FUNC_TRAIN_MAIN__from_sess(LIST_TRAIN__data[0])

	# split_number = int(len(LIST_TRAIN__data[0])*( 1- denoiser.TRAIN_DATA_SPLIT_PER ))
	# # X_TRAIN_data = LIST_TRAIN__data[0][: split_number]
	# # Y_TRAIN_data = LIST_TRAIN__data[1][: split_number]
	# X_TEST_data = LIST_TRAIN__data[0][split_number :]
	# X_TEST_DESCALE_data = LIST_TRAIN__data[1][split_number :]
	#
	# rand_indexes = np.random.choice(len(X_TEST_data), 6)
	#
	# denoiser.FUNC_PREDICT_MAIN__sess()




def SESS_parse_answer_data_from_sqlite(tmp_whole_df, datetime_obj_now, min_duration_forward):
	tmp_dictionary_for_return = {}

	# import pandas as pd
	# head_string = 'SELECT * FROM '
	# tmp_table_name_sql = "'" + str(stock_code) + "'"
	# tmp_whole_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top_connection, index_col=None)
	# tmp_whole_df['date'] = pd.to_datetime(tmp_whole_df['date'], format="%Y%m%d%H%M%S")

	if datetime_obj_now + datetime.timedelta(minutes=min_duration_forward) <= datetime_obj_now.replace(hour=15,
																									   minute=30):
		df_target = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_obj_now) & (
				tmp_whole_df.date < datetime_obj_now + datetime.timedelta(minutes=min_duration_forward))]

		dict_target = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_target), datetime_obj_now,
													  datetime_obj_now + datetime.timedelta(
														  minutes=min_duration_forward))

		tmp_dictionary_for_return.update(dict_target)

	else:
		tmp_delta_time_forward_now = datetime_obj_now.replace(hour=15, minute=30, second=0,
															  microsecond=0) - datetime_obj_now
		tmp_delta_time_forward_in_minutes = divmod(tmp_delta_time_forward_now.total_seconds(), 60)[0]
		tmp_calc_minutes = min_duration_forward - tmp_delta_time_forward_in_minutes
		datetime_target = None
		if datetime_obj_now.weekday() == 4:  # 금요일
			datetime_target = datetime_obj_now + datetime.timedelta(days=3)

		else:
			datetime_target = datetime_obj_now + datetime.timedelta(days=1)

		df_target_start_time = datetime_target.replace(hour=9, minute=0, second=0, microsecond=0)
		df_target_end_time = df_target_start_time + datetime.timedelta(minutes=tmp_calc_minutes)
		df_target = tmp_whole_df.loc[
			(tmp_whole_df.date >= df_target_start_time) & (tmp_whole_df.date <= df_target_end_time)]

		df_now = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_obj_now) & (
				tmp_whole_df.date < datetime_obj_now.replace(hour=15, minute=30, second=0, microsecond=0))]

		# dictionary
		dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now), datetime_obj_now,
													  datetime_obj_now.replace(hour=15, minute=30, second=0,
																			   microsecond=0))
		dict_df_target = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_target),
														 df_target_start_time, df_target_end_time)

		# @ update
		tmp_dictionary_for_return.update(dict_df_now)
		tmp_dictionary_for_return.update(dict_df_target)

	return tmp_dictionary_for_return


# @decorator_function
def SESS_parse_data_from_sqlite(tmp_whole_df, datetime_obj_now, hours_duration_back):
	tmp_dictionary_for_return = {}

	# import pandas as pd
	# head_string = 'SELECT * FROM '
	# tmp_table_name_sql = "'" + str(stock_code) + "'"
	# tmp_whole_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top_connection, index_col=None)
	# tmp_whole_df['date'] = pd.to_datetime(tmp_whole_df['date'], format="%Y%m%d%H%M%S")

	# print(f'at the start datetime_obj_now : {datetime_obj_now}')

	datetime_single_day__fix_obj = datetime_obj_now.replace(hour=9, minute=0, second=0, microsecond=0)
	tmp_time_forward_now = datetime_obj_now - datetime_single_day__fix_obj
	tmp_div_mod = divmod(tmp_time_forward_now.total_seconds(), 60)
	tmp_time_in_minutes = tmp_div_mod[0]

	# 총 시간은 600개 분봉 가져오면 됨 10시간이라서
	tmp_calc_hour = 60 * (hours_duration_back) - tmp_time_in_minutes
	tmp_div_mod_result = divmod(tmp_calc_hour, 391)
	# print(f'hours_duration_back : {hours_duration_back}, datetime_obj_now : {datetime_obj_now}, datetime_single_day__fix_obj : {datetime_single_day__fix_obj}, tmp_time_forward_now : {tmp_time_forward_now}, tmp_div_mod : {tmp_div_mod}, tmp_time_in_minutes: {tmp_time_in_minutes}, tmp_calc_hour : {tmp_calc_hour}, tmp_div_mod_result : {tmp_div_mod_result} ')

	if tmp_div_mod_result[0] == 1 and tmp_calc_hour > 0:  # two more df
		df_1_datetime_target = datetime_obj_now - datetime.timedelta(days=1)
		if df_1_datetime_target.weekday() not in [0, 1, 2, 3, 4]:
			df_1_datetime_target = df_1_datetime_target - datetime.timedelta(
				days=abs(df_1_datetime_target.weekday() - 4))
		df_2_datetime_target = df_1_datetime_target - datetime.timedelta(days=1)

		# @ df 1 설정
		df_1_end_time = df_1_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_1_start_time = df_1_datetime_target.replace(hour=9, minute=0, second=0, microsecond=0)
		df_1 = tmp_whole_df.loc[(tmp_whole_df.date >= df_1_start_time) & (tmp_whole_df.date <= df_1_end_time)]

		# @ df 2 설정
		df_2_end_time = df_2_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_2_start_time = df_2_end_time - datetime.timedelta(minutes=tmp_div_mod_result[1])
		df_2 = tmp_whole_df.loc[(tmp_whole_df.date >= df_2_start_time) & (tmp_whole_df.date <= df_2_end_time)]

		# @ df now 설정
		df_now = tmp_whole_df.loc[
			(tmp_whole_df.date >= datetime_single_day__fix_obj) & (tmp_whole_df.date < datetime_obj_now)]

		dict_df_1 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_1), df_1_start_time,
													df_1_end_time)
		dict_df_2 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_2), df_2_start_time,
													df_2_end_time)
		dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now),
													  datetime_single_day__fix_obj, datetime_obj_now)

		tmp_dictionary_for_return.update(dict_df_1)
		tmp_dictionary_for_return.update(dict_df_2)
		tmp_dictionary_for_return.update(dict_df_now)

	# print(f'df_1_end_time : {df_1_end_time}, df_1_start_time : {df_1_start_time}, df_2_end_time : {df_2_end_time}, df_2_start_time : {df_2_start_time}, datetime_single_day__fix_obj : {datetime_single_day__fix_obj}, datetime_obj_now : {datetime_obj_now}')

	elif tmp_div_mod_result[0] == 2 and tmp_calc_hour > 0:  # 3개

		df_1_datetime_target = datetime_obj_now - datetime.timedelta(days=1)
		if df_1_datetime_target.weekday() not in [0, 1, 2, 3, 4]:
			df_1_datetime_target = df_1_datetime_target - datetime.timedelta(
				days=abs(df_1_datetime_target.weekday() - 4))
		df_2_datetime_target = df_1_datetime_target - datetime.timedelta(days=1)
		df_3_datetime_target = df_1_datetime_target - datetime.timedelta(days=2)

		# @ df 1 설정
		df_1_end_time = df_1_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_1_start_time = df_1_datetime_target.replace(hour=9, minute=0, second=0, microsecond=0)
		df_1 = tmp_whole_df.loc[(tmp_whole_df.date >= df_1_start_time) & (tmp_whole_df.date <= df_1_end_time)]

		# @ df 2 설정
		df_2_end_time = df_2_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_2_start_time = df_2_datetime_target.replace(hour=9, minute=0, second=0, microsecond=0)
		df_2 = tmp_whole_df.loc[(tmp_whole_df.date >= df_2_start_time) & (tmp_whole_df.date <= df_2_end_time)]

		# @ df 3 설정
		df_3_end_time = df_3_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_3_start_time = df_3_end_time - datetime.timedelta(minutes=tmp_div_mod_result[1])
		df_3 = tmp_whole_df.loc[(tmp_whole_df.date >= df_3_start_time) & (tmp_whole_df.date <= df_3_end_time)]

		# @ df now 설정
		df_now = tmp_whole_df.loc[
			(tmp_whole_df.date >= datetime_single_day__fix_obj) & (tmp_whole_df.date < datetime_obj_now)]

		dict_df_1 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_1), df_1_start_time,
													df_1_end_time)
		dict_df_2 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_2), df_2_start_time,
													df_2_end_time)
		dict_df_3 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_3), df_3_start_time,
													df_3_end_time)
		dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now),
													  datetime_single_day__fix_obj, datetime_obj_now)

		tmp_dictionary_for_return.update(dict_df_1)
		tmp_dictionary_for_return.update(dict_df_2)
		tmp_dictionary_for_return.update(dict_df_3)
		tmp_dictionary_for_return.update(dict_df_now)

	# print(f'df_1_end_time : {df_1_end_time}, df_1_start_time : {df_1_start_time}, df_2_end_time : {df_2_end_time}, df_2_start_time : {df_2_start_time},  df_3_end_time : {df_3_end_time}, df_3_start_time : {df_3_start_time}, datetime_single_day__fix_obj : {datetime_single_day__fix_obj}, datetime_obj_now : {datetime_obj_now}')

	elif tmp_div_mod_result[0] == 0 and tmp_div_mod_result[1] > 0 and tmp_calc_hour > 0:  # one df
		df_1_datetime_target = datetime_obj_now - datetime.timedelta(days=1)
		if df_1_datetime_target.weekday() not in [0, 1, 2, 3, 4]:
			df_1_datetime_target = df_1_datetime_target - datetime.timedelta(
				days=abs(df_1_datetime_target.weekday() - 4))

		# @ df 1 설정
		df_1_end_time = df_1_datetime_target.replace(hour=15, minute=30, second=0, microsecond=0)
		df_1_start_time = df_1_end_time - datetime.timedelta(minutes=tmp_div_mod_result[1])
		df_1 = tmp_whole_df.loc[(tmp_whole_df.date >= df_1_start_time) & (tmp_whole_df.date <= df_1_end_time)]

		# @ df now 설정
		df_now = tmp_whole_df.loc[
			(tmp_whole_df.date >= datetime_single_day__fix_obj) & (tmp_whole_df.date < datetime_obj_now)]

		dict_df_1 = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_1), df_1_start_time,
													df_1_end_time)
		dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now),
													  datetime_single_day__fix_obj, datetime_obj_now)

		tmp_dictionary_for_return.update(dict_df_1)
		tmp_dictionary_for_return.update(dict_df_now)

	# print(f'df_1_end_time : {df_1_end_time}, df_1_start_time : {df_1_start_time}, datetime_single_day__fix_obj : {datetime_single_day__fix_obj}, datetime_obj_now : {datetime_obj_now}')

	else:
		df_now_start = datetime_obj_now - datetime.timedelta(minutes=60 * (hours_duration_back))

		df_now = tmp_whole_df.loc[(tmp_whole_df.date >= df_now_start) & (tmp_whole_df.date < datetime_obj_now)]

		dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now),
													  df_now_start, datetime_obj_now)

		tmp_dictionary_for_return.update(dict_df_now)

	# print(f'df_now_start : {df_now_start}, datetime_obj_now : {datetime_obj_now}')

	return tmp_dictionary_for_return


# tmp_single_day_df = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_single_start__fix_obj) & (tmp_whole_df.date < datetime_single_start__end_obj)]

def SESS__convert_dataframe_to_dic(dataframe):
	# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
	# print(f'★★★ convert_dataframe_to_dic len of dataframe : {len(dataframe)}')
	tmp_dictionary_return = {}

	for row_tuple in dataframe.itertuples():
		tmp_dictionary_return[row_tuple.date.strftime('%Y%m%d%H%M%S')] = {'price': row_tuple.open,
																		  'volume': row_tuple.volume}

	# print(f'☆☆☆ convert_dataframe_to_dic len of tmp_dictionary_return : {len(list(tmp_dictionary_return.keys()))}')
	# print(f'dataframe, tmp_dictionary_return : {dataframe, tmp_dictionary_return}')
	return tmp_dictionary_return


def SESS__fill_missing_data_in_dict(dictionary, start_time_obj, end_time_obj):
	####여기서 missing 나온다
	# try:
	tmp_return_dictionary = copy.deepcopy(dictionary)

	tmp_list_of_missing_datastamp = []

	tmp_datetime_stamp_list = list(dictionary.keys())
	# print(f'dictionary : {dictionary}')
	# print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')
	tmp_datetime_stamp_list.sort()

	# print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')

	#	if tmp_return_dictionary : # not empty

	tmp_start_datetime_stamp = tmp_datetime_stamp_list[0]  # 첫 데이터
	tmp_end_datetime_stamp = tmp_datetime_stamp_list[-1]  # 마지막 데이터
	tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)

	if tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj:
		before_price = None
		before_volume = None
		while tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj:  # datetime obj끼리 비교 while 문이라 위험??
			# @ 처음은 list에서 뽑아왔으므로 있다
			tmp_start_datetime_stamp_obj_convert = tmp_start_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
			if tmp_start_datetime_stamp_obj_convert in dictionary:
				before_price = dictionary[tmp_start_datetime_stamp_obj_convert]['price']
				before_volume = dictionary[tmp_start_datetime_stamp_obj_convert]['volume']
			else:
				tmp_list_of_missing_datastamp.append(tmp_start_datetime_stamp_obj_convert)
				tmp_return_dictionary[tmp_start_datetime_stamp_obj_convert] = {'price': before_price,
																			   'volume': 0}  # 'volume': before_volume

			tmp_start_datetime_stamp_obj = tmp_start_datetime_stamp_obj + datetime.timedelta(minutes=1)

	# 1) 뒤쪽에서 값이 missing된 경우
	tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_end_stub_price = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['price']
	tmp_end_stub_volume = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['volume']
	while tmp_end_datetime_stamp_obj < end_time_obj:
		tmp_end_datetime_stamp_obj_convert = tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
		if tmp_end_datetime_stamp_obj_convert in tmp_return_dictionary:
			pass
		else:
			tmp_return_dictionary[tmp_end_datetime_stamp_obj_convert] = {'price': tmp_end_stub_price,
																		 'volume': 0}  # 'volume': tmp_end_stub_volume
		tmp_end_datetime_stamp_obj = tmp_end_datetime_stamp_obj + datetime.timedelta(minutes=1)

	# 2) 앞쪽에서 값이 missing된 경우
	tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_start_stub_price = dictionary[tmp_start_datetime_stamp]['price']
	tmp_start_stub_volume = dictionary[tmp_start_datetime_stamp]['volume']
	tmp_end_time_obj = start_time_obj
	while tmp_end_time_obj <= tmp_start_datetime_stamp_obj:
		tmp_end_time_obj_convert = tmp_end_time_obj.strftime('%Y%m%d%H%M%S')
		if tmp_end_time_obj_convert in tmp_return_dictionary:
			pass
		else:
			tmp_return_dictionary[tmp_end_time_obj_convert] = {'price': tmp_start_stub_price,
															   'volume': 0}  # 'volume': tmp_start_stub_volume

		tmp_end_time_obj = tmp_end_time_obj + datetime.timedelta(minutes=1)

	# print(f'tmp_return_dictionary in  SESS__fill_missing_data_in_dict : \n{tmp_return_dictionary}')

	return tmp_return_dictionary


#	else:
#		return dictionary


# except Exception as e:
# 	print(f'something wrong in SESS__fill_missing_data_in_dict : {e}')
# 	traceback.print_exc()
# 	return dictionary


def SESS__save_image( start_day_str, prediction_dictionary, stock_code, episode_num):
	# tmp_return_list_for_drawing.append( (datetime_single_start__now_obj, tmp_return) )
	# tmp_return_dictionary_for_drawing[datetime_single_start__now_obj] = [tmp_input_list, tmp_max_value, tmp_return
	import matplotlib.pyplot as plt
	import random

	folder_location = (os.getcwd() + '\\DENOISER__Image_result').replace('/', '\\')  #
	file_location = folder_location + '\\' + str(stock_code) + '_' + str(
		start_day_str.strftime("%Y-%m-%d")) +  '__' + 'episode_num_' + str(episode_num) + '.png'

	fig = plt.figure(figsize=( 20, 20 * len(list(prediction_dictionary.keys()))  ))
	#ax1 = fig.add_subplot(111)

	plt.title('')

	i = 0
	tmp_plt_obj_list = []
	for datetime in prediction_dictionary:

		target_num = int( str( len(list(prediction_dictionary.keys())) ) + str(1) + str(i+1) )
		#plt.subplot(target_num)
		tmp_plt_obj_list.append(fig.add_subplot(target_num))

		# @ create x data
		x_t = []
		for j in range(len(prediction_dictionary[datetime][0])):
			x_t.append(j)

		y_1_t = [ x * prediction_dictionary[datetime][1] for x in prediction_dictionary[datetime][0] ]
		y_2_t = prediction_dictionary[datetime][2]

		# plt.plot(x_t, y_1_t, color = 'b', linestyle='solid')
		# plt.plot(x_t, y_2_t, color = 'r', linestyle='solid')
		tmp_plt_obj_list[i].plot(x_t, y_1_t, color = 'b', linestyle='solid')
		tmp_plt_obj_list[i].plot(x_t, y_2_t, color = 'r', linestyle='solid')

		i = i + 1

	try:
		fig.savefig(file_location, dpi=70)
		plt.close(fig)

		print(f'plotting successfully saved!')
	except Exception as e:
		import traceback
		print(f'failed to save the plotting...: {e}')
		traceback.print_exc()

	# @ try deleting them
	try:
		fig = None
		del fig

		print(f'successful deleting them in SESS__save_image')
	except Exception as e:
		print(f'error in  deleting them in SESS__save_image... {e}')

if __name__ == '__main__':
	Session(db_list_parsed=True)