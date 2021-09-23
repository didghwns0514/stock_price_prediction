
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # turn off gpu setting


import numpy as np
from collections import deque as dq


# @ tensorflow
import tensorflow as tf

# @ keras
from keras.models import Sequential, Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, Dropout,  Activation, Flatten, Reshape, Input
import keras.backend as K


# @ sklearn model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


# @ logger
from LOGGER_FOR_MAIN import pushLog

# @ other modules
import ENCODER__ML_MAIN as EN
import DENOISER__ML_MAIN as DE

# @ dt operation
from sub_function_configuration import *


class PrivTensorWrapper:

	PT_CREATION_DIC_CNT = {} # cnter dictionary per model
	NAME = 'PrivTensorWrapper_'
	GET_WEIGHTED_LOSS = False
	NUM_BATCH = int(100) # Batch size


	VERBOSE = int(1)

	TRAIN_DICT = {
		'initialized' : {
			'patience': int(50),
			'epoch' : int(100),
			'baseline' : None
		},
		'loaded' : {
			'patience': int(10),
			'epoch': int(50),
			'baseline': None
		},
		'trained' : {
			'patience': int(10),
			'epoch': int(50),
			'baseline': None
		}
	}

	MODEL_MODE = [ 'nn', 'lstm' ] # 0, 1
	MODE_SELECT = 'nn'


	def __init__(self,
				stock_code,
				code_folder_location,
				save_file_location,
				model_score_txt_lc,
				input_shape,
				output_shape,
				loading_bool=False):

		# cnter for model numbers
		if stock_code not in PrivTensorWrapper.PT_CREATION_DIC_CNT:
			PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code] = 0
		PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code] += 1
		"""
		PT_CREATION_DIC_CNT -> starts from 1
		"""

		## configuration per instance
		#----------------------------------------
		self._PT__config = tf.ConfigProto(
			device_count={'GPU': 0}
		)
		self._PT__GlobalCnt = 0 # global cnter
		self._PT__ifLoaded = False # if loaded, flag goes up
		#                          # do transfer learning
		self._PT__initTrain = False # if initial training is done
		self._PT__stock_code = stock_code
		self._PT__model_number = PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code] # marked model number
		self._PT__model_accuracy = 0
		self._PT__model_state = None # for saving state of the class
		self._input_shape = input_shape
		self._output_shape = output_shape

		self._model_mode = None
		#----------------------------------------

		## required locations
		self._PT__code_folder_location = code_folder_location
		self._PT__save_file_location = save_file_location
		self._PT__save_score_txt_location = model_score_txt_lc
		

		self._MAIN_GRAPH = tf.Graph()
		with self._MAIN_GRAPH.as_default() as g:
			self._MAIN_SESS = tf.Session(config=self._PT__config,
										 graph=g)
		self._MAIN_MODEL = None

		## set model mode
		self.PT__set_model_mode()

		## load/build model
		self.PT__handle_mode()

		## calc model state
		self.PT__calc_state()


	def PT__set_model_mode(self):
		"""
		function to set model mode
		"""
		assert PrivTensorWrapper.MODE_SELECT in PrivTensorWrapper.MODEL_MODE

		self._model_mode = PrivTensorWrapper.MODEL_MODE


	def PT__get_state(self):
		"""

		:return: Action - to get the state of the model
		"""
		return self._PT__model_state

	
	def PT__calc_state(self):
		"""
		param : None
				:: var self._PT__ifLoaded, self._PT__initTrain
		return: Action - bsaed on loading and after init training, update state
					   - result is saved in 
					   :: var self._PT__model_state
		reason : to save image as png, record the state change
		"""

		if self._PT__ifLoaded:
			self._PT__model_state = 'loaded'
		else:
			if not self._PT__initTrain : # init training not done
				self._PT__model_state = 'initialized'
			else: # after init training done
				self._PT__model_state = 'trained'

	#@pl.pushLog(dst_folder='PREDICTER_ML_CLASS')
	def PT__clear(self):
		"""
		param : None
		return : Action - clears 
						var:: self._MAIN_MODEL, 
							  self._MAIN_HISTORY,
							  self._MAIN_SESS
							  self._MAIN_GRAPH
							  PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code]
				 Return - if successful, Returns boolean
						  for NestedGraph day dictionary clear
						  // else try deleting the graph (key,value)
		"""
		
		## clear session
		try:
			self._MAIN_SESS.close()
			tf.keras.backend.clear_session()
			PrivTensorWrapper.PT_CREATION_DIC_CNT[self._PT__stock_code] =- 1
		except Exception as e:
			pushLog(dst_folder='PREDICTER__ML_CLASS',
					module='PT__clear',
					exception=True,
					exception_msg=e,
					memo=f'session clear success')
			

	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def PT_CLS__save_model(self):
		"""
		param : None
		return : Action - saves the model, Best model chosen outside the class
				 Return - None
		"""
		## lint
		assert self._MAIN_MODEL != None

		## save model
		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:
				self._MAIN_MODEL.save_weights( self._PT__save_file_location)


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def PT__handle_mode(self):
		"""
		param : None
		return : Action - load if model save file exist
						- differenct action based on Tensor usage
		"""

		## get active model
		if os.path.isfile(self._PT__save_file_location) and \
			PrivTensorWrapper.PT_CREATION_DIC_CNT[self._PT__stock_code]==1:
			with self._MAIN_GRAPH.as_default() as g:
				with self._MAIN_SESS.as_default() as sess:
					self._PT__ifLoaded = True
					#self._MAIN_MODEL = load_model(self._PT__save_file_location)
					self._MAIN_MODEL = load_model(self._PT__save_file_location,
												  custom_objects={'PT__custom_loss':self.PT__custom_loss})
					print(f'model has been loaded!')
					self._MAIN_MODEL.summary()


		else:
			with self._MAIN_GRAPH.as_default() as g:
				with self._MAIN_SESS.as_default() as sess:
					self._MAIN_MODEL = self.PT__build_model()
					print(f'model has been built!')
					self._MAIN_MODEL.summary()


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def PT__build_model(self):
		"""
		param : None
		return : Action - complies model and returns model
						- with differentiated loss
						- trainable layer for transfer learning named!
							=> this changes learning rate in the optimizer
		"""

		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:

				# @ configure model by selected configuration
				model=None
				if self._model_mode == 'nn':
					model = Sequential([

						Dense(120,
							  activation='relu',
							  input_shape=(self._input_shape,),
							  name=self.NAME + 'Dense_1'),
						Dense(120,
							  activation='relu',
							  name=self.NAME + 'Dense_2'),
						Dense(120,
							  activation='relu',
							  name=self.NAME + 'Dense_3'),
						Dense(self._output_shape,
							  activation='relu',
							  name=self.NAME + 'trainable')
					])

				elif self._model_mode == 'lstm':

				else:
					raise ValueError('wrong model mode configuration-1')


				# @ different optimizer for preloaded model instance
				optimizer = None
				if self._PT__ifLoaded:
					optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
				else:
					optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)


				# @ compile model
				model.compile(loss=self.PT__custom_loss,
							  optimizer=optimizer) # loss='mse'

				return model


	def PT__custom_loss(self, y_true, y_pred):

		# https://brunch.co.kr/@chris-song/34
		# https://uos-deep-learning.tistory.com/3
		# https://stackoverflow.com/questions/49729522/why-is-the-mean-average-percentage-errormape-extremely-high
		# https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b

		# y_true_f = K.flatten(y_true)
		# y_pred_f = K.flatten(y_pred)
		# length = K.intshape(y_true)[1]
		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:

				tot_sum = int((int(self._output_shape)) * (int(self._output_shape + 1)) / 2)
				lw = None
				if PrivTensorWrapper.GET_WEIGHTED_LOSS:
					lw = [ num / tot_sum for num in range(int(self._output_shape), 0, -1)]
				else:
					lw = [ 1  for num in range(int(self._output_shape), 0, -1)]

				# M  = (100 / length) * K.abs( 1 - K.sum(y_pred))
				diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
														K.epsilon(),
														None))

				diff = diff * lw

				return 100. * K.mean(diff, axis=-1)


	def PT__param_return(self):
		"""
		parameter : switch per trainable
		return : Action - returns callbacks for keras
		https://3months.tistory.com/424
		https://snowdeer.github.io/machine-learning/2018/01/09/find-best-model/
		"""
		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:

				early_stop, check_point, patience, epoch = None, None, None, None

				assert self._PT__model_state != None
				assert self._PT__model_state in PrivTensorWrapper.TRAIN_DICT

				train_Dict = PrivTensorWrapper.TRAIN_DICT[self._PT__model_state]

				early_stop = EarlyStopping(monitor='val_loss',
											mode='min',
											verbose=PrivTensorWrapper.VERBOSE,
											patience=train_Dict['patience'],
										    baseline=train_Dict['baseline']) # baseline = ~~ target value
				check_point = ModelCheckpoint(monitor='val_loss',
											  mode='min',
											  verbose=PrivTensorWrapper.VERBOSE,
											  save_best_only=True,
											  filepath=self._PT__save_file_location)

				return early_stop, check_point, train_Dict['epoch']


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def PT__train_model(self, X, Y):

		ensemble_num = PrivTensorWrapper.PT_CREATION_DIC_CNT[self._PT__stock_code]
		pushLog(dst_folder='PREDICTER__ML_CLASS',
				module='PT__train_model',
				memo=f'ensemble_num : {ensemble_num}')

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15)

		if self._model_mode == 'nn':
			X_train = np.array(X_train).reshape(-1, self._input_shape)
			X_test = np.array(X_test).reshape(-1, self._input_shape)
			Y_train = np.array(Y_train).reshape(-1, self._output_shape)
			Y_test = np.array(Y_test).reshape(-1, self._output_shape)

		elif self._model_mode == 'lstm':
			X_train = np.array(X_train).reshape(self._input_shape,1 ,1)
			X_test = np.array(X_test).reshape(self._input_shape,1 ,1)
			Y_train = np.array(Y_train).reshape(self._output_shape,1 ,1)
			Y_test = np.array(Y_test).reshape(self._output_shape,1 ,1)

		else:
			raise ValueError('wrong model mode configuration-2')


		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:

				## if want to do transfer learning
				""" if loaded or init training done! """
				if self._PT__ifLoaded or self._PT__initTrain: 
					for layer in self._MAIN_MODEL.layers:
						if 'trainable' in layer.name:
							layer.trainable = True
						else:
							layer.trainable = False

				early_stop, check_point, epoch = self.PT__param_return()

				self._MAIN_HISTORY = \
					self._MAIN_MODEL.fit(X_train, 
										Y_train,
										epochs=epoch,
										batch_size=PrivTensorWrapper.NUM_BATCH,
										shuffle=True,
										verbose=PrivTensorWrapper.VERBOSE,
										validation_data=(X_test, Y_test),
										callbacks=[early_stop,check_point])
				## calculate state change
				self.PT__calc_state()


				## refresh accuracy
				self._PT__model_accuracy = self.PT__acc_cal(Y_test, self._MAIN_MODEL.predict(X_test))

				## write accuracy
				try:
					with open(self._PT__save_score_txt_location, 'w') as f:
						f.write(str(self._PT__model_accuracy))
				except Exception as e:
					pushLog(dst_folder='PREDICTER__ML_CLASS', module='writing accuracy', exception=True, exception_msg=e, memo=f'fail to write model accuracy')

				## flag up
				self._PT__initTrain = True


	def PT__get_accuracy(self):
		"""

		:return: Action - returns accuracy instance in the class
		"""
		return self._PT__model_accuracy
	

	def PT__acc_cal(self, y_true, y_pred):
		# https://ebbnflow.tistory.com/123

		# 특이값 많은 경우 mae 사용
		# 아닌 경우 rmes 사용
		# return (1 / (mean_squared_error(y_true, y_pred)  + 1e-15) * 100)

		return ( 1 / (self.PT__custom_loss( y_true=y_true, y_pred=y_pred) + 1e-15) * 100  )


	def PT__predict_model(self, X_data):

		assert self._MAIN_SESS != None

		with self._MAIN_GRAPH.as_default() as g:
			with self._MAIN_SESS.as_default() as sess:

				X_data_conv = np.array(X_data).reshape(-1, self._input_shape)

				rtn = self._MAIN_MODEL.predict(X_data_conv) ## suppose is 2D list return

				return rtn[0]
	

class NestedGraph:
	
	LOOKUP = {}
	MAX_NUM_OF_ENSEM = 2
	LOOKUP_data = {}

	def __init__(self, shape_input, shape_output, minute_length, predict_length, dataque_length):

		## locations
		self.AT_SAVE_PATH__folder = str(os.getcwd() + "\\PREDICTER__MODEL_SAVE")
		if os.path.isdir(self.AT_SAVE_PATH__folder):
			pass
		else:
			os.mkdir(self.AT_SAVE_PATH__folder)


		## load other modules
		self.AGENT_SUB__encoder = EN.Autoencoder(module=True, simple=True)
		self.AGENT_SUB__denoiser = DE.Denoiser(module=True)

		## in / out shapes
		self.input_shape = shape_input
		self.output_shape = shape_output
		self.minute_length = minute_length
		self.predict_length = predict_length

		## rect day
		self.que_length = dataque_length
		self.dateKey = None


	def NG__check_stkcode(self, stock_code):
		"""

		:param stock_code: stock_code used
		:return: Action - check existance of the code in the nested graph dict
		"""

		if stock_code in NestedGraph.LOOKUP[self.NG__get_day()] and \
				stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]:
			return True
		else:
			return False


	def NG__set_day(self, _today):
		"""

		:param _today: date used to rectify and use as key
		:return:
		"""
		self.dateKey = FUNC_dtRect(_today, "00:00")


	def NG__get_day(self):
		"""

		:return: get the recified key date
		"""
		return self.dateKey


	def NG__clear(self, _today_date):
		"""
		param : None
		:param today_date : date value of datetime object
		return : Action - clean the model from the memory
						- clean the class variable dictionary
		"""
		print(f'enter NG__clear function')
		self.NG__clear_keras(_today_date=self.NG__get_day())
		self.NG__clear_data(_today_date=self.NG__get_day())


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__clear_keras(self, _today_date):
		"""
		param : None
		:param _today_date : date value of datetime object
		return : Action - clears keras graphs
		"""

		del_list__day = []
		for day in NestedGraph.LOOKUP:

			#if FUNC_dtRect(_today_date, "00:00") != day:
			if self.NG__get_day() != day:
				del_list__day.append(day)

				del_list__stkcode = []
				for stock_code in  NestedGraph.LOOKUP[day]:
					del_list__stkcode.append(stock_code)

					# clear session / graph in keras
					for _class in NestedGraph.LOOKUP[day][stock_code]:
						_class.PT__clear()

				# clear stock_codes in day key
				for d_stock_code in del_list__stkcode:
					print(f'deleting LOOKUP-keras : {d_stock_code}')
					del NestedGraph.LOOKUP[day][d_stock_code]

		# clear day in the lookup table
		for d_day in del_list__day:
			print(f'deleting LOOKUP : {d_day}')
			del NestedGraph.LOOKUP[d_day]


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__clear_data(self, _today_date):
		"""
		param : None
		:param _today_date : date value of datetime object
		return : Action - clear data savings for training
				 var :: self.LOOKUP_data
		"""
		del_list__day = []
		for day in NestedGraph.LOOKUP_data:

			#if FUNC_dtRect(_today_date, "00:00") != day:
			if self.NG__get_day() != day:
				del_list__day.append(day)

				del_list__stkcode = []
				for stock_code in NestedGraph.LOOKUP_data[day]:
					del_list__stkcode.append(stock_code)

				# delete stock code and dataset inside
				for d_stock_code in del_list__stkcode:
					print(f'deleting LOOKUP_data : {d_stock_code}')
					del NestedGraph.LOOKUP_data[day][d_stock_code]

		# delete day
		for d_day in del_list__day:
			print(f'deleting LOOKUP_data : {d_day}')
			del NestedGraph.LOOKUP_data[d_day]


	#@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__prediction_wrapper(self, stock_code, _day, X_data):
		"""
		param : stock_code
		:param _day: actual prediction datetime

		return : Action - returns mean value of predictions as an ensemble
		"""

		tmp_list = []
		assert stock_code in NestedGraph.LOOKUP[self.NG__get_day()]

		for stock_class in NestedGraph.LOOKUP[self.NG__get_day()][stock_code]:
			tmp_list.append(stock_class.PT__predict_model(X_data))

		#@ make array and take mean
		tmp_list = np.asarray(tmp_list)
		tmp_mean = np.mean(tmp_list, axis=0).tolist()


		#@ add prediction to the data class
		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]

		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code]
		data_class.DATA__save_prediction(datetime_obj=_day,
										 prediction_list=tmp_mean)

		return tmp_mean


	def NG__get_accuracy(self, stock_code):
		"""

		:param stock_code: stock code used
		:return: Action - returns accuracy of the predicters in the stock code, each in the list
						  is a container containing status of each model by (model state, model accuracy)
		"""

		tmp_return = []
		assert stock_code in NestedGraph.LOOKUP[self.NG__get_day()]

		for predicter in NestedGraph.LOOKUP[self.NG__get_day()][stock_code]:
			tmp_model_state = predicter.PT__get_state()
			tmp_model_acc = predicter.PT__get_accuracy()

			tmp_return.append([tmp_model_state, tmp_model_acc])


		return tmp_return



	def NG__get_prediction_dict(self, stock_code):
		"""
		:param stock_code: stock code used
		:return: Action - return prediction with raio calculated
		"""

		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]

		# @ data class
		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code]

		# @ dictionary
		tmp_savedDict = data_class.DATA__get_pred_saved()
		tmp_ratioDict = data_class.DATA__get_pred_ratio()


		return self.NG__calc_prediction_ratio(tmp_savedDict, tmp_ratioDict)


	def NG__calc_prediction_ratio(self, total_saved, total_ratio, get_latest : bool = False):
		"""

		:param total_saved: total prediction dataset, key : datetime.date
		:param total_ratio: total ratio of the start dataset at the key, key : str_datetime
		:param get_latest: boolean, if True / returns the latest prediction available
		:return:
		"""

		def decode_single_pred(pred_conved, ratio):
			rtn_list = [ (val + 1)*ratio for val in pred_conved ]
			return rtn_list

		if not get_latest:
			tmp_rtn_dict = { key : decode_single_pred(pred_conved=value, ratio=total_ratio[key]) \
							 for key, value in zip(total_saved.keys(), total_saved.values()) }

			return tmp_rtn_dict

		else: # get the latest prediction
			latest_key = sorted(total_saved.keys())[-1]
			#return  (total_saved[latest_key] + 1) * total_ratio[latest_key]
			return decode_single_pred(pred_conved=total_saved[latest_key], ratio=total_ratio[latest_key])



	def NG__training_wrapper(self, stock_code):
		"""

		:param stock_code: stock_code
		:return: Action - trains the graphs
		                  // if data was preped, returns according booleans
		"""

		assert stock_code in NestedGraph.LOOKUP[self.NG__get_day()]
		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]


		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code]
		X, Y = data_class.DATA__get_container()

		bool_check_train = True
		for stock_class in NestedGraph.LOOKUP[self.NG__get_day()][stock_code]:
			if X and Y : # non empty containers!
				stock_class.PT__train_model(X=X, Y=Y)
			else:
				bool_check_train = False

		if not bool_check_train:
			return False
		else:
			return True



	def NG__allocater(self, stock_code):
		"""
		:param stock_code
		:return: Action - create stock code folder, returns private tesnsor class
		"""

		# create stock folder location
		tmp_stock_code_folder = self.AT_SAVE_PATH__folder + '\\' + str(stock_code)
		if os.path.isdir(tmp_stock_code_folder):
			pass
		else:
			os.mkdir(tmp_stock_code_folder)
		
		# get save file location
		tmp_stock_code_checkpoint_file_location = tmp_stock_code_folder \
			  + '\\' + 'saved_model.h5'

		# get prediction score txt location
		tmp_sc_score_txt_location = tmp_stock_code_folder \
			+ '\\' + 'model_score.txt'

		# create instance and return
		rtn = PrivTensorWrapper(
			 stock_code=stock_code,
			 code_folder_location=tmp_stock_code_folder,
			 save_file_location=tmp_stock_code_checkpoint_file_location,
			 model_score_txt_lc=tmp_sc_score_txt_location,
			 input_shape=self.input_shape,
			 output_shape=self.output_shape)

		return rtn


	def NG__check_prep(self, stock_code):
		"""
		:param : stock_code, day - day value from datetime lib
		:return : Action - fills up self.LOOKUP dictionary
		"""

		## reset outdated graph / data
		self.NG__clear(_today_date=self.NG__get_day())

		print(f'enter NG__check_prep, allocate for stock_code : {stock_code}')
		print(f'enter NG__check_prep, day is : {self.NG__get_day()}')

		## allocate graph
		if self.NG__get_day() not in NestedGraph.LOOKUP:
			NestedGraph.LOOKUP[self.NG__get_day()] = {}
		
		if stock_code not in NestedGraph.LOOKUP[self.NG__get_day()]:
			NestedGraph.LOOKUP[self.NG__get_day()][stock_code] = []

		## make keras graph
		for trys in range(NestedGraph.MAX_NUM_OF_ENSEM):
			prd_made = len(NestedGraph.LOOKUP[self.NG__get_day()][stock_code])
			print(f'prd_made : {prd_made}')
			if  prd_made < NestedGraph.MAX_NUM_OF_ENSEM:
				NestedGraph.LOOKUP[self.NG__get_day()][stock_code].append(self.NG__allocater(stock_code=stock_code))
				print(f'passed prep!')
			else:
				pass
				print(f'skipped prep!')

		## allocate data class
		if self.NG__get_day() not in NestedGraph.LOOKUP_data:
			NestedGraph.LOOKUP_data[self.NG__get_day()] = {}

		if stock_code not in NestedGraph.LOOKUP_data[self.NG__get_day()]:
			NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code] = Dataset(stock_code=stock_code,
															   watch_length=self.que_length)


	def NG__wrapper(self, stock_code,
					stk_hashData=None,
					kospi_hashData=None,
					dollar_hashData=None):

		# @ prepare containers
		self.NG__check_prep(stock_code=stock_code)

		## assert stock code existance
		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]
		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code] # pointer

		if stk_hashData != None:
			data_class.DATA__stk_update(new_data=stk_hashData)

		if kospi_hashData != None:
			data_class.DATA__kospi_update(new_data=kospi_hashData)

		if dollar_hashData != None:
			data_class.DATA__dollar_update(new_data=dollar_hashData)


	def NG__check_article_first(self, stock_code, _day, article_hash, article_check):
		"""

		:param stock_code: stock_code
		:param _day: original dat / wo rectified
		:param article_pickle: article pickle
		:param article_check: either to check article existance or not
		:return: boolean for next step
		"""

		if article_check:


			rtn_article = self.NG__checkArticle(stock_code=stock_code,
												specific_time=_day,
												article_pickle=article_hash,
												request_time=_day)
			if rtn_article == None:  # no article exists
				return False

			else:
				return True

		else:
			return True


	def NG__getDatetime(self, datetime_obj):
		"""

		:param datetime_obj: datetime used
		"""

		return FUNC_getDatetimeConv(datetime_obj)


	def NG__dataCalculate(self, stock_code, _day, article_hash, article_check):
		"""

		:param stock_code: stock_code
		:param _day: original dat / wo rectified
		:param article_pickle: article pickle
		:param article_check: either to check article existance or not
		:return:
		"""

		## assert stock code existance
		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]
		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code] # pointer

		key__stkData = list(data_class._stk_dataset.keys())
		key__X_data = list(data_class._X_data.keys())

		## debuggers
		debug__data_skip = 0
		debug__article = 0
		debug__passed = 0


		## update needed datetime as list
		#update_needed = FUNC_dtLIST_str_sort(list( set(key__stkData) - set(key__X_data) ))
		update_needed = sorted(list(set(key__stkData) - set(key__X_data)))
		for i in range(self.minute_length, len(update_needed)-self.predict_length, 1 ):

			tmp_totContainer = []

			## skip existing datetime str if exists
			if data_class.DATA__check_existance(update_needed[i]):
				debug__data_skip += 1
				continue

			rtn_article = self.NG__checkArticle(stock_code=stock_code,
												specific_time=update_needed[i],
												article_pickle=article_hash,
												request_time=_day)
			if rtn_article == None : # no article exists
				if article_check :
					debug__article += 1
					continue
				else:
					# get dummy article
					rtn_article = self.NG__makeDummyArticle()
			data_class.DATA__article_update(new_data=rtn_article,
											_day=update_needed[i])

			rtn_X, rtn_Y = data_class.DATA__make_stock_set(date_list_data=update_needed[i-self.minute_length:i],
														   date_list_ans=update_needed[i:i+self.predict_length],
														   check_data_int=self.minute_length,
														   check_answer_int=self.predict_length,
														   datetime=update_needed[i])
			rtn_X_decoded = self.AGENT_SUB__denoiser.FUNC_PREDICT_MAIN__ontherun(\
			                     data_class.DATA__get_original_set( \
									 date_list_data=update_needed[i-self.minute_length:i],
									 check_data_int=self.minute_length))
			rtn_kospi = data_class.DATA__make_sub_set(date_list_data=update_needed[i-self.minute_length:i],
													  subset_type='kospi')
			rtn_dollar = data_class.DATA__make_sub_set(date_list_data=update_needed[i-self.minute_length:i],
													   subset_type='dollar')
			rtn_datetime = self.NG__getDatetime(datetime_obj=update_needed[i])

			debug__passed += 1

			## add date into the que to keep track on recent
			data_class.DATA__add_dateToSet(datetime=update_needed[i])


			## contain values
			tmp_totContainer.extend(rtn_X)
			tmp_totContainer.extend(rtn_X_decoded)
			tmp_totContainer.extend(rtn_kospi)
			tmp_totContainer.extend(rtn_dollar)
			tmp_totContainer.extend([rtn_datetime])
			tmp_totContainer.extend([rtn_article])

			## append data
			data_class.DATA__wrap_container(x_container=tmp_totContainer,
											y_container=rtn_Y,
											datetime=update_needed[i])

		pushLog(dst_folder='PREDICTER__ML_CLASS',
				lv='INFO', module='NG__dataCalculate', exception=True,
				memo=f'stock_code : {stock_code} \ndebug__passed : {debug__passed}, debug__article : {debug__article}, debug__data_skip : {debug__data_skip}')


	def NG__get_prediction_set(self, stock_code, _day, article_hash, article_check):
		"""

		:param stock_code: stock_code
		:param _day: actual datetime value of "now" for prediction
		:param article_check: boolean to check article
		:return: return the following data correct for _day variable
		"""

		## assert stock code existance
		assert stock_code in NestedGraph.LOOKUP_data[self.NG__get_day()]
		data_class = NestedGraph.LOOKUP_data[self.NG__get_day()][stock_code] # pointer
		key__stkData = list(data_class._stk_dataset.keys())

		## check _day existance in the dataset hash
		#targ_date = FUNC_dtSwtich(_day)
		#assert _day in data_class._stk_dataset

		tmp_totContainer = []

		rtn_article = self.NG__checkArticle(stock_code=stock_code,
											specific_time=_day,
											article_pickle=article_hash,
											request_time=_day)

		if rtn_article == None : # no article exists
			if article_check :
				pass
			else:
				# get dummy article
				rtn_article = self.NG__makeDummyArticle()

		## update needed datetime as list
		#update_needed = FUNC_dtLIST_str_sort(key__stkData)
		update_needed = sorted(key__stkData)

		rtn_X = data_class.DATA__get_prediction( \
			               date_list_data=update_needed[len(update_needed) - self.minute_length:],
						   check_data_int=self.minute_length,
						   datetime=_day)
		rtn_X_decoded = self.AGENT_SUB__denoiser.FUNC_PREDICT_MAIN__ontherun( \
			data_class.DATA__get_original_set( \
				date_list_data=update_needed[len(update_needed) - self.minute_length:],
				check_data_int=self.minute_length))
		rtn_kospi = data_class.DATA__make_sub_set(date_list_data=update_needed[len(update_needed) - self.minute_length:],
												  subset_type='kospi')
		rtn_dollar = data_class.DATA__make_sub_set(date_list_data=update_needed[len(update_needed) - self.minute_length:],
												   subset_type='dollar')
		rtn_datetime = self.NG__getDatetime(datetime_obj=_day)


		## contain values
		tmp_totContainer.extend(rtn_X)
		tmp_totContainer.extend(rtn_X_decoded)
		tmp_totContainer.extend(rtn_kospi)
		tmp_totContainer.extend(rtn_dollar)
		tmp_totContainer.extend([rtn_datetime])
		tmp_totContainer.extend([rtn_article])

		return tmp_totContainer



	def NG__checkArticle(self, stock_code, specific_time, request_time,
					  article_loc=None, article_pickle=None):
		"""

		:param stock_code: stock_code
		:param specific_time: time now to retrieve article of net 5days
		:param request_time: to check article in the weekend // maybe not needed!
		:param article_loc:
		:param article_pickle:
		 -> will check article_loc and article_pickle if both are None

		:return: wrapper for reading article, returns calculated result
		"""



		rtn = self.AGENT_SUB__encoder.FUNC_SIMPLE__read_article(article_loc=article_loc,
																article_pickle=article_pickle,
																stock_code=stock_code,
																_specific_time=specific_time)

		return rtn


	def NG__makeDummyArticle(self):
		"""

		:return: incase dummy return needed, process it and return.
		"""

		rtn = self.AGENT_SUB__encoder.FUNC_SIMPLE__dummy_calc()

		return rtn


class Dataset:

	_kospi_dataset = {}
	_dollar_dataset = {}


	def __init__(self, stock_code, watch_length):
		self._stock_code = stock_code
		self._stk_dataset = {}
		self._article_dataset = {}
		
		self._ratio_train_dataset = {} # start data[0] record
		self._ratio_predict_dataset = {}

		self._prediction_dataset = {}

		self._datetime_que = dq(maxlen=int(watch_length))

		self._X_data = {}
		self._Y_data = {}


	def DATA__get_pred_ratio(self):
		"""

		:return: Action - returns saved start value ratio of the predict dataset
		"""

		return self._ratio_predict_dataset


	def DATA__get_pred_saved(self):
		"""
		:return: Action - returns saved prediction values in the data class
		"""

		return self._prediction_dataset


	def DATA__save_prediction(self, datetime_obj : datetime.date, prediction_list : list):
		"""

		:param datetime_obj: datetime object as input
		:param prediction_list: prediction list containing predicted values
		:return: Action - to save predict values
		"""

		self._prediction_dataset[datetime_obj] = prediction_list


	def DATA__get_prediction(self, date_list_data, check_data_int, datetime):
		"""

		:param date_list_data: date list to attrive back
		:param check_data_int: to check length of parsed data
		:param datetime_str: key to record first val of prediction dataset
		:return: create data
		"""
		assert len(date_list_data) == check_data_int

		standard_first_val = self._stk_dataset[date_list_data[0]]['price']

		rtn_list_data = []
		tmp_data_price = [ ( self._stk_dataset[date]['price'] / standard_first_val) - 1 for  \
					         n, date in enumerate(date_list_data) ]
		tmp_data_volume = [  self._stk_dataset[date]['volume'] for n, date in enumerate(date_list_data) ]
		_ = [rtn_list_data.extend([t_p, t_v]) for t_p, t_v in zip(tmp_data_price, tmp_data_volume)]
		# rtn_list_data.extend(tmp_data_price)
		# rtn_list_data.extend(tmp_data_volume)

		assert len(rtn_list_data) == len(date_list_data) * 2

		#@ record first val
		self._ratio_predict_dataset[datetime] = standard_first_val


		return  rtn_list_data


	def DATA__add_dateToSet(self, datetime):
		"""
		:param datetime: datetime value to put to queue
		:return: Action - add datetime object into the
		                  tobe used to limit train data length
		"""
		if datetime in self._datetime_que:
			pass
		else:
			self._datetime_que.append(datetime)



	def DATA__get_container(self):
		"""

		:return: get _X / _Y for training / returns list
		"""
		tmp_X = [ data for key1, key2, data in zip(self._X_data.keys(), self._Y_data.keys(), self._X_data.values()) \
				  if key1 == key2 if key1 in self._datetime_que]

		tmp_Y = [ data for key1, key2, data in zip(self._Y_data.keys(), self._X_data.keys(), self._Y_data.values()) \
				  if key1 == key2 if key1 in self._datetime_que]

		return tmp_X, tmp_Y


	def DATA__wrap_container(self, x_container, y_container, datetime):
		"""

		:param x_container: x value to append to _X_data
		:param y_container: y vale to append to _Y_data
		:return: add _X / _Y dictionary it's new values
		"""
		# @ skip existing data
		if datetime in self._X_data or datetime in self._Y_data:
			return

		self._X_data[datetime] = x_container
		self._Y_data[datetime] = y_container


	def DATA__make_stock_set(self, date_list_data, date_list_ans, check_data_int, check_answer_int, datetime):
		"""

		:param date_list_data: original stock parse date list
		:param date_list_ans: original stock parse answer list
		:param check_data_int: to check length of parsed data
		:param check_answer_int: to check length of parsed answer
		:return: create X, Y data partials from current stock data
		"""

		assert len(date_list_data) == check_data_int
		assert len(date_list_ans) == check_answer_int

		standard_first_val = self._stk_dataset[date_list_data[0]]['price']

		rtn_list_data = []
		tmp_data_price = [ ( self._stk_dataset[date]['price'] / standard_first_val) - 1 for  \
					         n, date in enumerate(date_list_data) ]
		tmp_data_volume = [  self._stk_dataset[date]['volume'] for n, date in enumerate(date_list_data) ]
		# rtn_list_data.extend(tmp_data_price)
		# rtn_list_data.extend(tmp_data_volume)
		_ = [ rtn_list_data.extend([t_p, t_v]) for t_p, t_v in zip(tmp_data_price, tmp_data_volume) ]


		rtn_list_answer = []
		tmp_answer_price = [ ( self._stk_dataset[date]['price'] / standard_first_val) - 1 for  \
					           n, date in enumerate(date_list_ans) ]
		
		#@ record ratio -> time when it was made
		self._ratio_train_dataset[datetime] = standard_first_val
		
		assert len(rtn_list_data) == len(date_list_data) * 2
		assert len(tmp_answer_price) == len(date_list_ans)

		return rtn_list_data, tmp_answer_price


	def DATA__get_original_set(self, date_list_data, check_data_int):
		"""

		:param date_list_data: datetime list to parse from (specific window to work with)
		"param check_data_int: to check length of parsed data
		:return:
		"""

		assert len(date_list_data) == check_data_int

		tmp_data_price = [(self._stk_dataset[date]['price']) for \
						  n, date in enumerate(date_list_data)]

		assert len(tmp_data_price) == check_data_int

		return tmp_data_price


	def DATA__make_sub_set(self, date_list_data, subset_type):
		"""

		:param date_list_data: datetime list to parse from (specific window to work with)
		:param subset_type:
		:return: make calculation based on mean values of kospi and dollar
		"""
		assert subset_type in ['kospi', 'dollar']

		def return_market_status(hash, _date_list_data):

			assert len(list(hash.keys())) > 0

			#tmp_hash_key_srted = FUNC_dtLIST_str_sort(list(hash.keys()))
			tmp_hash_key_srted = sorted(list(hash.keys()))
			tmp_filtered_recent = [ data for data in tmp_hash_key_srted if data in _date_list_data]

			## 1st layer
			price = []
			for dt in tmp_filtered_recent:
				price.append(hash[dt]['price'])

			mean = sum(price) / len(price)
			latest_price = price[-1]

			return [(latest_price - mean) / mean]

		if subset_type == 'kospi':
			return return_market_status(Dataset._kospi_dataset, date_list_data)

		elif subset_type == "dollar":
			return return_market_status(Dataset._dollar_dataset, date_list_data)



	def DATA__check_existance(self, datetime_str):
		"""

		:param datetime_str: specific datetime to check
		:return: if the datetime exists, returns False / else True
		"""
		tmp_bool = False
		if datetime_str in self._X_data:
			tmp_bool = True
		else:
			tmp_bool = False

		return tmp_bool


	def DATA__stk_update(self, new_data:dict):
		"""
		:param new_data : input from outside, type dict
		:return : Action - update new data on interest stock _stk_dataset instance
		"""
		assert isinstance(new_data, dict)
		self._stk_dataset.update(new_data)


	def DATA__kospi_update(self, new_data:dict):
		"""
		:param new_data : input from outside, type dict
		:return : Action - update new data on interest stock _kospi_dataset instance
		"""
		assert isinstance(new_data, dict)
		self._kospi_dataset.update(new_data)


	def DATA__dollar_update(self, new_data:dict):
		"""
		:param new_data : input from outside, type dict
		:return : Action - update new data on interest stock _dollar_dataset instance
		"""
		assert isinstance(new_data, dict)
		self._dollar_dataset.update(new_data)


	def DATA__article_update(self, new_data:float, _day):
		"""
		:param new_data : input from outside, type float
		:return : Action - update new data on interest stock _article_dataset instance
		"""
		assert isinstance(new_data, float) or isinstance(new_data, int)
		#_day_str = FUNC_dtSwtich(datetime_item=_day)
		self._article_dataset.update({_day : new_data})



