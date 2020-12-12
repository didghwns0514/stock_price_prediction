
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # turn off gpu setting

import pickle
import numpy as np

# @ xgboost
import xgboost

# @ tensorflow
import tensorflow as tf

# @ keras
from keras.models import Sequential, Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, Dropout,  Activation, Flatten, Reshape, Input
import keras.backend as K


# @ sklearn model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


# @ logger
from LOGGER_FOR_MAIN import pushLog


class PrivTensorWrapper:

	PT_CREATION_DIC_CNT = {} # cnter dictionary per model
	NAME = 'PrivTensorWrapper_'
	GET_WEIGHTED_LOSS = True
	NUM_BATCH = int(100) # Batch size


	def __init__(self,
				stock_code,
				code_folder_location,
				save_file_location,
				model_score_txt_lc,
				input_shape,
				output_shape,
				loading_bool = False):

		# cnter for model numbers
		if stock_code not in PrivTensorWrapper.PT_CREATION_DIC_CNT:
			PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code] = 0
		PrivTensorWrapper.PT_CREATION_DIC_CNT[stock_code] += 1

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
		#----------------------------------------

		## required locations
		self._PT__code_folder_location = code_folder_location
		self._PT__save_file_location = save_file_location
		self._PT__save_score_txt_location = model_score_txt_lc
		

		self._MAIN_GRAPH = tf.Graph()
		self._MAIN_SESS = tf.Session(config=self._PT__config)
		self._MAIN_MODEL = None

		## load/build model
		self.PT__handle_mode()

	
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
		except Exception as e:
			pushLog(dst_folder='PREDICTER__ML_CLASS',module='PT__clear',exception=True, exception_msg=e,memo=f'session clear success')
			

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
				self._MAIN_MODEL.save_weights(self._PT__save_file_location)


	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def PT__handle_mode(self):
		"""
		param : None
		return : Action - load if model save file exist
						- differenct action based on Tensor usage
		"""

		## get active model
		if os.path.isfile(self._PT__save_file_location) and \
			PrivTensorWrapper.PT_CREATION_DIC_CNT[self._PT__stock_code]==0:
			with self._MAIN_GRAPH.as_default() as g:
				with self._MAIN_SESS.as_default() as sess:
					self._PT__ifLoaded = True
					self._MAIN_MODEL = load_model(self._PT__save_file_location)

		else:
			with self._MAIN_GRAPH.as_default() as g:
				with self._MAIN_SESS.as_default() as sess:
					self._MAIN_MODEL = self.PT__build_model()


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
				
				lw = None
				if PrivTensorWrapper.GET_WEIGHTED_LOSS: # diffrentiated loss
					lw = list(range(self._output_shape, 0, -1))
				else:
					lw = [1 for _ in range(0, self._output_shape,1)]

				## build model
				model = Sequential([
					Dense(120, 
						  activation='relu', 
						  input_shape=[self._input_shape], 
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

				## different optimizer for preloaded model instance
				optimizer = None
				if self._PT__ifLoaded:
					optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
				else:
					optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

				# model.compile(loss=tf.keras.losses.MeanAbsoluteP, 
				#               optimizer=optimizer,
				#               loss_weights=lw) # loss='mse'
				model.compile(loss=self.PT__custom_loss, 
							  optimizer=optimizer,
							  loss_weights=lw) # loss='mse'

				return model


	def PT__custom_loss(self, y_true, y_pred):

		# https://brunch.co.kr/@chris-song/34
		# https://uos-deep-learning.tistory.com/3
		# https://stackoverflow.com/questions/49729522/why-is-the-mean-average-percentage-errormape-extremely-high

		# y_true_f = K.flatten(y_true)
		# y_pred_f = K.flatten(y_pred)
		# length = K.intshape(y_true)[1]

		# M  = (100 / length) * K.abs( 1 - K.sum(y_pred))
		diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
												K.epsilon(),
												None))
		return 100. * K.mean(diff, axis=-1)       


	def PT__param_return(self, transfer=False):
		"""
		parameter : switch per trainable
		return : Action - returns callbacks for keras
		https://3months.tistory.com/424
		https://snowdeer.github.io/machine-learning/2018/01/09/find-best-model/
		"""
		early_stop, check_point, patience, epoch = None, None, None, None

		if transfer:
			patience = int(10)
			epoch = int(50)
		else:
			patience = int(50)
			epoch = int(100)
			
		early_stop = EarlyStopping(monitor='val_loss', 
									mode='min', 
									verbose=0, 
									patience=patience) # baseline = ~~ target value
		check_point = ModelCheckpoint(monitor='val_loss', 
									  mode='min',
									  verbose=0, 
									  save_best_only=True, 
									  filepath=self._PT__save_file_location)
		
		return early_stop, check_point, epoch



	def PT__train_model(self, X, Y):

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15)

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

				early_stop, check_point, epoch = self.PT__param_return(transfer=self._PT__ifLoaded)

				self._MAIN_HISTORY = \
					self._MAIN_MODEL.fit(X_train, 
										Y_train,
										epochs=epoch,
										batch_size=PrivTensorWrapper.NUM_BATCH,
										shuffle=True,
										verbose=0,
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
	

	def PT__acc_cal(self, y_true, y_pred):
		# https://ebbnflow.tistory.com/123

		# 특이값 많은 경우 mae 사용
		# 아닌 경우 rmes 사용
		 return mean_squared_error(y_true, y_pred)


	def PT__predict_model(self, X_data):

		assert self._MAIN_SESS != None

		rtn = self._MAIN_MODEL.predict(X_data) ## suppose is 2D list return

		return rtn[0]
	

class NestedGraph:
	
	LOOKUP = {}
	MAX_NUM_OF_ENSEM = 2
	LOOKUP_data = {}

	def __init__(self, shape_input, shape_output):

		## locations
		self.AT_SAVE_PATH__folder = str(os.getcwd() + "\\PREDICTER__MODEL_SAVE")
		if os.path.isdir(self.AT_SAVE_PATH__folder):
			pass
		else:
			os.mkdir(self.AT_SAVE_PATH__folder)

		## in / out shapes
		self.input_shape = shape_input
		self.output_shape = shape_output


	def NG__clear(self):
		"""
		param : None
		return : Action - clean the model from the memory
						- clean the class variable dictionary
		"""

		self.NG__clear_keras()
		self.NG__clear_data()

	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__clear_keras(self):
		"""
		param : None
		return : Action - clears keras graphs
		"""
		for day in NestedGraph.LOOKUP:
			for stock_code in  NestedGraph.LOOKUP[day]:
				for _class in NestedGraph.LOOKUP[day][stock_code]:
					_class.PT__clear() # clear session / graph in keras

				# clear stock_codes in day key
				del NestedGraph.LOOKUP[day][stock_code]

			# clear day in the lookup table
			del NestedGraph.LOOKUP[day]

	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__clear_data(self):
		"""
		param : None
		return : Action - clear data savings for training
				 var :: self.LOOKUP_data
		"""
		for day in NestedGraph.LOOKUP_data:
			for stock_code in NestedGraph.LOOKUP_data[day]:
				for datset in NestedGraph.LOOKUP_data[day][stock_code]:

					# delete dataset
					del NestedGraph.LOOKUP_data[day][stock_code][datset]

				# delete stock code
				del NestedGraph.LOOKUP_data[day][stock_code]
			
			# delete day
			del NestedGraph.LOOKUP_data[day]
				

	@pushLog(dst_folder='PREDICTER__ML_CLASS')
	def NG__prediction_wrapper(self, stock_code, X_data):
		"""
		param : stock_code
		return : Action - returns mean value of predictions as an ensemble
		"""
		tmp_list = []
		for stock_code in NestedGraph.LOOKUP:
			tmp_list.append(NestedGraph.LOOKUP[stock_code].PT__predict_model(X_data))
		
		tmp_list = np.asarray(tmp_list)

		return np.mean(tmp_list, axis=1)


	def NG__normalize_data(self, X_data_list, Y_data_list):
		"""
		param : data to be normalized,
				input - CNN encoded + news + normalized original of 3 critical
					  - save data for training!
					  var :: self.LOOKUP_data
					  => 위에서 해야 될 듯?
		"""

		pass


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
	

	def NG__check_graph(self, stock_code, day):
		"""
		param : stock_code, day - day value from datetime lib
		return : Action - fills up self.LOOKUP dictionary
		"""

		if day not in NestedGraph.LOOKUP:
			NestedGraph.LOOKUP[day] = {}
		
		if stock_code not in NestedGraph.LOOKUP[day]:
			NestedGraph.LOOKUP[day][stock_code] = []
		
		if len(NestedGraph.LOOKUP[day][stock_code])  \
			 < self.MAX_NUM_OF_ENSEM:
				NestedGraph.LOOKUP[day][stock_code].append(self.NG__allocater(stock_code=stock_code))
