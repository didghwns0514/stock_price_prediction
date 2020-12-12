# -*-coding: utf-8-*-

# @ normal library
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
import traceback
import joblib
import pickle
import codecs
import sqlite3

# # @ keras
# import keras
# from keras.models import Sequential, Model, load_model
# from keras.optimizers import Adam
# from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
# from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, UpSampling1D
# from keras.utils import np_utils
# from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# from keras import backend as K
# import h5py
# from keras.backend.tensorflow_backend import set_session


# # @ tensorflow
# import tensorflow as tf


# @ outside module
import ENCODER__ML_MAIN as EN
import DENOISER__ML_MAIN as DE
import sub_function_configuration as SUB_F
import PREDICTER__ML_CLASS as PCLS
from LOGGER_FOR_MAIN import pushLog as pl




class Stock_prediction:

	## Wrapper Name
	NAME = 'stock_prediction_'

	## Must watch list
	WATCH_LIST = ["226490", "261250", "252670"]

	LENGTH__MINUTE_DATA = int(60 * 4) # 3 data used, stock / kospi / dollar-mearchant
	LENGTH__NEWS_ENCODED = int(20)
	LENGTH__ALL_INPUT = int(LENGTH__MINUTE_DATA * 3) \
						+ int(LENGTH__NEWS_ENCODED) \
						+ int(1200)
	LENGTH__ALL_OUTPUT = int(30)



	def __init__(self, module=True):

		# @ previous declarations
		self.AGENT_SUB__encoder = EN.Autoencoder(module=True)
		self.AGENT_SUB__denoiser = DE.Denoiser(module=True)
		self.nestgraph = PCLS.NestedGraph(shape_input=Stock_prediction.LENGTH__ALL_INPUT,
		shape_output=Stock_prediction.LENGTH__ALL_OUTPUT)

		#self.options = Options(self.envs)
		self.module = module


	def _getDay(self, datetime_obj = None):
		"""
		param : datetime_obj
		return : returns 'day only object' time 
		"""
		if datetime_obj:
			_datetime_obj = datetime_obj
		else:
			_datetime_obj = datetime.datetime.now()
		return _datetime_obj.replace(hour=0,
									 second=0,
									 minute=0,
									 mirosecond=0)



def Session():

	def sqlite_capture(db_loc):

		sqlite_conTop = sqlite3.connect(db_loc)
		sqlite_curTop = sqlite_conTop.cursor()

		def wrapper(stock_code=None, get_codes=True):

			if get_codes:
				tmp_codes__obj = sqlite_curTop.execute("SELECT name FROM sqlite_master WHERE type='table';")
				tmp_codes = tmp_codes__obj.fetchall()
				tmp_codes = [ list(value)[0] for value in tmp_codes ]

				return tmp_codes

			else:
				assert stock_code != None
				head_string = 'SELECT * FROM '
				tmp_selected = "'" + str(stock_code) + "'"
				tmp_df = pd.read_sql(head_string + tmp_selected, sqlite_conTop, index_col=None)

				if (not tmp_df.empty)  and  len(tmp_df) >= int(900 * 0.99):
					rtn_df = tmp_df.loc[  (tmp_df['open'] >= 5000) \
									    & (np.mean(tmp_df['volume'])>=500 )
							           ].copy()
					if rtn_df.empty : # 빈 데이터
						return None
					else:
						return rtn_df
				else: # not enough data / or no data at all
					return None

		return wrapper



	# # @ import for db
	# import sqlite3
	# import pickle
	
	#@ stock prediction wrapper class
	#sp = Stock_prediction(module=True)

	## current working python directory
	current_wd = os.getcwd().replace('/', '\\')

	## directory tobe used
	dir_db__folder = current_wd + '\\' + 'PREDICTER__DATABASE_single'
	dir_article__folder = current_wd + '\\' + 'PREDICTER__ARTICLE_check'

	## db and pickle
	dir_db__file = dir_db__folder + '\\' + 'SINGLE_DB.db'
	dir_pickle__file = dir_db__folder + '\\' + 'parsed_list_pickle.p'
	dir_article__file = dir_db__folder + '\\' + 'pickle.p'
	dir_pickle_skipped__file = dir_db__folder + '\\' + 'stock_list_pickle__skipped.p'


	# @ Sqlite object
	sqlite_closure = sqlite_capture(dir_db__file)
	sqlite_closure(get_codes=True)


	#####################
	## var for loading!
	pickle_article = None
	with open(dir_article__file, 'rb') as file:
		pickle_article = copy.deepcopy(pickle.load(file))

	pickle_visited_list = None
	with open(dir_pickle_skipped__file, 'rb') as file:
		pickle_visited_list = copy.deepcopy(pickle.load(file))
	#####################












if __name__ == '__main__':

	# training begin
	Session()




