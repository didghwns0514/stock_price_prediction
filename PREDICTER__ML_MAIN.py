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
from LOGGER_FOR_MAIN import pushLog
from DATA_OPERATION import *





class Stock_prediction:

	## Wrapper Name
	NAME = 'stock_prediction_'

	## Must watch list
	WATCH_LIST = ["226490", "261250" ]#, "252670"]
	# KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X

	LENGTH__MINUTE_DATA = int(60 * 4) # 3 data used, stock / kospi / dollar-mearchant
	LENGTH__NEWS_ENCODED = int(20)
	LENGTH__ALL_INPUT = int(LENGTH__MINUTE_DATA * 3) \
						+ int(LENGTH__NEWS_ENCODED) \
						+ int(1200)
	LENGTH__ALL_OUTPUT = int(30)

	HOURS_WATCH = int(13)



	def __init__(self, module=True):

		# @ previous declarations
		self.AGENT_SUB__encoder = EN.Autoencoder(module=True,simple=True)
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

		return SUB_F.FUNC_dtRect(_datetime_obj, "00:00")
		# return _datetime_obj.replace(hour=0,
		# 							 second=0,
		# 							 minute=0,
		# 							 mirosecond=0)

	def _checkArticle(self, stock_code, specific_time,
					  article_loc=None, article_pickle=None):
		"""

		:param stock_code: stock_code
		:param specific_time: time now to retrieve article of net 5days
		:param article_loc:
		:param article_pickle:
		 -> will check article_loc and article_pickle if both are None
		:return: wrapper for reading article, returns calculated result
		"""

		rtn = self.AGENT_SUB__encoder.FUNC_SIMPLE__read_article(article_loc=article_loc,
														  article_pickle=article_pickle,
														  stock_code=stock_code,
														  specific_time=specific_time)

		return rtn

	def _checkStock(self):
		pass



def Session():



	def sqlite_capture(db_loc):

		sqlite_conTop = sqlite3.connect(db_loc)
		sqlite_curTop = sqlite_conTop.cursor()

		def wrapper(_stock_code=None, get_codes=False):

			if get_codes:
				tmp_codes__obj = sqlite_curTop.execute("SELECT name FROM sqlite_master WHERE type='table';")
				tmp_codes = tmp_codes__obj.fetchall()
				tmp_codes = [ list(value)[0] for value in tmp_codes ]
				return tmp_codes

			else:
				stock_code = str(_stock_code)
				assert stock_code != None

				head_string = 'SELECT * FROM '
				tmp_selected = "'" + str(stock_code) + "'"
				tmp_df = pd.read_sql(head_string + tmp_selected, sqlite_conTop, index_col=None)

				if (not tmp_df.empty)  and  len(tmp_df) >= int(900 * 0.99):
					rtn_df = tmp_df.loc[  (tmp_df['open'] >= 5000) \
									    & (np.mean(tmp_df['volume'])>=500 )
							           ].copy()
					if rtn_df.empty : # 빈 데이터
						return pd.DataFrame() # return real empty df
					else:
						## change date to datetime
						rtn_df['date'] = pd.to_datetime(rtn_df['date'],  format="%Y%m%d%H%M%S" )
						print(f'stock code that was selected... : {stock_code}')
						print(f'{rtn_df.head()}')
						return rtn_df
				else: # not enough data / or no data at all
					return pd.DataFrame() # return real empty df

		return wrapper


	def sweep_day(df, start_date, end_date, hours_back = int(13), minute_forward = int(30),_type='data'):
		"""

		:param df: dataframe input
		:param start_date:
		:param end_date:
		:return:
		"""

		assert _type in ['data', 'answer']

		tmp_dt_start = start_date
		tmp_dt_sweep = start_date
		tmp_dt_end = end_date
		return_hash = None

		while tmp_dt_sweep <= tmp_dt_end:

			## calculate back
			tmp_backwards = SUB_F.FUNC_datetime_backward(
				datetime_now__obj_=tmp_dt_sweep,
				hours_back=hours_back
			)
			if _type == 'answer':
				return_hash = SQ__parse_answer(ori_df_whole=df,
											  dt_now__obj=tmp_dt_sweep,
											  minute_forward=minute_forward)
			else:
				return_hash = SQ__parse_sqData(ori_df_whole=df,
											   dt_now__obj=tmp_dt_sweep,
											   hours_duration_back=hours_back)
			## right now
			tmp_dt_sweep += datetime.timedelta(minutes=1)

			yield return_hash




	# # @ import for db
	import sqlite3

	
	#@ stock prediction wrapper class
	sp = Stock_prediction(module=True)

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
	list_of_codes = sqlite_closure(get_codes=True)


	#####################
	## var for loading!
	pickle_article = None
	with open(dir_article__file, 'rb') as file:
		pickle_article = copy.deepcopy(pickle.load(file))

	pickle_visited_list = None
	with open(dir_pickle_skipped__file, 'rb') as file:
		pickle_visited_list = copy.deepcopy(pickle.load(file))
	#####################

	df__kospi = sqlite_closure(_stock_code=str(226490))
	df__dollar = sqlite_closure(_stock_code=str(261250))
	MUST_WATCH_LIST = ["226490", "261250"] # "252670"
	"""                KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X"""


	for stock_code in list_of_codes:
		pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
				exception=True,
				memo=f'entered stock : {str(stock_code)}')

		# @ skip must watch list
		if stock_code in MUST_WATCH_LIST:
			pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
					exception=True,
					memo=f'stock code in MUST_WATCH_LIST')
			continue

		# @ check empty dataframe
		main_Stk_df = sqlite_closure(_stock_code=str(stock_code))
		if main_Stk_df.empty:
			pushLog(dst_folder='SESSION__PREDICTER__ML_MAIN',
					exception=True,
					memo=f'stock code did not match standards, returned None type')
			continue

		mainStk_dt_start__obj = main_Stk_df.date.min() + datetime.timedelta(days=3)
		mainStk_dt_end__obj = main_Stk_df.date.max() - datetime.timedelta(days=1)


		#################################
		# If all has passed : start from here!
		#################################
		while mainStk_dt_start__obj <= mainStk_dt_end__obj:
			tmp_dt_start__obj = SUB_F.FUNC_dtRect(mainStk_dt_start__obj,"9:00")




			# add day
			mainStk_dt_start__obj += datetime.timedelta(days=1)

	print(f'total execution finished!')
















if __name__ == '__main__':

	# training begin
	Session()




