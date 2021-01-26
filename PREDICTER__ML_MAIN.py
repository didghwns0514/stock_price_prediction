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
import sub_function_configuration as SUB_F
import PREDICTER__ML_CLASS as PCLS
from LOGGER_FOR_MAIN import pushLog
from DATA_OPERATION import *
from RETURN_WRAPPER import ReturnWrap





class Stock_prediction:

	## Wrapper Name
	NAME = 'stock_prediction_'

	## Must watch list
	WATCH_LIST = ["226490", "261250" ]#, "252670"]
	# KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X

	## article strict match
	ARTICLE_CHECK = True

	LENGTH__MINUTE_DATA = int((60 * 4)) # 3 data used, stock / kospi / dollar-mearchant
	LENGTH__NEWS_ENCODED = int(1)
	LENGTH__ALL_INPUT = int(LENGTH__MINUTE_DATA * 2) + int(1*2) \
						+ int(LENGTH__NEWS_ENCODED) \
						+ int(1200)
	LENGTH__ALL_OUTPUT = int(30)

	HOURS_WATCH = int(13)



	def __init__(self, module=True):

		# @ previous declarations
		self.nestgraph = PCLS.NestedGraph(shape_input=Stock_prediction.LENGTH__ALL_INPUT,
										  shape_output=Stock_prediction.LENGTH__ALL_OUTPUT,
										  minute_length=Stock_prediction.LENGTH__MINUTE_DATA,
										  predict_length=Stock_prediction.LENGTH__ALL_OUTPUT)

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




	def _checkStock(self, stock_code, hash_stock, hash_kospi, hash_dollar, hash_article, _today):
		"""
		:param stock_code: stock_code
		:param hash_stock: hash data of stock
		:param hash_kospi: hash data of kospi
		:param hash_dollar: hash data of dollar
		:param _today: datetime obj of date only
		:param hash_answer: 
		"""
		이부분, today가 주말일 경우와, 다른 부분에서 사용되는 곳에서
		주말일 경우, 주말 article 까지 포함해야 하므로, 그부분 확인해서 wrapper 작성할 것!
		artc_rtn = self.nestgraph.NG__check_article_first(stock_code=stock_code,
											   _day=_today,
											   article_hash=hash_article,
											   article_check=Stock_prediction.ARTICLE_CHECK)
		if artc_rtn:

			self.nestgraph.NG__wrapper(stock_code=stock_code,
									   stk_hashData=hash_stock,
									   kospi_hashData=hash_kospi,
									   dollar_hashData=hash_dollar
									   )

			self.nestgraph.NG__dataCalculate(stock_code=stock_code,
											 _day=_today,
											 article_hash=hash_article,
											 article_check=Stock_prediction.ARTICLE_CHECK)

			bool_trainable = self.nestgraph.NG__training_wrapper(stock_code=stock_code)

			if bool_trainable:

				#rtn_dataForPredic 이부분 체크, None 계속 들어간다
				rtn_dataForPredic = self.nestgraph.NG__get_prediction_set(stock_code=stock_code,
																		  _day=_today,
																		  article_hash=hash_article,
																		  article_check=Stock_prediction.ARTICLE_CHECK)
				if rtn_dataForPredic != None: # article existance problem
					rtn_predicted = self.nestgraph.NG__prediction_wrapper(stock_code=stock_code,
																		  _day=_today,
																		  X_data=rtn_dataForPredic)
					print(f'Predictable is True')
					return ['Predictable', rtn_predicted]

				else:
					print(f'rtn_dataForPredic if None : {rtn_dataForPredic}')
					return ['No-prediction_set', None]

			else:
				print(f'bool_trainable is false : {bool_trainable}')
				return ['Not-traininable', None]

		else:
			print(f'check article is false : {artc_rtn}')
			return ['No-article', None]

		
	def _stock_op_wrapper(self, stock_code, hash_stock, hash_kospi, hash_dollar, _today, hash_article):

		# @ set rect day in the NG
		self.nestgraph.NG__set_day(_today=_today)

		# @ operate on the stock
		rtn = self._checkStock(stock_code=stock_code,
						   hash_stock=hash_stock,
						   hash_kospi=hash_kospi,
						   hash_dollar=hash_dollar,
						   _today=_today,
						   hash_article=hash_article)


		return rtn




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


	def sweep_day(df, stock_code, start_date, end_date,
				  hours_back = int(13), minute_forward = int(30), _type='data',
				  min_dur=int(1)):
		"""

		:param df: dataframe input
		:param start_date:
		:param end_date:
		:return: yields hashs until stopiteration exception
			=> except StopIteration:: happends at the end of the function
		"""

		assert _type in ['data', 'answer']

		tmp_dt_start = copy.deepcopy(start_date)
		tmp_dt_sweep = copy.deepcopy(start_date)
		tmp_dt_end = end_date
		return_hash = None

		print(f'begin parsing stock_code : {stock_code}')

		while tmp_dt_sweep <= tmp_dt_end:

			## get hash
			if _type == 'answer':
				return_hash = SQ__parse_answer(ori_df_whole=df,
											  dt_now__obj=tmp_dt_sweep,
											  minute_forward=minute_forward,
											  stock_code=stock_code)
			else:
				return_hash = SQ__parse_sqData(ori_df_whole=df,
											   dt_now__obj=tmp_dt_sweep,
											   hours_duration_back=hours_back,
											   stock_code=stock_code)

			yield return_hash, tmp_dt_sweep

			## right now -> update after yield keyword
			tmp_dt_sweep += datetime.timedelta(minutes=min_dur)

		print(f'ended parsing stock_code : {stock_code}')




	# # @ import for db
	import sqlite3

	
	#@ stock prediction wrapper class
	prediction_agent = Stock_prediction(module=True)

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

		mainStk_dt_start__obj = main_Stk_df.date.min() + datetime.timedelta(days=10)
		mainStk_dt_end__obj = main_Stk_df.date.max() - datetime.timedelta(days=1)



		#################################
		# If all has passed : start from here!
		#################################
		while mainStk_dt_start__obj <= mainStk_dt_end__obj:
			tmp_dt_start__obj = SUB_F.FUNC_dtRect(mainStk_dt_start__obj,"9:00")
			tmp_dt_end__obj = SUB_F.FUNC_dtRect(mainStk_dt_start__obj,"15:30")

			# checking break in the middle
			tmp_break_bool = False

			print('\n'*2)
			print(f'=#'*20)
			print(f'tmp_dt_start__obj : {tmp_dt_start__obj}')
			print(f'tmp_dt_end__obj : {tmp_dt_end__obj}')

			f_kospi = sweep_day(df=df__kospi,
						stock_code=str(226490),
						start_date=tmp_dt_start__obj,
						end_date=tmp_dt_end__obj,
						_type='data',
						)

			f_dollar = sweep_day(df=df__dollar,
						stock_code=str(261250),
						start_date=tmp_dt_start__obj,
						end_date=tmp_dt_end__obj,
						_type='data',
						)

			# f_ans = sweep_day(df=main_Stk_df,
			# 			stock_code=stock_code,
			# 			start_date=tmp_dt_start__obj,
			# 			end_date=tmp_dt_end__obj,
			# 			_type='answer',
			# 			)

			f_data = sweep_day(df=main_Stk_df,
						stock_code=stock_code,
						start_date=tmp_dt_start__obj,
						end_date=tmp_dt_end__obj,
						_type='data',
						)


			while True:
				try:
					hash_kospi, t1 = f_kospi.__next__()
					hash_dollar, t2 = f_dollar.__next__()
					hash_data, t3 = f_data.__next__()
					#hash_ans, t4 = f_ans.__next__()

					if len(list(set([t1,t2,t3]))) != 1:
						pushLog(dst_folder='PREDICTER__ML_MAIN', 
						        lv='ERROR',
								module='Session', 
								exception=True,
								memo=f'time stamp different by generators')
						tmp_break_bool = True
						print(f'datestamp list isnt in sink...!')
						break

					elif list(filter(lambda x : x == None , [hash_kospi, hash_dollar, hash_data ])):
						tmp_break_bool = True
						print(f'value returned from generator is corrupt...!')
						break


					if SQ_check_opDay(t1):
						print(f'weekday, proceeding...!')
					else:
						print(f'weekend, skipping...!')
						tmp_break_bool = True
						break
					rtn = prediction_agent._stock_op_wrapper(stock_code=stock_code,
													   hash_stock=hash_data,
													   hash_kospi=hash_kospi,
													   hash_dollar=hash_dollar,
													   _today=t1,
													   hash_article=pickle_article)

					log, data = rtn  ## data 분석 proceed -> plot graph
					rtn_checker = ReturnWrap._type(_type='PREDICTER_TEST', rtn_val=rtn)
					if rtn_checker:
						print(f'predictable! -> log message : {log}')

					else: # unpredictable
						print(f'un-predictable! -> log message : {log}')
						tmp_break_bool = True
						break


					## add success log
					################################################
					pushLog(dst_folder='PREDICTER__ML_MAIN', 
							lv='INFO',
							module='Session', 
							exception=True,
							memo=f'date used : {str(t1)} \
								  \n- normally passed')

				except StopIteration as se:
					print(f'error in sweeping singe day : {se}')
					traceback.print_exc()
					pushLog(dst_folder='PREDICTER__ML_MAIN', 
							lv='ERROR',
							module='Session', 
							exception=True,
							exception_msg=str(se),
							memo=f'StopIteration exception')
					input(f'type any to continue to next!')
					tmp_break_bool = True
					break

			# @ get prediction result
			check_stock_code = prediction_agent.nestgraph.NG__check_stkcode(stock_code)
			if check_stock_code:
				print(f'stock code exists in nested graph')

				tmp_pred_datetime_dict = \
					prediction_agent.nestgraph.NG__get_prediction_dict(stock_code)

				tmp_model_status = prediction_agent.nestgraph.NG__get_accuracy(stock_code)


				if tmp_pred_datetime_dict:
					# @ plot graph and save
					############################
					# Plotting only done in the session
					#
					############################
					print(f'start plotting graph')
					FUNC__save_image(start_day_str=tmp_dt_start__obj,
									 model_status=tmp_model_status,
									 dataframe=main_Stk_df,
									 pred_dict=tmp_pred_datetime_dict,
									 stock_code=stock_code)
					pass

				else:
					print(f'iter to next day - no predictions available')
			else:
				print(f'no stock code available in nested graph')

			# add day
			mainStk_dt_start__obj += datetime.timedelta(days=1)

	print(f'total execution finished!')




def FUNC__save_image(start_day_str, model_status, dataframe, pred_dict, stock_code):
	"""

	:param start_day_str: predction training datetime
	:param model_status: model status container, follow hierarchy call tree to check
						 -> [ [ model state, model accuracy  ], [ , ] ..... [ , ] ]
	:param dataframe:
	:param pred_dict:
	:param stock_code:
	:return:
	"""

	import matplotlib.pyplot as plt
	import random

	folder_location = (os.getcwd() + '\\PREDICTER__Image_result').replace('/', '\\')
	if not os.path.isdir(folder_location):
		os.mkdir(folder_location)
	file_location = folder_location + '\\' + str(stock_code) + '_' + str(
		start_day_str.strftime("%Y-%m-%d")) + '.png'


	# @ picture
	fig = plt.figure(figsize=(100, 50))
	ax1 = fig.add_subplot(111)
	
	
	# @ set title
	tmp_title = ''
	for stat, acc in model_status:
		tmp_title += 'state : ' + str(stat)
		tmp_title += '\n' + 'acc : ' + str(acc)
	plt.title(tmp_title)

	# @ add original dateimte - value plot
	# ax1 = dataframe.plot(x='date', y='open', figsize = (100, 50), grid=True, Linewidth=1, fontsize=5)
	dataframe.plot(x='date', y='open', figsize=(80, 50), grid=True, Linewidth=1, fontsize=5, ax=ax1)

	# https://datascienceschool.net/view-notebook/372443a5d90a46429c6459bba8b4342c/
	# @ plot rand
	tmp_plot_item = ['b', 'r', 'g', 'k', 'c', 'm']

	for date_key in pred_dict:
		dot_color = random.choice(tmp_plot_item)
		tmp_x = [ date_key + datetime.timedelta(minutes=i+1) for i in range(0, len(pred_dict[date_key]) ) ]
		tmp_y = pred_dict[date_key]

		ax1.plot_date(tmp_x, tmp_y,
					  color=dot_color,
					  marker='o',
					  linestyle='solid',
					  alpha=0.5)  # marker='None', marker='o'
		ax1.axvline(x=date_key,
					color='r',
					linestyle='--',
					linewidth=1,
					alpha=0.3)
		ax1.axvline(x=date_key + datetime.timedelta(minutes=len(tmp_y)),
					color='b',
					linestyle='--',
					linewidth=1,
					alpha=0.3)

	try:
		fig.savefig(file_location, dpi=150)
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

		print(f'successful deleting them in FUNC__save_image')
	except Exception as e:
		print(f'error in  deleting them in FUNC__save_image... {e}')



if __name__ == '__main__':

	# training begin
	Session()




