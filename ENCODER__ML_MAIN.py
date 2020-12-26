

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

from scipy.stats import norm

from sub_function_configuration import *




class Autoencoder:

	DATE_LENGTH = int(4)

	def __init__(self, module = True, simple=True):

		pass

	def FUNC_SIMPLE__read_article(self, specific_time, stock_code,
								  article_loc=None, article_pickle=None,hours_back=int(13)):
		"""

		:param specific_time: specific time to parse 4 days
		:param stock_code: stock code to search through
		:param article_loc: pickle file location
		:param article_pickle: if input can be itself hash of article, the hash
		:return: calculates simple article result based on stock code
				 // if retrun is None, skip!
		"""
		# @ check other path
		assert bool(
				  not(    (article_pickle != None and article_loc != None)
					   or (article_pickle == None and article_loc == None)
					 )
					)

		def search_name(pickled, code):
			target_name = None
			for stock_names in pickled:  # first hash is stock name
				tmp_stock_code = pickle_file[stock_names][0]
				if tmp_stock_code == code:
					target_name = stock_names
					break
			else:
				return target_name

			return target_name

		rtn = None
		srch_name = None
		if article_loc != None and article_pickle == None:
			with open(article_loc, 'wb') as file:
				pickle_file = pickle.load(file)
				srch_name = search_name(pickled=pickle_file, code=stock_code)
				tmp_rtn = parse_four_days(specific_time=specific_time,
									  stock_name=srch_name,
									  pickle_data=pickle_file)
				rtn = self.FUNC_SIMPLE__calculate(tmp_rtn)
		elif article_loc == None and article_pickle != None:
			srch_name = search_name(pickled=article_pickle, code=stock_code)
			tmp_rtn = parse_four_days(specific_time=specific_time,
								  stock_name=srch_name,
								  pickle_data=article_pickle)
			rtn = self.FUNC_SIMPLE__calculate(tmp_rtn)
		else:
			raise ValueError('check assertion in FUNC_SIMPLE__read_article')

		return rtn


	def FUNC_SIMPLE__calculate(self, rtn_list):
		"""
		:param : rtn_list : input from FUNC_SIMPLE__read_article
		:return: None if None type or nothing in the list
		"""
		if rtn_list != None or len(rtn_list) == 0:
			return None

		tmp_filtered = np.asarray( [ time * score for time, score in rtn_list] )
		mean, std = norm.fit(tmp_filtered)

		if mean < 0:
			mean = 0

		return mean

	
def zero_padding(list_obj):
	"""

	:param list_obj: list object to trim, at given number of articles
	:return:
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
		tmp_list_original = sorted(tmp_list_original, key=lambda x: x[0])[:100] # duration 순 정리

	return tmp_list_original


def parse_four_days(specific_time, stock_name, pickle_data,
					judge_zero_article_bool = True, n_days=int(Autoencoder.DATE_LENGTH)):

	"""
	@ dictionary 형태
	{
		종목이름1 : [ 종목코드 , { 기사주소 : [ ['2020-04-29 15:25', 0.23208051174879074, 기사내용], ... , [,,] ] } ]
	  , 종목이름2 : ...
	  , ....
	}

	중요 -> 점수 범위를 양수로 해야  encoding 가능, -범위 있으면 loss 때문에 판단 못함!

	"""
	# @ set parameter

	time_now__obj = specific_time # 지금 시점
	time_start__obj = (specific_time - datetime.timedelta(days=n_days)).replace(hour=0, minute=0) # 4일 전 정각
	
	tmp_result_list = [] # 임시 해당 부분 담을 list
	article_hash = pickle_data[stock_name][1] #기사 hash list
	
	for article_web in article_hash:
		tmp_article_list = article_hash[article_web] # hash value 값 list 전체
		for i in range(len(tmp_article_list)):
			article_time__obj = FUNC_dtSwtich(tmp_article_list[i][0], string_method="%Y-%m-%d %H:%M")

			if      (article_time__obj >= time_start__obj) \
				and (article_time__obj <= time_now__obj) : # 4일치 window에 포함된다면

				tmp_time_delta = time_now__obj - article_time__obj
				tmp_duration_in_hour = return_exp( tmp_time_delta.total_seconds() / (60*60), days = n_days )
				tmp_result_list.append( [tmp_duration_in_hour, (tmp_article_list[i][1] + 1)*0.5 ] )
				# 점수 10~20사이로 세팅, 10이 lowest, 20가 maximum socre

	if judge_zero_article_bool == True:
		if len(tmp_result_list) == 0:
			#raise RuntimeError('parse_four_days - error!') # ValueError
			return None
		else:
			return tmp_result_list#zero_padding(tmp_result_list)
	else:
		return tmp_result_list#zero_padding(tmp_result_list)


def return_exp(value, days):
	"""

	:param value: value of the hours passed from now, when the article was written
	:param days: days as standard to calculate from
	:return: set the return value to flat out in specific time
	"""
	conv_to_hour = int((days + 1) * 24)
	if value <= conv_to_hour:
		return (conv_to_hour - value)
	if value > conv_to_hour:
		return 0


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
		# copy_obj = copy.deepcopy(file)
		# global copy_obj
		copy_obj = copy.deepcopy(pickle.load(file))

	with open(pickle_path_2, 'rb') as file_2:
		# copy_obj = copy.deepcopy(file)
		# global copy_obj
		copy_obj_2 = copy.deepcopy(pickle.load(file_2))

	if copy_obj == None or copy_obj_2 == None:
		raise ValueError('wrong get pickle...')
	else:
		print('successfully loaded pickle file...')
	return copy_obj, copy_obj_2


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