# -*- coding: utf-8 -*-

import os
import time
# import multiprocessing
import copy
import asyncio
import threading
# import nest_asyncio
# nest_asyncio.apply()

import time
import traceback
import pickle
import datetime
import sub_function_configuration as SUB_F
'''
https://winterflower.github.io/2015/01/27/communication-between-two-python-scripts/

https://hyojabal.tistory.com/7
https://soooprmx.com/archives/8762
https://soooprmx.com/archives/6436
https://zeromq.org/languages/python/
https://stackoverflow.com/questions/16213235/communication-between-two-python-scripts
'''
##################################################################################

##################################################################################

class Bit_64(threading.Thread):
	NUM_OF_MAX_REPEAT_IN_AT_STAGE = 5 # 최대로 반복할 stage 기준 값, 넘으면 error 띄운다.
	NUM_OF_MAX_WAIT_IN_AT_STAGE = 100 # 최대로 대기할 wait

	MUST_WATCH_LIST = ["226490", "261250", "252670"]
	#ㅋㅋ# KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X

	def __init__(self, test):
		threading.Thread.__init__(self)
		
		self.TEST = test

		# @ ml components
		self.AGENT_trader = None
		self.AGENT_predicter = None
		self.AGENT_encoder = None
	
		self.counter_error = 0
		self.counter_wait = 0
		
		self.stage = 0
		self.dictionary_stage = {
									0 : 'P32_TO_P64_READ_READY',
									1 : 'P32_INPUT',
									2 : 'P64_TO_P32_SEND_RECIEVE',
									3 : 'WORKING',
									4 : 'P64_OUTPUT',
									5 : 'P64_TO_P32_SEND_READY',
									6 : 'P32_TO_P64_READ_RECIEVE'
		}
		
		self.comms_32_dictionary = {'P32_TO_P64_SEND_READY':False , 'P32_TO_P64_SEND_RECIEVE':False} # 32bit 파일 준비완료 알림 / 64에게 받았다고 알림
		self.comms_64_dictionary = {'P64_TO_P32_SEND_READY':False , 'P64_TO_P32_SEND_RECIEVE':False} # 64bit 파일 준비완료 알림 / 32에게 받았다고 알림
		
		# @ bool for activation
		self.ACTIVATION_BOOL__order = False

		# @ article
		self.COMMS_DICTIONARY__article = None

		# @ other dictionaries
		self.COMMS_DICTIONARY__input = None
		self.COMMS_DICTIONARY__output = {'prediction':{}, 'trade':{} , 'date':None}

		

		# @ paths
		self.COMMS_PATH__send = None
		self.COMMS_PATH__read = None
		self.COMMS_PATH__order = None # 32bit에서 불러와서 작업을 하는 것인지 확인할 때
		self.COMMS_PATH__article = None # 기사 path
		self.COMMS_PATH__output = None
		self.COMMS_PATH__input = None
		self.func_COMM_PICKLE__file_path()
	
	def func_COMM_STOCK_DICTIONARY__fill_missing_data(self):
		"""
		COMMS_DICTIONARY__input 사용, STOCK_DICTIONARY_FROM_BE__real_time_data_MIN 은 32bit name
		self.COMMS_DICTIONARY__input 사용
		FUNC_Bit64__fill_missing_data_in_dict 사용
		FUNC_Bit64__return_datetime_targets 사용
		"""
		ㅋㅋ
		try:
			pass
		except Exception as e:
			print(f'error in func_COMM_STOCK_DICTIONARY__fill_missing_data : {e}')
			return 'AT_FAIL'

	def func_COMM_STOCK_DICTIONARY__prediction(self):
		"""
		32bit 단에서는 : 
		STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code][datetime_obj] = {'price': XXX } 로 구현되어야 함
		self.COMMS_DICTIONARY__input 사용
		def FUNC_Bit64__stock_operation(dictionary_target, datetime_now, hours_duration_back ):
		"""
		ㅋㅋ
		try:
			pass
		except Exception as e:
			print(f'error in func_COMM_STOCK_DICTIONARY__prediction : {e}')
			return 'AT_FAIL'

	
	def func_COMM_STOCK_DICTIONARY__trade(self):
		"""
		STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] = XXX 로 구현되어야 함
		self.COMMS_DICTIONARY__input 사용
		def FUNC_Bit64__stock_operation(dictionary_target, datetime_now, hours_duration_back ):
		"""
		try:
			pass
		except Exception as e:
			print(f'error in func_COMM_STOCK_DICTIONARY__trade : {e}')
			return 'AT_FAIL'

	
	def FUNC_COMM_STOCK_DICTIONARY__order_calc(self):
		"""
		func_COMM_STOCK_DICTIONARY__prediction, func_COMM_STOCK_DICTIONARY__trade
		사용
		->
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN = {'prediction':{}, 'trade':{} , 'date':None} 에 넣음

		가격 예측 및 주문량 넣기 -> 전체 작업
		self.COMMS_DICTIONARY__input
		= (equals)
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN = {

			'STOCK_MIN' : {},
			'SQLITE' : {},
			'FILTER' : {},
			'BUDGET' : {},
			'OWNING' : {},
			'AVERAGE_PRICE':{}
		} # SQLITE로 boolian True일 때 만 작업함
		"""
		try:
			# @ initialize
			self.COMMS_DICTIONARY__output = {'prediction':{}, 'trade':{} , 'date':None}

			if 'STOCK_MIN' in self.COMMS_DICTIONARY__input:
				for stock_codes in self.COMMS_DICTIONARY__input['STOCK_MIN']:
					if stock_codes in self.COMMS_DICTIONARY__input['FILTER'] : # 존재여부 확인
						if not self.COMMS_DICTIONARY__input['FILTER'][stock_codes] == False: # 필터 false 면 계산 안함

							###################################
							# 여기서 계산, Must watch list 포함시켜서!
							# article result 까지 포함
							###################################
							pass
							
						else:
							pass
					else:
						if 'FILTER' in self.COMMS_DICTIONARY__input:
							print(f'failure... find out in FUNC_COMM_STOCK_DICTIONARY__order_calc (2) ')
							return 'AT_FAIL'
						else:
							print(f'failure... find out in FUNC_COMM_STOCK_DICTIONARY__order_calc (3) ')
							return 'AT_FAIL'

			else:
				print(f'failure... find out in FUNC_COMM_STOCK_DICTIONARY__order_calc (1) ')
				return 'AT_FAIL'

		except Exception as e:
			print(f'error in func_COMM_STOCK_DICTIONARY__order_calc : {e}')
			return 'AT_FAIL'
	
	def func_COMM__pickle_dump_from__32(self):
		"""
		self.COMMS_DICTIONARY__input
		->
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN = {

			'STOCK_MIN' : {},
			'SQLITE' : {},
			'FILTER' : {}, -> watching 여부 / init 시점에 다 예측할 때 쓰임
			'BUDGET' : {},
			'OWNING' : {},
			'AVERAGE_PRICE':{}
		} # SQLITE로 boolian True일 때 만 작업함
		"""
		try:
			with open(self.COMMS_PATH__input, 'rb') as file:
				self.COMMS_DICTIONARY__input = copy.deepcopy(pickle.load(file))
		except Exception as e:
			print(f'error in func_COMM__pickle_dump_from__32 : {e}')
			return 'AT_FAIL'

	def func_COMM__pickle_dump_for__32(self):
		"""
		self.COMMS_DICTIONARY__output
		->
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN = {'prediction':{}, 'trade':{} , 'date':None}
		"""
		try:
			with open(self.COMMS_PATH__output, 'wb') as file:
				pickle.dump(self.COMMS_DICTIONARY__output, file)
		except Exception as e:
			print(f'error in func_COMM__pickle_dump_for__32 : {e}')
			return 'AT_FAIL'

	def func_COMM_PICKLE__file_path(self):
		"""
		communication 용 pickle 파일 경로 지정부분
		"""
		python_path = os.getcwd()
		db_path = str(python_path + '\\KIWOOM_API__ML__COMMON').replace('/', '\\')
		article_path = str(python_path + '\\CRAWLER__pickle').replace('/', '\\')
		input_path = str(python_path+ '\\KIWOOM_API__to_ML').replace('/', '\\')
		output_path = str(python_path + '\\KIWOOM_API__from_ML').replace('/', '\\')
		
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path) # 경로 생성

		file_path_read = db_path + '\\' + 'COMM_32.p' # file_path_read
		file_path_send = db_path + '\\' + 'COMM_64.p'
		file_path_order = db_path + '\\' + 'COMM_32_ORDER.p' # 명령어!
		file_path_article = article_path + '\\' + 'pickle.p'
		file_path_input = input_path + '\\' + 'MINUTE_DATA.p'
		file_path_output = output_path + '\\' + 'ML_DATA.p'

		self.COMMS_PATH__send = copy.deepcopy(file_path_send)
		self.COMMS_PATH__read = copy.deepcopy(file_path_read)
		self.COMMS_PATH__order = copy.deepcopy(file_path_order)
		self.COMMS_PATH__article = copy.deepcopy(file_path_article)
		self.COMMS_DICTIONARY__input = copy.deepcopy(input_path)
		self.COMMS_DICTIONARY__output = copy.deepcopy(file_path_output)


	def func_COMM_PICKLE__send_64(self):
		"""
		64 bit 보내는 것!
		self.comms_64_dictionary 를 self.COMMS_PATH__send 에다가
		"""
		try:
			
			with open(self.COMMS_PATH__send, 'wb') as file:
				pickle.dump(self.comms_64_dictionary, file)
			print('successful pickle save in Bit_64 - func_COMM_PICKLE__send_64')
		
		except Exception as e:
			print('error in Bit_64 - func_COMM_PICKLE__send_64 :: ', e)
			return 'AT_FAIL'
				
	def func_COMM_PICKLE__read_32(self):
		"""
		32 bit 읽는 것!
		self.comms_32_dictionary 를 self.COMMS_PATH__read 에서
		"""
		try:
			with open(self.COMMS_PATH__read, 'rb') as file:
				self.comms_32_dictionary = copy.deepcopy(pickle.load(file))
			print('successful pickle read from Bit_32 - func_COMM_PICKLE__read_32')

		except Exception as e:
			print('error in Bit_64 - func_COMM_PICKLE__read_32 :: ', e)
			return 'AT_FAIL'	

	
	async def FUNC_ASYNC_CHECK__runnable(self):
		"""
		self.ACTIVATION_BOOL__order = False
		self.COMMS_PATH__order
		"""
		delay = 1
		
		while True:
			await asyncio.sleep(delay)
			
			try:
				with open(self.COMMS_PATH__order, 'rb') as file:
					self.ACTIVATION_BOOL__order = copy.deepcopy(pickle.load(file))
			except Exception as e:
				print(f'error in FUNC_ASYNC_CHECK__runnable : {e}')
				return 'COMM_FAIL'
			############################################
			# anything error with 64bit operation!!!
			############################################
	
	async def FUNC_ASYNC_COMM__wrapper(self):
		"""
		func_proceed 사용, 필요 함수 구동!
		"""
		pass

	
	async def FUNC_ASYNC_COMM__article_get(self):

		delay = 1
		"""
		article 계속 가져오는 부분
		self.COMMS_PATH__article
		self.COMMS_DICTIONARY__article
		"""
		while True:
			await asyncio.sleep(delay)
			
			try:
				with open(self.COMMS_PATH__article, 'rb') as file:
					self.COMMS_DICTIONARY__article = copy.deepcopy(pickle.load(file))
			except Exception as e:
				print(f'error in FUNC_ASYNC_COMM__article_get : {e}')
				return 'COMM_FAIL'


	def func_proceed(self, **kwargs): # **kwarg로 받아서 매 stage run할 함수들 가변해서 돌림
		"""
		다음 스테이지로 넘어가는 부분
		
		0 : 'P32_TO_P64_READ_READY',
		1 : 'P32_INPUT',
		2 : 'P64_TO_P32_SEND_RECIEVE',
		3 : 'WORKING',
		4 : 'P64_OUTPUT',
		5 : 'P64_TO_P32_SEND_READY',
		6 : 'P32_TO_P64_READ_RECIEVE'
		"""
		def run_kwargs(kwargs, tmp_stage_string):
			tmp_list_return = []
			if kwargs is not None:
				for key, value in kwargs.items():
					if key == tmp_stage_string :
						if value != None :
							print(f'execute func in stage : {key} - function : {value}')
							try:
								tmp_value = value() # 상위레벨에서 '이미' exception으로 감싸여있음
								tmp_list_return.append(tmp_value) # execute function
							except:
								tmp_list_return.append('AT_FAIL')
								traceback.print_exc()
						else:
							tmp_list_return.append('AT_SUCCESS') # dummy
					else:
						#raise ValueError('Wrong stage input from func_proceed wrapper.... :: (key, stage_now)', key, tmp_stage_string )
						pass
				
			else:
				tmp_list_return.append('AT_SUCCESS') # dummy 리턴
			
			return tmp_list_return

		def return_upper_wrapping_stage(kwargs):
			# fe단 주문 들어온 stage
			tmp_list = []
			if kwargs is not None:
				for key, value in kwargs.items():
					tmp_list.append(key.split('+',1)[0].strip())
			tmp_list = copy.deepcopy(list(dict.fromkeys(tmp_list)))

			return tmp_list

		try:
			Kiwoom꺼랑 비교해서 확인하기
			
			tmp_stage_string = self.func_get_stage()
			tmp_upper_stage = return_upper_wrapping_stage(kwargs)
			tmp_return = []
			print(f'tmp_stage_string : {tmp_stage_string}  \ntmp_upper_stage : {tmp_upper_stage} \nkwargs : {kwargs}')
			
			if 'P32_TO_P64_READ_READY' in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string) #pass # working 끝나고 마지막에 부르면 될 듯
				tmp_return.append(self.func_COMM_PICKLE__read_32())
				if self.comms_32_dictionary['P32_TO_P64_SEND_READY'] == False: # 32에서 받은게 
					tmp_return.append('WAITING')
				else:
					self.counter_wait = 0 # waiting 카운터 reset

			
			elif 'P32_INPUT' in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				# func_COMM__pickle_dump_from__32 -> 이거 wrapping에 넣기
				
			elif "P64_TO_P32_SEND_RECIEVE" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				
				self.comms_64_dictionary['P64_TO_P32_SEND_RECIEVE'] = True # P32_TO_P64_SEND_RECIEVE 사용 reset
				self.comms_64_dictionary['P64_TO_P32_SEND_READY'] = False # 보낼 값
				tmp_return.append(self.func_COMM_PICKLE__send_64()) # 보내고 local에 기록하고
				
			
			elif "WORKING" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				# tmp_return.append(self.func_COMM_PICKLE__read_64())
				# if self.TEST == False:
				# 	if self.comms_64_dictionary['P32_SEND_RECIEVE'] == False: # 64에서 받은게 확인되어야 함
				# 		tmp_return.append('WAITING')
				# 	else:
				# 		self.comms_32_dictionary['P32_TO_P64_SEND_READY'] = False	# ram에서 init
				# 		tmp_return.append(self.func_COMM_PICKLE__send_32()) # 보내고 local에서도 init 기록하고
				# 		self.counter_wait = 0 # waiting 카운터 reset

				# else:
				# 	pass
			
			elif "P64_OUTPUT" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				# func_COMM__pickle_dump_for__32 -> 사용 wrapping에서 

				# tmp_return.append(self.func_COMM_PICKLE__read_64())
				# if self.TEST == False:
				# 	if self.comms_64_dictionary['P64_SEND_READY'] == False:
				# 		tmp_return.append('WAITING')
				# 	else:
				# 		self.counter_wait = 0 # waiting 카운터 reset
				# else:
				# 	pass
			
			elif "P64_TO_P32_SEND_READY" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				self.comms_64_dictionary['P64_TO_P32_SEND_READY'] = True
				self.comms_64_dictionary['P64_TO_P32_SEND_RECIEVE'] = False
				tmp_return.append(self.func_COMM_PICKLE__send_64())
				
			elif "P32_TO_P64_READ_RECIEVE" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return.append(self.func_COMM_PICKLE__read_32())
				tmp_return = run_kwargs(kwargs, tmp_stage_string)		
				if self.comms_32_dictionary['P32_TO_P64_SEND_RECIEVE'] == False: # 32에서 받은게 
					tmp_return.append('WAITING')
				else:
					self.counter_wait = 0 # waiting 카운터 reset

			else:
				tmp_return.append('AT_STAGE_MISMATCH')
				#tmp_return.append('WAITING')
				
			
			# @ return 값 확인
			tmp_stage_fail = False # fail이 나면 True, 다음 stage로 못 넘어가야 함
			tmp_stage_repeat = False # True 나오면 다시 시작해야됨 stage를
			tmp_stage_mismatch = False # 다른 stage에서 함수 call 되었을 때 확인
			for return_value in tmp_return : 
				if return_value == "AT_FAIL":
					tmp_stage_fail = True
					#break
				elif return_value == "WAITING":
					tmp_stage_repeat = True
				elif return_value == "AT_STAGE_MISMATCH":
					tmp_stage_mismatch = True
				else:
					pass


			print(f'tmp_return in bit32 comms :: {tmp_return}')
			print('\n' * 2)
			
			if tmp_stage_mismatch == False:
				if tmp_stage_fail == False : # fail 아니면 -> 이게 먼저 맞아
					if tmp_stage_repeat == False : # repeat 하는게 아니면
						tmp_list_stage = copy.deepcopy(list(self.dictionary_stage.keys()))
						tmp_list_stage.sort()
						if self.stage == tmp_list_stage[-1] : # 제일 마지막 stage
							self.stage = tmp_list_stage[0]
						else:
							self.stage = self.stage + 1
					else:
						print('waiting for the response in AT - func_proceed :: ' + str(tmp_stage_string))
						print(' ')
						self.counter_wait = self.counter_wait + 1
						if self.counter_wait >= self.NUM_OF_MAX_WAIT_IN_AT_STAGE:
							self.counter_wait = 0
							self.stage = 0
				
				else: # 반복으로 에러 띄울지 확인
					print('\n' * 2)
					self.counter_error = self.counter_error + 1
					if self.counter_error >= self.NUM_OF_MAX_REPEAT_IN_AT_STAGE : # 허용 COMMS 에러 넘음
						#self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
						return 'COMM_FAIL'
			else: # mismatch인 경우 그냥 수행자체를 안한다
				pass
				return 'COMM_FAIL'
			
		except Exception as e:
			print('error in AT - Stage 32 func_proceed :: ', e)
			traceback.print_exc()
			#self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
			return 'COMM_FAIL'
	
	def func_backward(self):
		"""
		이전 스테이지로 돌아가는 부분
		"""		
		tmp_list_stage = copy.deepcopy(list(self.dictionary_stage.keys()))
		tmp_list_stage.sort()
		if self.stage == tmp_list_stage[0] : # 제일 첫 stage
			self.stage = tmp_list_stage[-1]
		else:
			self.stage = self.stage - 1
	

	# ======================================================


	async def bit64_main(self):
		t0 = asyncio.ensure_future(self.FUNC_ASYNC_CHECK__runnable())
		t1 = asyncio.ensure_future(self.FUNC_ASYNC_COMM__wrapper())
		t2 = asyncio.ensure_future(self.FUNC_ASYNC_COMM__article_get())

		await asyncio.gather(t0, t1, t2)

	def run(self):
		asyncio.run(self.bit64_main())

	def FUNC_Bit64__stock_operation(self, dictionary_target, datetime_now, hours_duration_back ):
		#tmp_list_of_target_datetimes = None
		tmp_list_of_target_datetimes = self.FUNC_Bit64__return_datetime_targets(datetime_now, hours_duration_back)

		tmp_dictionary_for_return = {}

		for item in tmp_list_of_target_datetimes:
			# dictionary, start_time_obj, end_time_obj
			if datetime_now < datetime_now.replace(hour=9, minute=0, second=0, microsecond=0):
				if item[0] == item[1]:
					pass
			else:
				tmp_split_dictionary_return = self.FUNC_Bit64__fill_missing_data_in_dict(dictionary_target, item[0], item[1])
				tmp_dictionary_for_return.update(tmp_split_dictionary_return)
		
		return tmp_dictionary_for_return


	def FUNC_Bit64__return_datetime_targets(self, datetime_obj_now, hours_duration_back ):
		#ㅋㅋㅋ 여기 df_3까지 들어와야 함
		"""
		dictionary 값으로 부터 datetime 개수 도출해놓고, 
		FUNC_Bit64__fill_missing_data_in_dict 사용해서 합치기!
		""" 

		return SUB_F.FUNC_datetime_backward(datetime_obj_now, hours_duration_back )

	def FUNC_Bit64__fill_missing_data_in_dict(self, dictionary, start_time_obj, end_time_obj):

		####여기서 missing 나온다
		#try:
		tmp_return_dictionary = {}

		for datetime_str in dictionary:
			tmp_datetime = datetime.datetime.strptime(datetime_str, "%Y%m%d%H%M%S").replace(
			second=0, microsecond=0)
			if tmp_datetime >= start_time_obj and tmp_datetime <= end_time_obj:
				tmp_return_dictionary[datetime_str] = dictionary[datetime_str]

		#tmp_list_of_missing_datastamp = []

		tmp_datetime_stamp_list = list(tmp_return_dictionary.keys())
		#print(f'dictionary : {dictionary}')
		#print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')
		tmp_datetime_stamp_list.sort()

		# print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')

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
					#tmp_list_of_missing_datastamp.append(tmp_start_datetime_stamp_obj_convert)
					tmp_return_dictionary[tmp_start_datetime_stamp_obj_convert] = {'price': before_price,
																				'volume': 0} # 'volume': before_volume

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
																			'volume': 0} # 'volume': tmp_end_stub_volume
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
																'volume': 0}#'volume': tmp_start_stub_volume

			tmp_end_time_obj = tmp_end_time_obj + datetime.timedelta(minutes=1)


		#print(f'tmp_return_dictionary in  SESS__fill_missing_data_in_dict : \n{tmp_return_dictionary}')

		return tmp_return_dictionary


#____________________________________________________________________________________________
# Using of async
# HERE ▼
if __name__ == '__main__':

	t1 = Bit64_thread()
	t1.start()


#____________________________________________________________________________________________

	
# 	time.sleep(2)
# 	while True:
# 	## ^ sending 10 requests
# 		print("Sending request %s …" % request)
# 		socket.send(b"1Hello")
		
# 		# get the reply
# 		message = socket.recv()
# 		print("Received reply %s [ %s ]" % (request, message))
# 		import time.sleep(0.01)
'''
https://stackoverflow.com/questions/36275217/inter-process-communication-using-zeromq-transferring-large-arrays
http://blog.naver.com/PostView.nhn?blogId=parkjy76&logNo=30142067471&redirect=Dlog&widgetTypeCall=true&directAccess=false
'''
"""