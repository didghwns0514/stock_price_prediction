 # -*- coding: utf-8 -*-

# where you can decode and encode messages

#api request data
#api sends data

import time

#32bit 
#current = 0
y_val_32 = None
class Switch32 :
	global current,y_val_32
	def __init__(self): #name as input
		self.name = 'B32_0_get_api_variables_to_32'
		self.current = 0
		
		self.enter_time = time.time()

		self.err_count = 0

		###############
		#self.value = self.value_tbl() #result
		
	def value_tbl(self):
		global y_val_32
		
		####
		# reset stage after 10 secs
		self.check_stage_stay_time()
		self.check_err_count()
		####
		
		# self.value = value_tbl(self.name, self.current)
		name = str(self.name) #for pre-caution if its not a string
		y_val_32 = {
				 'B32_0_get_api_variables_to_32' :0,
				 'B32_1_logics_for_32' :1,
				 'B32_2_clear_write_confirm_data_32' :2,
				 'B32_3_data_ready_for_64' :3,
				 'B32_4_wait_for_64_reply' :4,
				 'B32_5_request_from_64' :5,
				 'B32_6_reply_to_confirm_64' :6,
				 'B32_7_request_api_new_BSH_32':7
				 }
		##^ 3 ,4 ,5  는 64 bit 통신 용임
		y_num = int(y_val_32.get(name))
		x_num = int(self.current)

		if y_num == x_num :
			print('    - 32bit now stage : ', name)
			return 1
		else:
			return 0

	
	def proceed(self):
		global y_val_32
		print('32bit just left stage : ', (list(y_val_32)[self.current]))
		if self.current <= 6 :
			self.current = self.current + 1
		elif self.current == 7:
			self.current = 0
		# 마지막에 도달하면 0으로 바꿔주어야 한다
		
		self.enter_time = time.time() # 넘어갈 때 마다 stamp찍음


	def backward(self):
		global y_val_32
		print('32bit recoverd from stage : ', (list(y_val_32)[self.current]))
		if self.current <= 7 and self.current >= 1 :
			self.current = self.current - 1
		elif self.current == 0:
			self.current = 7
		# 마지막에 도달하면 0으로 바꿔주어야 한다
		
		self.enter_time = time.time() # 넘어갈 때 마다 stamp찍음


	def check_stage_stay_time(self):
		
		if time.time() - self.enter_time >10 : # 10초 넘기면
			self.current = 0 # 초기화
			self.enter_time = time.time()
			print('-----======TIME======-----')
			print('-----======STAGE RESET======-----')

	def check_err_count(self):
		if self.err_count > 10 :
			self.current = 0 # 초기화
			self.err_count = 0
			print('-----======COUNTER======-----')
			print('-----======STAGE RESET======-----')
		
switch32 = Switch32()

def err_counter_stage32():
	global switch32
	switch32.err_count = switch32.err_count + 1

def get_stage32(name):
	global swtich32
	switch32.name = name
	
	return switch32.value_tbl()

def proceed_32():
	global switch32
	switch32.proceed()

def backward_32():
	global switch32
	switch32.backward()

y_val_64 = None
class Switch64:
	global current,y_val_64

	def __init__(self):  # name as input
		self.name = 'B64_0_data_confirm_from_32'
		self.current = 0
		
		self.enter_time = time.time()

		self.err_count = 0

	###############
	 # self.value = self.value_tbl() #result

	def value_tbl(self):
		global y_val_64
		
		####
		# reset stage after 10 secs
		self.check_stage_stay_time()
		self.check_err_count()
		####
		
		# self.value = value_tbl(self.name, self.current)
		name = str(self.name)  # for pre-caution if its not a string
		y_val_64 = {
			'B64_0_data_confirm_from_32': 0,
			'B64_1_data_reply_for_32': 1,
			'B64_2_read_data_file': 2,
			'B64_3_eraise_data_file': 3,
			'B64_4_set_variables_train_AI':4,
			'B64_5_get_result_action': 5,
			'B64_6_request_BSH_for_32': 6,
			'B64_7_wait_reply_from_confirmation_32': 7,
		 }
		y_num = int(y_val_64.get(name))
		x_num = int(self.current)

		if y_num == x_num:
			print('    - 64bit now stage : ', name)
			return 1
		else:
			return 0

	def proceed(self):
		global y_val_64
		print('64bit just left stage : ', (list(y_val_64)[self.current]))
		if self.current <= 6:
			self.current = self.current + 1
		elif self.current == 7:
			self.current = 0
	# 마지막에 도달하면 0으로 바꿔주어야 한다
		self.enter_time = time.time() # 넘어갈 때 마다 stamp찍음

	def backward(self):
		global y_val_64
		print('32bit recoverd from stage : ', (list(y_val_64)[self.current]))
		if self.current <= 7 and self.current >= 1 :
			self.current = self.current - 1
		elif self.current == 0:
			self.current = 7
		# 마지막에 도달하면 0으로 바꿔주어야 한다
		self.enter_time = time.time() # 넘어갈 때 마다 stamp찍음
		
	def check_stage_stay_time(self):
		
		if time.time() - self.enter_time >10 : # 10초 넘기면
			self.current = 0 # 초기화
			self.enter_time = time.time()
			print('-----======TIME======-----')
			print('-----======STAGE RESET======-----')

	def check_err_count(self):
		if self.err_count > 10 :
			self.current = 0 # 초기화
			self.err_count = 0
			print('-----======COUNTER======-----')
			print('-----======STAGE RESET======-----')

switch64 = Switch64()

def err_counter_stage64():
	global switch64
	switch64.err_count = switch64.err_count + 1

def get_stage64(name):
	global swtich64
	switch64.name = name

	return switch64.value_tbl()

def proceed_64():
	global switch64
	switch64.proceed()

def backward_64():
	global switch64
	switch64.backward()