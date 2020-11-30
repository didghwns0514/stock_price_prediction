# -*- coding: utf-8 -*-

# import os
# import time
# import multiprocessing
import zmq
import asyncio
import threading
# import nest_asyncio
# nest_asyncio.apply()

import kiwoom_api as ki
import message as Msg #< 32bit and 64bit seperately uses it
import stage as Stg #< 32bit and 64bit seperately uses it
import bit_64_data_read
import reinforcement_learning as RL
import time
import traceback
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


class bit64_client:
	# 32bit에게 64bit가 데이터 읽어 가라는 부분

	def __init__(self):
		# @ socket
		self.context32 = zmq.Context()
		self.socket32 = self.context32.socket(zmq.PUSH)

		# @ set socket properties
		self.socket32.setsockopt(zmq.BACKLOG, 1)
		self.socket32.setsockopt(zmq.CONFLATE, 1)
		self.socket32.setsockopt(zmq.IMMEDIATE, 1)
		self.socket32.setsockopt(zmq.LINGER, 0)  # ms value

		# @ message to send
		self.message32 = None

	def initialize(self):  # 다시 연결해주는 부분
		print('bit64_client initialization proceed...')
# 		self.message32 = None
# 		if Stg.get_stage64('B64_1_data_reply_for_32') or Stg.get_stage64('B64_6_request_BSH_for_32'):
# 			try:
# 				self.socket32.connect("tcp://localhost:50165")
# 				print('established bit64_client connection')
# 			except Exception as e:
# 				print('error in bit32_client - initialize')
		self.message32 = None
		try:
			self.socket32.connect("tcp://localhost:50165")
			print('established bit64_client connection')
		except Exception as e:
			print('error in bit32_client - initialize')

	def send_and_recieve(self):
		# initialize
		self.message32 = None
		
		#####
		# 32bit를 위한 데이터 제공 처리 부분
		#####
		print('bit64_client s&r proceed...')
		self.do_job()

		if Stg.get_stage64('B64_1_data_reply_for_32'):
			try:
				#self.socket32.send(Msg.enc('message_for_32:B64_1'), zmq.NOBLOCK)
				self.socket32.send(Msg.enc('message_for_32:B64_1'))

				Stg.proceed_64()  # Trusted that message is in the QUEUE
				# B64_7_wait_reply_from_confirmation_32
				# 3번까지 갈 수 있는지 판단은 2번 stage에서 확인, 안되면 다시 1번으로 돌아온다.
			except Exception as e:
				print('error in bit64 client send and recieve - 2nd :: ', e)
				#traceback.print_exc()
		
		elif Stg.get_stage64('B64_6_request_BSH_for_32'):
			try:
				self.socket32.send(Msg.enc('message_for_32:B64_6'), zmq.NOBLOCK)
				Stg.proceed_64()  # Trusted that message is in the QUEUE
				# B64_7_wait_reply_from_confirmation_32
			except Exception as e:
				print('error in bit64 client send and recieve - 1st :: ', e)

		else:
			print('bit64 client waiting for reading of data by 32bit')

	def do_job(self):
		pass  # save csv ( data file for 64bit )

	def quit(self):
		print('bit64_client quit proceed...')
# 		if Stg.get_stage64('B64_7_wait_reply_from_confirmation_32'):
# 			try:
# 				# self.socket32.close()
# 				self.socket32.disconnect()
# 				print('remove bit64_client connection')
# 				Stg.proceed_64()
# 			except Exception as e:
# 				print('error in bit64_client - closing socket')
		try:
			if Stg.get_stage64('B64_2_read_data_file') or Stg.get_stage64('B64_0_data_confirm_from_32'):
				# self.socket32.close()
				# print('remove bit64_client connection')
				
# 				self.socket32.send(Msg.enc('DUMMY_MESSAGE'), zmq.NOBLOCK)
# 				print('sent dummy message by 64bit')
				
				self.socket64.disconnect()
				print('64bit client disconnect')
			
			#Stg.proceed_32()
		except Exception as e:
			print('error in bit32_client - closing socket')


class bit64_server:
	# 32bit에서 제공된 데이터를 가져가라고 알려주는 부분

	def __init__(self):
		# @ socket
		self.context64 = zmq.Context()
		#self.socket64 = self.context64.socket(zmq.REP)
		self.socket64 = self.context64.socket(zmq.PULL)

		# @ set socket properties
		self.socket64.setsockopt(zmq.BACKLOG, 1)
		self.socket64.setsockopt(zmq.CONFLATE, 1)
		self.socket64.setsockopt(zmq.IMMEDIATE, 1)
		self.socket64.setsockopt(zmq.LINGER, 0)  # ms value

		# @ message recieved
		self.message64 = None  # message from 64bit client

	def initialize(self):  # 다시 연결 하는 부분
# 		self.message64 = None
# 		if Stg.get_stage64('B64_0_data_confirm_from_32') or Stg.get_stage64('B64_7_wait_reply_from_confirmation_32'):  # 32bit가 제작이 끝났으니..!
# 			try:
# 				self.socket64.bind("tcp://*:50160")
# 			except Exception as e:
# 				print('error in bit64_server - initialize')
		self.message64 = None
		try:
			self.socket64.bind("tcp://*:50160")
		except Exception as e:
			print('error in bit64_server - initialize')

				
	def recieve_and_send(self):
		# iniitialzie
		self.message64 = None


		if Stg.get_stage64('B64_0_data_confirm_from_32'):  # 32bit가 제작이 끝났으니..!
			try:
				#if not (self.message64 in list(Msg.tbl_32)):
				self.message64 = Msg.dec(self.socket64.recv(zmq.NOBLOCK))  # no waiting...
				print('        @@@@', self.message64, '@@@@    ')
				if not (self.message64 == 'message_for_64:B32_3'):
					#raise ValueError('Wrong bi64.py fail-1')
					self.message64 = None
					#Stg.err_counter_stage64()
					Stg.backward_64()
				else:
					Stg.proceed_64()  # to reply to confirm
					pass

				#####
				# 32bit의 데이터 읽어가는 부분
				self.do_job()
			#####

			except Exception as e:
				print('error in bit64_server - recieve and sending error 1st :: ',e)
		
		elif Stg.get_stage64('B64_7_wait_reply_from_confirmation_32'):
			try:
				#self.message64 = Msg.dec(self.socket64.recv(zmq.NOBLOCK))  # no waiting...
				#if not (self.message64 in list(Msg.tbl_32)):
				self.message64 = Msg.dec(self.socket64.recv(zmq.NOBLOCK))  # no waiting...
				print('        @@@@', self.message64, '@@@@    ')
				if not (self.message64 == 'message_for_64:B32_6' ):
					#raise ValueError('Wrong bi64.py fail-1')
					self.message64 = None
					Stg.err_counter_stage64()
					Stg.backward_64()
				else:
					Stg.proceed_64()  # to reply to confirm
					pass
			
			except Exception as e:
				print('error in bit64_server - recieve and sending error 2nd :: ',e)
				#Stg.backward_64() # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

		else:
			print('bit64 server waiting for request of data by 32bit')

	def do_job(self):
		# 원래 해야되는 일
		pass  # read 64bit data

	def quit(self):  # 소켓 close하는 부분
		if Stg.get_stage64('B64_1_data_reply_for_32'):
			try:
				# self.socket32.close()
				# self.socket64.disconnect()
				# Stg.proceed_64()
				pass
			except Exception as e:
				print('error in bit64_server - closing socket')


class Bit64_thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		
		# Create contexts
		self.context32 = None
		self.socket32 = None
		self.context64 = None
		self.socket64 = None

		#Message bag
		self.Mto32bit = None # 64bit에 보낼 때
		self.Mfrom32bit = None # 64bit한테 받은 것
		self.message32 = None # 64bit client 용
		self.message64 = None # 64bit server 용

		#Global for agent actions
		# : buy, sell, hold, sell_panic, sell_all
		#                     ^ sell all and abort
		#                               ^ just selling all for NOW
		self.action_out = None
		self.data_transfer_64bit_data = None #data csv에서 읽어와서 AI에 전달할 global 변수
		# clinet, server class objects

		self.bit64_server_obj = bit64_server()
		self.bit64_client_obj = bit64_client()
	
	async def check_runnable(self):
		delay = 1
		
		while True:
			await asyncio.sleep(delay)
			if ki.curr_time > ki.finish_time:
				pass
			
			############################################
			# anything error with 64bit operation!!!
			############################################
			
	async def read_data_file(self):

		delay = 0.01

		while True:
			await asyncio.sleep(delay)
			if Stg.get_stage64('B64_2_read_data_file'):
				###########
				# if check datafile exists -> do reading
				###########
				if 1: 
					self.data_transfer_64bit_data = bit_64_data_read.read_data()
					Stg.proceed_64()
				
				else:
					#Stg.backward_64()
					pass
				##########
				# else
				# Stg.backward_64()
				##########

	async def confirm_data_file(self):
		
		delay = 0.1
		while True:
			await asyncio.sleep(delay)
			if Stg.get_stage64('B64_3_eraise_data_file'):
				if (1): 
					###################################### 
					#confirm the csv file exists -> os isfile 함수 적용
					######################################
					
					######################################
					# 1) get the data, give it to class variable
					# 2) eraise the csv
					######################################
					
					Stg.proceed_64() 
				else:
					print('no data file to read in bit64 - confirm_data_file...')

	async def stock_AI(self):

		delay = 0.001

		while True:
			await asyncio.sleep(delay)
			if Stg.get_stage64('B64_4_set_variables_train_AI'):
				RL.train(self.data_transfer_64bit_data) #결정을 하는 부분
				####################
				# ^ 이 부분은 AI관련 로직 전부 돌리기 위함. 이거 ram에 올리기 위해서 structure가 필요할 듯
				# ★이 부분 또한 class로 선언하여서, 수행 후 결과를 class에 저장, 밑에 get result to 64에서 가져가는 형식으로.
				####################
				
				self.data_transfer_64bit_data = None # reset the variable for next step usage
				Stg.proceed_64()

	async def get_result_to_64(self):

		delay = 0.01

		while True:
			await asyncio.sleep(delay)
			if Stg.get_stage64('B64_5_get_result_action'):
				self.action_out = None  # reset the variable just before right assignment
				self.action_out = RL.RL_action_result
				######################
				# class로 선언하여서 결과 가져오는 부분
				######################
				RL.RL_action_result = None # reset the global variable in RL file
				Stg.proceed_64()

	async def run_server_64(self):

		delay = 0.5
		#delay = 1

		while True:
			#32bit 데이터 준비되었다는 메세지 받고 64bit 동작하는 부분...!!
			await asyncio.sleep(delay)
			#time.sleep(1)
			self.bit64_server_obj.initialize()
			self.bit64_server_obj.recieve_and_send()
			self.bit64_server_obj.quit()



	async def run_client_64(self):
		

		delay = 0.5
		#delay = 1

		while True:
			# 64bit가 32bit에게 자료 준비되었다고 하는 부분
			await asyncio.sleep(delay)
			#time.sleep(1)
			self.bit64_client_obj.initialize()
			self.bit64_client_obj.send_and_recieve()
			self.bit64_client_obj.quit()

	async def bit64_main(self):
		t0 = asyncio.ensure_future(self.check_runnable())
		t1 = asyncio.ensure_future(self.read_data_file())
		t2 = asyncio.ensure_future(self.confirm_data_file())
		t3 = asyncio.ensure_future(self.stock_AI())
		t4 = asyncio.ensure_future(self.get_result_to_64())
		t5 = asyncio.ensure_future(self.run_server_64())
		t6 = asyncio.ensure_future(self.run_client_64())

		await asyncio.gather(t0, t1, t2, t3, t4, t5, t6)

	def run(self):
		asyncio.run(self.bit64_main())

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