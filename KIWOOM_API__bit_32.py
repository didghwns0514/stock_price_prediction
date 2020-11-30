# -*- coding: utf-8 -*-

# import time
# import os
# import multiprocessing as mp
import zmq
import asyncio
import queue
# import datetime
# import nest_asyncio
# #nest_asyncio.apply()

import cmd_open
import kiwoom_api as ki
import bit_32_data_create
import message as Msg #< 32bit and 64bit seperately uses it
import stage as Stg #< 32bit and 64bit seperately uses it
import threading
import copy

import time
import traceback

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QAxContainer import *

'''
https://dojang.io/mod/page/view.php?id=2469
https://soooprmx.com/archives/6882
https://stackoverflow.com/questions/44982332/asyncio-await-and-infinite-loops
https://wikidocs.net/21046

https://docs.python.org/ko/3/library/asyncio-task.html
'''

'''
https://stackoverflow.com/questions/7538988/zeromq-how-to-prevent-infinite-wait
'''
##################################################################################
# Globals for 32_64 coms
# 32 as a server

#API for main 32bit thread
# ki.app, ki.myWindow = ki.api_32bit()
# ki.myWindow.show()
# ki.app.exec_()



##################################################################################

class bit32_client:
	# 32bit에서 데이터가 있으니 64bit에게 읽어가라는 부분

	def __init__(self):
		#@ socket
		self.context64 = zmq.Context()
		#self.socket64 = self.context64.socket(zmq.REQ)
		self.socket64 = self.context64.socket(zmq.PUSH) # push to pull model

		#@ set socket properties
		self.socket64.setsockopt(zmq.BACKLOG,1)
		self.socket64.setsockopt(zmq.CONFLATE,1)
		self.socket64.setsockopt(zmq.IMMEDIATE,1)
		self.socket64.setsockopt(zmq.LINGER, 0) # ms value

		#@ message to send
		self.message64 = None

	def initialize(self): # 다시 연결해주는 부분
		print('bit32_client initialization proceed...')
# 		self.message64 = None
# 		if Stg.get_stage32('B32_3_data_ready_for_64') or Stg.get_stage32('B32_6_reply_to_confirm_64'):
# 			try:
# 				self.socket64.connect("tcp://localhost:50160")
# 				print('established bit32_client connection')
# 			except Exception as e:
# 				print('error in bit32_client - initialize')
		self.message64 = None

		try:
			self.socket64.connect("tcp://localhost:50160")
			print('established bit32_client connection')
		except Exception as e:
			print('error in bit32_client - initialize')

	def send_and_recieve(self):
		#####
		# 64bit를 위한 데이터 제공 처리 부분
		#####
		print('bit32_client s&r proceed...')
		self.do_job()


		if Stg.get_stage32('B32_3_data_ready_for_64'):
			try: # 안보내지면 다음 스테이지로 안넘어가기는 하는데.. connection없어도 안보내지는게
				# 맞는지 아직 모름
				self.socket64.send(Msg.enc('message_for_64:B32_3'), zmq.NOBLOCK)
				Stg.proceed_32() # Trusted that message is in the QUEUE and sent -> 에러나면 이 자체 exception으로 빠져서 실행 안된다.

				# 'B32_4_wait_for_64_reply'
			except Exception as e:
				print('error in bit32 client send and recieve - 1st :: ', e)


		# confirm the message

		elif Stg.get_stage32('B32_6_reply_to_confirm_64'):
			try:
				#self.socket64.send(Msg.enc(Msg.tbl_64.get(self.message64)))
				#self.socket64.send(Msg.enc('message_for_64:B32_6'), zmq.NOBLOCK)
				self.socket64.send(Msg.enc('message_for_64:B32_6'))
				
				Stg.proceed_32()  # Trusted that message is in the QUEUE

			except Exception as e:
				print('error in bit32 client send and recieve - 2nd :: ', e)

		else:
			print('bit32 client waiting for reading of data by 64bit')

	def do_job(self):
		pass # save csv ( data file for 64bit )

	def quit(self):
		print('bit32_client quit proceed...')
		# if Stg.get_stage32('B32_4_wait_for_64_reply'):
		# 	try:
		# 		#self.socket64.close()
		# 		self.socket64.disconnect()
		# 		print('remove bit32_client connection')
		# 		Stg.proceed_32()
		# 	except Exception as e:
		# 		print('error in bit32_client - closing socket')

		try:
			# B32_5_request_from_64
			if Stg.get_stage32('B32_5_request_from_64') or Stg.get_stage32('B32_7_request_api_new_BSH_32') :
				# 메세지 보낸것을 confirm 되었으면!
				# self.socket64.close()
				# print('remove bit32_client connection')

# 				self.socket64.send(Msg.enc('DUMMY_MESSAGE'), zmq.NOBLOCK)
# 				print('sent dummy message by 32bit')
		
				self.socket64.disconnect()
				print('32bit client disconnect')

			#Stg.proceed_32()
		except Exception as e:
			print('error in bit32_client - closing socket')

class bit32_server:
	# 먼저 메세지를 받고나서,
	# 다음 메세지를 주고 close 처리
	# 64bit에서 제공되는 데이터를 읽어가면 되는 부분

	def __init__(self):
		#@ socket
		self.context32 = zmq.Context()
		self.socket32 = self.context32.socket(zmq.PULL)

		#@ set socket properties
		self.socket32.setsockopt(zmq.BACKLOG,1)
		self.socket32.setsockopt(zmq.CONFLATE,1)
		self.socket32.setsockopt(zmq.IMMEDIATE,1)
		self.socket32.setsockopt(zmq.LINGER, 0) # ms value

		#@ message recieved
		self.message32 = None # message from 64bit client

	def initialize(self): # 다시 연결 하는 부분
# 		self.message32 = None
# 		if Stg.get_stage32('B32_4_wait_for_64_reply') or Stg.get_stage32('B32_5_request_from_64'): #64bit가 제작이 끝났으니..!
# 			try:
# 				self.socket32.bind("tcp://*:50165")
# 			except Exception as e:
# 				print('error in bit32_server - initialize')
		self.message32 = None
		try:
			self.socket32.bind("tcp://*:50165")
		except Exception as e:
			print('error in bit32_server - initialize')

	def recieve_and_send(self):
		# iniitialzie
		self.message32 = None


		if Stg.get_stage32('B32_5_request_from_64'): #64bit가 제작이 끝났으니..!
			try:
				#if not( self.message32 in list(Msg.tbl_64)):
				self.message32 = Msg.dec(self.socket32.recv(zmq.NOBLOCK))  # no waiting...
				print('        @@@@', self.message32, '@@@@    ')
				if not( self.message32 == 'message_for_32:B64_6'):
					#raise ValueError('Wrong bi32.py fail-1')
					# 'message_for_32:S'
					self.message32 = None
					Stg.backward_32()
				else:
					Stg.proceed_32()
					pass

				#####
				# 64bit의 데이터 읽어가는 부분
				self.do_job()
				#####
			except Exception as e:
				print('error in bit32 server recieve and send - 1st :: ', e)
				

		elif Stg.get_stage32('B32_4_wait_for_64_reply'):
			try:
				#self.message32 = Msg.dec(self.socket32.recv(zmq.NOBLOCK))
				#if not Msg.tbl_32.get('data_ready_for_64:S') == self.message32: #confirm the reply
				self.message32 = Msg.dec(self.socket32.recv(zmq.NOBLOCK))  # no waiting...
				print('        @@@@', self.message32, '@@@@    ')
				if not( self.message32 == 'message_for_32:B64_1'):
					self.message32 = None
					Stg.err_counter_stage32()
					Stg.backward_32()
				else:
					Stg.proceed_32()
					# 'B32_5_request_from_64'
			except Exception as e:
				print('error in bit32 server recieve and send - 2nd :: ', e)
				#Stg.backward_32() # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		# 	try:
		# 		Stg.proceed_32()
		# 	except Exception as e:
		# 		print('error in bit32 client send and recieve - 2nd :: ', e)

		else:
			print('bit32 server waiting for request of data by 64bit')

	def do_job(self):
		# 원래 해야되는 일
		pass # read 64bit data

	def quit(self): # 소켓 close하는 부분
		if Stg.get_stage32('B32_6_reply_to_confirm_64'):
			try:
				#self.socket32.close()
				#self.socket32.disconnect()
				#Stg.proceed_32()
				pass
				# stage proceed done in recieve!!
			except Exception as e:
				print('error in bit32_server - closing socket')



class Bit32_thread(threading.Thread):

	def __init__(self,  api_to_bit32_Q, bit32_to_api_Q ):
		threading.Thread.__init__(self)
		# Create contexts
		self.context32 = None
		self.socket32 = None
		self.context64 = None
		self.socket64 = None

		# Message bag
		self.Mto64bit = None  # 64bit에 보낼 때
		self.Mfrom64bit = None  # 64bit한테 받은 것
		self.message32 = None  # 32bit server용 의 64bit REQ
		self.message64 = None  # 32bit client용
		self.data_transfer_32bit_variables = None  # for data transfer for bit_32_csv
		self.data_transfer_32bit_data = None

		# clinet, server class objects
		self.bit32_server_obj = bit32_server()
		self.bit32_client_obj = bit32_client()

		# class queue
		self.input_queue = api_to_bit32_Q
		self.output_queue = bit32_to_api_Q



	async def check_runnable(self):

		delay = 1

		while True :
			await asyncio.sleep(delay)
			# check for time
			if ki.curr_time > ki.finish_time:
				pass
				# raise a value flag for end
			
			#######################
			# Check for dangers -> error in any while-True -> skip , sell everything and stop traiding
			#######################


	async def get_api_data_loop(self):
		#^  계속적으로 api에서 32bit으로 data를 가져오는 while문
		# pyqt5 login
		# self.execute_pyq('login_instance')
		# self.execute_pyq('get_all_code_instance')
		######
		# Queue 에서 값을 읽어온다!!!!!!!
		# Queue 에서 값을 읽어온다!!!!!!!
		# Queue 에서 값을 읽어온다!!!!!!!
		######
		tmp_val = None

		delay = 0.2

		while True:
			await asyncio.sleep(delay) # 아니면 function 자체를 await하게 __await__()변환을 하는게..
			#tmp_val = copy.deepcopy(ki.request_data())  # 무조건... 돌린다
			#^^ #################
			# confirmation MUST EXIST!!!!!!!
			#####################

			if Stg.get_stage32('B32_0_get_api_variables_to_32') :
				self.data_transfer_32bit_variables = None  # for safety, reset the variable
				# 32bit 내부로 가져올 필요가 있을 때
				self.data_transfer_32bit_variables = copy.deepcopy(tmp_val)
				Stg.proceed_32()

	async def api_data_inner_logic(self):
		#^ 데이터가 32bit 글로벌로 받아오고 interface 해주는 부분

		delay = 0.1

		while True :
			await asyncio.sleep(delay)
			self.data_transfer_32bit_data = None  # for safety, reset the variable
			if Stg.get_stage32('B32_1_logics_for_32') :
				#await asyncio.sleep(delay)
				self.data_transfer_32bit_data = bit_32_data_create.inner_logic(self.data_transfer_32bit_variables) #로직 돌려서 필요한 보조 지표 등등 생성...
									 # 아니면 키움증권에서 받아올수는 있을 것 같음
															##^ maybe a class variable
				self.data_transfer_32bit_variables = None  # for safety, reset the variable
				Stg.proceed_32()



	async def data_creation_32(self): # 데이터가 32bit 글로벌로 받아왔다면, 내부 로직 돌리는 부분

		delay = 0.1

		while True :
			await asyncio.sleep(delay)
			if Stg.get_stage32('B32_2_clear_write_confirm_data_32') : # pickle data로 만들어 주는 부분
				bit_32_data_create.write_to_file(self.data_transfer_32bit_data)
				self.data_transfer_32bit_data = None
				Stg.proceed_32()
				####################
				# clear file, write to file + inner-logics, confirm it is complete,
				# set flag lv_Data_ready_to_64bit = 1,
				#
				####################
				await asyncio.sleep(delay)

	async def set_api_data(self):

		delay=0.1 # get api, send mesg to 64, retrieve, then set api

		while True:
			await asyncio.sleep(delay)
			if Stg.get_stage32('B32_7_request_api_new_BSH_32'):
				###################################
				# queue 써서 전달하는 방식이 나을 듯? -> global 변수를 큐로 설정해서..!
				# 여러개에 대한 작업이 진행되어야 하니깐...
				###################################
				
				"""
				if self.message32 == 'sell:S' :
					ki.request_sell()
					## ^ confirmation of new requests must be done in api.py
				elif self.message32 == 'buy:S':
					ki.request_buy()
					## ^ confirmation of new requests must be done in api.py
				elif self.message32 == 'hold:S':
					pass
				elif self.message32 == 'sell_panic:S': # 나중에 하드 코딩으로 구현
					ki.request_sell_panic()
				elif self.message32 == 'sell_all:S':
					ki.request_sell_all()
				#^^ #################
				# confirmations MUST EXIST!!!!!!! -> 
				#####################
				"""
				print('send action to kiwoom-api')

				Stg.proceed_32()


	async def run_server_32(self):

		delay = 0.5
		#delay = 1

		while True:
			# 32bit가 64bit의 자료 읽기 전에 컨펌 해주는 부분...!!
			await asyncio.sleep(delay)
			#time.sleep(1)
			self.bit32_server_obj.initialize()
			self.bit32_server_obj.recieve_and_send()
			self.bit32_server_obj.quit()



	async def run_client_32(self):

		delay = 0.5
		#delay = 1

		while True:
			# 32bit가 64bit에게 자료 준비되었다고 하는 부분...!!
			await asyncio.sleep(delay)
			#time.sleep(1)
			self.bit32_client_obj.initialize()
			self.bit32_client_obj.send_and_recieve()
			self.bit32_client_obj.quit()


	async def bit32_main(self):
		t0 = asyncio.ensure_future(self.check_runnable())
		t1 = asyncio.ensure_future(self.get_api_data_loop())
		t2 = asyncio.ensure_future(self.api_data_inner_logic())
		t3 = asyncio.ensure_future(self.data_creation_32())
		t4 = asyncio.ensure_future(self.set_api_data())
		t5 = asyncio.ensure_future(self.run_server_32())
		t6 = asyncio.ensure_future(self.run_client_32())


		await asyncio.gather(t0, t1, t2, t3, t4, t5, t6)

	def run(self):
		asyncio.run(self.bit32_main())
		#asyncio.run(self.bit32_main())
		#loop = asyncio.get_event_loop()
		#result = loop.run_until_complete(self.bit32_main())

	def execute_pyq(self, name):
		print('entered : ' + str(name) + '...')
		app = QApplication(sys.argv)
		class_ = ki.pyq_object(str(name))
		class_.pyq_exec()
		print('exiting : ' + str(name) + '...')
		#sys.exit(app.exec_())

#____________________________________________________________________________________________
# Using of async
# HERE ▼
if __name__ == '__main__':

	###
	# Queue 선언부
	###
	api_to_bit32_Q = queue.Queue()
	bit32_to_api_Q = queue.Queue()
	api_to_gui_Q = queue.Queue()
	gui_to_api_Q = queue.Queue()


	####
	#GUI 세팅 부분
	####
	#t0 = threading.Thread(target = ki.Main_window, args = [api_to_gui_Q, gui_to_api_Q]) # gui
	t1 = Bit32_thread(api_to_bit32_Q, bit32_to_api_Q) # 32bit
	#t2 = threading.Thread(target=ki.PyQ_wrapper, args=[api_to_bit32_Q, bit32_to_api_Q, api_to_gui_Q, gui_to_api_Q]) #api
	t2 = threading.Thread(target=ki.PyQ_wrapper,  args=[api_to_bit32_Q, bit32_to_api_Q])  # api + gui
	#t0.start()
	t1.start()
	t2.start()
	#t0.join()
	t1.join()
	t2.join()
	########################
	# 신문 crwaling 부분 + pickle data 만드는 부분 + 그걸 가지고 auto encoding해서 받아오는 부분 -> global queue로 넣으면 될 듯
	# api 연결부분????
	########################




#____________________________________________________________________________________________

# async def main_():
	
# 	Get_api = asyncio.ensure_future(get_api_data())
# 	Set_api = asyncio.ensure_future(set_api_data())
# 	#Server = asyncio.ensure_future(run_server()) # 일반적 방법
# 	try:
# 		#Server = asyncio.wait_for(run_server(),0.5)
# 		#
# 		Server = asyncio.ensure_future(run_server())
# 		#pass
# 	except:
# 		print('Time Out!')
	#await asyncio.gather(Tick, Get_api, Set_api, Server)

#asyncio.run(main_())
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
# loop.close

'''
https://stackoverflow.com/questions/32761095/python-asyncio-run-forever-or-while-true
https://stackoverflow.com/questions/41063331/how-to-use-asyncio-with-existing-blocking-library

____________________________________________________________________
Note: asyncio.create_task() was introduced in Python 3.7. In Python 3.6 or lower, use asyncio.ensure_future() in place of create_task().
____________________________________________________________________
'''
