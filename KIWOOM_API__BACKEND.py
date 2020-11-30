# -*- coding: utf-8 -*-

import sys
import time
import datetime

from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import traceback



#import decorators
def call_printer(original_func):
	"""original 함수 call 시, 현재 시간과 함수 명을 출력하는 데코레이터"""

	def wrapper(*args, **kwargs):
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
		print('[{:.22s}] func `{}` is called'.format(timestamp, original_func.__name__))
		return original_func(*args, **kwargs)

	return wrapper



class KiwoomAPI(QAxWidget):
	TR_REQ_TIME_INTERVAL = 0.15
	"""
	3 : 작동은 함 1000개 request 걸림
	4 : 작동은 함, 1000 넘는거 한번 확인
	5 : 작동 - 거의 무조건 계속가능한듯
	"""

	def __init__(self):
		print('init backend kiwoom api!!!')
		super().__init__()
		self.num_com_rq_data = 0 # 전체 require 갯수 설정
		self._create_kiwoom_instance()
		self._set_signal_slots()

		# @ BackEnd error counter
		self.error_counter_BE = 0

		# @ event loop
		self.login_event_loop = None # login 시 사용할 event loop
		self.tr_event_loop = None # Transaction data 가져올 시 사용할 event loop


		# @ BackEnd event_loop exit
		self.timer_out = None
		
		# @ backend단의 signal들
		self.my_be_signal = MY_SIG_BE_global()

		# @ backend단의 flag들
		self.flag_1st_login = False
		
		# @ 서버 통신 1회 요청을 위한 flag들
		self.flag_comm_connect_check = False
		self.flag_comm_rq_data_check = None
		
		# @ backend 단의 ram 변수들
		"""
		★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
		항상 이 변수들 쓸 때는 front end 단에서 reset을 해주도록 한다( 쓰고난 직후에 )
		★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
		"""
		self.latest_buy_sell_result_message = None
		self.latest_cancle_buy_sell_result_message = None
		self.latest_balance_check_with_order_message = None
		self.latest_check_owning_stock_message = None
		self.latest_buy_sell_result_first_data = None
		self.latest_buy_sell_result_second_data = None
		self.latest_tr_data = None
		self.latest_balance_normal_data = None
		self.latest_balance_with_order_data = None
		self.latest_owning_stocks_data = None
		self.latest_stock_additional_info = None
		self.latest_stock_unmet_order_data = None
		
		
		self.flag_1st_login = False
		self.flag_1st_login = False
		
		self.flag_check_noraml_chejen = False # 정상 trade시 msg1 msg2 True, 끝나면 False
		self.flag_sudden_send_order_message = False # 비정상 수신일 때 True 바꿔줌

		self.is_tr_data_remained = None

		self.comm_rq_data_result = None # com_rq_data error 반환 보려고
		"""
		OP_ERR_SISE_OVERFLOW – 과도한 시세조회로 인한 통신불가
		OP_ERR_RQ_STRUCT_FAIL – 입력 구조체 생성 실패
		OP_ERR_RQ_STRING_FAIL – 요청전문 작성 실패
		OP_ERR_NONE – 정상처리
		"""
		self.latest_stock_realtime_data = {}
		self.flag_latest_stock_realtime_data = False
		self.send_order_1st_message_store = {}


	def reset_rq_data(self):
		self.num_com_rq_data = 0

	def _create_kiwoom_instance(self):
		self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

	def _set_signal_slots(self):
		# Login 요청 후 서버가 발생시키는 이벤트의 핸들러 등록
		self.OnEventConnect.connect(self._on_event_connect)

		# 조회 요청 후 서버가 발생시키는 이벤트의 핸들러 등록
		self.OnReceiveTrData.connect(self._on_receive_tr_data)

		# 체결 후 잔고 가져오는 데이터(주문 체결시 실행되는 event)
		self.OnReceiveChejanData.connect(self._on_receive_chejan_data)

		# 서버에서 보내는 메세지 받는 용도
		self.OnReceiveMsg.connect(self._on_receive_message_data)
		#--------# https://goni9071.tistory.com/264

		# 실시간 사세 데이터 받아오는 부분
		self.OnReceiveRealData.connect(self._on_receive_real_data)

		# 키움 사용자 조건검색식 수신 관련 이벤트가 발생할 경우 receive_condition_var 함수 호출
		self.OnReceiveRealCondition.connect(self._on_receive_real_condition)

		# 키움 사용자 조건검색식 초기 조회 시 반환되는 값이 있을 경우 receive_tr_condition 함수 호출
		self.OnReceiveTrCondition.connect(self._on_receive_tr_condition)

	def _on_receive_tr_condition(self):
		pass

	def _on_receive_real_condition(self):
		pass

	def _on_receive_real_data(self, stock_code, realtype, realdata):


		"""
		여기 매수 매도 일 때는 realtime 감시 안해야되고, rqname이 get_real_time_data 일 때 할 것임...
		-------------------------------------------
		          [GetCommRealData() 함수]
          
          GetCommRealData(
          BSTR strCode,   // 종목코드
          long nFid   // 실시간 타입에 포함된FID
          )
          
          OnReceiveRealData()이벤트가 호출될때 실시간데이터를 얻어오는 함수입니다.
          이 함수는 반드시 OnReceiveRealData()이벤트가 호출될때 그 안에서 사용해야 합니다.
          
          ------------------------------------------------------------------------------------------------------------------------------------
          
          [주식체결 실시간 데이터 예시]
          
          if(strRealType == _T("주식체결"))
          {
            strRealData = m_KOA.GetCommRealData(strCode, 10);   // 현재가
            strRealData = m_KOA.GetCommRealData(strCode, 13);   // 누적거래량
            strRealData = m_KOA.GetCommRealData(strCode, 228);    // 체결강도
            strRealData = m_KOA.GetCommRealData(strCode, 20);  // 체결시간
          }
          
          

          [OnReceiveRealData()이벤트]
          
          OnReceiveRealData(
          BSTR sCode,        // 종목코드
          BSTR sRealType,    // 리얼타입 -> ("주식체결") / ("주식당일거래원") 등등...
          BSTR sRealData    // 실시간 데이터 전문 -> 이거 원문이 뭔지.,? 이거로 해석하는건 아닌듯
          밑에서 getcommrealdata로 잘라서 가져오면 됨
          )
          
          실시간 데이터 수신할때마다 호출되며 SetRealReg()함수로 등록한 실시간 데이터도 이 이벤트로 전달됩니다.
          GetCommRealData()함수를 이용해서 실시간 데이터를 얻을수 있습니다.
		"""
		import KIWOOM_API__tr_receive_handler as tr
		print('_on_receive_real_data in BE activated...')

		self.flag_latest_stock_realtime_data = True # violation 막기 위함
		
		# @ 가져온 시점 기록
		tmp_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
		
		# @ comm get real 부름
		try:
			print(f'In _on_receive_real_data in BE \nstock_code : {stock_code}')
			print(f'realdata : {realdata}')
			print(f'realtype : {realtype}')
			tmp_dict = tr.on_receive_realtime_data(self, stock_code, realtype)
			print(f'tmp_dict from getcomm real data : {tmp_dict}')

		except Exception as e:
			print('error..._on_receive_real_data : ', e)
			traceback.print_exc()
		
		# @ realtype에 따른 구분
		if realtype == "순간체결량":
			print('enter real time data (1)')
			if stock_code not in self.latest_stock_realtime_data: # 코드 자체가 처음 들어감
				#self.latest_stock_realtime_data[stock_code] = {tmp_datetime :  realdata} # FID 코드일 것으로 추정은 되는데...
				self.latest_stock_realtime_data[stock_code] = {tmp_datetime: tmp_dict}  # FID 코드일 것으로 추정은 되는데...
			else: # 코드가 있음
				if tmp_datetime in self.latest_stock_realtime_data[stock_code] and tmp_dict: # 만약에 timestamp에 대한 기록 있고 tmp_dict 가 비지 않았으면
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['price'] = tmp_dict['price'] + self.latest_stock_realtime_data[stock_code]['price']
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['volume'] = tmp_dict['volume'] + self.latest_stock_realtime_data[stock_code]['volume']
				else: # 해당 시간 unique timestamp 임
					self.latest_stock_realtime_data[stock_code][tmp_datetime] = tmp_dict
			
			print(self.latest_stock_realtime_data)

		elif realtype == "주식체결":
			print('enter real time data (2)')
			if stock_code not in self.latest_stock_realtime_data:  # 코드 자체가 처음 들어감
				# self.latest_stock_realtime_data[stock_code] = {tmp_datetime :  realdata} # FID 코드일 것으로 추정은 되는데...
				self.latest_stock_realtime_data[stock_code] = {tmp_datetime: tmp_dict}  # FID 코드일 것으로 추정은 되는데...
			else:  # 코드가 있음
				if tmp_datetime in self.latest_stock_realtime_data[stock_code]:  # 만약에 timestamp에 대한 기록 있고 tmp_dict 가 비지 않았으면
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['price'] = tmp_dict['price'] + self.latest_stock_realtime_data[stock_code]['price']
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['volume'] = tmp_dict['volume'] + self.latest_stock_realtime_data[stock_code]['volume']
				else:  # 해당 시간 unique timestamp 임
					self.latest_stock_realtime_data[stock_code][tmp_datetime] = tmp_dict

			print(self.latest_stock_realtime_data)

		elif realtype == "주식예상체결":
			print('enter real time data (3)')
			if stock_code not in self.latest_stock_realtime_data:  # 코드 자체가 처음 들어감
				# self.latest_stock_realtime_data[stock_code] = {tmp_datetime :  realdata} # FID 코드일 것으로 추정은 되는데...
				self.latest_stock_realtime_data[stock_code] = {tmp_datetime: tmp_dict}  # FID 코드일 것으로 추정은 되는데...
			else:  # 코드가 있음
				if tmp_datetime in self.latest_stock_realtime_data[stock_code]:  # 만약에 timestamp에 대한 기록 있고 tmp_dict 가 비지 않았으면
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['price'] = tmp_dict['price'] + self.latest_stock_realtime_data[stock_code]['price']
					self.latest_stock_realtime_data[stock_code][tmp_datetime]['volume'] = tmp_dict['volume'] + self.latest_stock_realtime_data[stock_code]['volume']
				else:  # 해당 시간 unique timestamp 임
					self.latest_stock_realtime_data[stock_code][tmp_datetime] = tmp_dict
			print(self.latest_stock_realtime_data)

		else:
			pass
		
		self.flag_latest_stock_realtime_data = False
	
	def set_real_register(self, screen_no, stock_code_list, fid_list, opt_type):
		"""
		이미 할당중인지는 FE 단에서 확인하여 돌리는 것으로..!
		opt_type : 0 -> 할당 즉시 나머지 제외하고 마지막시간에 등록한 list만 실시간 수신
		"""
		print('function set_real_register in BE activated...')
		tmp_return = self.dynamicCall("SetRealReg(QString, QString, QString, QString)", screen_no, stock_code_list, fid_list, opt_type)
		return tmp_return


	def set_real_remove(self, screen_no, stock_code):
		"""
		각 화면당 개별적으로 code 번호 지울 것임
		"""
		print('function set_real_remove in BE activated...')
		tmp_return = self.dynamicCall("SetRealRemove(QString, QString, QString, QString)", screen_no, stock_code)
		return tmp_return
	
	
	def _on_receive_message_data(self, screen_no, rqname, trcode, message):

		if rqname == "send_buy_order_req" or rqname == "send_sell_order_req":
			self.latest_buy_sell_result_message = None
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}

			self.latest_buy_sell_result_message = ohlcv
			print('self.latest_buy_sell_result_message BE :: ', self.latest_buy_sell_result_message)

		elif rqname == "cancle_buy_order_req" or rqname == "cancle_sell_order_req":
			self.latest_cancle_buy_sell_result_message = None
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}

			self.latest_cancle_buy_sell_result_message = ohlcv
			print('self.latest_cancle_buy_sell_result_message BE :: ', self.latest_cancle_buy_sell_result_message)
			
		elif rqname == "balance_check_normal" :
			self.latest_balance_check_normal_message = None
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}
			self.latest_balance_check_normal_message = ohlcv
			print('self.latest_balance_check_normal_message BE :: ', self.latest_balance_check_normal_message)
		
		elif rqname == "balance_check_with_order" :
			self.latest_balance_check_with_order_message = None
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}
			self.latest_balance_check_with_order_message = ohlcv
			print('self.latest_balance_check_with_order_message BE :: ', self.latest_balance_check_with_order_message)

		elif rqname == "check_owning_stocks":
			self.latest_check_owning_stock_message = None
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}
			self.latest_check_owning_stock_message = ohlcv
			print('self.latest_check_owning_stock_message BE :: ', self.latest_check_owning_stock_message)

		else:
			ohlcv = {'screen_no': screen_no,
					 'rqname' : rqname,
					 'trcode' : trcode,
					 'message' : message}
			print('un-recognized message :::: ', ohlcv)

	def _on_receive_chejan_data(self, gubun, item_cnt, fid_list):
		# https://wikidocs.net/5931
		print('_on_receive_chejan_data activated in BE...')

		# self.chejen_event_loop = QEventLoop()
		# self.chejen_event_loop.exec_()
		try:
			if self.flag_check_noraml_chejen == False:
				# 비정상 갑작스러운 수신으로 들어오는 부분
				self.flag_sudden_send_order_message = True # 이거 false는 FE단에서 데이터 request 날리고 false로 바꿔줌
		
		
			if gubun == '0':  # 주문 체결 통보
				print('_on_receive_chejan_data 1st path in BE...')
				self.latest_buy_sell_result_first_data = None  # ram 변수 reset in the class, 체결 결과 받아오는 부분
				ohlcv = {}
				"""
				================================================
				'order_num': [], 9203 -> 주문번호
				'stock_name': [], 302 -> 종목명
				'order_num_count': [], 900 -> 주문수량
				'order_price':[], 901 -> 주문가격
				'unmet_order_num' : [], 902 -> 미체결수량
				'original_order_num' : [], 904 -> 원주문번호
				'order_state' : [], 905 -> 주문구분
				'order_done_time':[], 908 -> 주문/체결시간
				'order_met_num' : [], 909 -> 체결번호
				'order_met_price' : [], 910 -> 체결가
				'order_met_num_count' : [] 911 -> 체결량
				현재가, 체결가, 실시간 종가 : 10
				"""

				ohlcv['gunbun']=str(gubun.strip())
				ohlcv['item_cnt']= int(item_cnt.strip())
				ohlcv['fid_list']= str(fid_list.strip())
				#print(gubun, item_cnt, fid_list)
				# -> 의미  :  https://ldgeao99.tistory.com/559?category=880439

				ohlcv['order_num'] = str(self.get_chejan_data(9203).strip())
				ohlcv['stock_name']= str(self.get_chejan_data(302).strip())
				ohlcv['order_num_count']= int(self.get_chejan_data(900).strip())
				ohlcv['order_price']= float(self.get_chejan_data(901).strip())
				ohlcv['unmet_order_num']= int(self.get_chejan_data(902).strip())
				ohlcv['original_order_num']= int(self.get_chejan_data(904).strip())
				ohlcv['order_state']= str(self.get_chejan_data(905).strip())
				ohlcv['order_done_time'] = str(self.get_chejan_data(908).strip())
				ohlcv['order_met_num']= int(self.get_chejan_data(909).strip())
				ohlcv['order_met_price']= float(self.get_chejan_data(910).strip())
				ohlcv['order_met_num_count']=int(self.get_chejan_data(911).strip())
				ohlcv['order_3_price_info']=str(self.get_chejan_data(10).strip())


				self.latest_buy_sell_result_first_data = ohlcv
				
				# @ 1st message 저장 수행
				"""
				tmp_hash_container__order_num = {ohlcv['order_num'] : {
																		'unmet_order_num' : ohlcv['unmet_order_num'],
																		'original_order_num' : ohlcv['original_order_num'],
																		'order_met_price' : ohlcv['order_met_price'],
																		
																		
																		
																		}}
				if ohlcv['stock_name'] in self.send_order_1st_message_store: # 이미 주식 포함되어있으면 order_num기준으로 최신으로 update
					self.send_order_1st_message_store[ohlcv['stock_name']].update(tmp_hash_container__order_num)
				else: # 첫 input
					self.send_order_1st_message_store[ohlcv['stock_name']] = tmp_hash_container__order_num
				"""

					

			elif gubun == '1': # 잔고 통보
				print('_on_receive_chejan_data 2nd path in BE...')
				self.latest_buy_sell_result_second_data = None  # ram 변수 reset in the class, 체결 결과 받아오는 부분
				ohlcv = {}
				"""
				================================================
				'stock_name': [], 302 -> 종목명
				'stock_num_count': [], 930 -> 보유수량
				'stock_buy_price':[], 931 -> 매입단가
				'stock_sum_buy_price' : [], 932 -> 총매입가
				'singleday_pure_buy_num' : [], 945 -> 당일순매수량
				'buy_sell_which' : [], 946 -> 매도매수 구분
				'singleday_pure_sell_profit':[], 950 -> 당일 총 매도 손익
				'budget' : [], 951 -> 예수금
				'profit_rate' : [], 8019 -> 손익률
				현재가, 체결가, 실시간 종가 : 10
				"""

				ohlcv['gunbun']= str(gubun.strip())
				ohlcv['item_cnt']= int(item_cnt.strip())
				ohlcv['fid_list']= str(fid_list.strip())
				#print(gubun, item_cnt, fid_list)
				# -> 의미  :  https://ldgeao99.tistory.com/559?category=880439

				ohlcv['stock_name']= str(self.get_chejan_data(302).strip())
				ohlcv['stock_num_count']=int(self.get_chejan_data(930).strip())
				ohlcv['stock_buy_price']=float(self.get_chejan_data(931).strip())
				ohlcv['stock_sum_buy_price']=float(self.get_chejan_data(932).strip())
				ohlcv['singleday_pure_buy_num']=float(self.get_chejan_data(945).strip())
				ohlcv['buy_sell_which']=str(self.get_chejan_data(946).strip())
				ohlcv['singleday_pure_sell_profit']=float(self.get_chejan_data(950).strip())
				ohlcv['budget']=float(self.get_chejan_data(951).strip())
				ohlcv['profit_rate']=str(self.get_chejan_data(8019).strip())
				ohlcv['order_3_price_info']=str(self.get_chejan_data(10).strip())

				self.latest_buy_sell_result_second_data = ohlcv
				#print('2'*40,ohlcv)

			else:
				print('_on_receive_chejan_data 3rd path in BE...')
				pass
				"""
				error flag 값 띄워야함!
				"""
			try:

				if self.latest_buy_sell_result_first_data != None and self.latest_buy_sell_result_second_data != None and self.latest_buy_sell_result_message != None : # 세가지 모두 받았다면 exit 시도
					self.chejen_event_loop.exit()
					self.chejen_event_loop = None
					self.flag_check_noraml_chejen = False # 정상 request 종료
			except Exception as e:
				self.flag_check_noraml_chejen = False # 정상 request 종료
				print('error in _on_receive_chejan_data BE - exiting self.chejen_event_loop:: ', e)
		except Exception as e:
			print('error in _on_receive_chejan_data BE :: ', e)
			self.flag_check_noraml_chejen = False # 정상 request 종료
			traceback.print_exc()

	def _on_receive_condition_ver(self):
		pass


	def _on_event_connect(self, err_code):
		if err_code == 0:
			print("connected")
			self.flag_1st_login = True
			self.flag_comm_connect_check = False # connect 동작 이후, reset
		else:
			print("disconnected")
			self.flag_1st_login = False
			self.flag_comm_connect_check = False # connect 동작 이후, reset
		try:
			self.login_event_loop.exit()
			self.login_event_loop = None
			self.flag_comm_connect_check = False # connect 동작 이후, reset
		except Exception as e:
			print('error in _on_event_connect BE - already exit complete :: ', e)
			self.flag_comm_connect_check = False # connect 동작 이후, reset
			traceback.print_exc()

	def _on_receive_tr_data(self, screen_no, rqname, trcode, record_name, next,
							unused1, unused2, unused3, unused4):
		import KIWOOM_API__tr_receive_handler as tr
		print('_on_receive_tr_data activated...')
		
		# @ initialize
		self.latest_tr_data = None
		self.is_tr_data_remained = None
		self.latest_balance_normal_data = None
		self.latest_balance_with_order_data = None
		self.latest_owning_stocks_data = None
		self.latest_stock_additional_info = None
		self.latest_stock_unmet_order_data = None

		if next == '2':
			self.is_tr_data_remained = True
		else:
			self.is_tr_data_remained = False

		if rqname == "opt10081_req":
			self.latest_tr_data = tr.on_receive_opt10081(self, rqname, trcode)
		elif rqname == "opt10080_req":
			self.latest_tr_data = tr.on_receive_opt10080(self, rqname, trcode)

		elif rqname == "send_buy_order_req":
			print('tr - send_buy_order_req path')
		elif rqname == "send_sell_order_req" :
			print('tr - send_sell_order_req path')
		elif rqname == "cancle_sell_order_req" :
			print('tr - cancle_sell_order_req path')
		elif rqname == "cancle_buy_order_req":
			print('tr - cancle_buy_order_req path')
			

			
		elif rqname == "balance_check_normal": # 예수금 확인
			print('tr - check balance normal path')
			self.latest_balance_normal_data = tr.on_receive_balance_check_normal(self, rqname, trcode)
			if self.latest_balance_check_normal_message != None and self.latest_balance_normal_data != None : # 둘다 받아야 빠져나옴
				try:
					self.tr_event_loop.exit()
					self.tr_event_loop = None
					self.flag_comm_rq_data_check = False
				except Exception as e:
					print('error in _on_receive_tr_data BE - already exit complete :: ', e)
					self.flag_comm_rq_data_check = False
					traceback.print_exc()
		
		elif rqname == "balance_check_with_order" : # Transaction 과 함께 가져오기 위함
			print('tr - check balance with order path')
			self.latest_balance_with_order_data = tr.on_receive_balance_check_with_order(self, rqname, trcode)
			"""
			>>>
			이 바로 윗부분 역시 수정해주어야함
			# 서버에서 100% 증거금 조회
			"""
			if self.latest_balance_check_with_order_message != None and self.latest_balance_with_order_data != None : # 둘다 받아야 빠져나옴
				try:
					self.tr_event_loop.exit()
					self.tr_event_loop = None
					self.flag_comm_rq_data_check = False
				except Exception as e:
					print('error in _on_receive_tr_data BE - already exit complete :: ', e)
					self.flag_comm_rq_data_check = False
					traceback.print_exc()
					
		elif rqname == "check_owning_stocks" : # 보유 종목 조회하는 부분
			print('tr - check owning stocks path')
			self.latest_owning_stocks_data = tr.on_receive_owning_stocks(self, rqname, trcode)
			"""
			>>>
			메세지 받는거 체크,
			메세지 수신 맞으면 밑에서 빼고 여기서 loop exit 구현
			"""
		
		elif rqname == "check_unmet_order" : # 미체결 종목 조회하는
			print('tr - check unmet order path')
			self.latest_stock_unmet_order_data = tr.on_recieve_unmet_order(self, rqname, trcode)
			"""
			>>>
			메세지 받는거 체크,
			메세지 수신 맞으면 밑에서 빼고 여기서 loop exit 구현 -> 매수 매도 관련이라 미수라고 체결 정보 알려줄 듯
			"""


		elif rqname == "stock_additional_info_tr" : # 추가 정보 요구하는 부분
			# 메시지 받아오는 부분 아님..! 돈거래가 아니라서 그런듯? 정보라..
			print('_on_receive_tr_data in BE activated - stock_additional_info_tr ')
			self.latest_stock_additional_info = tr.on_receive_additional_info_tr(self, rqname, trcode)
			"""
			여기 말고 onrecieverealtime에서 받아야 하지 않나? + 화면 종료나 이런 것은 FE에서 명령받는 것으로 
			https://m.blog.naver.com/PostView.nhn?blogId=jhsgo&logNo=221526307126&proxyReferer=https:%2F%2Fwww.google.com%2F
			"""
		
		
		# @ opt10081 / opt10080 / opw00004
		try:
			if rqname in ["opt10081_req", "opt10080_req", "check_owning_stocks", "stock_additional_info_tr"] : # 이쪽으로 들어와야 event loop 있음
				try:
					self.tr_event_loop.exit()
					self.tr_event_loop = None
					self.flag_comm_rq_data_check = False
				except Exception as e:
					print('error in _on_receive_tr_data BE - already exit complete :: ', e)
					self.flag_comm_rq_data_check = False
					traceback.print_exc()
			else:
				pass
		# except AttributeError:
		# 	print('error in _on_receive_tr_data')
		except Exception as e:
			print('error in _on_receive_tr_data - ', e)
			self.flag_comm_rq_data_check = False
			traceback.print_exc()

	def comm_connect_Timer(self):
		try:
			if self.login_event_loop.isRunning():
				print('un-safe exit in comm_connect BE...')
				self.error_counter_BE = self.error_counter_BE + 1
				self.login_event_loop.exit()
				self.login_event_loop = None
				self.flag_comm_connect_check = False

			else:
				print('safe exit in comm_connect BE...')
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
		except Exception as e:
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
			print('error in comm_connect_Timer BE :: ', e)
			traceback.print_exc()

	def comm_connect(self):
		print("Login 요청 후 서버가 이벤트 발생시킬 때까지 대기하는 메소드")
		# timer_out_loop = QTimer() # exit용 타이머 설정
		# timer_out_loop.start()
		try:
			print('aaa-1')
			timer_out = QTimer() # timer 설정
			print('aaa-2')
			self.login_event_loop = QEventLoop() # event loop 설정
			timer_out.singleShot(int(20000), self.comm_connect_Timer)  # 20초 후 exit, timer 연결
			print('aaa-3')
			if not self.flag_comm_connect_check: # 두번 보내는 것 방지 위함
				comm_connect_data_result = self.dynamicCall("CommConnect()")
				self.flag_comm_connect_check = True # connect 동작 이후 올림
			else:
				pass
			print('aaa-4')
			self.login_event_loop.exec_() # 여기서 대기하고 event 발생시 thread로 event connect된 애가 열리면서 exit 하게 되는 듯...! : share 한다고 해야 하나? 그런 느낌
			print('aaa-5')
			print('comm_connect in BE return value :: ', comm_connect_data_result)
			return comm_connect_data_result

		except Exception as e:
			print('error in comm_connect BE :: ', e)
			traceback.print_exc()
			return None


	def comm_rq_data_Timer(self):
		try:
			if self.tr_event_loop.isRunning():
				print('un-safe exit in comm_rq_data BE...')
				self.error_counter_BE = self.error_counter_BE + 1
				self.tr_event_loop.exit()
				self.tr_event_loop = None
				self.flag_comm_rq_data_check = False

			else:
				print('safe exit in comm_rq_data BE...')
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
		except Exception as e:
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
			print('error in comm_rq_data_Timer BE :: ', e)
			traceback.print_exc()

	@call_printer
	def comm_rq_data(self, rqname, trcode, next, screen_no, state_time, weekday_num):
		#self.num_com_rq_data = self.num_com_rq_data + 1

		"""
		서버에 조회 요청을 하는 메소드
		이 메소드 호출 이전에 set_input_value 메소드를 수차례 호출하여 INPUT을 설정해야 함
		"""
		#timer_out_loop = QTimer()  # exit용 타이머 설정
		#timer_out_loop.start()
		try:
			comm_rq_data_result = None # initialize RAM variable
			
			print('bbb-1')
			timer_out = QTimer()
			print('bbb-2')
			self.tr_event_loop = QEventLoop()
			timer_out.singleShot(int(3000), self.comm_rq_data_Timer)  # 3초 후 exit, timer 연결
			print('bbb-3')
			if not self.flag_comm_rq_data_check: # 1회만 요청하기 위해서
				comm_rq_data_result = self.dynamicCall("CommRqData(QString, QString, int, QString)", rqname, trcode, next, screen_no)
				print('bbb-ㄱ')
				self.flag_comm_rq_data_check = True
			else:
				print('bbb-ㄴ')
				pass

			print('bbb-4')
			self.tr_event_loop.exec_()
			print('bbb-5')
			print('comm_rq_data in BE return value :: ', comm_rq_data_result)



			# @ wait for request time and return the result
			# 키움 Open API는 시간당 request 제한이 있기 때문에 딜레이를 줌
			if not(state_time == "개장중" and (weekday_num >= 0 and weekday_num <= 4 )): # 주중 개장이 아닐 때 -> 그외 모든 시간
				if rqname in ["opt10081_req", "opt10080_req" ] : # 분봉 / 일봉 조회
					self.rest_for_request(self.TR_REQ_TIME_INTERVAL)
				else: # 아닐 시 1초당 5번 이하로 request
					self.rest_for_request(0.21)
					# https://smbyeon.github.io/2019/12/06/kiwoom-graph.html
			else:
				self.rest_for_request(0.41)

			return comm_rq_data_result


		except Exception as e:
			print('error in comm_rq_data BE :: ', e)
			traceback.print_exc()
			return None
	def get_condition(self):
		"""
		조건검색의 조건목록 요청
		:return:
		"""
		ret = self.dynamicCall("GetConditionLoad()")
		return ret


	def get_chejan_data(self, fid):
		ret = self.dynamicCall("GetChejanData(int)", int(fid))
		return ret.strip()

	def get_comm_data(self, trcode, record_name, index, item_name):
		"""
		ACCOUNT BALANCE 가져오는 부분
		
		:param trcode: 
		:param record_name: 
		:param index: 
		:param item_name: 
		:return: 
		"""
		ret = self.dynamicCall("GetCommData(QString, QString, QString, int, QString)", trcode,
							   record_name, index, item_name)
		return ret.strip()

	"""
		def _on_receive_tr_data(self, screen_no, rqname, trcode, record_name, next,
							unused1, unused2, unused3, unused4):
	"""

	def comm_get_data(self, code, real_type, field_name, index, item_name):
		"""
		이거 용도에 맞는 애로 바꿔야 한다 GetCommData 로!!!
		"""
		ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", code,
							   real_type, field_name, index, item_name)
		return ret.strip()
	
	def get_comm_real_data(self, stock_code, fid_num):
		ret = self.dynamicCall("GetCommRealData(QString, int)", stock_code, fid_num)
		
		return ret.strip()

	def send_order_Timer(self):
		try:
			if self.chejen_event_loop.isRunning():
				print('un-safe exit in send_order BE...')
				self.error_counter_BE = self.error_counter_BE + 1
				self.chejen_event_loop.exit()
				self.chejen_event_loop = None
				self.send_order_result = None
				self.flag_check_noraml_chejen = False # 정상 request 종료 표시

			else:
				print('safe exit in send_order BE...')
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
		except Exception as e:
			print('self.error_counter_BE value in BE - counter up :: ', self.error_counter_BE)
			print('error in send_order_Timer BE :: ', e)
			traceback.print_exc()


	def send_order(self, rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no):

		self.send_order_result = None
		self.flag_check_noraml_chejen = True # 정상 request 올라감

		# @ Q-event loop
		timer_out = QTimer()
		self.chejen_event_loop = QEventLoop()
		timer_out.singleShot(int(2000), self.send_order_Timer) # 2초

		self.send_order_result = self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
						 [rqname, screen_no, acc_no, order_type, code, quantity, price, hoga, order_no])
		print('self.send_order_result in BE :: ', self.send_order_result)
		print('successfully sent order request in BE...')

		# 1초당 5회 제한 있음
		self.rest_for_request(0.3)

		print('send order event waiting in BE...')
		if (self.latest_buy_sell_result_message != None) and ('장종료' in self.latest_buy_sell_result_message['message']):  
			# 장 종료 상태로 wait 할 필요 없음 - 메세지 단에서 짤라버리고 하단에서 exec 하지 않음
			# '[00Z218] 모의투자 장종료 상태입니다'
			self.flag_check_noraml_chejen = False # 정상 request 종료 표시
		else:
			self.chejen_event_loop.exec_()
		print('end of send_order in BE...')

		return self.send_order_result

	def get_master_stock_state(self, code):
		"""
		입력한 종목의 증거금 비율, 거래정지, 관리종목, 감리종목, 투자융의종목, 담보대출, 액면분할, 신용가능 여부를 전달합니다.
		:param code:
		:return:
		"""
		state = self.dynamicCall("GetMasterStockState(QString)", code)
		#print(state)
		return state

	def get_master_construction(self, code):
		"""
		입력한 종목코드에 해당하는 종목의 감리구분(정상, 투자주의, 투자경고, 투자위험, 투자주의환기종목)을 전달합니다.
		:param code:
		:return:
		"""
		construction = self.dynamicCall("GetMasterConstruction(QString)", code)
		#print(construction)
		return construction


	def get_code_list_by_market(self, market):
		"""market의 모든 종목코드를 서버로부터 가져와 반환하는 메소드"""
		code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
		code_list = code_list.split(';')
		return code_list[:-1]

	def get_master_code_name(self, code):
		"""종목코드를 받아 종목이름을 반환하는 메소드"""
		code_name = self.dynamicCall("GetMasterCodeName(QString)", code)
		return code_name

	def get_connect_state(self):
		"""서버와의 연결 상태를 반환하는 메소드"""
		ret = self.dynamicCall("GetConnectState()")
		return ret

	def set_input_value(self, input_dict):
		"""
		CommRqData 함수를 통해 서버에 조회 요청 시,
		요청 이전에 SetInputValue 함수를 수차례 호출하여 해당 요청에 필요한
		INPUT 을 넘겨줘야 한다.
		"""
		for key, val in input_dict.items():
			self.dynamicCall("SetInputValue(QString, QString)", key, val)

	def get_repeat_cnt(self, trcode, rqname):
		ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
		return ret

	def get_server_gubun(self):
		"""
		실투자 환경인지 모의투자 환경인지 구분하는 메소드
		실투자, 모의투자에 따라 데이터 형식이 달라지는 경우가 있다. 대표적으로 opw00018 데이터의 소수점
		"""
		self.login_gunbun = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
		return self.login_gunbun

	def get_login_info(self, tag):
		"""
		계좌 정보 및 로그인 사용자 정보를 얻어오는 메소드
		"""
		ret = self.dynamicCall("GetLoginInfo(QString)", tag)
		return ret

	def rest_for_request(self, secs):
		"""
		지정된 시간동안 병렬로 execute로 gui동작 허용하면서 시간 멈춤
		"""
		loop = QEventLoop()
		QTimer.singleShot(int(secs * 1000), loop.quit) # ms
		loop.exec_()
	
	

class MY_SIG_BE_global(QObject): # 사용자 정의 시그널 포함하는 class
	sig_build_database = pyqtSignal() #

	def run_sig_name(self, func_name): # 함수 명으로 함수 수행하는 함수
		func = getattr(KiwoomAPI, func_name)
		func(self)

	@pyqtSlot()
	def build_database(self): # 데이터베이스 시그널 송출
		self.sig_build_database.emit()





# C++과 python destructors 간의 충돌 방지를 위해 전역 설정
# garbage collect 순서를 맨 마지막으로 강제함
# 사실, 이 파일을 __main__으로 하지 않는경우에는 고려 안해도 무방
app = None


def main():
	global app
	app = QApplication(sys.argv)
	kiwoom = KiwoomAPI()
	kiwoom.comm_connect()


if __name__ == "__main__":
	main()