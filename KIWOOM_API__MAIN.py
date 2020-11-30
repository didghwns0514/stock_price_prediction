# -*- coding: utf-8 -*-
import time
import os
import sys


import tracemalloc
#import sklearn.externals import joblib
import joblib
#impot klepto
#https://stackoverflow.com/questions/17513036/pickle-dump-huge-file-without-memory-error

from PyQt5.QtWidgets import *
from PyQt5.QtChart import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QAxContainer import *
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
#from utils import TimeAxisItem, timestamp


import datetime
import copy
import pandas as pd
import pickle
import _pickle as fpk # fast pickle
import sqlite3
import re
import pandas as pd
import datetime
import traceback
import numpy as np
import inspect
from transitions import Machine
import random
import gc
import multiprocessing as mp


from KIWOOM_API__BACKEND import KiwoomAPI
#import KIWOOM_API__autotrade_manager as at


import queue
# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj

###########################################
# Globals of api



###########################################
def return_status_msg_setter(original_func):
	"""
	original 함수 exit 후, QMainWindow 인스턴스의 statusbar에 표시할 문자열을 수정하는 데코레이터
	이 데코레이터는 QMainWindow 클래스의 메소드에만 사용하여야 함
	"""

	def wrapper(self):
		ret = original_func(self)

		timestamp = datetime.datetime.now().strftime('%H:%M:%S')

		# args[0]는 인스턴스 (즉, self)를 의미한다.
		self.DISP_STRING__display_current_job = '`{}` 수행중[{}]'.format(original_func.__name__, timestamp)
		return ret

	return wrapper


def PyQ_wrapper(api_to_bit32_Q, bit32_to_api_Q):
	app = QApplication(sys.argv)
	myWindow = App_wrapper(api_to_bit32_Q, bit32_to_api_Q)
	myWindow.show()
	#myWindow.check_queue()  ->  자동으로 계속 돌아가는 내부 함수 만들어야함
	app.exec_()


class App_wrapper(QMainWindow):
	def __init__(self, api_to_bit32_Q, bit32_to_api_Q):
		super().__init__()
		self.title = "Py_stock_trader"
		self.left = 100
		self.top = 70
		self.width = 860
		self.height = 900
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.table_widget = pyq_object(self, api_to_bit32_Q, bit32_to_api_Q)
		self.setCentralWidget(self.table_widget)

		#self.show()


class pyq_object(QWidget, QObject):  # this is API
	# QAxWidget 클래스로부터 dynamicCall, setControl, OnEventConnect 를 상속받음

	"""
	https://zapary.blogspot.com/2015/12/qt5-using-activex.html
	https://doc.qt.io/qt-5/qaxwidget.html

	★ https://decdream08.tistory.com/19

	QAxWidget -> active X 사용하기 위한 class가 있음, 상속해서 사용
	
	"""
	#MUST_WATCH_LIST = ["226490", "261250"]
	MUST_WATCH_LIST = ["226490", "261250", "252670"]
	#ㅋㅋ# KODEX 코스피, KODEX 미국달러선물 레버리지, KODEX 200선물 인버스 2X

	REQUEST_MAX_NUM = 98 #self.REQUEST_MAX_NUM
	REQUEST_MAX_NUM_ON_THE_RUN = 998 # 돌고있는 와중에 request number for tr data
	STOCK_AT_MAX_NUM = 10 # 최대로 들고 있을 주식 갯수
	STOCK_AT_WATCH_MAX_NUM = 20 + int(len(MUST_WATCH_LIST))#최대로 realtime 받을 종목 수
	STOCK_SECOND_DATA_LEN = 3 #관리할 초의 length in minutes

	STOCK_TARGET_PROFIT = 0.03 # 목표 수익률
	STOCK_TARGET_MICRO_PROFIT = 0.03 # 30분간 목표 수익률
	STOCK_TARGET_MINUS_PROFIT = -0.03 # SEC 데이터에서 이거 이하로 내려가면 sell 한다

	ARTICLE_MIN_NUMBER = 1 # 최소 들고있어야 할 기사 개수
	ARTICLE_MIN_NUMBER_ONGOING = 10 # 개장중 안보고 있었을 때 뉴스기사 생성되면 db 만들고 scrno 등록할 최소 개수

	STOCK_AT_TIME_WINDOW_HOUR = 10 # 최대 감시해서 들고있을 주식 시간 width
	STOCK_BUDGET_AT_LEAST = 100000 # 최소 계좌에 들고있어야 하는 금액
	
	AUTO_TRADE_FEE = 0.0034792 # 대략



	# https://www.google.com/search?q=python+class+slot&rlz=1C1GCEA_enKR869KR869&oq=python+class+slot&aqs=chrome..69i57j0l7.3873j0j4&sourceid=chrome&ie=UTF-8
	__slots__= ['TEST', 'MINIMIZED_STOCK_LIST','GC_THRESHOLD__old', 'GC_THRESHOLD__new', 'name', 'test_fe', 'SIGNAL_MINE', 'layout', 'STATE_TIME', 'KI_MESSAGE', 'SCREEN_NO', 'STOCK_IN_ATTENTION', 'STATE_MACHINE', 'input_queue', 'output_queue', 'DISP_STRING__display_current_job', 'STOCK_DICTIONARY_FROM_BE__real_time_data_SEC', 'STOCK_DICTIONARY_FROM_BE__real_time_data_MIN', 'STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML', 'STOCK_DICTIONARY_FROM_ML__real_time_data_MIN', 'STOCK_DICTIONARY_FROM_ML__path_for_32bit', 'TEMP_CHECK_2_STOCK_IN_ATTENTION', 'STOCK_LIST__for_total_display', 'STOCK_DICTIONARY__name_to_code', 'STOCK_DICTIONARY__code_to_name', 'STOCK_DICTIONARY_NAME__article_dump', 'STOCK_DICTIONARY_NAME__article_result', 'STOCK_PICKLE__path_for_article', 'STOCK_DICTIONARY_NAMES__owning_stocks', 'STOCK_DICTIONARY_NAMES__basic_info', 'STOCK_DICTIONARY_NAMES__additional_info_tr', 'STOCK_DICTIONARY_NAMES__unmet_order', 'STOCK_FLAG__additional_info_creation_in_progress', 'STOCK_FLAG__additional_info_is_stalled', 'STOCK_FLAG__started_getting_additional_info_tr', 'STOCK_FLAG__when_unmet_order_made', 'STOCK_PICKLE__path_for_additional_info', 'SQLITE_LIST__folder_sub_file_path', 'SQLITE_PICKLE__path_for_db_update_date', 'SQLITE_DICTIONARY__db_update_date', 'SQLITE__con_top', 'SQLITE__cur_top', 'SQLITE_LIST__stocks_already_updated', 'BALANCE_PICKLE__path_for_balance_update_date', 'BALANCE_DICTIONARY__for_normal_update', 'ACCOUNT__code_of_my_account', 'ACCOUNT_LIST__for_accounts_owned', 'ACCOUNT__user_id', 'STOCK_LIST__all_kospi', 'STOCK_LIST__all_kosdq', 'ERROR_COUNTER_BE__request_num', 'ERROR_COUNTER_BE__front_be_counter_previous', 'ERROR_DICTIONARY__backend_and_critical', 'flag_FUNC_STOCK__enable_auto_trading', 'ERROR_FLAG__sell_all_err_critical_check_3', 'SQLITE_FLAG__database_creation_in_progress','SQLITE_FLAG__database_is_stalled', 'DISP_FLAG__test_code_look_up_sucess', 'SQLITE_FLAG__latest_database_data_checked', 'COUNTER_GLOBAL','CHECK_1_FLAG__ALL_login','CHECK_1_RESULT__api_real_or_try','CHECK_1_LOGICAL__windows_1st_login','CHECK_1_LOGICAL__ki_connect_state','CHECK_2_FLAG__every_data_get_success', 'CHECK_2_FLAG_BALANCE__get_success', 'CHECK_2_FLAG_STOCK__get_all_stock_codes', 'CHECK_2_FLAG_SQLITE__first_database_create_success', 'CHECK_2_FLAG_STOCK__owning_stock_get_success', 'CHECK_2_FLAG_STOCK__unmet_order_success', 'CHECK_2_FLAG__news_article_pickle_ready', 'CHECK_2_FLAG_STOCK__basic_info', 'CHECK_2_FLAG_STOCK__additional_info_tr', 'CHECK_FILTER_FLAG__at_initialize', 'CHECK_3_FLAG__ALL_auto_trade_ready', 'CHECK_3_FLAG__ALTIMATE_AUTO_ON', 'CHECK_SELL_ALL_FLAG__restart_api', 'AT_STOCK_CLASS__wrapper', 'AT_FLAG__very_first_init_func_called', 'AT_TUPLE__profit_record_watch', 'AT_TUPLE__profit_record_trans', 'FLAG__FIRST_TIME_REACHED_FILTER_STAGE', 'TOTAL_DICT', 'COMM_32','tabs', 'tab_login', 'tab_cockpit', 'tab_stats','tab_database','tab_test', 'KIWOOM', 'CHECK_1_LOGICAL__windows_1st_login', 'timer', 'timer_num_req']
	

	def __init__(self, parent, api_to_bit32_Q, bit32_to_api_Q):
		#super().__init__()
		super(QWidget, self).__init__(parent)
		super(QObject, self).__init__(self)

		#########################
		# TEST
		self.TEST = True
		tracemalloc.start(10)
		#self.tr_mlc__1 = tracemalloc.take_snapshot()

		# GC SETTING
		self.GC_THRESHOLD__old = gc.get_threshold()
		self.GC_THRESHOLD__new = None
		print(f'self.GC_THRESHOLD__old value : {self.GC_THRESHOLD__old}')
		print(f'setting up hard gc threashold...')
		#self.GC_THRESHOLD__new = (int(self.GC_THRESHOLD__old[0]/2), int(self.GC_THRESHOLD__old[1]/2), int(self.GC_THRESHOLD__old[2]/2) )
		self.GC_THRESHOLD__new = (0,0,0)
		print(f'self.GC_THRESHOLD__new value : {self.GC_THRESHOLD__new}')

		# @ minimized stock list
		self.MINIMIZED_STOCK_LIST = True

		#########################
		
		self.name = None # 함수 이름으로 동작
		self.test_fe = 'Hi- this is fe'
		
		# @ pyqt5 GUI 생성
		#_________________________________________________________________________________
		#============================================================================================
		self.SIGNAL_MINE = MY_SIG_global() # 사용자 정의 시그널 class 인스턴스 생성
		self.layout = QVBoxLayout(self)

		# @ Thread 선언, wrapping하기 위한 부분
		#self.thread_wrapper = Thread_wrapper(self)

		# @ stage today 단계별 추적
		self.STATE_TIME = Time_stage()

		# @ kiwoom message 해독용
		self.KI_MESSAGE = Message()

		# @ Screen number 부여
		self.SCREEN_NO = Screen_no(self.STOCK_AT_WATCH_MAX_NUM)
		
		# @ STOCK IN ATTENTION
		self.STOCK_IN_ATTENTION = STOCK_IN_ATTENTION()

		# @ state machine
		self.STATE_MACHINE = Machine_state()
		print('STATE_MACHINE :: ', self.STATE_MACHINE.state)
		#time.sleep(5)

#		self.test = at.TEST_AT()

		# @ Request Wrapper
		#self.request_wrapper = Stock_wrapper() # 각 쓰레드의 error값들과 전달할 개별 종목의 class 가지고 있는 class # 반드시 deep copy로 할 것, 이것이 큐를 왔다갔다 한다!
		# _________________________________________________________________________________
		# ============================================================================================

		# @ queue
		self.input_queue = bit32_to_api_Q
		self.output_queue = api_to_bit32_Q


		# @ 무조건 사용하는 변수 // error들
		#_________________________________________________________________________________
		#============================================================================================
		#1) 결과를 쓸 변수들
		self.DISP_STRING__display_current_job = ''  # 현재 작업
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC = { } # BE 단에서 받아온 realtime 데이터 second
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN = {

			'STOCK_MIN' : {},
			'SQLITE' : {},
			'FILTER' : {},
			'BUDGET' : {},
			'OWNING' : {}
		} # SQLITE로 boolian True일 때 만 작업함
		self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML = None
		
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN = {'prediction':{}, 'trade':{} , 'date':None}
		self.STOCK_DICTIONARY_FROM_ML__path_for_32bit = None # ml에서 보내준거 읽어올 주소, self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN에 엎어침
		
		self.TEMP_CHECK_2_STOCK_IN_ATTENTION = None # 2단계에서 3단계 자동트레이팅 넘어갈 때 저장할 STOCK_IN_ATTENTION -> 되는 건지 모르겠음??

		"""
		STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']
		STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']
		"""

		self.STOCK_LIST__for_total_display = None # 전체 종목 list //self.STOCK_LIST__for_total_display//
		self.STOCK_DICTIONARY__name_to_code = {} # 이름에 대한 code값 (사용자 편의)
		self.STOCK_DICTIONARY__code_to_name = {}
		self.STOCK_DICTIONARY_NAME__article_dump = {} # 기사 article 엎어치는 부분 ----------------------------------------------------- 64bit 에서 축약본 가져오기
		self.STOCK_DICTIONARY_NAME__article_result = {} # 기사 개수 세는 부분
		self.STOCK_PICKLE__path_for_article = None # 기사 pickle 주소
		self.STOCK_DICTIONARY_NAMES__owning_stocks = {} # 보유 주식 dictionary
		self.STOCK_DICTIONARY_NAMES__basic_info = {}
		self.STOCK_DICTIONARY_NAMES__additional_info_tr = {} # tr에서 구할 수 있는 더 세밀한 정보
		self.STOCK_DICTIONARY_NAMES__unmet_order = { } # unmet order 나올 때 마다 담아놓을 부분
		
		self.STOCK_FLAG__additional_info_creation_in_progress = False # 이거 정보 가져온다고 올린다
		self.STOCK_FLAG__additional_info_is_stalled = False # 다시 시작했으므로 멈춘다
		self.STOCK_FLAG__started_getting_additional_info_tr = False
		self.STOCK_FLAG__when_unmet_order_made = False # unmet order 생성시마다 올리는 부분
		self.STOCK_PICKLE__path_for_additional_info = None # 경로

		#self.pickle_last_update_record = None # db 업데이트 시점 pickle 데이터 저장하는 부분
		self.SQLITE_LIST__folder_sub_file_path = []
		self.SQLITE_PICKLE__path_for_db_update_date = None # db pickle file 경로
		self.SQLITE_DICTIONARY__db_update_date = {}  # 실제 db_pickle file 담을 ram 변수
		self.SQLITE__con_top = None #sqlite
		self.SQLITE__cur_top = None #sqlite
		self.SQLITE_LIST__stocks_already_updated = [] # 이미 getting 한 code 저장하는 list

		self.BALANCE_PICKLE__path_for_balance_update_date = None # balance db file 경로
		self.BALANCE_DICTIONARY__for_normal_update = {'balance':None, 'date':None} # balance를 normal로 조회한 시점 저장
		#self.account_balance = None # 실제 account의 잔고 int 저장 부분


		#2) Error 체크할 변수들
		######################
		# kiwoom 증권 문서 봐서 return값으로 확인해야 됨
		######################
		self.ACCOUNT__code_of_my_account = None # self.KIWOOM.dynamicCall("GetLoginInfo(QString)", ["ACCNO"])
		self.ACCOUNT_LIST__for_accounts_owned = [] # account 정보 가져와서 받는 list
		self.ACCOUNT__user_id = None # self.KIWOOM.dynamicCall("GetLoginInfo(QString)", ["USER_ID"])
		self.STOCK_LIST__all_kospi = None # self.KIWOOM.dynamicCall("GetCodeListByMarket(QString)", ['10'])  # 코스피
		self.STOCK_LIST__all_kosdq = None # self.KIWOOM.dynamicCall("GetCodeListByMarket(QString)", ['0'])  # 코스닥
		#self.tmp_ret_value = None # 특별한 변수들을 제외하고 tmp로 담을 container, return 값 감시
		self.ERROR_COUNTER_BE__request_num = 0 # backend에서 돌아가고있는 request 갯수 체크 99 이상이면 로그인부터 다시
		self.ERROR_COUNTER_BE__front_be_counter_previous = 0 # 전단계
		####중요####
		self.ERROR_DICTIONARY__backend_and_critical={
			'error_backend'  : 0,
			'error_critical' : 0
		}
		############

		
		#3) flag 변수들
		#self.flag_FUNC_STOCK__enable_auto_trading = False # 모든걸 전자동화 한다는 flag
		self.ERROR_FLAG__sell_all_err_critical_check_3 = False # 에러 하나라도 있으면 up flag -> selling 진입


		self.SQLITE_FLAG__database_creation_in_progress  = False # 자동으로 1분 데이터 만드는 flag 올림 -> 이 작업이 진행 중이라는 말
		self.SQLITE_FLAG__database_is_stalled = False # 카운터 다 차서 그만 해야할 때 올림
		#self.SIG_database_create = pyqtSignal(self)
		self.DISP_FLAG__test_code_look_up_sucess = False # 조회 완료된거면 flag 올려서 매수 버튼 클릭 가능하도록 함
		self.SQLITE_FLAG__latest_database_data_checked = False # 데이터 베이스 local에서 필요한 부분 탐색 완료시 올림
		

		
		
		#4) Counter 변수들
		self.COUNTER_GLOBAL = 0 # 전체 카운터
		
		#5) CHECK 변수들
		self.CHECK_1_FLAG__ALL_login = False #  모든 로그인 성공이면 올린다

		self.CHECK_1_RESULT__api_real_or_try = None  # api가 모의투자인지 아닌지
		self.CHECK_1_LOGICAL__windows_1st_login = None # self.KIWOOM.dynamicCall("CommConnect()")
		self.CHECK_1_LOGICAL__ki_connect_state = None # self.KIWOOM.dynamicCall("GetConnectState()")
		#-------------------------------------------------------------------------------
		self.CHECK_2_FLAG__every_data_get_success = False  # 모든 데이터베이스 가져오면 올린다

		self.CHECK_2_FLAG_BALANCE__get_success = False # 처음 python api 가동시, balance 확인부분
		self.CHECK_2_FLAG_STOCK__get_all_stock_codes = False # 처음 python api 가동시, 모든 종목 코드 가져오는 부분
		self.CHECK_2_FLAG_SQLITE__first_database_create_success = False # 처음 api 가동시 database 한번 가져오고 true로 올림
		self.CHECK_2_FLAG_STOCK__owning_stock_get_success = False # 보유 종목 체크 확인시 true
		self.CHECK_2_FLAG_STOCK__unmet_order_success = False # 미체결 종목 확인 여부 flag 올림
		self.CHECK_2_FLAG__news_article_pickle_ready = False  # 뉴스 pickle 데이터베이스 데이터를 가져오는데 성공하면, 올린다.
		self.CHECK_2_FLAG_STOCK__basic_info = False # 기본 감리정보 등 가져오는 부분
		self.CHECK_2_FLAG_STOCK__additional_info_tr = False # TR 에서 추가 정보 가져오도록 요청하는 부분 (세부 디테일)...!
		
		# -------------------------------------------------------------------------------
		self.CHECK_FILTER_FLAG__at_initialize = False # 종목 필터링 끝나면 True로 올릴 부분

		# -------------------------------------------------------------------------------
		self.CHECK_3_FLAG__ALL_auto_trade_ready = False # 종목코드까지 가져오는 것 성공하였으면, 올린다.
		self.CHECK_3_FLAG__ALTIMATE_AUTO_ON = False # GUI 테스트와 분리하기 위함 - all auto 버튼 클릭시 true로 올라옴 -> 나중에 시간으로 조정
		# -------------------------------------------------------------------------------
		self.CHECK_SELL_ALL_FLAG__restart_api = False #  모두 팔아버리는 flag
		# -------------------------------------------------------------------------------
		
		
		# 6) automation 변수들
		#self.AT_STOCK_CLASS__wrapper = None
		# =============================================================================================
		self.AT_FLAG__very_first_init_func_called = False
		self.AT_TUPLE__profit_record_watch = []
		self.AT_TUPLE__profit_record_trans = []
		self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE = False

		self.TOTAL_DICT = {}
		self.COMM_32 = None

		# =============================================================================================
		# create tabs
		self.tabs = QTabWidget()
		self.tab_login = QWidget()
		self.tab_cockpit = QWidget()
		self.tab_stats = QWidget()
		self.tab_database = QWidget()
		self.tab_test = QWidget()
		#self.tab_database = QListWidget()
		self.tabs.resize(800, 900)

		# Add tabs
		self.tabs.addTab(self.tab_login, "로그인")
		self.tabs.addTab(self.tab_cockpit, "칵핏")
		self.tabs.addTab(self.tab_stats, "보유종목/금액")
		self.tabs.addTab(self.tab_database, "데이터베이스")
		self.tabs.addTab(self.tab_test, "테스트 수행")

		# individual build of tabs
		self.build()

		# Add tabs to widget
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)

		# ============================================================================================
		#_________________________________________________________________________________

		# @ class 생성시 자동 login 수행할 부분까지 자동화 -> 3 대장 부름
		self.KIWOOM = KiwoomAPI()
		self.CHECK_1_LOGICAL__windows_1st_login = self.KIWOOM.comm_connect()
		self.FUNC_CHECK_DISP__all_login_process()

		self.func_SQLITE_LIST__for_path_to_db()
		self.func_SQLITE_PICKLE__create_db_update_date()
		self.func_BALANCE_PICKLE__create_balance_update_date()
		self.func_STOCK_PICKLE__for_create_additional_info()
		self.func_STOCK_DICTIONARY_PICKLE__article_location()
		self.func_STOCK_DICTIONARY_PICKLE__min_data_dump_for_ML()
		self.func_STOCK_DICTIONARY_PICKLE__FROM_ML__path_for_32bit()


		# @ 자동 Event 감지지
		#self.KIWOOM.OnReceiveTrData.connect(self.receive_trdata)

		# @ QTimer 1sec
		self.timer = QTimer(self)
		self.timer.setInterval(1000) # ms
		self.timer.timeout.connect(self.QTIMER__periodic_state_checker_1s)
		self.timer.start()

		# @ QTimer for request counter
		self.timer_num_req = QTimer(self)
		self.timer_num_req.setInterval(100) # ms
		self.timer_num_req.timeout.connect(self.QTIMER__periodic_worker)
		self.timer_num_req.start()

	# def EMIT_flag_database_create_auto(self):
	# 	self.SIG_database_create.emit()

	def FUNC__restart_api(self): # 재접속 하는 부분
		print('FE - initiate re-login proces....')
		# @ FE 먼저 초기화
		#===================================================================
		# @ BE error 값 저장
		self.ERROR_COUNTER_BE__front_be_counter_previous = self.KIWOOM.error_counter_BE + self.ERROR_COUNTER_BE__front_be_counter_previous

		# @ BE realtime data 저장
		self.FUNC_STOCK_BE__update_real_time_for_FE()


		# @ DB pickle update 시점 저장
		try:  # save new pickle db date that was updated..
			with open(self.SQLITE_PICKLE__path_for_db_update_date, 'wb') as file:
				pickle.dump(self.SQLITE_DICTIONARY__db_update_date, file)
				print('successfully save new db date pickle file...')
		except Exception as e:
			print('error in FUNC_STOCK_DATABASE_SQLITE__create - failed to save db pickle date data :: ', e)

		# @ 각종 변수 초기화
		self.CHECK_1_LOGICAL__windows_1st_login = None # self.KIWOOM.dynamicCall("CommConnect()")
		self.CHECK_1_LOGICAL__ki_connect_state = None # self.KIWOOM.dynamicCall("GetConnectState()")
		self.ACCOUNT__code_of_my_account = None # self.KIWOOM.dynamicCall("GetLoginInfo(QString)", ["ACCNO"])
		self.ACCOUNT__user_id = None # self.KIWOOM.dynamicCall("GetLoginInfo(QString)", ["USER_ID"])

		# @ Thread 초기화
		#self.thread_wrapper.__init__()

		print('FE - request 카운터 0으로 리셋')
		self.ERROR_COUNTER_BE__request_num = 0
		self.KIWOOM.reset_rq_data()  # 0 으로 만들어줌
		# ===================================================================

		# @ backend 인스턴스 초기화
		print('FE - 키움 인스턴스 초기화 진행')

		self.KIWOOM.__init__()
		self.FUNC_PYQT__rest_timer(1) # 시간초 대기


		print('FE - 윈도우와 연결상태 다시 생성')
		self.CHECK_1_LOGICAL__windows_1st_login = self.KIWOOM.comm_connect() # 윈도우와 연결상태 다시 만들기
		print('FE - reset 함수에서 연결상태 체크')
		self.FUNC_CHECK__ki_connect_state()  # 연결상태 체크 - 다시 하고 기다리기
		self.FUNC_PYQT__rest_timer(1) # 시간초 대기


		# @ 데이터베이스 종료
		try:
			self.SQLITE__con_top.close()
			self.SQLITE__con_top = None
			del self.SQLITE__con_top
		except Exception as e:
			print('error in FUNC__restart_api - failed to close sqlite connection :: ', e)

		#self.FUNC_CHECK__ki_connect_state()
		print('finished api - FUNC__restart_api')
	
	def func_STOCK_DICTIONARY_PICKLE__FROM_ML__path_for_32bit(self):
		"""
		ML 제공하는 Package64의 위치
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN 에 dump로 가져옴
		self.STOCK_DICTIONARY_FROM_ML__path_for_32bit 가 위치
		"""
		try:
			python_path = os.getcwd()
			db_path = str(python_path + '\\KIWOOM_API__from_ML').replace('/', '\\')
			file_path = db_path + '\\' + 'ML_DATA.p'
			self.STOCK_DICTIONARY_FROM_ML__path_for_32bit = copy.deepcopy(file_path)
			
		except Exception as e:
			print('error in func_STOCK_DICTIONARY_PICKLE__FROM_ML__path_for_32bit :: ', e)
	
	
	def func_STOCK_DICTIONARY_PICKLE__min_data_dump_for_ML(self):
		"""
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN  덤프 떠서
		self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML 주소에 넣어줌
		"""
		try:
			python_path = os.getcwd()
			db_path = str(python_path + '\\KIWOOM_API__to_ML').replace('/', '\\')
			if os.path.isdir(db_path): # 경로 존재하는지 확인
				pass
			else:
				os.mkdir(db_path) # 경로 생성
			file_path = db_path + '\\' + 'MINUTE_DATA.p'
			self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML = copy.deepcopy(file_path)
			
		except Exception as e:
			print('error in func_STOCK_DICTIONARY_PICKLE__min_data_dump_for_ML :: ', e)
		
	
	
	def func_STOCK_DICTIONARY_PICKLE__article_location(self):
		# article pickle 주소 확인하고 가져오는 부분
		# //pickle//pickle.p
		python_path = os.getcwd()
		db_path = str(python_path + '\\CRAWLER__pickle').replace('/', '\\')
		file_path = db_path + '\\pickle.p'
		self.STOCK_PICKLE__path_for_article = copy.deepcopy(file_path) # 어차피 고정
		
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
			if os.path.isfile(file_path):
				print('article file existance confirmed')
				#self.STOCK_PICKLE__path_for_article = copy.deepcopy(file_path)
			else:
				print("article file doesn' exist")
				self.CHECK_2_FLAG__news_article_pickle_ready = False # 없는 거니깐
		else:
			# creation 건너뜀, 여기서 항 일은 아니고 news crawler 에서 수행시
			self.CHECK_2_FLAG__news_article_pickle_ready = False # 없는 거니깐
			

	def func_BALANCE_PICKLE__create_balance_update_date(self):
		# balance 조회시 update 부분 : 첫 시작 시 FUNC_CHECK_BALANCE__normal 돌아가면 기록
		python_path = os.getcwd()
		db_path = str(python_path + '\\KIWOOM_API__DATABASE_pickle_balance_date').replace('/', '\\')
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path) # 경로 생성
		file_path = db_path + '\\' + 'PICKLE_BALANCE.p'
		self.BALANCE_PICKLE__path_for_balance_update_date = file_path


	def func_SQLITE_PICKLE__create_db_update_date(self): # db 업데이트 시점 pickle file path 작업
		python_path = os.getcwd()
		db_path = str(python_path + '\\KIWOOM_API__DATABASE_pickle_db_date').replace('/', '\\')
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path) # 경로 생성

		file_path = db_path + '\\' + 'PICKLE_DB.p'
		self.SQLITE_PICKLE__path_for_db_update_date = file_path



	def func_SQLITE_LIST__for_path_to_db(self): # 모든 db 주소 가져오는 함수
		python_path = os.getcwd()
		db_path = str( python_path + '\\KIWOOM_API__DATABASE' ).replace('/','\\')
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path)  # 경로 생성
		file_path = db_path + '\\' + 'SINGLE_DB.db'
		file_path = copy.deepcopy(file_path.strip()).replace('\\','\\\\')
		file_path = re.escape(file_path)
		self.SQLITE_LIST__folder_sub_file_path.append(db_path + '\\' + 'SINGLE_DB.db')


		print('func_SQLITE_LIST__for_path_to_db - Process Done ...@!!')
		#print(self.SQLITE_LIST__folder_sub_file_path)

	def func_STOCK_PICKLE__for_create_additional_info(self): # 모든 db 주소 가져오는 함수
		python_path = os.getcwd()
		db_path = str( python_path + '\\KIWOOM_API__ADDITIONAL_INFO' ).replace('/','\\')
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path)  # 경로 생성
		file_path = db_path + '\\' + 'PICKLE_ADDITIONAL.p'

		self.STOCK_PICKLE__path_for_additional_info = file_path


		print('func_SQLITE_LIST__for_path_to_db - Process Done ...@!!')
		#print(self.SQLITE_LIST__folder_sub_file_path)

	def QTIMER__periodic_worker(self):
		"""
		실제 CHECK_1 , CHECK_2, CHECK_3 단계 잡일 수행하는 부분
		:return:
		"""
		print('do periodc work [100ms] active #$#')

		try:

			if self.STOCK_IN_ATTENTION.state == "WAKEUP":
				if self.CHECK_1_FLAG__ALL_login == False:
					print('periodic_work_wrapper - 3 path')
					self.FUNC__restart_api()

			# @ 자동 데이터 만드는 부분 : 현재 2개
			if self.STOCK_IN_ATTENTION.state == "CHECK_1":
				if self.STOCK_FLAG__additional_info_creation_in_progress == True : # 추가 information 가져오는 부분
					if self.STOCK_FLAG__additional_info_is_stalled == True :
						print('re enter creating additional info process by signal...')
						self.FUNC_PYQT__rest_timer(3)  # 시간초 대기
						self.SIGNAL_MINE.addtional_info()
				if self.SQLITE_FLAG__database_creation_in_progress == True: # 자동 데이터베이스 만드는 중임 표기
					if self.SQLITE_FLAG__database_is_stalled == True:
						print('re enter creating database process by signal...')
						self.FUNC_PYQT__rest_timer(3) # 시간초 대기
						self.SIGNAL_MINE.database()

				else: # 처음 데이터 베이스 전체 가져오기 가동
					pass


			if self.STOCK_IN_ATTENTION.state == "CHECK_1": # 기본 로그인 성공하고 자동투자 준비단계로 나가는 부분
				if self.CHECK_2_FLAG_BALANCE__get_success == False:
					print('periodic_work_wrapper - 1 path')
					print('self.ACCOUNT__code_of_my_account ::', self.ACCOUNT__code_of_my_account)
					self.FUNC_CHECK_BALANCE__normal()
				if self.CHECK_2_FLAG_STOCK__get_all_stock_codes == False:
					print('periodic_work_wrapper - 2 path')
					self.FUNC_STOCK_DATABASE__get_all_codes_api()
				if self.CHECK_2_FLAG_STOCK__owning_stock_get_success == False:
					print('periodic_work_wrapper - 4 path')
					self.FUNC_CHECK_STOCK__owning()
					self.func_CHECK_STOCK_DISP__owning() # 보유 주식 disp
				if self.CHECK_2_FLAG_STOCK__unmet_order_success == False:
					print('periodic_work_wrapper - 5 path')
					self.FUCN_CHECK_STOCK__unmet_order()
					self.func_CHECK_STOCK_DISP__unmet() # 미체결 주식 disp

				if self.CHECK_2_FLAG__news_article_pickle_ready == False:
					
					if self.TEST == False :
						print('periodic_work_wrapper - 6 path')
						self.FUNC_CHECK_ARTICLE__get()
						#self.func_CHECK_ARTICLE__date_validation() # 위에것이 선행
					else:
						self.CHECK_2_FLAG__news_article_pickle_ready = True

				if self.CHECK_2_FLAG_STOCK__basic_info == False:
					print('periodic_work_wrapper - 7 path')
					self.FUNC_CHECK_STOCK__basic_info()

				if self.CHECK_2_FLAG_STOCK__additional_info_tr == False:
					print('periodic_work_wrapper - 8 path')
					#self.FUNC_CHECK_STOCK__additional_info_tr()
					self.CHECK_2_FLAG_STOCK__additional_info_tr = True

				# 반드시 마지막
				elif self.CHECK_2_FLAG_SQLITE__first_database_create_success == False:
					print('periodic_work_wrapper - final path')
					# @ reset minute data dictionary
					# proc_1 = mp.Process(target = self.FUNC_STOCK_DATABASE_SQLITE__create)
					# proc_1.start()
					# proc_1.join()
					self.FUNC_STOCK_DATABASE_SQLITE__create()


			# @
			if self.STOCK_IN_ATTENTION.state == "CHECK_2": # 자동 트레이딩 준비 상태
				
				
				
				if self.COUNTER_GLOBAL % (30) == 0: # 30초에 한번
					self.FUNC_CHECK_ARTICLE__get() # 기사 pickle 읽어옴 ->  항상 해야 됨
				
				# if self.COUNTER_GLOBAL % (60*10) == 0: # 10분에 한번
				# 	self.func_CHECK_ARTICLE__date_validation()

				if self.COUNTER_GLOBAL % (60 * 60 * 1) == 0: # 카운터 업데이트 보다가 1시간마다 업데이트 해줌
					self.FUNC_STOCK_DATABASE__get_all_codes_api()

				if self.COUNTER_GLOBAL % (60 * 60 * 4) == 0: # 4 시간 마다 잔고 update
					self.FUNC_CHECK_BALANCE__normal()


				# @ 1시간 마다 database 게팅 reset
				# 개장 1시간 반 전까지만 수행함
				if (self.COUNTER_GLOBAL % (60 * 60 * 1) == 0) :
					if self.STATE_TIME.weekday_num != 5 and self.STATE_TIME.weekday_num != 6:
						# 전체 자동 트레이딩 enable 되고, 데이터 베이스 처음 가져오기 on 일때, 주말 아니면
						if (self.STATE_TIME.stage == "개장전" and (self.STATE_TIME.func_time_now_to_sec() < (self.STATE_TIME.timesec__9 - (60*1.5))) ) or self.STATE_TIME.stage == "개장후":
							self.SIGNAL_MINE.database()
						else:
							pass
					else:  # 주말인 경우
						self.SIGNAL_MINE.database()

				# @ 주식 기본정보 다시 가져오는 부분 , 2시간 반마다, 개장 3시간 전까지만 수행
				if(self.COUNTER_GLOBAL % (60 * 60 * 2.5) == 0) :
					if self.STATE_TIME.weekday_num != 5 and self.STATE_TIME.weekday_num != 6:
						if (self.STATE_TIME.stage == "개장전" and (self.STATE_TIME.func_time_now_to_sec() < (self.STATE_TIME.timesec__9 - (60*3)))) or self.STATE_TIME.stage == "개장후":
							self.SIGNAL_MINE.addtional_info()
						elif (self.STATE_TIME.stage == "개장전" and (self.STATE_TIME.func_time_now_to_sec() >= (self.STATE_TIME.timesec__9 - (60*3)))) : # 개장 전인데, 시간이 개장전 3시간 이전일 때
							try:
								if os.path.isfile(self.STOCK_PICKLE__path_for_additional_info):  # 존재하면
									with open(self.STOCK_PICKLE__path_for_additional_info, 'rb') as file:
										self.STOCK_DICTIONARY_NAMES__additional_info_tr = copy.deepcopy(pickle.load(file))
							except Exception as e:
								print('error in pre-loading additional info..')

					else: # 주말
						self.SIGNAL_MINE.addtional_info()

				if self.STOCK_FLAG__when_unmet_order_made == True: # 준비상태에서 test될 때도 확인하려고, check3상태에서도 가능하도록
					self.FUCN_CHECK_STOCK__unmet_order() # 미수체결 확인 -> 어쨋든 부름 (거의 매번 불릴 때 최신이다.)
					self.FUNC_CHECK_BALANCE__with_order() # 잔고 재 확인
					self.FUNC_CHECK_STOCK__owning() # 보유 종목 재 확인
					self.STOCK_FLAG__when_unmet_order_made = False

				if(self.COUNTER_GLOBAL % (60) == 0) : # 1분에 한번씩 업데이트
					if self.line_test_text_input.text() in self.STOCK_DICTIONARY__name_to_code:
						tmp_stock_code = self.STOCK_DICTIONARY__name_to_code[ self.line_test_text_input.text()]
						self.FUNC_STOCK__draw_daily_graph(tmp_stock_code) # 그래프 그리기
					self.func_CHECK_STOCK_DISP__owning() # 보유 주식 disp
				
				if self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE == True : # 첫 할당 아닐 때
					self.AT_INIT_EVERYTHING() # init 해줌
					self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE = False
					self.AT_FLAG__very_first_init_func_called = False

				if self.AT_FLAG__very_first_init_func_called == False:
					self.AT_INIT_EVERYTHING()  # init 해줌
				
				
				else: # 할당 안된 상태는 pass -> filter stage에서 할당할 것이므로
					pass # 
			
			if self.STOCK_IN_ATTENTION.state == "FILTER":
			
				if self.STATE_TIME.weekday_num != 5 and self.STATE_TIME.weekday_num != 6: # 주중
					# 개장 1시간 전에 수행 한다. 대략 8시 즈음?!
					#self.AT_STOCK_CLASS__wrapper = at.Stock_wrapper(self.STOCK_TARGET_PROFIT) # 1차 AT 부분 선언

					# TEST
					if self.TEST == True:
						if self.CHECK_FILTER_FLAG__at_initialize == False:
							self.AT_INIT_EVERYTHING()  # init 해줌
							self.AT_FUNC_PACKAGE__wake_up()
					else:
						# REAL
						
						if self.AT_FLAG__very_first_init_func_called == False and (self.STATE_TIME.stage == "개장전" or self.STATE_TIME.stage == "개장중"): # 아직 첫 init function 안불리면
							self.AT_INIT_EVERYTHING() # init 해줌
						
						if (self.STATE_TIME.stage == "개장전" and (self.STATE_TIME.func_time_now_to_sec() < (self.STATE_TIME.timesec__9 - (60*1)))):
							if self.CHECK_FILTER_FLAG__at_initialize == False:
								self.AT_FUNC_PACKAGE__wake_up() # 안에서 self.CHECK_FILTER_FLAG__at_initialize True 올림
						
						elif (self.STATE_TIME.stage == "개장전" and (self.STATE_TIME.func_time_now_to_sec() > (self.STATE_TIME.timesec__9 - (60*1))) and (self.CHECK_FILTER_FLAG__at_initialize == False )):
							self.AT_FUNC_PACKAGE__wake_up()
						
						elif self.STATE_TIME.stage == "개장중" and (self.CHECK_FILTER_FLAG__at_initialize == False ): # initialize 의미 0..
							pass # trade 안하고 켜져있기만 한 상태...
							
						
						else:
							pass
					
					
						# ★★★★★★
						# ★★★★★★
						# ★★★★★★
						# >>>
						# 여기 나중에 시간 끝나면 FILTER에서 빠져나오도록 구현해야됨
						# worker 말고 stage 관리 QTimer에서 bool로 시간으로 가져와서 stage넘어가는거 구현

				self.FUNC_PYQT__rest_timer(0.05)

				if 1: # 주가 예측 + 트레이팅 수 들고 오는 부분
					pass # FUNCTION_PACKAGE로 수행

			if self.STOCK_IN_ATTENTION.state == "CHECK_3": # 자동 트레이딩 상태
				
				if self.ERROR_FLAG__sell_all_err_critical_check_3 == True:
					#####################
					# call function to sell everything
					#####################
					self.FUNC_STOCK__handle_sell_every_stock() # 전부 매도
					self.FUNC_CHECK_BALANCE__with_order() # 잔고 재 확인
					self.FUNC_CHECK_STOCK__owning() # 보유 종목 재 확인
					self.ERROR_FLAG__sell_all_err_critical_check_3 = False
				
				self.FUNC_STOCK_BE__update_real_time_for_FE() # 자동으로 BE 단 realtime 데이터 가지고 옴.. 복사

				try: # BE sec realtime information reset
					if(self.COUNTER_GLOBAL % (60 * 2) == 0 ): # 2분마다 reset
						if self.KIWOOM.flag_latest_stock_realtime_data == False: # protection
							print('RESET in real time data in BE begin...')
							self.KIWOOM.latest_stock_realtime_data = copy.deepcopy({}) # reset done
							print('RESET in real time data in BE has been done...')

					if (self.COUNTER_GLOBAL % (60 * 3) == 0):  # 2분마다 reset
						print('FORCED RESET in real time data in BE begin...')
						self.KIWOOM.latest_stock_realtime_data = copy.deepcopy({})  # forced reset done!
						print('FORCED RESET in real time data in BE has been done...')
				except:
					pass
				
				if self.TEST == True:
					pass
				else:
					if self.COUNTER_GLOBAL % (30) == 0: # 30초에 한번
						self.FUNC_CHECK_ARTICLE__get() # 기사 pickle 읽어옴 ->  항상 해야 됨
					
					# if self.COUNTER_GLOBAL % (60*10) == 0: # 10분에 한번
					# 	self.func_CHECK_ARTICLE__date_validation() # 최신 데이터인지 판단!

				# @
				"""
				여기서 buy sell 수행 AUTO backend에서 수행 - 편의 위해 구분해놓은 부분
				"""
				if self.KIWOOM.flag_sudden_send_order_message == True: # 갑자기 들어온 send order 리턴 message있으면
					self.KIWOOM.flag_sudden_send_order_message = False # BE reset, taken care
					self.FUCN_CHECK_STOCK__unmet_order() # 미수체결 확인 -> 어쨋든 부름 (거의 매번 불릴 때 최신이다.)
					self.FUNC_CHECK_BALANCE__with_order() # 잔고 재 확인
					self.FUNC_CHECK_STOCK__owning() # 보유 종목 재 확인


				if self.STOCK_FLAG__when_unmet_order_made == True: # 준비상태에서 test될 때도 확인하려고, check3상태에서도 가능하도록
					self.FUCN_CHECK_STOCK__unmet_order() # 미수체결 확인 -> 어쨋든 부름 (거의 매번 불릴 때 최신이다.)
					self.FUNC_STOCK__handle_unmet_order() # 미수체결 제거
					self.FUNC_CHECK_BALANCE__with_order() # 잔고 재 확인
					self.FUNC_CHECK_STOCK__owning() # 보유 종목 재 확인
					self.STOCK_FLAG__when_unmet_order_made = False

				if(self.COUNTER_GLOBAL % (60) == 0) : # 1분에 한번씩 업데이트
					if self.line_test_text_input.text() in self.STOCK_DICTIONARY__name_to_code:
						tmp_stock_code = self.STOCK_DICTIONARY__name_to_code[self.line_test_text_input.text()]
						self.FUNC_STOCK__draw_daily_graph(tmp_stock_code) # 그래프 그리기
					self.func_CHECK_STOCK_DISP__owning() # 보유 주식 disp

				if self.COUNTER_GLOBAL % (58) == 0:  # 58초마다 주문 list 작업 수행
					self.AT_FUNC_PACKAGE__on_the_run_PERIODIC()
				
				else:
					if self.COUNTER_GLOBAL % (60*60) == 0: # 1시간 마다
						try:
							gc.collect()
						except Exception as e:
							print('error in gc.collect in perodic worker Stage check 3')
					pass # 1분동안 사이에 다른 AT 단 periodic 작업 수행


			if self.STOCK_IN_ATTENTION.state == "ERROR_SELL_ALL":
				if self.CHECK_SELL_ALL_FLAG__restart_api == True:
					self.FUNC__restart_api() # 재시작
					self.CHECK_SELL_ALL_FLAG__restart_api = False
				pass
				# 1) 32bit에 큐를 보내지 않고, sell 동작을 하는 것으로!
				# 2) self.FUNC__restart_api() 로 다시시작
				# error sell all에서 wakeup으로 돌아가는거 짜야됨

			else:
				pass

				# -----------------------------------------------------------------------------------
				# -----------------------------------------------------------------------------------

		except Exception as e:
			print('error in periodic num count - ',e)
			traceback.print_exc()

	def QTIMER__periodic_state_checker_1s(self):
		"""
		1)queue 두개에다가, article database를 위한 전체 종목 output 생성해야됨
		  queue 는 기본적으로 api, 32bit, 64bit 공용 class로 만들어야됨
		2)error return value 모두 체크
		3)news article pickle database 읽는 과정 포함
		:return:
		"""
		print('*' * 40)
		print('        -period process [1sec] now time stamp : ', datetime.datetime.now())
		print('*' * 40)

		try:
			# @ 요청 갯수 표기
			self.statusbar_req_num.showMessage(str(self.ERROR_COUNTER_BE__request_num) + '개 의 BE 요청 갯수')

			# @ 카운터 올리기 + reset
			if self.COUNTER_GLOBAL >= (23 * 60 * 60):
				self.COUNTER_GLOBAL = 1
			else:
				self.COUNTER_GLOBAL = self.COUNTER_GLOBAL + 1

			self.FUNC_CHECK__ki_connect_state() # 연결상태 체크
			self.FUNC_CHECK_DISP__all_login_process()
			self.FUNC_CHECK_ERROR__all()
			self.STATE_TIME.func_today_stage()
			self.STATE_TIME.func_week_num_stage() # 주중 계산, 월 : 0 ~ 일 : 6
			

			# @ ADDITIONAL CALCULATION
			# 1) 시간
			if self.STATE_TIME.func_today_stage() == "개장중" and self.STATE_TIME.weekday_num not in [5,6]: # 시간 여부에 따라 state 움직임 추가
				tmp_flag_time_for_check_3 = True
			else:
				tmp_flag_time_for_check_3 = False
			if self.TEST == True:
				tmp_flag_time_for_check_3 = True # 지금은 test 위해서 그냥 박아놓음

			# 2) error

			if self.ERROR_DICTIONARY__backend_and_critical['error_critical'] > 0 and self.CHECK_SELL_ALL_FLAG__restart_api == False: # 1개 이상이라도 나오면
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = 0 # reset
				self.CHECK_SELL_ALL_FLAG__restart_api = True # worker에서 init으로 씀
				self.ERROR_FLAG__sell_all_err_critical_check_3 = True # 이것 있으면 전부 sell
			else:
				"""
				non - critical back_end 단 error logic
				"""
				if self.STATE_TIME.func_today_stage() == "개장중":
					pass
			tmp_critical_error_rest = False # 자동 트레이딩 중일 때랑 아니랑 구분
			if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL":
				if self.ERROR_DICTIONARY__backend_and_critical['error_critical'] > 0: # 1개 이상이라도 나오면
					self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = 0 # reset
					tmp_critical_error_rest = True
			
			# 3) filter stage 시간
			tmp_flag_time_for_filter = False
			if self.STATE_TIME.func_today_stage() == "개장전" or self.STATE_TIME.func_today_stage() == "개장중":
				tmp_flag_time_for_filter = True

			#===========================================================================
			"""
			해당 단계로 넘어가기위해서 만족해야 하는 부분
			"""
			tmp_list_check_1 = [self.CHECK_1_LOGICAL__windows_1st_login,
								self.CHECK_1_LOGICAL__ki_connect_state,
								self.CHECK_1_FLAG__ALL_login, not(tmp_critical_error_rest)] # wakeup to data creation able

			tmp_list_check_2 = [self.CHECK_2_FLAG_STOCK__get_all_stock_codes,
								self.CHECK_2_FLAG_STOCK__owning_stock_get_success,
								self.CHECK_2_FLAG_STOCK__unmet_order_success,
								self.CHECK_2_FLAG__news_article_pickle_ready,
								self.CHECK_2_FLAG_SQLITE__first_database_create_success,
								self.CHECK_2_FLAG_STOCK__basic_info,
								self.CHECK_2_FLAG_STOCK__additional_info_tr] # data creation able to autotrade able

			
			if self.TEST == False:
				# real
				tmp_list_filter  = [self.AT_FLAG__very_first_init_func_called, tmp_flag_time_for_filter]
			else:
				# test
				tmp_list_filter = [self.AT_FLAG__very_first_init_func_called, True]
			
			tmp_list_check_3 = [self.CHECK_FILTER_FLAG__at_initialize,
								self.CHECK_3_FLAG__ALTIMATE_AUTO_ON,
								self.CHECK_2_FLAG__news_article_pickle_ready,
								tmp_flag_time_for_check_3] # auto trade mode

			tmp_list_check_error_sell = [self.CHECK_SELL_ALL_FLAG__restart_api]

			# @ Stage 계산부분
			self.STATE_MACHINE.FUNC__main(tmp_list_check_1, tmp_list_check_2, tmp_list_filter, tmp_list_check_3, tmp_list_check_error_sell)
			self.STOCK_IN_ATTENTION.state = self.STATE_MACHINE.state
			print('self.STOCK_IN_ATTENTION.state :: ', self.STOCK_IN_ATTENTION.state)


			# @ STAGE 관련 label update
			if  self.STOCK_IN_ATTENTION.state == "WAKEUP":
				self.label_auto_enabled_check.setText('로그인 수행')
			elif self.STOCK_IN_ATTENTION.state == "CHECK_2":
				self.label_auto_enabled_check.setText('자동투자 - 준비')
			elif self.STOCK_IN_ATTENTION.state == "FILTER":
				self.label_auto_enabled_check.setText('자동투자 - 종목선정')				
			elif self.STOCK_IN_ATTENTION.state == "CHECK_3":
				self.label_auto_enabled_check.setText('자동투자 - 시작')
			elif self.STOCK_IN_ATTENTION.state == "ERROR_SELL_ALL":
				self.label_auto_enabled_check.setText('자동투자 - 전체 매도')
			else:
				self.label_auto_enabled_check.setText('자동투자 - 불가')

		except Exception as e:
			print('periodic process error - ',e)
			traceback.print_exc()


	def FUNC_CHECK_ERROR__all(self): # checking 에러 in every dynamic instance calls

		# @ 1차 backend error 개수 표시
		tmp_text = "백엔드에서의 에러 개수 : " + str(self.ERROR_COUNTER_BE__front_be_counter_previous)
		self.label_error_count.setText(tmp_text)

		# @ error dictionary update
		self.ERROR_DICTIONARY__backend_and_critical['error_backend'] = self.ERROR_COUNTER_BE__front_be_counter_previous

	def FUNC_STOCK_BE__update_real_time_for_FE(self):
		try:
			"""
			sec 단위 : 2분정도
			min 단위 
			두가지로 따로 저장
			
			# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
			# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj
			"""
			# @ second용 작업 먼저
			#--------------------------------------
			# 1) dictionary 비어있는지 확인 + 2) update
			print('FUNC_STOCK_BE__update_real_time_for_FE entered...')
			if self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC : # 비어있는지 확인 -> 안비어있으면!
				"""
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC = self.KIWOOM.latest_stock_realtime_data
				"""
				if self.KIWOOM.latest_stock_realtime_data: # BE 단이 비지 않았으면
					for stock_code in self.KIWOOM.latest_stock_realtime_data: #BE 에서 들고있는 것들에 한해서 FE 작업
						if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC: # FE 단 상태 확인, 들고있는 주식 코드이면
							# for datetime_data in self.KIWOOM.latest_stock_realtime_data[stock_code]: # BE단 real data로
							# 	if datetime_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # FE단에 stock code에 datetime 이미 있는 애면
							# 		pass
							# 	else:
							# 		self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][datetime_data] = copy.deepcopy(self.KIWOOM.latest_stock_realtime_data[stock_code][datetime_data]) # FE단 stock_code로 access 한 날짜 hash update.
							self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code].update(self.KIWOOM.latest_stock_realtime_data[stock_code]) # dictionary 그냥 업데이트
						else: # 없다면
							self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code] = copy.deepcopy(self.KIWOOM.latest_stock_realtime_data[stock_code])
				else: # BE 단이 비어있음
					pass
			else: # FE단 sec데이터 비어있으면!
				if self.KIWOOM.latest_stock_realtime_data:  # BE 단이 비지 않았으면
					self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC = copy.deepcopy(self.KIWOOM.latest_stock_realtime_data) # 그냥 복사
				else: #BE 단이 비어있으면
					pass

			# 2) 관리할 길이로 짜름
			# STOCK_SECOND_DATA_LEN
			tmp_list_of_datetime_to_remove = [] # 제거할 대상 다 넣는다
			if self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC:  # 비어있는지 확인 -> 안비어있으면!
				# @ 제거 대상 찾음
				for stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC: # FE sec data에 대해
					tmp_list_of_datetime_to_remove = [] # reset for next for loop
					for datetime_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # 개별 stock안의 date stamp에 대해
						tmp_datetime_obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S")
						if datetime.datetime.now() - tmp_datetime_obj > datetime.timedelta(seconds=self.STOCK_SECOND_DATA_LEN*60):  # 관심 길이보다 크면
							tmp_list_of_datetime_to_remove.append(datetime_data)
						
						#@ del 관리
						try:
							tmp_datetime_obj = None
							del tmp_datetime_obj
						except Exception as e:
							print('error in del FUNC_STOCK_BE__update_real_time_for_FE (1) : ', e)

					# @ remove 진행
					for datetime_stamp_for_remove in tmp_list_of_datetime_to_remove:
						if datetime_stamp_for_remove in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # 있는 datetime stamp이면
							try:
								self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][datetime_stamp_for_remove] = None
								del self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][datetime_stamp_for_remove] # 삭제과정 수행
							except Exception as e:
								print('error in FUNC_STOCK_BE__update_real_time_for_FE (1) del :: ', e)
						else:
							pass # 위에서 바로 했으므로 unreachable 일 것..
			else: # 아무것도 없으므로 관리할 필요가 없음
				pass

			# @ minute data 업데이트 -> 위에서 사용한 걸루!
			"""
			Volume은 합산해야됨
			"""
			# 볼륨합산! -> 볼륨은 -이면 매수, + 매도  // price는 전일가 보다 낮으면 - 붙음, ABS 내장 함수 써야할 수도? -> receive handler에서 이미 abs로 받아옴 매도/매수 표기 - 알필요 없어서.
			if  self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'] : # 빈 dictionary가 아님!
				for stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC: # sec 데이터에서 min을 업데이트 할 것 이므로
					if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']: # 이미 있는데 업데이트 함
						for datetime_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # sec data에서 가져오는 부분

							# @ 정각 계산용
							tmp_datetime_from_sec__microsec_z__obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S").replace(microsecond=0)
							tmp_datetime_from_sec__sec_z__obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S").replace(second=0, microsecond=0)

							if tmp_datetime_from_sec__microsec_z__obj == tmp_datetime_from_sec__sec_z__obj: # sec 데이터 second정각
								if datetime_data not in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code]: # min 데이터에 없는 경우에
									# @ price 업데이트
									self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_data]['price'] = self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][datetime_data]['price']

									# @ volume 업데이트
									tmp_volume_value__window_1_min = 0 # volume 기록용
									tmp_while_fali_safe = 0
									tmp_datetime_min_target__obj = copy.deepcopy((tmp_datetime_from_sec__microsec_z__obj - datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)) #1분전 datetime_obj
									while tmp_datetime_min_target__obj < tmp_datetime_from_sec__microsec_z__obj and tmp_while_fali_safe <= 60:
										tmp_while_fali_safe = tmp_while_fali_safe + 1 # fail safe counter up
										# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
										# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj
										tmp_datetime_min_target__obj_str = tmp_datetime_min_target__obj.strftime('%Y%m%d%H%M%S')
										if tmp_datetime_min_target__obj_str in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]:
											tmp_volume_value__window_1_min = tmp_volume_value__window_1_min + self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][tmp_datetime_min_target__obj_str]['volume']
										tmp_datetime_min_target__obj = tmp_datetime_min_target__obj + datetime.timedelta(seconds=1)
									else:
										self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_data]['volume'] = tmp_volume_value__window_1_min



								else: # min에 이미 있는 데이터면 작업을 하지 않는다
									pass
							else: # 비 정각시 min에 대해서 작업을 하지 않는다
								pass

							# @ del
							try:
								tmp_datetime_from_sec__microsec_z__obj = None
								tmp_datetime_from_sec__sec_z__obj = None
								tmp_volume_value__window_1_min = None
								tmp_while_fali_safe = None
								tmp_datetime_min_target__obj = None
								tmp_datetime_min_target__obj_str = None

								del tmp_datetime_from_sec__microsec_z__obj
								del tmp_datetime_from_sec__sec_z__obj
								del tmp_volume_value__window_1_min
								del tmp_while_fali_safe
								del tmp_datetime_min_target__obj
								del tmp_datetime_min_target__obj_str

							except Exception as e:
								print('error in FUNC_STOCK_BE__update_real_time_for_FE (2) del :: ', e)

					else: # 없는 stock_code 인데 api 실시간 요청 띄워서 sec 데이터에서 새로 받음
						self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code] = {}
						for datetime_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # sec data에서 가져오는 부분
							# @ 정각 계산용
							tmp_datetime_from_sec__microsec_z__obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S").replace(microsecond=0)
							tmp_datetime_from_sec__sec_z__obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S").replace(second=0, microsecond=0)

							if tmp_datetime_from_sec__microsec_z__obj == tmp_datetime_from_sec__sec_z__obj: # sec 데이터 second정각
								if datetime_data not in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code]: # min 데이터에 없는 경우에
									# @ price 업데이트
									self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_data]['price'] = self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][datetime_data]['price']

									# @ volume 업데이트
									tmp_volume_value__window_1_min = 0 # volume 기록용
									tmp_while_fali_safe = 0
									tmp_datetime_min_target__obj = copy.deepcopy((tmp_datetime_from_sec__microsec_z__obj - datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)) #1분전 datetime_obj
									while tmp_datetime_min_target__obj < tmp_datetime_from_sec__microsec_z__obj and tmp_while_fali_safe <= 60:
										tmp_while_fali_safe = tmp_while_fali_safe + 1 # fail safe counter up
										# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
										# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj
										tmp_datetime_min_target__obj_str = tmp_datetime_min_target__obj.strftime('%Y%m%d%H%M%S')
										if tmp_datetime_min_target__obj_str in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]:
											tmp_volume_value__window_1_min = tmp_volume_value__window_1_min + self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][tmp_datetime_min_target__obj_str]['volume']
										tmp_datetime_min_target__obj = tmp_datetime_min_target__obj + datetime.timedelta(seconds=1)
									else:
										self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_data]['volume'] = tmp_volume_value__window_1_min



								else: # min에 이미 있는 데이터면 작업을 하지 않는다
									pass
							else: # 비 정각시 min에 대해서 작업을 하지 않는다
								pass

							# @ del
							try:
								tmp_datetime_from_sec__microsec_z__obj = None
								tmp_datetime_from_sec__sec_z__obj = None
								tmp_volume_value__window_1_min = None
								tmp_while_fali_safe = None
								tmp_datetime_min_target__obj = None
								tmp_datetime_min_target__obj_str = None

								del tmp_datetime_from_sec__microsec_z__obj
								del tmp_datetime_from_sec__sec_z__obj
								del tmp_volume_value__window_1_min
								del tmp_while_fali_safe
								del tmp_datetime_min_target__obj
								del tmp_datetime_min_target__obj_str

							except Exception as e:
								print('error in FUNC_STOCK_BE__update_real_time_for_FE (3) del :: ', e)

			else:
				# minute data from second data
				"""
				일어나지 않을 것, FUNC_STOCK_DICTIONARY__parse_from_sqlite 에서 이미 수행을 할 것이므로
				"""
				pass # 없어야 정상
						
			
			# @ min data missing point 작업해주는 부분
			#self.func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup(volume_zero_padding = False) -> 이건 64bit에서 담당하는 것으로!

			# @ MIN data 10시간 안쪽으로 관리할 것!
			# self.STOCK_AT_TIME_WINDOW_HOUR 사용
			tmp_list_of_datetime_to_remove_in_min = []
			for stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']:
				tmp_list_of_datetime_to_remove_in_min = []
				tmp_hash_of_datetime = self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code]
				tmp_keys_from_hash = copy.deepcopy(list(tmp_hash_of_datetime.keys()))
				tmp_keys_from_hash.sort()

				# @ 가져온 날짜
				tmp_most_past_hash = tmp_keys_from_hash[0]
				tmp_most_latest_hash = tmp_keys_from_hash[-1]
				tmp_most_past_hash__obj = datetime.datetime.strptime(tmp_most_past_hash, "%Y%m%d%H%M%S")
				tmp_window_from_latest_hash__obj = datetime.datetime.strptime(tmp_most_latest_hash, "%Y%m%d%H%M%S") - datetime.timedelta(hours = self.STOCK_AT_TIME_WINDOW_HOUR)

				tmp_fail_safe = 0
				while tmp_fail_safe <= 60*10 and tmp_most_past_hash__obj < tmp_window_from_latest_hash__obj:
					tmp_fail_safe = tmp_fail_safe + 1
					tmp_list_of_datetime_to_remove_in_min.append(tmp_most_past_hash__obj.strftime('%Y%m%d%H%M%S'))
					tmp_most_past_hash__obj = tmp_most_past_hash__obj + datetime.timedelta(minutes=1)

				else: # end of while loop

					# @ del items in list
					for datetime_to_remove in tmp_list_of_datetime_to_remove_in_min:
						try:
							self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_to_remove] = None
							del self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][datetime_to_remove]
						except Exception as e:
							print('error in FUNC_STOCK_BE__update_real_time_for_FE (4) del :: ', e)
					
					# @ del for while loop
					try:
						tmp_fail_safe = None
						del tmp_fail_safe

					except Exception as e:
						print('error in FUNC_STOCK_BE__update_real_time_for_FE (5) del :: ', e)
			
				# del for forloop
				try:
					tmp_list_of_datetime_to_remove_in_min = None
					tmp_hash_of_datetime = None
					tmp_keys_from_hash = None
					tmp_most_past_hash = None
					tmp_most_latest_hash = None
					tmp_most_past_hash__obj = None
					tmp_window_from_latest_hash__obj = None

					del tmp_list_of_datetime_to_remove_in_min
					del tmp_hash_of_datetime
					del tmp_keys_from_hash
					del tmp_most_past_hash
					del tmp_most_latest_hash
					del tmp_most_past_hash__obj
					del tmp_window_from_latest_hash__obj

				except Exception as e:
					print('error in FUNC_STOCK_BE__update_real_time_for_FE (6) del :: ', e)



		except Exception as e:
			print('error in FUNC_STOCK_BE__update_real_time_for_FE :: ', e)
			traceback.print_exc()

	def func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup(self, volume_zero_padding = False):
		"""
		min data missing point 작업해주는 부분 -> 64비트로 옮기자
		:return:
		"""
		try:
			# @ start cleaning up process
			tmp_list_stock_codes_to_clear = []
			for stock_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']:
				if stock_data not in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']:
					tmp_list_stock_codes_to_clear.append(stock_data)
				else:
					if self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_data] == True:
						pass
					else:
						tmp_list_stock_codes_to_clear.append(stock_data)

			for stock_data in tmp_list_stock_codes_to_clear:
				if stock_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']: # 존재하는 data 확인
					try:
						self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data] = None
						del self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data]
						print('successfully deleted unecessary STOCK_MIN data : ', stock_data)
					except Exception as e:
						print('error in func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup - delete unused stock_code :: ', e, ' stock_code : ', stock_data)
			
			

			# @ 빈 데이터프레임 채워주기
			tmp_counter = 0
			for stock_data in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']:

				if stock_data not in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']: #일어나지 않아야 하는데, boolian 안찾았으면
					continue # next iter로 넘어감
					#return None # 함수종료
				else: # 이 경우로 들어가야 함
					if self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_data] == True: # sqlite로부터 okay
						# TEST
						pass
						print('func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup survived : ', stock_data)
						tmp_counter = tmp_counter + 1


						# REAL
						# https://www.google.com/search?rlz=1C1SQJL_koKR876KR876&q=%ED%8C%8C%EC%9D%B4%EC%8D%AC+del+%EB%A9%94%EB%AA%A8%EB%A6%AC&sa=X&ved=2ahUKEwjMxLmJyM_qAhXEMN4KHbYaBr8Q1QIoAHoECAsQAQ&biw=863&bih=1102
						# https://www.google.com/search?rlz=1C1SQJL_koKR876KR876&biw=863&bih=1102&ei=VRsPX_znKM66mAWJi6XoDw&q=python+del+vs+gc&oq=python+del+gc&gs_lcp=CgZwc3ktYWIQAxgCMgIIADICCAAyBggAEAgQHjoECAAQQzoHCAAQsQMQQzoICAAQsQMQgwE6BQgAELEDOgoIABCxAxCDARBDOgQIABAeUOOpP1jZwz9g2eE_aABwAHgAgAGtAYgB8QySAQQwLjE0mAEAoAEBqgEHZ3dzLXdpeg&sclient=psy-ab
						# https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python
						"""
						
						print('func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup : ', stock_data)
						# 이부분부터 구현하면 됨 --------------------------------------
						#tmp_datetime_stamp_list = copy.deepcopy(list(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data].keys())).sort() # 소팅으로 뒤쪽 list일 수록, 최신 데이터임
						tmp_hash = copy.deepcopy(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data])
						tmp_datetime_stamp_list = copy.deepcopy(list(tmp_hash.keys()))
						tmp_datetime_stamp_list.sort()# 소팅으로 뒤쪽 list일 수록, 최신 데이터임
						tmp_start_datetime_stamp = tmp_datetime_stamp_list[0] #첫 데이터 
						tmp_end_datetime_stamp = tmp_datetime_stamp_list[-1] #마지막 데이터
						tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)
						tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(second=0,microsecond=0)
						#before_price = None
						#before_volume = None
						if tmp_start_datetime_stamp_obj < tmp_end_datetime_stamp_obj:
							while tmp_start_datetime_stamp_obj < tmp_end_datetime_stamp_obj : # datetime obj끼리 비교 while 문이라 위험??
								# @ 처음은 list에서 뽑아왔으므로 있다
								tmp_start_datetime_stamp_obj_convert = tmp_start_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
								if tmp_start_datetime_stamp_obj_convert in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data]:
									before_price = copy.deepcopy(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data][tmp_start_datetime_stamp_obj_convert]['price'])
									before_volume =  copy.deepcopy(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data][tmp_start_datetime_stamp_obj_convert]['volume'])
								else: # 없는 timestamp이면
									if volume_zero_padding == True:
										before_volume = 0
									self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_data][tmp_start_datetime_stamp_obj_convert] = copy.deepcopy({'price':before_price, 'volume':before_volume}) # 빈 곳에 값 할당

								# @ time stamp start 다음 min으로 옮김
								tmp_start_datetime_stamp_obj = tmp_start_datetime_stamp_obj + datetime.timedelta(minutes=1)

								# @ while loop gc/del
								try:
									tmp_start_datetime_stamp_obj_convert = None

									del tmp_start_datetime_stamp_obj_convert

								except Exception as e:
									print('error in del method in func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup - while loop : ', e)
						
						else:
							pass
						"""
					else: # sqdata 판별로 거르는 부분 False일시
						#tmp_list_stock_codes_to_clear.append(stock_data)
						continue #함수종료 -> sqlite에서 거래 정지 등의 사유.. data 모자라거나?
				
				# @ forloop gc/del
				try:
					tmp_hash = None
					tmp_datetime_stamp_list = None
					tmp_start_datetime_stamp = None
					tmp_end_datetime_stamp = None
					tmp_start_datetime_stamp_obj = None
					tmp_end_datetime_stamp_obj = None

					before_price = None
					before_volume = None
					
					del tmp_hash
					del tmp_datetime_stamp_list
					del tmp_start_datetime_stamp
					del tmp_end_datetime_stamp
					del tmp_start_datetime_stamp_obj
					del tmp_end_datetime_stamp_obj

					del before_price, before_volume

				except Exception as e:
					print('error in del method in func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup - for loop : ', e)

			# @ 10시간안으로 관리 -> 30분, 20배 차이

			print('total length of filtered by sqlite database : ', tmp_counter)

		except Exception as e:
			print('error in func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup :: ', e)
			traceback.print_exc()

	
	def FUNC_STOCK_DICTIONARY__parse_from_sqlite(self, string_val='multi'): # multi or single
		"""
		sq lite db에서 가져오는 부분, 
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'] 첫 생성 부분
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_data] -> 에 bool로 기록하면서 넘어간다
		
		multi : self.STOCK_IN_ATTENTION.stock_list_for_sqlite 로 함수 돌림
		single : self.STOCK_IN_ATTENTION.code 로 함수 돌림
		"""
		try:

			print('entering FUNC_STOCK_DICTIONARY__parse_from_sqlite...')

			try:
				# @ SQLITE connection 세팅
				self.SQLITE__con_top = sqlite3.connect(self.SQLITE_LIST__folder_sub_file_path[0])
			except Exception as e:
				print('self.SQLITE__con_top error in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)
			
			# @ 작업할 list 세팅
			if string_val == "multi":
				list_to_parse_to_dict = copy.deepcopy(self.STOCK_IN_ATTENTION.stock_list_for_sqlite)
			elif string_val == "single":
				list_to_parse_to_dict = [self.STOCK_IN_ATTENTION.code]
			else:
				raise ValueError()
			
			# @ list에서 parsing, 하나의 stock에 대해서 수행
			#print('list_to_parse_to_dict :: ', list_to_parse_to_dict)
			for stock_code in list_to_parse_to_dict:
				# @ SQLITE 검색 부분
				print(stock_code)
				head_string = 'SELECT * FROM '
				tmp_table_name_sql = "'" + str(stock_code) + "'"
				#df = pd.read_sql(head_string + tmp_table_name_sql, self.SQLITE__con_top, index_col=None)  # string 자체는 받아낼 수 있는듯
				#tmp_df = df.copy()  # 변하는 것 같아서!!
				tmp_df = pd.read_sql(head_string + tmp_table_name_sql, self.SQLITE__con_top, index_col=None)  # string 자체는 받아낼 수 있는듯

				# try:
				# 	df = None
				# 	del df
				# 	print('successful del of df in FUNC_STOCK_DICTIONARY__parse_from_sqlite')
				# except Exception as e:
				# 	print('error in del of df in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)
				
				# @ 주가 + 거래량 dictionary 담는 부분
				#df_date_list = copy.deepcopy(tmp_df['date'].tolist())
				# date, open close, high, low, volume -> dataframe 에서는 0번 row가 최신 데이터임 시간순으로..!
				# https://medium.com/swlh/how-to-efficiently-loop-through-pandas-dataframe-660e4660125d
				#---------------------------------------------- 
				# dictionary에 담을지 판단하는 부분
				#---------------------------------------------- 
				

				# @ raw 데이터에서 확인해서 기록할 부분
				"""
				900 분봉 : iter 1번해서 가져오기 때문,
				95퍼 이상 데이터 존재하면 pass로 할것임
				"""
				# logical 하게 판단할 부분
				# 1) 빈 데이터 프레임 확인
				bool_1 = (len(tmp_df) >= 900 * 0.95) and (len(tmp_df) != 0) # 비지 않고 900*0.9 이상


				# 2) 최신 데이터 여부 확인
				bool_2 = False
				if  not ((self.STATE_TIME.stage == "개장중" and (self.STATE_TIME.weekday_num != 5 or self.STATE_TIME.weekday_num != 6) ) and not tmp_df.empty ):
					"""
					전일 개장후 ~ 금일 개장전에 불리는 함수이므로 
					"""
					today_date = datetime.datetime.now()
					flag_change_date = None
					#date_list = copy.deepcopy(list(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code].keys()))
					date_list = tmp_df['date'].tolist()
					#print('FUNC_STOCK_DICTIONARY__parse_from_sqlite - date_list :: ', date_list)
					#date데이터
					if len(date_list) != 0: # 비어있지 않으면
						date_list.sort()
						convert_date = datetime.datetime.strptime(date_list[-1], "%Y%m%d%H%M%S") #최신 날짜 convert
						
						if self.STATE_TIME.weekday_num == 5: # 토요일
							today_date = today_date - datetime.timedelta(days=1)
							today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0)
							flag_change_date = True
						elif self.STATE_TIME.weekday_num == 6: # 일요일
							today_date = today_date - datetime.timedelta(days=2)
							today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0)
							flag_change_date = True
						
						else: # 다른 주중 요일
							pass
						# ->  월 00:00 ~ 금 24:00 까지 재조정, 토/일-> 금 15:30
						today_15_30 = self.STATE_TIME.timesec__15_30
						today_24    = self.STATE_TIME.timesec__24
						tomorrow_9 =  self.STATE_TIME.timesec__9
						today_sec = (((today_date.hour) * 60) * 60) + (today_date.minute * 60) + today_date.second
						
						if today_sec < tomorrow_9 : # 개장전
							if self.STATE_TIME.weekday_num == 0: # 월요일 -> 전주 금요일 3시 30분 이상에 포함되는 log있으면 됨
								tmp_today_date = (today_date - datetime.timedelta(days=3)).replace(hour= 15, minute= 30, second=0, microsecond=0) # 금요일
								if convert_date >= tmp_today_date:
									bool_2 = True
							else: # 주중 월요일 아닌 다른 요일
								if flag_change_date == True: # 토 / 일은 금요일로 set되었는데 다시 금요일에서 -1 할 수 없음
									back_date = 0
								else:
									back_date = 1
									tmp_today_date = (today_date - datetime.timedelta(days=back_date)).replace(hour=15,minute=30,second=0,microsecond=0)  # 하루전
									if convert_date >= tmp_today_date:
										bool_2 = True
						elif (today_sec > today_15_30 and today_sec < today_24) or (flag_change_date == True):
							# 개장후(토/일은 금요일 15:30으로 변경), 바뀌면 금요일 장이후 취급
							tmp_today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0) # 금요일
							if convert_date >= tmp_today_date:
								bool_2 = True
					else: # 빈 데이터 
						bool_2 = False


				# @ 3) 가격
				bool_3 = True
				if not tmp_df.empty:
					for row_tuple in tmp_df.itertuples(): 
						if row_tuple.open < 5000:
							bool_3 = False
							break


				bool_4 = True
				tmp_counter = 0
				tmp_volume_sum = 0
				if not tmp_df.empty : # 빈 데이터프레임 아니면
					for row_tuple in tmp_df.itertuples(): 
						tmp_counter = tmp_counter + 1
						tmp_volume_sum = row_tuple.volume + tmp_volume_sum

					if tmp_counter == 0: # avoid division by zero
						bool_4 = False
					else:
						if tmp_volume_sum / tmp_counter < 500:
							bool_4 = False

				else:
					bool_4 = False


				bool_5 = True
				if self.STOCK_DICTIONARY__code_to_name[stock_code] in self.STOCK_DICTIONARY_NAMES__basic_info:
					bool_5 = self.STOCK_DICTIONARY_NAMES__basic_info[self.STOCK_DICTIONARY__code_to_name[stock_code]]['result']
				else:
					bool_5 = False				

				bool_6 = True
				# if self.STOCK_DICTIONARY__code_to_name[stock_code] in self.STOCK_DICTIONARY_NAMES__additional_info_tr:
				# 	tmp_tr_number= self.STOCK_DICTIONARY_NAMES__additional_info_tr[self.STOCK_DICTIONARY__code_to_name[stock_code]]['tr_number']
				# 	tmp_tr_number_compare= self.STOCK_DICTIONARY_NAMES__additional_info_tr[self.STOCK_DICTIONARY__code_to_name[stock_code]]['tr_number_compare']
				# 	if tmp_tr_number_compare >= - tmp_tr_number * 0.5 and tmp_tr_number != '' and tmp_tr_number_compare != '': # 전일 대비 거래량 -50 퍼 이상 거래됨 -2: 600중반정도
				# 		bool_6 = True
				# 	else:
				# 		bool_6 = False
						

				# else:
				# 	bool_6 = False

				# ------------------------------------------------------------------------------
				# 계산 부분
				if bool(bool_1 * bool_2 * bool_3 * bool_4 * bool_5 * bool_6) or stock_code in self.MUST_WATCH_LIST :
					self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code] = True
					self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code] = { }

					for row_tuple in tmp_df.itertuples():  
						# @ 실제 넣는 부분
						self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code][row_tuple.date] = {'price' : float(row_tuple.open) , 'volume':float(row_tuple.volume)}

				else:
					self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code] = False


				#@ tmp_df del
				try:
					tmp_df = None
					del tmp_df
					print('successful del of tmp_df in FUNC_STOCK_DICTIONARY__parse_from_sqlite')
				except Exception as e:
					print('error in del of FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)

				try:
					bool_1 = None
					bool_2 = None
					bool_3 = None
					bool_4 = None
					bool_5 = None
					bool_6 = None
					dates = None

					del bool_1, bool_2, bool_3, bool_4, bool_5, bool_6, dates

				except Exception as e:
					print('error in del of series of bools in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)

				
				try:
					row_tuple = None
					del row_tuple
					print('successful del of row_tuple in FUNC_STOCK_DICTIONARY__parse_from_sqlite')
				except Exception as e:
					print('error in del row_tuple in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)


			try:
				# @ close SQLITE
				self.SQLITE__con_top.close()
				self.SQLITE__con_top = None
				del self.SQLITE__con_top
			except Exception as e:
				print('self.SQLITE__con_top closing error in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)
			
		except Exception as e:
			print('error in FUNC_STOCK_DICTIONARY__parse_from_sqlite :: ', e)
			traceback.print_exc()
		
		


	def FUNC_CHECK_ARTICLE__get(self): # pickle data를 news에서 읽어온다.
		#self.CHECK_2_FLAG__news_article_pickle_ready = True
		pass
		"""
		if -> 읽을 수 있으면 + 데이터베이스 시간 체크 : self.CHECK_2_FLAG__news_article_pickle_ready = True
		   -> 없으면 : self.CHECK_2_FLAG__news_article_pickle_ready = False 로 한다. (데이터베이스에 시간항목 추가로...?!)
		   {stock_name : [stock_num, {article_address : [[article_time, article_score, article_sentence] , [...] , ....,  ... ]} ] }
		   
		   2020-04-29 16:16 as string
		   self.STOCK_PICKLE__path_for_article : pickle 주소
		   self.STOCK_DICTIONARY_NAME__article_dump : article dump update 해줄 부분
		   self.STOCK_DICTIONARY_NAME__article_result : 여기에 갯수 기록할 거임
		   5
		   5
		   5
		   5
		   5
		   5
		   여기 64bit에서 요약본 구현한거 갖다가 써야될 듯, ram 최적화를 위해서
		   -> ram 최적화도 최적화인데 나중에 kakao -asw 통신위해서 전체를 가져옴
		"""
		try:
			
			try:
				with open(self.STOCK_PICKLE__path_for_article, 'rb') as file:
					self.STOCK_DICTIONARY_NAME__article_dump = copy.deepcopy(file) # 읽어들임
			except Exception as e:
				print('error in FUNC_CHECK_ARTICLE__get - opening and dumping article pickle file in "rb" : ', e)
			
			tmp_counter = 0
			if self.STOCK_DICTIONARY_NAME__article_dump: # 빈 dictionary 가 아니다
	
				for stock_name in self.STOCK_DICTIONARY_NAME__article_dump:
					tmp_counter = 0 # reset
					tmp_article_dict = self.STOCK_DICTIONARY_NAME__article_dump[stock_name][1]
					for article_address in tmp_article_dict:
						tmp_parsed_content_list = tmp_article_dict[article_address]
						tmp_counter = tmp_counter + len(tmp_parsed_content_list)
					
					# result dictionary 에 기록
					self.STOCK_DICTIONARY_NAME__article_result[stock_name] = int(tmp_counter)# 개수 기록!
				# 작업완료 이후
				self.CHECK_2_FLAG__news_article_pickle_ready = True 
					
			else: # 비었음
				print('empty pickle dictionary')
				self.CHECK_2_FLAG__news_article_pickle_ready = False
				
		except Exception as e:
			print('error in FUNC_CHECK_ARTICLE__get :: ', e)
			self.CHECK_2_FLAG__news_article_pickle_ready = False
			traceback.print_exc()
	
	def func_CHECK_ARTICLE__date_validation(self): # 뉴스 데이터 정합성 (날짜 근처인가 확인)
		"""
		self.STOCK_DICTIONARY_NAME__article_dump : article dump update 해줄 부분
		 {stock_name : [stock_num, {article_address : [[article_time, article_score, article_sentence] ,[...] , ....,  ... ]} ] }
		 
		# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
		# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj
		2020-04-29 16:16 as string
		"""
		try:
			tmp_list_of_dates = []
			if self.STOCK_DICTIONARY_NAME__article_dump: # 빈 dictionary 아니면
				for stock_name in self.STOCK_DICTIONARY_NAME__article_dump: # iteration
					tmp_article_dict = self.STOCK_DICTIONARY_NAME__article_dump[stock_name][1]
					for article_address in tmp_article_dict: # 두번 째 hash에 대해 iteration
						tmp_parsed_content_list = tmp_article_dict[article_address]
						for container_list in  tmp_parsed_content_list:
							tmp_date_target = container_list[0] # 해당 날짜 access
							tmp_date_target_obj = datetime.datetime.strptime(tmp_date_target, "%Y-%m-%d %H:%M") # sting to obj
							tmp_date_target_convert = tmp_date_target_obj.strftime('%Y%m%d%H%M%S') # obj to string -> 이거로 통일!(sorting 위해서)
							tmp_list_of_dates.append(tmp_date_target_convert)
				
				# finished parsing all the dates
				tmp_list_of_dates = copy.deepcopy(list(dict.fromkeys(tmp_list_of_dates))) # remove duplicate
				tmp_list_of_dates.sort() # sorting
				
				tmp_latest_article_date = tmp_list_of_dates[-1] # 최신 날짜 가져온 것 -> 타임 맞출 필요는 없을 듯? today 기준으로만 해도 될거임
				"""
				어차피 autotrade able은 시간쪽에서 하면 되니깐?
				"""
				tmp_datetime_now__obj = datetime.datetime.now()
				tmp_latest_article_date__obj = datetime.datetime.strptime(tmp_latest_article_date, "%Y-%m-%d %H:%M")
				
				if tmp_datetime_now__obj - tmp_latest_article_date__obj <= datetime.timedelta(seconds=60*15) : # 15분 편차 이내 최신 article이면
					self.CHECK_2_FLAG__news_article_pickle_ready = True
				else:
					self.CHECK_2_FLAG__news_article_pickle_ready = False
			else:
				self.CHECK_2_FLAG__news_article_pickle_ready = False
				print('empty self.CHECK_2_FLAG__news_article_pickle_ready to check date validation')
		except Exception as e:
			print('error in func_CHECK_ARTICLE__date_validation :: ', e)
			traceback.print_exc()


	def build(self): # build basic pyqt gui
		try:
			self.build_login()
			self.build_cockpit()
			self.build_stats()
			self.build_database()
			self.build_test()
		except Exception as e:
			print('error in build() :: ', e)
			traceback.print_exc()

	def build_login(self): # login 창 생성

		# @ 전체 layout 구성
		self.tab_login.layout = QGridLayout(self)


		# @ 로그인 결과 표시
		self.label_auto_enabled_check = QLabel('Empty', self.tab_login)
		self.tab_login.layout.addWidget(self.label_auto_enabled_check,0,0)

		# @ 로그인 결과 표시
		self.label_login_1 = QLabel('Empty', self.tab_login)
		self.tab_login.layout.addWidget(self.label_login_1,0,1)
		#self.label_login_1.setAlignment(Qt.AlignVCenter)

		# @ 계좌정보 표시
		self.label_account_info_ACCNO = QLabel('Empty', self.tab_login)
		self.tab_login.layout.addWidget(self.label_account_info_ACCNO,1,0)
		self.label_account_info_USERID = QLabel('Empty', self.tab_login)
		self.tab_login.layout.addWidget(self.label_account_info_USERID,1,1)

		# @ 계좌 콤보박스
		self.combo_ACCNO = QComboBox()
		self.combo_ACCNO.addItems(self.ACCOUNT_LIST__for_accounts_owned)
		self.combo_ACCNO.insertSeparator(int(len(self.ACCOUNT_LIST__for_accounts_owned)-1)) # 구분선 추가
		self.tab_login.layout.addWidget(self.combo_ACCNO,2,0)

		# @ 계좌정보 조회 버튼
		self.button_try_login = QPushButton("Login")
		self.tab_login.layout.addWidget(self.button_try_login,3,0)
		self.button_try_login.clicked.connect(self.FUNC_CHECK__ki_connect_state)

		self.button_login_process = QPushButton("Check-Status")
		self.tab_login.layout.addWidget(self.button_login_process,3,1)
		self.button_login_process.clicked.connect(self.FUNC_CHECK_DISP__all_login_process)

		# @ 전자동화 버튼 구성
		self.button_login_auto_all = QPushButton("All - Auto 시작")
		self.tab_login.layout.addWidget(self.button_login_auto_all,4,0) #FUNC_STOCK__enable_auto_trading
		self.button_login_auto_all.clicked.connect(self.FUNC_STOCK__enable_auto_trading)
		
		self.button_login_auto_all_disable = QPushButton("All - Auto 중지")
		self.tab_login.layout.addWidget(self.button_login_auto_all_disable,4,1) #FUNC_STOCK__enable_auto_trading
		self.button_login_auto_all_disable.clicked.connect(self.FUNC_STOCK__disable_auto_trading)

		self.button_kill_backend = QPushButton('Kill - Backend')
		self.tab_login.layout.addWidget(self.button_kill_backend,4,2)
		self.button_kill_backend.clicked.connect(self.FUNC__restart_api)

		# @ 에러 결과 표시
		self.label_error_count = QLabel('Empty', self.tab_login)
		self.tab_login.layout.addWidget(self.label_error_count,5,0)

		# @ Request 갯수 표시 부분
		self.statusbar_req_num = QStatusBar()
		self.tab_login.layout.addWidget(self.statusbar_req_num,5,1)

		# @ api 모드 표시 부분
		self.statsbar_login_mode = QStatusBar()
		self.tab_login.layout.addWidget(self.statsbar_login_mode,5,2)

		# @ 전체 결과 + time 표시 부분
		self.statusbar_login = QStatusBar()
		self.tab_login.layout.addWidget(self.statusbar_login,6,0)

		# @ 전체 layout 할당 부분?
		self.tab_login.setLayout(self.tab_login.layout)

	def build_cockpit(self): # 조종할 cockpit

		# @ 전체 layout 구성
		self.tab_cockpit.layout = QGridLayout(self)

		# @ SQLITE BOOL 결과 표시
		self.label_sqlite_bool_title = QLabel("SQLITE bool title : ")
		self.tab_cockpit.layout.addWidget(self.label_sqlite_bool_title, 0, 0)
		self.label_sqlite_bool_result = QLabel("Empty...")
		self.tab_cockpit.layout.addWidget(self.label_sqlite_bool_result, 0, 1)

		# @ 그래프 결과 표시
		self.plot_sqlite_database = pg.PlotWidget(title="TODAY_MIN_GRAPH",labels={'left': 'price'}, axisItems={'bottom': TimeAxisItem(orientation='bottom')})
		self.plot_sqlite_database.showGrid(x=True, y=True)
		#self.plot_sqlite_database__y_obj = self.plot_sqlite_database.plot(pen='y') # PlotDataItem obj 반환.

		
		#self.plot_sqlite_database = QtCharts.QCandlestickSeries()
		self.tab_cockpit.layout.addWidget(self.plot_sqlite_database, 1,0, 5, 4)
		self.plot_sqlite_database_bar = pg.PlotWidget()
		self.tab_cockpit.layout.addWidget(self.plot_sqlite_database_bar,6,0, 8, 4)



		# @ 전체 layout 할당 부분?
		self.tab_cockpit.setLayout(self.tab_cockpit.layout)



	def build_stats(self): # 수익률 및 통계들
		"""
		여기에 들고있는 주식, 잔고 등등 가져와야됨
		-> 방향 0, 1, 2 :: 열 배치는 2번째 값
		밑 방향 0, 1, 2 :: 행 배치는 1번째 값
		
		:return: 
		"""
		# @ 전체 layout 구성
		self.tab_stats.layout = QGridLayout(self)

		# @ 계좌 잔고 + 그 결과 값
		self.label_balance_title = QLabel("예수금 잔고 : ")
		self.tab_stats.layout.addWidget(self.label_balance_title, 0, 0)
		self.label_balance_value = QLabel("Empty...")
		self.tab_stats.layout.addWidget(self.label_balance_value, 0, 1)

		# @ 계좌 잔고 가져오는 버튼
		self.button_balance_get = QPushButton("예수금 확인")
		self.tab_stats.layout.addWidget(self.button_balance_get, 1, 0)
		self.button_balance_get.clicked.connect(self.FUNC_CHECK_BALANCE__normal)

		# @ check_balance_with_order
		self.button_balance_get_with_order = QPushButton("예수금 - 주문 함께 확인(종목이름 필수)")
		self.tab_stats.layout.addWidget(self.button_balance_get_with_order, 1, 1)
		self.button_balance_get_with_order.clicked.connect(self.FUNC_CHECK_BALANCE__with_order)
		
		# @ 보유주식 label
		self.label_owning_stocks_title = QLabel("[보유주식]")
		self.tab_stats.layout.addWidget(self.label_owning_stocks_title, 2, 0)
		
		# @ 보유주식 조회 button
		self.button_owning_stocks = QPushButton("보유종목 확인")
		self.tab_stats.layout.addWidget(self.button_owning_stocks, 2, 1)
		self.button_owning_stocks.clicked.connect(self.func_CHECK_STOCK_DISP__owning)
		
		# @ 보유 주식 table
		self.table_owning_stocks = QTableWidget(self)
		self.tab_stats.layout.addWidget(self.table_owning_stocks, 3, 0, 5, 3)
		self.table_owning_stocks.clear() # clear 처리 한번 해줌 -> update는 self.FUNC_CHECK_STOCK__owning() 함수 수행시 해줌

		#--------------------------------
		# @ 미체결 주식 label
		self.label_unmet_order_title = QLabel("[미체결목록]")
		self.tab_stats.layout.addWidget(self.label_unmet_order_title, 6, 0)

		# @ 미체결 주식 button
		self.button_unmet_order = QPushButton("미체결 확인")
		self.tab_stats.layout.addWidget(self.button_unmet_order, 6, 1)
		self.button_unmet_order.clicked.connect(self.func_CHECK_STOCK_DISP__unmet)		
		
		# @ 미체결 주식 table
		self.table_unmet_order = QTableWidget(self)
		self.tab_stats.layout.addWidget(self.table_unmet_order, 7, 0, 9, 3)
		self.table_unmet_order.clear()

		# @ 전체 layout 할당 부분?
		self.tab_stats.setLayout(self.tab_stats.layout)

	def build_database(self): # 데이터베이스 만드는 용

		# @ 전체 layout 구성0
		self.tab_database.layout = QVBoxLayout(self)

		# @ 가져온 전체 종목 갯수 표시할 창
		self.label_database_stock_list_num = QLabel('Empty')
		self.tab_database.layout.addWidget(self.label_database_stock_list_num)

		# @ 종목 정보들 표시할 list 창
		self.table_stock_list = QListWidget(self.tab_database)
		self.tab_database.layout.addWidget(self.table_stock_list)

		# @ 종목 fetch 버튼
		self.button_get_stock_list = QPushButton("종목 리스트 가져오기 (코스피+코스닥)")
		self.tab_database.layout.addWidget(self.button_get_stock_list)
		self.button_get_stock_list.clicked.connect(self.FUNC_STOCK_DATABASE__get_all_codes_api)

		# @ 종목 database 1분봉 만들 버튼
		self.button_create_database = QPushButton("SQ_lite 데이터베이스 만들기(분봉)")
		self.tab_database.layout.addWidget(self.button_create_database)
		self.button_create_database.clicked.connect(self.FUNC_STOCK_DATABASE_SQLITE__create) #EMIT_flag_database_create_auto
		self.SIGNAL_MINE.sig_database.connect(self.FUNC_STOCK_DATABASE_SQLITE__create) # 사용자 정의 시그널과 연결

		self.tab_database.setLayout(self.tab_database.layout)
	
	def build_test(self): # 테스트 하는 tab 구성
		# @ 전체 layout 구성0
		self.tab_test.layout = QGridLayout(self)

		# @ 입력 및 구매 수행하는 부분?!
		self.label_test_title = QLabel("종목이름")
		self.tab_test.layout.addWidget(self.label_test_title,0,0)
		self.label_test_title.move(20, 20)
		#---------- 텍스트 입력
		self.line_test_text_input = QLineEdit("")
		self.tab_test.layout.addWidget(self.line_test_text_input,0,1)
		self.line_test_text_input.textChanged.connect(self.FUNC_STOCK__look_up_txt_input)
		self.line_test_text_input.move(80,20)
		#---------- 종목 코드 조회 버튼 / 표시 버튼
		self.button_stock_look_up = QPushButton("조회")
		self.tab_test.layout.addWidget(self.button_stock_look_up,0,2)
		self.button_stock_look_up.clicked.connect(self.FUNC_STOCK__look_up_txt_input)
		self.label_stock_look_up = QLabel("조회 실패")
		self.tab_test.layout.addWidget(self.label_stock_look_up,0,3)


		# --------- 버튼 구성
		self.button_stocks_condition = QPushButton("종목 Condtion Check")
		self.tab_test.layout.addWidget(self.button_stocks_condition, 1, 0)
		self.button_stocks_condition.clicked.connect(self.FUNC_CHECK_STOCK__basic_info)


		# --------- 버튼 구성
		self.button_stocks_additional_info = QPushButton("종목 Additional Info")
		self.tab_test.layout.addWidget(self.button_stocks_additional_info, 1, 1)
		self.button_stocks_additional_info.clicked.connect(self.FUNC_CHECK_STOCK__additional_info_tr)
		self.SIGNAL_MINE.sig_addtional_info_tr.connect(self.FUNC_CHECK_STOCK__additional_info_tr)  # 사용자 정의 시그널과 연결

		# --------- 버튼 구성
		# self.label_stock_look_up_basic_info = QLabel("Blank...")
		# self.tab_test.layout.addWidget(self.label_stock_look_up_basic_info, 1, 1)
		self.button_stocks_set_real_reg = QPushButton("종목 실시간 ON")
		self.tab_test.layout.addWidget(self.button_stocks_set_real_reg, 2, 0)
		self.button_stocks_set_real_reg.clicked.connect(self.FUNC_STOCK_DATABASE__get_real_time)
		self.button_stocks_disable_real_reg = QPushButton("종목 실시간 OFF")
		self.tab_test.layout.addWidget(self.button_stocks_disable_real_reg, 2, 1)
		self.button_stocks_disable_real_reg.clicked.connect(self.FUNC_STOCK_DATABASE__disable_real_time)

		#---------- 버튼 구성
		self.button_request_action = QPushButton("Action-수행")
		self.tab_test.layout.addWidget(self.button_request_action,3,0)
		self.button_request_action.clicked.connect(self.FUNC_STOCK__do_action)
		
		#---------- action 수행 종료 combo box
		self.combo_action = QComboBox()
		self.combo_action.addItems(["매수","매도","매수취소", "매도취소"])
		self.combo_action.insertSeparator(1) # 구분선 추가
		self.combo_action.currentIndexChanged.connect(self.FUNC_STOCK__action_decision)
		self.tab_test.layout.addWidget(self.combo_action,3,1)
		self.STOCK_IN_ATTENTION.action = self.combo_action.currentText() # attention에 추가 -> 한번만 수행
		
		#---------- 오더 갯수
		self.line_order_num = QLineEdit("")
		self.tab_test.layout.addWidget(self.line_order_num,3,2)
		self.line_order_num.textChanged.connect(self.FUNC_STOCK__order_num_decision)
		

		# --------- 결과표시 창 0
		self.text_browser_condition_check_print = QTextBrowser()  # 결과표시 창
		self.text_browser_condition_check_print.setText("Empty condition check result")
		self.tab_test.layout.addWidget(self.text_browser_condition_check_print, 4, 0, 5, 4)

		#--------- 결과표시 창 1
		self.text_browser_additional_result_print = QTextBrowser() # 결과표시 창
		self.text_browser_additional_result_print.setText("Empty additional info result") # 텍스트 표시부분
		self.tab_test.layout.addWidget(self.text_browser_additional_result_print,5,0,6,4)


		#--------- 결과표시 창 2
		self.text_browser_request_action_result_print = QTextBrowser() # 결과표시 창
		self.text_browser_request_action_result_print.setText("Empty Transaction result") # 텍스트 표시부분
		self.tab_test.layout.addWidget(self.text_browser_request_action_result_print,6,0,7,4)



		self.tab_test.setLayout(self.tab_test.layout)
	

		

	def FUNC_STOCK__enable_auto_trading(self): # 오토로 전체 process 실행하는 부분 -> button 으로 동작

		print('＠'*40)
		print('!!!FUNC_STOCK__enable_auto_trading activated!!!')
		# @ reset the attention
		self.TEMP_CHECK_2_STOCK_IN_ATTENTION = copy.deepcopy(self.STOCK_IN_ATTENTION)
		self.STOCK_IN_ATTENTION.__init__()

		# @ 0) set the flag
		# = True
		self.CHECK_3_FLAG__ALTIMATE_AUTO_ON = True

		# @ 1) disable other buttons
		self.FUNC_DISP__set_button_available(False)

		self.FUNC_PYQT__rest_timer(1) # 시간초 대기


		"""
		>>>
		1) 여기서 매수금 10만원 이상 시 매수 하는거 setting
		2) opt10075 미수 체결 발생 시 대응하기! -> 메세지 등으로 확인?
		"""

	
	def FUNC_STOCK__disable_auto_trading(self) : # 오토 enable 동작 그만두는 부분

		print('＠'*40)
		print('!!!disabled FUNC_STOCK__enable_auto_trading!!!')

		#self.flag_FUNC_STOCK__enable_auto_trading = False
		self.CHECK_3_FLAG__ALTIMATE_AUTO_ON = False
		
		# @ 1) disable other buttons
		self.FUNC_DISP__set_button_available(True)

		# @ 2) reset the STOCK IN INTEREST
		self.STOCK_IN_ATTENTION.__init__()
		self.STOCK_IN_ATTENTION = copy.deepcopy(self.TEMP_CHECK_2_STOCK_IN_ATTENTION)

	def FUNC_DISP__set_button_available(self, boolian): # AUTO-Login에 대해서 버튼 활성/비활성화 하는 부분
		### a) login
		self.button_try_login.setEnabled(boolian)
		self.button_login_process.setEnabled(boolian)
		### b) database
		self.button_get_stock_list.setEnabled(boolian)
		#self.button_create_database.setEnabled(boolian) -> 데이터베이스 만드는 용도!!

	def FUNC_CHECK__ki_connect_state(self): # window loging status 체크하는 부분

		state = self.KIWOOM.get_connect_state()

		if state == 1:
			self.CHECK_1_LOGICAL__ki_connect_state = 0
			login_state_msg = '서버 연결 중'
			print('서버 연결 중')
		
		else:
			self.CHECK_1_LOGICAL__ki_connect_state = 1
			login_state_msg = '서버 미 연결'
			print('서버 미 연결')

		curr_time_loging = QTime.currentTime()
		text_time = curr_time_loging.toString("hh:mm:ss")
		statsbar_login = text_time + " | " + login_state_msg + self.DISP_STRING__display_current_job

		self.statusbar_login.showMessage(statsbar_login)
		print('FE - login instance 함수 실행 종료')



	def FUNC_CHECK_DISP__all_login_process(self): # 로그인 확인하는 부분 (window, api 전체)
		print('login checking process...')
		if self.CHECK_1_LOGICAL__windows_1st_login == 0:
			if self.CHECK_1_LOGICAL__ki_connect_state == 1:
				self.label_login_1.setText("접속상태 : 윈도우 성공 + 현재 접속 실패")
				self.statsbar_login_mode.showMessage('Blank...')
				self.CHECK_1_FLAG__ALL_login = False

			else:
				tmp_accno_list = []
				tmp_2_accno_list =  self.KIWOOM.get_login_info(["ACCNO"]).split(';') # account 정보 받아옴
				for item in tmp_2_accno_list:
					if item != '':
						tmp_accno_list.append(item) # -> ; 로 자르고 나서 공백 string 없애기 위함

				for accno in tmp_accno_list:  # 새로 받아온 list
					if accno not in self.ACCOUNT_LIST__for_accounts_owned : # 전에 것과 봐서 다르다면 업데이트
						self.ACCOUNT_LIST__for_accounts_owned = copy.deepcopy(tmp_accno_list)
						self.combo_ACCNO.clear()
						self.combo_ACCNO.addItems(self.ACCOUNT_LIST__for_accounts_owned)
						self.combo_ACCNO.insertSeparator(int(len(self.ACCOUNT_LIST__for_accounts_owned) - 1))  # 구분선 추가
						break # for loop 빠져나옴
				
				# @ 로그인 성공 정보 ram 변수에 탑재
				self.ACCOUNT__code_of_my_account = self.combo_ACCNO.currentText()
				self.STOCK_IN_ATTENTION.accno = str(self.combo_ACCNO.currentText())
				self.ACCOUNT__user_id = self.KIWOOM.get_login_info(["USER_ID"])

				# @ 모투인지 아닌지...
				self.CHECK_1_RESULT__api_real_or_try = self.KIWOOM.get_server_gubun()
				
				# @ 로그인 성공 정보 flag 올림
				self.label_login_1.setText("접속상태 : 윈도우 성공 + 현재 접속 성공")
				if self.ACCOUNT__code_of_my_account and self.ACCOUNT__user_id and self.CHECK_1_RESULT__api_real_or_try:
					self.label_account_info_ACCNO.setText('accnount No. : ' + str(self.ACCOUNT__code_of_my_account))
					self.label_account_info_USERID.setText('user id No. : ' + str(self.ACCOUNT__user_id))
					if self.CHECK_1_RESULT__api_real_or_try: # non null string value
						self.statsbar_login_mode.showMessage('모의투자 모드')
					else:
						self.statsbar_login_mode.showMessage('실제투자 모드')

					self.CHECK_1_FLAG__ALL_login = True
				else:
					self.CHECK_1_FLAG__ALL_login = False
					print('error in FUNC_CHECK_DISP__all_login_process - unexpected error ')

		else:
			self.label_login_1.setText("접속상태 : 윈도우 실패 + 현재 접속 실패")
			self.statsbar_login_mode.showMessage('Blank...')
			self.CHECK_1_FLAG__ALL_login = False


	def FUNC_CHECK_STOCK__basic_info(self):
		"""
		주식 기본 정보들 요청
		ex)
		광동제약 stock names in BE...
		정상  :: ['투자주의', '투자경고',  '투자위험'] 거르기
		증거금30%|담보대출|신용가능  :: ['거래정지'] 포함하는 string일 시 거르기
		:return: 
		"""
		first_filter_1 = [['정상'],['투자주의', '투자경고',  '투자위험']]
		first_filter_2 = [[], ['거래정지','관리종목']]
						 # '증거금100%'
		self.STOCK_DICTIONARY_NAMES__basic_info ={}
		tmp_list_positive = first_filter_1[0] + first_filter_2[0]
		tmp_list_negative = first_filter_1[1] + first_filter_2[1]

		try:
			for stock_names in self.STOCK_DICTIONARY__name_to_code:
				
				# @ 정상 이고 + 투자주의환기종목 / 
				data_1 = self.KIWOOM.get_master_construction(self.STOCK_DICTIONARY__name_to_code[stock_names]) #

				# @ 증거금100% 이고 + 거래정지 아니고 + 관리종목 아니어야함
				data_2 = self.KIWOOM.get_master_stock_state(self.STOCK_DICTIONARY__name_to_code[stock_names])

				data_total = data_1 + data_2

				# @ 결과 기록
				tmp_flag_positive = True
				tmp_flag_negative = True
				for item in tmp_list_positive:
					tmp_flag_positive = bool(tmp_flag_positive * (item in data_total)) # and 형식
				for item in tmp_list_negative:
					tmp_flag_negative = bool(tmp_flag_negative * (item not in data_total))

				tmp_flag_result = bool(tmp_flag_positive * tmp_flag_negative)

				self.STOCK_DICTIONARY_NAMES__basic_info[stock_names] = {'result' : tmp_flag_result, 'master_construction' : data_1, 'master_stock_state' : data_2}

			#print('self.stocks_condition :: ', self.stocks_condition)

			self.CHECK_2_FLAG_STOCK__basic_info = True
		except Exception as e:
			print('error in FUNC_CHECK_STOCK__basic_info :: ',e)
			self.CHECK_2_FLAG_STOCK__basic_info = False
			traceback.print_exc()

	def FUNC_CHECK_STOCK__additional_info_tr(self):
		"""
		 [ opt10001 : 주식기본정보요청 ]

		 1. Open API 조회 함수 입력값을 설정합니다.
			종목코드 = 전문 조회할 종목코드
			SetInputValue("종목코드"	,  "입력값 1");


		 2. Open API 조회 함수를 호출해서 전문을 서버로 전송합니다.
			CommRqData( "RQName"	,  "opt10001"	,  "0"	,  "화면번호");
		:return:
		"""
		# @ dictionary load 시도
		try:
			if os.path.isfile(self.STOCK_PICKLE__path_for_additional_info) : # 존재하면
				with open(self.STOCK_PICKLE__path_for_additional_info, 'rb') as file:
					self.STOCK_DICTIONARY_NAMES__additional_info_tr = copy.deepcopy(pickle.load(file))
					print('successful loading additional info')
			else:
				if len(list(self.STOCK_DICTIONARY_NAMES__additional_info_tr.keys())) == 0:
					self.STOCK_DICTIONARY_NAMES__additional_info_tr = {}
					print('un-successful loading additional info')
		except Exception as e:
			print('error in loading additional info in FUNC_CHECK_STOCK__additional_info_tr :: ', e)
			if len(list(self.STOCK_DICTIONARY_NAMES__additional_info_tr.keys())) == 0:
				self.STOCK_DICTIONARY_NAMES__additional_info_tr = {}


		self.STOCK_FLAG__additional_info_creation_in_progress = True # 이거 정보 가져온다고 올린다
		self.STOCK_FLAG__additional_info_is_stalled = False # 다시 시작했으므로 멈춘다
		# self.STOCK_FLAG__started_getting_additional_info_tr
		try:
			if self.STOCK_FLAG__started_getting_additional_info_tr == False:
				self.tmp_list_additional_info_not_yet_updated = []
				# print('self.STOCK_DICTIONARY__name_to_code :: ', self.STOCK_DICTIONARY__name_to_code)
				# print('self.STOCK_DICTIONARY_NAMES__additional_info_tr :: ', self.STOCK_DICTIONARY_NAMES__additional_info_tr)
				for names in self.STOCK_DICTIONARY__name_to_code:
					if names in self.STOCK_DICTIONARY_NAMES__additional_info_tr: # load 한것에 있다면
						tmp_additional_info_date =  self.STOCK_DICTIONARY_NAMES__additional_info_tr[names]['date']
						tmp_additional_info_date_convert = datetime.datetime.strptime(tmp_additional_info_date, "%Y%m%d%H%M%S")
						#if datetime.datetime.now() - tmp_additional_info_date_convert > datetime.timedelta(days=1): # 있다면 하루 이상일 때
						if datetime.datetime.now() - tmp_additional_info_date_convert > datetime.timedelta(hours=10):
							self.tmp_list_additional_info_not_yet_updated.append(self.STOCK_DICTIONARY__name_to_code[names]) # 하루 지난건 가져오도록 하기
					else: # 가져온 적이 없으므로 가져오기
						self.tmp_list_additional_info_not_yet_updated.append(self.STOCK_DICTIONARY__name_to_code[names])
			else:
				pass
		except Exception as e:
			print('error in FUNC_CHECK_STOCK__additional_info_tr - updating list', e)
			traceback.print_exc()

		try:
			print('request stock_additional_info_tr ... ')
			if  self.STOCK_FLAG__started_getting_additional_info_tr == False:
				self.STOCK_FLAG__started_getting_additional_info_tr = True

			tmp_list_for_additional_info_done = []
			for codes in self.tmp_list_additional_info_not_yet_updated:
				input_dict = {} # input 설정
				input_dict["종목코드"] = codes
				self.KIWOOM.set_input_value(input_dict)
				print('name/code : ', self.STOCK_DICTIONARY__code_to_name[codes], codes)

				tmp_return = self.KIWOOM.comm_rq_data("stock_additional_info_tr", "opt10001", 0, self.SCREEN_NO.stock_additional_info_tr, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
				print('tmp_return in FUNC_CHECK_STOCK__additional_info_tr :: ', tmp_return)
				self.ERROR_COUNTER_BE__request_num = self.ERROR_COUNTER_BE__request_num + 1 # 정상수신 yes / no 관계 없이 수행
				if tmp_return == 0 : # 정상수신
					tmp_name_of_stock = self.STOCK_DICTIONARY__code_to_name[codes]
					#self.STOCK_DICTIONARY_NAMES__additional_info_tr[tmp_name_of_stock] = copy.deepcopy(self.KIWOOM.latest_stock_additional_info) # 받아옴
					if self.KIWOOM.latest_stock_additional_info != None:
						self.STOCK_DICTIONARY_NAMES__additional_info_tr.update(self.KIWOOM.latest_stock_additional_info)
						print('★'*40)
						print('length of saved information in pickle data :: ', len(list(self.STOCK_DICTIONARY_NAMES__additional_info_tr.keys())))
						print('name/code : ', self.STOCK_DICTIONARY__code_to_name[codes], codes)


						print('★' * 40)

						# @ 기록
						tmp_list_for_additional_info_done.append(codes)
				else:
					print('ERROR_COUNTER_BE__request_num :: ', self.ERROR_COUNTER_BE__request_num)

				if self.ERROR_COUNTER_BE__request_num >= 19: # 이거 좀... 빨리 차는 듯?
					self.STOCK_FLAG__additional_info_is_stalled = True
					break
			try:
				with open(self.STOCK_PICKLE__path_for_additional_info, 'wb') as file:
					pickle.dump(self.STOCK_DICTIONARY_NAMES__additional_info_tr, file)
					print('stock additional information successfully saved...!')
			except Exception as e:
				print('error in dumping additional info as pickle... : ', e)


			if self.STOCK_FLAG__additional_info_is_stalled == True:
				tmp_updated_list_for_additional_info_not_yet_updated = []  # tmp로 완료항목 제거하고 다시
				for codes in self.tmp_list_additional_info_not_yet_updated:
					if codes not in tmp_list_for_additional_info_done: # 완료된거 기록한거에서 안된거 다시 뺌
						tmp_updated_list_for_additional_info_not_yet_updated.append(codes)

				# @ remove duplicates
				self.tmp_list_additional_info_not_yet_updated = copy.deepcopy(list(dict.fromkeys(self.tmp_list_additional_info_not_yet_updated)))

				self.tmp_list_additional_info_not_yet_updated = copy.deepcopy(tmp_updated_list_for_additional_info_not_yet_updated) # 미완료 업데이트
				print('len(self.tmp_list_additional_info_not_yet_updated) :: ', len(self.tmp_list_additional_info_not_yet_updated))
				self.FUNC__restart_api()

			else: # 문제 없이 완료하였음
				print('successful creation of additional info in FUNC_CHECK_STOCK__additional_info_tr')
				self.CHECK_2_FLAG_STOCK__additional_info_tr = True
				self.STOCK_FLAG__additional_info_creation_in_progress = False  # 이거 정보 가져온다고 올린다
				self.STOCK_FLAG__additional_info_is_stalled = False  # 다시 시작했으므로 멈춘다
				self.STOCK_FLAG__started_getting_additional_info_tr = False
				self.tmp_list_additional_info_not_yet_updated = []
				

		except Exception as e:
			print('error in FUNC_CHECK_STOCK__additional_info_tr :: ', e)
			traceback.print_exc()
			self.CHECK_2_FLAG_STOCK__additional_info_tr = False

	def FUNC_CHECK_STOCK__owning(self):
		"""
		보유 종목 확인
		:return:
		----------------------------------------------
	 [ OPW00004 : 계좌평가현황요청 ]

	 1. Open API 조회 함수 입력값을 설정합니다.
		계좌번호 = 전문 조회할 보유계좌번호
		SetInputValue("계좌번호"	,  "입력값 1");

		비밀번호 = 사용안함(공백)
		SetInputValue("비밀번호"	,  "입력값 2");

		상장폐지조회구분 = 0:전체, 1:상장폐지종목제외
		SetInputValue("상장폐지조회구분"	,  "입력값 3");

		비밀번호입력매체구분 = 00
		SetInputValue("비밀번호입력매체구분"	,  "입력값 4");


	 2. Open API 조회 함수를 호출해서 전문을 서버로 전송합니다.
		CommRqData( "RQName"	,  "OPW00004"	,  "0"	,  "화면번호");

		"""

		# @ input dictionary 설정
		input_dict ={}
		input_dict["계좌번호"] = self.STOCK_IN_ATTENTION.accno
		input_dict["상장폐지조회구분"] = 0
		input_dict["비밀번호입력매체구분"] = '00'
		self.KIWOOM.set_input_value(input_dict)

		try:
			tmp_check_owning_stocks_result = self.KIWOOM.comm_rq_data("check_owning_stocks", "opw00004", 0, self.SCREEN_NO.check_owning_stocks, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
			print('tmp_check_owning_stocks_result :: ', tmp_check_owning_stocks_result)
			if tmp_check_owning_stocks_result == 0 : # 통신 성공
				self.STOCK_DICTIONARY_NAMES__owning_stocks = copy.deepcopy(self.KIWOOM.latest_owning_stocks_data)
				print('self.STOCK_DICTIONARY_NAMES__owning_stocks :: ', self.STOCK_DICTIONARY_NAMES__owning_stocks)

				if type(self.STOCK_DICTIONARY_NAMES__owning_stocks) == type(dict()) : # 빈 dictionary가 아니면
					print('owning search done 1...')
					self.CHECK_2_FLAG_STOCK__owning_stock_get_success = True


				else:
					print('owning search done 2...')
					self.CHECK_2_FLAG_STOCK__owning_stock_get_success = False
					self.table_owning_stocks.clear()
			else:
				# 통신 실패
				print('owning search done 3...')
				self.CHECK_2_FLAG_STOCK__owning_stock_get_success = False
				self.table_owning_stocks.clear()

		except Exception as e:
			print('error in FUNC_CHECK_STOCK__owning :: ', e)
			self.table_owning_stocks.clear()
			traceback.print_exc()
	
	def func_CHECK_STOCK_DISP__owning(self): # display만 담당
		try:		
			if self.CHECK_2_FLAG_STOCK__owning_stock_get_success == True:
				# @ table update
				#--------------------------------------------------------------------------------------
				if (self.STOCK_DICTIONARY_NAMES__owning_stocks): # 보유 종목이 있다면
					self.table_owning_stocks.clear() # 먼저 table clear 함
					# -> 주식이름 /
					tmp_dict_row_names = copy.deepcopy(list(self.STOCK_DICTIONARY_NAMES__owning_stocks.keys())) # 종목이름
					tmp_dict_col_names = copy.deepcopy(list(self.STOCK_DICTIONARY_NAMES__owning_stocks[tmp_dict_row_names[0]].keys())) # 열 값들 제목

					col_num = len(tmp_dict_col_names)
					row_num = len(tmp_dict_row_names) + 1 # 1칸은 제목
					
					# @ 테이블 사용전에 크기 지정
					self.table_owning_stocks.setRowCount(row_num)
					self.table_owning_stocks.setColumnCount(col_num)
					

					for col in range(col_num):
						for row in range(row_num):
							if row == 0: # 1번째 행일 때 제목 적음
								self.table_owning_stocks.setItem(row, col, QTableWidgetItem(tmp_dict_col_names[col]))
							else:
								tmp_col_search_field = tmp_dict_col_names[col]
								tmp_row_search_field = tmp_dict_row_names[row-1] # stock code
								self.table_owning_stocks.setItem(row, col, QTableWidgetItem(str(self.STOCK_DICTIONARY_NAMES__owning_stocks[tmp_row_search_field][tmp_col_search_field])))
					
					
					# --------------------------------------------------------------------------------------
			else:
				self.table_owning_stocks.clear()
		except Exception as e:
			print('error in func_CHECK_STOCK_DISP__owning :: ', e)

	def FUNC_CHECK_BALANCE__with_order(self):
		"""
		>>>
		여기 input dic 고쳐야 할듯?
		"""
		try:
			# @ input dictionary 설정
			input_dict = {} # 사용할 input dictionary 선언,
			input_dict['계좌번호'] = self.STOCK_IN_ATTENTION.accno
			input_dict['비밀번호입력매체구분'] = '00' # 조회구분 = 1:추정조회, 2:일반조회
			input_dict['종목번호'] = self.STOCK_IN_ATTENTION.code
			print('FUNC_CHECK_BALANCE__with_order - input dic :: ', input_dict)
			self.KIWOOM.set_input_value(input_dict)

			# @ STOCK in attention 부분 코드 None일시 예외 처리
			if self.DISP_FLAG__test_code_look_up_sucess == True:  # 코드 조회 성공일 시 or 전자동화 수행중
				pass
			else: # 코드 조회창에서 조회 안되었을 시
				if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL":
					print('check_balance_with_order - non auto / enter correct code to proceed')
					return None
				else:
					print('check_balance_with_order - auto procedure running')
					pass

			# @ 데이터 송신
			tmp_balance_with_order_result = self.KIWOOM.comm_rq_data("balance_check_with_order", "opw00011", 0, self.SCREEN_NO.balance_check_with_order, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
			print('tmp_balance_with_order_result :: ', tmp_balance_with_order_result)

			if tmp_balance_with_order_result == 0 : # 정상 수신을 함 -> error가 아닌 것은 아님
				print('self.KIWOOM.latest_balance_with_order_data :: ', self.KIWOOM.latest_balance_with_order_data)
				#return self.KIWOOM.latest_balance_with_order_data
				# 자동 차감을 여기서 하면 될 듯?
				if type(self.KIWOOM.latest_balance_with_order_data) == type(dict()) and 'balance_2' in self.KIWOOM.latest_balance_with_order_data: # 딕셔너리 타입이고 데이터가 존재함
					tmp_balance_with_order = copy.deepcopy(float(self.KIWOOM.latest_balance_with_order_data['balance_2'][0])) # 미수 불가 주문 금액 -> 실제랑 가장 가까움
					self.BALANCE_DICTIONARY__for_normal_update['balance'] =   tmp_balance_with_order # balance dic update
					self.BALANCE_DICTIONARY__for_normal_update['date'] = datetime.datetime.now().strftime(
						'%Y%m%d%H%M%S')  # balance dic update - date
					self.STOCK_IN_ATTENTION.balance = tmp_balance_with_order  # latest interest 값 설정
					#self.account_balance = tmp_balance_with_order
					self.STOCK_IN_ATTENTION.balance = tmp_balance_with_order
					self.label_balance_value.setText("조회 성공!-kiwoom second : " + str(self.STOCK_IN_ATTENTION.balance)) # GUI 표기 업데이트

					try:
						with open(self.BALANCE_PICKLE__path_for_balance_update_date, 'wb') as file:  # balance dic 저장
							pickle.dump(self.BALANCE_DICTIONARY__for_normal_update, file)
							print('balance_pickle_dic successfully saved...!')
					except Exception as e:
						print('error in dumping pickle in balance update : ', e)

			else: # 비정상 수신
				pass

		except Exception as e:
			print('error in FUNC_CHECK_BALANCE__with_order :: ', e)
			traceback.print_exc()
		
	
	def FUNC_CHECK_BALANCE__normal(self):
		"""
		1) 계좌 잔고 확인 -> 맨 처음, 나중 매수 매도 하고 나서 받는 것은 action 부분에서 업데이트
		2) 보유 종목 확인
		:return: 
		"""
		try:
			# self.counter_account_balance_check = self.counter_account_balance_check + 1 # 카운터 증가
			# -> periodic_1s 에서 담당
			
			tmp_flag_check_balance_pickle = False # new 조회 수행 by opw00001
			try:
				with open(self.BALANCE_PICKLE__path_for_balance_update_date, 'rb') as file:  # pickel path에서 읽어온다
					self.BALANCE_DICTIONARY__for_normal_update = copy.deepcopy(pickle.load(file))
					if type(self.BALANCE_DICTIONARY__for_normal_update) == type(dict()) and str('balance') in self.BALANCE_DICTIONARY__for_normal_update : # pickle datat 정합성 확인부분
						print('successful deep copy of balance!')
					else: # 정합성 fail
						tmp_flag_check_balance_pickle = True
						
			except Exception as e: # 파일 존재하지 않음
				print('un-successful deep copy of balance! :: ', e)
				tmp_flag_check_balance_pickle = True
			
			if tmp_flag_check_balance_pickle : # 조회를 하겠다 -> pickle file 이 없기 때문
				
				# @ input dictionary 설정
				input_dict = {} # 사용할 input dictionary 선언,
				input_dict['계좌번호'] = self.ACCOUNT__code_of_my_account
				input_dict['조회구분'] = 2 # 조회구분 = 1:추정조회, 2:일반조회


				self.KIWOOM.set_input_value(input_dict) # Backend단 commrqdata 위해 세팅
				tmp_balance_result = self.KIWOOM.comm_rq_data("balance_check_normal", "opw00001", 0, self.SCREEN_NO.balance_check_normal, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
				#tmp_balance_result = self.KIWOOM.comm_rq_data("balance_check", "opw00011", 0, self.SCREEN_NO.balance_check, self.STATE_TIME.state)
				
				# @ 메세지 처리
				tmp_message = self.KIWOOM.latest_balance_check_normal_message
				self.KIWOOM.latest_balance_check_normal_message = None

				if tmp_balance_result == 0 : # 정상수신
					print('self.KIWOOM.latest_balance_normal_data :: ',self.KIWOOM.latest_balance_normal_data)
					# @ int 값으로 balance ram 변수에 올림

					if type(self.KIWOOM.latest_balance_normal_data) == type(dict()) and str('balance') in self.KIWOOM.latest_balance_normal_data : # 딕셔네리 형태이고, balance 받은 상태
						"""
						>>>
						메세지 체크 if문 안에 넣기
						"""
						self.STOCK_IN_ATTENTION.balance = copy.deepcopy(float(self.KIWOOM.latest_balance_normal_data['balance'][0]))
						self.label_balance_value.setText("조회 성공!-kiwoom first : " + str(self.STOCK_IN_ATTENTION.balance))
						self.CHECK_2_FLAG_BALANCE__get_success = True
						self.BALANCE_DICTIONARY__for_normal_update['balance'] = self.STOCK_IN_ATTENTION.balance # balance dic update
						self.BALANCE_DICTIONARY__for_normal_update['date'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S') # balance dic update - date
						#self.STOCK_IN_ATTENTION.balance = self.account_balance # latest interest 값 설정

						try:
							with open(self.BALANCE_PICKLE__path_for_balance_update_date, 'wb') as file: # balance dic 저장
								pickle.dump(self.BALANCE_DICTIONARY__for_normal_update, file)
								print('balance_pickle_dic successfully saved...!')
						except Exception as e:
							print('error in dumping balance pickle 2 : ', e)
						
					else:
						self.label_balance_value.setText("조회 실패!")
						self.CHECK_2_FLAG_BALANCE__get_success = False

				else:
					print("tmp_balance_result :: " , tmp_balance_result)
					self.label_balance_value.setText("조회 실패!")
					self.CHECK_2_FLAG_BALANCE__get_success = False
			
			else: # balance pickle 있어서 거기서 load 함
				"""
				self.BALANCE_PICKLE__path_for_balance_update_date = None # balance db file 경로
				self.BALANCE_DICTIONARY__for_normal_update = {} # balance를 normal로 조회한 시점 저장
				self.account_balance = None # 실제 account의 잔고 int 저장 부분
				"""
				self.STOCK_IN_ATTENTION.balance = self.BALANCE_DICTIONARY__for_normal_update['balance']
				#self.STOCK_IN_ATTENTION.balance = self.account_balance  # latest interest 값 설정
				self.label_balance_value.setText("조회 성공!-pickle : " + str(self.STOCK_IN_ATTENTION.balance))
				self.CHECK_2_FLAG_BALANCE__get_success = True
				
			
		except Exception as e:
			print('error in FUNC_CHECK_BALANCE__normal :: ', e)
			traceback.print_exc()
	
	
	
	def FUNC_CHECK_STOCK__condition_rq_in_api(self): # 조건여부 충족하는 종목 선정 부분
		"""
		키움증권 1.5 설명서에 get_ 이하 함수 쓸 수 있을 듯 + db에서 고르기
		:return: 
		"""
		pass
	

		
	def FUNC_STOCK__action_decision(self):
		"""
		action을 drop combo box에서 고른 경우
		:return: 
		"""
		print('FUNC_STOCK__action_decision - activated...')

		# if self.STOCK_IN_ATTENTION.state != "CHECK_3" : # 전자동화 수행중 아니면
		if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL": # 전자동화 수행중 아니면
			self.STOCK_IN_ATTENTION.action = self.combo_action.currentText()
			print('self.combo_action.currentText() :: ', self.combo_action.currentText())

		else:
			pass
	
	def FUNC_STOCK__order_num_decision(self):
		if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL": # 전자동화 수행중 아니면
			self.STOCK_IN_ATTENTION.order_num = int(self.line_order_num.text())
			print('self.line_order_num.currentText() :: ', self.line_order_num.text())

	def FUNC_STOCK__get_send_order_1st_msg(self):
		"""
		저장된 1st 메세지 계속 업데이트 하는 부분
		unmet order와 비교하려고!
		"""
		try:
			pass
			#ㅋㅋ여기 계속 업데이트 해주어야함!!
		
		
		except Exception as e:
			print('error in FUNC_STOCK__get_send_order_1st_msg :: ', e)


	def FUCN_CHECK_STOCK__unmet_order(self): # 미체결 요청 해결 부분
		"""
		 [ opt10075 : 실시간미체결요청 ] -> 미수 대응하는 부분 // 미수시 발생 메세지 체크해서 대응할 수도? 이거는 메세지
		                                   자체를 체크해야 할듯 (주문 넣는 곳에서)
		                                   KIWOOM QNA 게시판 이용

		 1. Open API 조회 함수 입력값을 설정합니다.
			계좌번호 = 전문 조회할 보유계좌번호
			SetInputValue("계좌번호"	,  "입력값 1");

			전체종목구분 = 0:전체, 1:종목
			SetInputValue("전체종목구분"	,  "입력값 2");

			매매구분 = 0:전체, 1:매도, 2:매수
			SetInputValue("매매구분"	,  "입력값 3");

			종목코드 = 전문 조회할 종목코드
			SetInputValue("종목코드"	,  "입력값 4");

			체결구분 = 0:전체, 2:체결, 1:미체결
			SetInputValue("체결구분"	,  "입력값 5");


		 2. Open API 조회 함수를 호출해서 전문을 서버로 전송합니다.
			CommRqData( "RQName"	,  "opt10075"	,  "0"	,  "화면번호");
		:return:
		------------------------------------------------------------------------------------
		FUNC_STOCK__do_action -> 미수 확인 -> FUNC_STOCK__do_action로 주문 cancle 과정
		------------------------------------------------------------------------------------
		"""
		try:
			print('FUCN_CHECK_STOCK__unmet_order activated...')
			# @ input dictionary 설정합니다
			# https://m.blog.naver.com/PostView.nhn?blogId=wjs0906&logNo=221258810025&proxyReferer=https:%2F%2Fwww.google.com%2F
			input_dict = {}
			input_dict["계좌번호"] = self.STOCK_IN_ATTENTION.accno
			input_dict["체결구분"] = "1" # 1이면 미체결 조회
			input_dict["매매구분"] = "0" # 0이면 전체 매수 매도 주문 정보 요청하는
			
			self.KIWOOM.set_input_value(input_dict) # input 전송
			
			tmp_check_unmet_order = self.KIWOOM.comm_rq_data("check_unmet_order", "opt10075", 0, self.SCREEN_NO.check_unmet_order, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
			
			if tmp_check_unmet_order == 0: # 정상수신
				"""
				double hash :
				종목이름 -> 주문번호 -> 나머지 순서
				"""	
				self.STOCK_DICTIONARY_NAMES__unmet_order = copy.deepcopy(self.KIWOOM.latest_stock_unmet_order_data) # update!
				if type(self.STOCK_DICTIONARY_NAMES__unmet_order) == type(dict()):
					self.CHECK_2_FLAG_STOCK__unmet_order_success = True
				else: # 잘못된 수신 : 빈칸??? 정상수신인데 데이터 안들어옴
					self.CHECK_2_FLAG_STOCK__unmet_order_success = False
			else: # 비정상 수신
				self.CHECK_2_FLAG_STOCK__unmet_order_success = False
		
		except Exception as e:
			print('error in FUCN_CHECK_STOCK__unmet_order :: ', e)
			self.CHECK_2_FLAG_STOCK__unmet_order_success = False

	def func_CHECK_STOCK_DISP__unmet(self):
		"""
		self.STOCK_DICTIONARY_NAMES__unmet_order -> double hash
		종목이름 -> 주문번호 -> 나머지 순서
		"""
		try:
			print('func_CHECK_STOCK_DISP__unmet activated...')
			if self.CHECK_2_FLAG_STOCK__unmet_order_success == True:
				# @ table update
				#--------------------------------------------------------------------------------------
				if (self.STOCK_DICTIONARY_NAMES__unmet_order): # 미체결 종목이 있다면
					self.table_unmet_order.clear() # 먼저 table clear 함
					# -> 주식이름 /
					tmp_dict_stock_names = copy.deepcopy(list(self.STOCK_DICTIONARY_NAMES__unmet_order.keys())) # 종목이름
					tmp_dict_row_names = [] # 주문번호 + 종목이름이 y table이라서
					tmp_dict_col_names = []
					tmp_bool_for_col_name = False
					for stock_names in self.STOCK_DICTIONARY_NAMES__unmet_order:
						for order_num in self.STOCK_DICTIONARY_NAMES__unmet_order[stock_names]:
							tmp_dict_row_names.append(order_num)
							if tmp_bool_for_col_name == False:
								tmp_dict_col_names = copy.deepcopy(list(self.STOCK_DICTIONARY_NAMES__unmet_order[stock_names][order_num].keys()))
								if len(tmp_dict_col_names) != 0 : # 비지 않았다면
									tmp_bool_for_col_name = True
								
					#tmp_dict_col_names = copy.deepcopy(list(self.STOCK_DICTIONARY_NAMES__unmet_order[tmp_dict_stock_names[0]][tmp_dict_row_names[0]].keys())) # 열 값들 제목
					
					
					col_num = len(tmp_dict_col_names)
					row_num = len(tmp_dict_row_names) + 1 # 1칸은 제목
					
					# @ 테이블 사용전에 크기 지정
					self.table_unmet_order.setRowCount(row_num)
					self.table_unmet_order.setColumnCount(col_num)
					
					tmp_bool_for_next = False
					for col in range(col_num):
						tmp_bool_for_next = False # reset
						for row in range(row_num):
							if row == 0: # 1번째 행일 때 제목 적음
								self.table_unmet_order.setItem(row, col, QTableWidgetItem(tmp_dict_col_names[col]))
							else:
								tmp_col_search_field = tmp_dict_col_names[col]
								tmp_row_search_field = tmp_dict_row_names[row-1] # 주문번호
								
								# 주문번호로 찾는다
								for stock_name in self.STOCK_DICTIONARY_NAMES__unmet_order:
									for order_num in self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name]:
										if order_num == tmp_row_search_field:
											tmp_hash = self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name][order_num] # 찾은것 temp hash
											self.table_unmet_order.setItem(row, col, QTableWidgetItem(str(tmp_hash[tmp_col_search_field])))
											tmp_bool_for_next = True
											break
									if tmp_bool_for_next == True: # 위 for문에서 찾았으면
										break
								
					
					
					# --------------------------------------------------------------------------------------
			else:
				self.table_unmet_order.clear()
		except Exception as e:
			print('error in func_CHECK_STOCK_DISP__owning :: ', e)	

	
	def FUNC_STOCK__handle_sell_all_single_stock(self):
		"""
		stock attention의 code로 수행
		필요없는 하나의 stock 모두 처분
		:return: 
		"""
		try:
			
			if self.STOCK_DICTIONARY_NAMES__owning_stocks: # 빈 딕셔너리가 아니면
				if self.STOCK_DICTIONARY__code_to_name[ self.STOCK_IN_ATTENTION.code ] in self.STOCK_DICTIONARY_NAMES__owning_stocks:

					self.STOCK_IN_ATTENTION.order_num = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.STOCK_DICTIONARY__code_to_name[ self.STOCK_IN_ATTENTION.code ]]['number_owned']
					self.STOCK_IN_ATTENTION.action = "매도"

					self.FUNC_STOCK__do_action()
					
		except Exception as e:
			print('error in FUNC_STOCK__handle_sell_all_single_stock :: ', e)
	
	def FUNC_STOCK__handle_sell_every_stock(self):
		"""
		보유중인 모든 주식 판매함
		
		:return: 
		"""
		try:

			if self.STOCK_DICTIONARY_NAMES__owning_stocks:  # 빈 딕셔너리가 아니면
				for stock_name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.STOCK_IN_ATTENTION.code = self.STOCK_DICTIONARY__name_to_code[ self.STOCK_DICTIONARY_NAMES__owning_stocks[stock_name]['stock_name'] ]
					self.STOCK_IN_ATTENTION.order_num = self.STOCK_DICTIONARY__name_to_code[ self.STOCK_DICTIONARY_NAMES__owning_stocks[stock_name]['stock_name'] ]['number_owned']
					self.STOCK_IN_ATTENTION.action = "매도"
					self.FUNC_STOCK__do_action()  # 종목 전체 매도 수행
					

		except Exception as e:
			print('error in FUNC_STOCK__handle_sell_every_stock :: ', e)
	

	def FUNC_STOCK__handle_unmet_order(self):
		"""
		미체결 값 확인해서 sell order 해주는 부분
		
		self.STOCK_DICTIONARY_NAMES__unmet_order
		double hash :
		종목이름 -> 주문번호 -> 나머지 순서
		"""
		try:
			tmp_STOCK_IN_ATTENTION__save = copy.deepcopy(self.STOCK_IN_ATTENTION) # 혹시 모르니 save
			if self.STOCK_DICTIONARY_NAMES__unmet_order: #하나라도 값이 있으면
				
				for stock_name in self.STOCK_DICTIONARY_NAMES__unmet_order:
					#tmp_order_num_hash = self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name]
					for order_num_in_stock in self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name]: # double hash 끝단
						# @ 주문개수 설정
						tmp_number_of_unmet_order = self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name][order_num_in_stock]['order_num_count']
						self.STOCK_IN_ATTENTION.order_num = tmp_number_of_unmet_order
						
						# @ 주문상태 확인
						tmp_order_state = self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name][order_num_in_stock]['order_state']
						# @ 반대 action 수행
						if "매수" in tmp_order_state: # 매수면 매수취소
							self.STOCK_IN_ATTENTION.action = "매수취소"
						elif "매도" in tmp_order_state: # 매도면 매도취소
							self.STOCK_IN_ATTENTION.action = "매수취소"
						
						# @ 취소하는 stock code
						tmp_stock_code = self.STOCK_DICTIONARY_NAMES__unmet_order[stock_name][order_num_in_stock]['stock_code']
						self.STOCK_IN_ATTENTION.code = tmp_stock_code
						
						# @ 주문 취소할 주문번호
						tmp_order_num = order_num_in_stock
						self.STOCK_IN_ATTENTION.orderno = tmp_order_num
						
						self.FUNC_STOCK__do_action() # 취소 수행
				
						
				self.STOCK_IN_ATTENTION = copy.deepcopy(tmp_STOCK_IN_ATTENTION__save)# stock in attention reset 수행
			
		
		except Exception as e:
			print('error in FUNC_STOCK__handle_unmet_order :: ', e)
			
	
	def FUNC_STOCK__do_action(self): # test 탭에서 구매 수행

		try:
			print('Run action func in front end activated...')
			
			

			# Ram 변수에 올릴 ohlcv 값 2개 선언
			ohlcv_1_data = None
			ohlcv_2_data = None
			ohlcv = None
			
			combo_box_which = self.STOCK_IN_ATTENTION.action # "매수" / "매도" - all 시장가

			print('Run by func in front end 2nd path activated...')
			order_type_lookup = {'신규매수': 1, '신규매도': 2, '매수취소': 3, '매도취소': 4}
			if combo_box_which == "매수":
				order_type = '신규매수'
				string_name_for_req = "send_buy_order_req"
				original_order_num = ''
				
				
			elif combo_box_which == "매도":
				order_type = '신규매도'
				string_name_for_req = "send_sell_order_req"
				original_order_num = ''
			
			elif combo_box_which == "매도취소":
				order_type = "매도취소"
				string_name_for_req = "cancle_sell_order_req"
				original_order_num = self.STOCK_IN_ATTENTION.orderno # 주문번호 -> cancle 용
				
			elif combo_box_which == "매수취소":
				order_type = "매수취소"
				string_name_for_req = "cancle_buy_order_req"
				original_order_num = self.STOCK_IN_ATTENTION.orderno # 주문번호 -> cancle 용
			
			else:
				# 에러 올린다
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				return None # 함수 종료

			hoga_lookup = {'지정가': "00", '시장가': "03"}
			hoga = '시장가'
			price = int(0)
			if hoga == '시장가':
				price = int(0)
				
			code = self.STOCK_IN_ATTENTION.code
			num = int(self.STOCK_IN_ATTENTION.order_num) # 매수 갯수
			# @ num 정한 것에서 현재가 확인하고 주문 가능 수량까지 한정시킴
			try:
				if self.STOCK_IN_ATTENTION.action == "매수":
					if self.STOCK_IN_ATTENTION.balance >= self.STOCK_BUDGET_AT_LEAST : # 한도 금액 이상 계좌에 있어야 함
						if code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC and (self.STOCK_IN_ATTENTION.state == "CHECK_3" or self.STOCK_IN_ATTENTION.state == "ERROR_SELL_ALL" ): # 최신 데이터 조회구분
							tmp_datetime_stamp_hash = list(self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[code].keys()).sort()
							tmp_latest_time_stamp = tmp_datetime_stamp_hash[-1] # 최신 time stamp
							tmp_latest_time_stamp_obj = datetime.datetime.strptime(tmp_latest_time_stamp, "%Y%m%d%H%M%S")
							# tmp_datetime_obj = datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S")
							if datetime.datetime.now() - tmp_latest_time_stamp_obj < datetime.timedelta(seconds=10): # 10초 안의 최신 데이터 가지고 있을 때만
								tmp_price_latest = self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[code][tmp_latest_time_stamp]['price']
								tmp_num = copy.deepcopy(num)
								while (tmp_price_latest*tmp_num > self.STOCK_IN_ATTENTION.balance and tmp_num > 0):
									tmp_num = tmp_num - 1 # 하나씩 개수 줄임
									if tmp_num < 1 : # 1개도 못사면
										break
								if tmp_num < 1 : # 못산다
									self.text_browser_request_action_result_print.setText('problem 매수 - 1')
									return None # 주문 못하고 나옴
								else:
									if self.STOCK_IN_ATTENTION.balance - tmp_price_latest*tmp_num < self.STOCK_BUDGET_AT_LEAST: # 구매 후 남은 돈이 기준값 미만
										self.text_browser_request_action_result_print.setText('problem 매수 - 2')
										return None
									else:
										num = tmp_num
										self.STOCK_IN_ATTENTION.order_num = tmp_num # update 하여 차후에 사용
							else:
								self.text_browser_request_action_result_print.setText('problem 매수 - 3')
								# 최신 데이터가 없다
							
						else:
							self.text_browser_request_action_result_print.setText('problem 매수 - 3')
							return None # 함수 끝, 주문가능 몇개인지 알 수 없으므로...
					return None # 한도 금액 이상 없어서 주문 취소
				
				elif self.STOCK_IN_ATTENTION.action == "매도": # self.STOCK_DICTIONARY_NAMES__owning_stocks
					if self.STOCK_DICTIONARY__code_to_name[code] in self.STOCK_DICTIONARY_NAMES__owning_stocks: # 들고있지 않다면 return None
						tmp_num = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.STOCK_DICTIONARY__code_to_name[code]]['number_owned'] # local에서 한번 받아온 보유 결과
						if tmp_num >= num : # 보유 수량보다 적게 팔려고 한다는
							pass
						else: # tmp_num < num
							num = copy.deepcopy(tmp_num) # 들고있는거 다 팔도록 바꿔줌
							self.STOCK_IN_ATTENTION.order_num = tmp_num # update 하여 차후에 사용

					else: # 들고있지 않은데 팔려고 함
						self.text_browser_request_action_result_print.setText('problem 매도 - 1')
						return None
				else: # 매도취소, 매수취소는 함수 밖에서 정해진 값으로 넣어줌
					self.text_browser_request_action_result_print.setText('problem 매도 - 2')
					pass
					
			except Exception as e:
				print('error converting possible num value in FUNC_STOCK__do_action :: ', e)
					
			
			# @ DISP 세팅 바꿔줌
			if self.DISP_FLAG__test_code_look_up_sucess == True : # 코드 조회 성공일 시 or 전자동화 수행중
				pass
			else:
				if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL":
					print('Run by func in front end 3rd path activated...')
					self.text_browser_request_action_result_print.setText('ACTION RESULT - non-auto / enter correct code to proceed')
					return None
				else:
					print('Run by func in front end 4th path activated...')
					self.text_browser_request_action_result_print.setText('ACTION RESULT - auto procedure running')
					pass

			print(" in the FUNC_STOCK__do_action :: ", string_name_for_req, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], original_order_num)

			# @ order 보내는 부분, 순서 있음
			#####################
			"""
			1)메세지 받음
			2)tr 값 받음
			3)chej 값 받음
			"""
			#####################
			tmp_send_order_result = self.KIWOOM.send_order(string_name_for_req, self.SCREEN_NO.send_order, self.ACCOUNT__code_of_my_account, order_type_lookup[order_type], code, num, price, hoga_lookup[hoga], original_order_num)
			print('setting -2 in FE FUNC_STOCK__do_action finished...')
			self.FUNC_PYQT__rest_timer(0.05) # wait just in case
			print('setting -3 in FE run_buy finished...')
			print('tmp_send_order_result :: ', tmp_send_order_result)
				
			# @ recieve message
			ohlcv = copy.deepcopy(self.KIWOOM.latest_buy_sell_result_message) # 정상 거래 메세지
			self.KIWOOM.latest_buy_sell_result_message = None # reset
			
			ohlcv_cancle = copy.deepcopy(self.KIWOOM.latest_cancle_buy_sell_result_message)
			self.KIWOOM.latest_cancle_buy_sell_result_message = None # reset

			if ohlcv : # 메세지 blank 아님
				tmp_mssg = self.KI_MESSAGE.decode_message(ohlcv)
				if tmp_send_order_result == 0 : # 정상수신
					if tmp_mssg == "매수성공" or tmp_mssg == "매도성공" : # -> 메세지에서 매도/매수 성공함
						#print('   &&& latest_buy_sell_result_message :: ',ohlcv['message'])
						########
						ohlcv_1_data = copy.deepcopy(self.KIWOOM.latest_buy_sell_result_first_data)
						ohlcv_2_data = copy.deepcopy(self.KIWOOM.latest_buy_sell_result_second_data)

						# @ if 분기문으로 메세지 상태별 대응
						if ohlcv_1_data == None and ohlcv_2_data == None: # 두 메세지 모두 timeout으로 못받음
							self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1  # critical error 증가 -> message 1개 라도 받아야 함
							self.text_browser_request_action_result_print.setText('ACTION RESULT - fail :: ' + str(ohlcv))

						elif ohlcv_1_data == None and ohlcv_2_data != None: # 한 메세지 timeout으로 못받음
							self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1  # critical error 증가 -> message 1개 라도 
							self.text_browser_request_action_result_print.setText('ACTION RESULT - partial fail_1 :: ' + str(ohlcv))
							print('ohlcv_1_data :', ohlcv_1_data)
							print('ohlcv_2_data :' ,ohlcv_2_data)


						elif ohlcv_1_data != None and ohlcv_2_data == None: # 한 메세지 timeout으로 못받음
							self.text_browser_request_action_result_print.setText('ACTION RESULT - partial fail_2 :: ' + str(ohlcv))
							print('ohlcv_1_data :', ohlcv_1_data)
							print('ohlcv_2_data :' ,ohlcv_2_data)

							
						elif ohlcv_1_data != None and ohlcv_2_data != None:  # 두 메세지 모두 들어옴
							self.text_browser_request_action_result_print.setText('ACTION RESULT - success :: ' + str(ohlcv))
							print('ohlcv_1_data :', ohlcv_1_data)
							print('ohlcv_2_data :' ,ohlcv_2_data)
							"""
							>>>
							메세지 받아와서 self.account_balance 차감해야됨
							메세지 container value access 해서 가져올 때 strip 쓰기
							미수처리 메세지? -> kiwoom QNA 게시판 확인
							"""
						if ohlcv_1_data != None: # msg1은 받아야함 -> 계좌 작업 시작
							
							# ------------------------------------------
							"""
							1) 매수 / 매도에 따른 로직 바뀜
							2) 주문 캔슬인 경우에 바뀜
							3) 1), 2)에 의해 업데이트 항목이 달라짐
							''''''''''''''''''''''''''''''''''''''''
							self.STOCK_DICTIONARY_NAMES__owning_stocks
							
							"""
							# ------------------------------------------
							# @ msg_1에서 받아오는 체결가 * 체결수량
							tmp_met_price = 0
							if combo_box_which == "매수":
								tmp_met_price = -(ohlcv_1_data['order_met_price'] * ohlcv_1_data['order_met_num_count'])
								self.STOCK_DICTIONARY_NAMES__owning_stocks[self.STOCK_DICTIONARY__code_to_name[self.STOCK_IN_ATTENTION.code]]['number_owned'] = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.STOCK_DICTIONARY__code_to_name[self.STOCK_IN_ATTENTION.code]]['number_owned'] + ohlcv_1_data['order_met_num_count']
							elif combo_box_which == "매도":
								tmp_met_price = (ohlcv_1_data['order_met_price'] * ohlcv_1_data['order_met_num_count'])
								self.STOCK_DICTIONARY_NAMES__owning_stocks[
									self.STOCK_DICTIONARY__code_to_name[self.STOCK_IN_ATTENTION.code]]['number_owned'] = \
								self.STOCK_DICTIONARY_NAMES__owning_stocks[
									self.STOCK_DICTIONARY__code_to_name[self.STOCK_IN_ATTENTION.code]][
									'number_owned'] - ohlcv_1_data['order_met_num_count']
							
							
							# 1) 1차 계좌 값 후보 -> 이거로 밀고 나가자
							tmp_STOCK_IN_ATTENTION__balance = self.STOCK_IN_ATTENTION.balance + (tmp_met_price*(1+self.AUTO_TRADE_FEE)) # 세금 + 거래세 포함한 금액
							self.STOCK_IN_ATTENTION.balance = tmp_STOCK_IN_ATTENTION__balance # STOCK_IN_ATTENTION에 대입

							# 2) 종목에 대한 잔고현황 확인
							if ohlcv_2_data != None:
								tmp_target_sum_buy_price = ohlcv_2_data['stock_sum_buy_price']
								self.STOCK_DICTIONARY_NAMES__owning_stocks[self.STOCK_DICTIONARY__code_to_name[self.STOCK_IN_ATTENTION.code]]['buy_price'] = tmp_target_sum_buy_price

							else:
								self.STOCK_FLAG__when_unmet_order_made = True # false여부 상관 없음, tr조회로 다시 확인하는 부분
							

							
							# save) 2차 후보
							#self.FUNC_CHECK_BALANCE__with_order() # 서버에서 미수불가증거금 조회 -> self.STOCK_IN_ATTENTION.balance update
							#print('tmp_balance_with_order_result :: ', tmp_balance_with_order_result)
							
							self.BALANCE_DICTIONARY__for_normal_update['balance'] = tmp_STOCK_IN_ATTENTION__balance
							self.BALANCE_DICTIONARY__for_normal_update['date'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

							try:
								with open(self.BALANCE_PICKLE__path_for_balance_update_date, 'wb') as file: # balance dic 저장
									pickle.dump(self.BALANCE_DICTIONARY__for_normal_update, file)
									print('balance_pickle_dic successfully saved...!')
							except Exception as e:
								print('error in dumping balance pickle in action fuction : ', e)

							# @ 미체결 수량 확인
							# self.STOCK_DICTIONARY_NAMES__unmet_order -> update 해줌
							"""
							flag 값으로 해주는게 맞는듯?
							"""
							tmp_num_of_unmet_order = ohlcv_1_data['unmet_order_num'] # 미체결 수량
							if tmp_num_of_unmet_order != 0 : #0개가 아니면 flag 띄운다
								self.STOCK_FLAG__when_unmet_order_made = True # false여부 상관 없음
							
							############################
							# AT 단에서 쓰기 위해 return 필요없음, ohlcv_2로 해결
							############################
							
							


						# @ BE reset 해줌
						self.KIWOOM.latest_buy_sell_result_first_data = None
						self.KIWOOM.latest_buy_sell_result_second_data = None

					elif tmp_mssg == "매수취소성공" or tmp_mssg == "매도취소성공" :
						self.text_browser_request_action_result_print.setText('ACTION RESULT - 2nd path :: ' + str(ohlcv_cancle))
				
				else: # 비정상 수신을 받음
					self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
					self.text_browser_request_action_result_print.setText('ACTION RESULT - 3rd path fail :: ' + str(ohlcv))
			elif ohlcv_cancle: # 매수취소 매도취소 메세지 받는것이 None 아니면, 	
				tmp_mssg = self.KI_MESSAGE.decode_message(ohlcv_cancle)
				if tmp_send_order_result == 0: # 정상수신
					if tmp_mssg == "매수취소성공" or tmp_mssg == "매도취소성공" :
						self.text_browser_request_action_result_print.setText('ACTION RESULT - 4th path :: ' + str(ohlcv_cancle))
				else: # 주문 취소 비정상 수신
					self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
					self.text_browser_request_action_result_print.setText('ACTION RESULT - 5th path fail :: ' + str(ohlcv_cancle))
			else:
				print('server message is blank...')
				self.text_browser_request_action_result_print.setText('ACTION RESULT - server message is blank...')
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1 # critical error 증가
				self.KIWOOM.latest_buy_sell_result_first_data = None
				self.KIWOOM.latest_buy_sell_result_second_data = None


		except Exception as e:
			traceback.print_exc()
			print('error in run_buy :: ', e)
			
			
	def FUNC_STOCK__draw_daily_graph(self, stock_code): # test 탭에서 고른 항목 그래프 그림
		# self.label_sqlite_bool_result
		"""
		https://www.learnpyqt.com/courses/graphics-plotting/plotting-pyqtgraph/
		https://freeprog.tistory.com/373
		:param stock_code:
		:return:
		"""
		# legacy clear
		self.plot_sqlite_database.clear()
		self.plot_sqlite_database_bar.clear()

		try:
			if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']: # dictionary에 존재하는 경우만
				tmp_hash_stock_data = copy.deepcopy(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code])
				#print('tmp_hash_stock_data :: ', tmp_hash_stock_data)
				# -> datatime : { 'price' : , 'volume' :  }
				# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
				# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S") : string to obj

				# x_val = []
				# for datetime_stamp in tmp_hash_stock_data:
				# 	x_val.append(datetime_stamp)
				# x_val.sort()

				x_val = copy.deepcopy(list(tmp_hash_stock_data.keys()))
				x_val.sort()

				#print('x_val :: ', x_val)

				x_val_real = []
				for i in range(len(x_val)): # string value
					tmp_datetime_obj = int(datetime.datetime.strptime(x_val[i], "%Y%m%d%H%M%S").timestamp())
					#x_val_real.append(i)
					x_val_real.append(tmp_datetime_obj)


				y_price = []
				y_volume = []

				for x_point in x_val:
					y_price.append(tmp_hash_stock_data[x_point]['price'])
					y_volume.append(tmp_hash_stock_data[x_point]['volume'])
				#print('y_volume :: ', y_volume)
				
				# @ 그림 그림
				#self.plot_sqlite_database.plot(x_val_real, y_price)
				#self.plot_sqlite_database.setTicks([x_val_real, []])
				self.plot_sqlite_database.setXRange(x_val_real[0], x_val_real[-1], padding=0)
				self.plot_sqlite_database.plot(x_val_real, y_price)
				#self.plot_sqlite_database__y_obj.setData(x_val, y_price)
				self.plot_sqlite_database.enableAutoRange()
				self.plot_sqlite_database_bar.plot(x_val_real, y_volume)
				self.plot_sqlite_database_bar.enableAutoRange()

				#self.plot_sqlite_database.re

				# @  volume 결과 bool 표시
				self.label_sqlite_bool_result.setText(str(self.STOCK_DICTIONARY__code_to_name[stock_code])+ ' - ' + str(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code]) + ' :: ' + str(len(x_val)))
			
			else:
				if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']:
					if self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code]: # 있어야 함
						self.label_sqlite_bool_result.setText(str(self.STOCK_DICTIONARY__code_to_name[stock_code]) + ' - ' + 'wrong deletion of stock data...')
					else:
						self.label_sqlite_bool_result.setText(str(self.STOCK_DICTIONARY__code_to_name[stock_code]) + ' - ' + 'deleted by filtering...')
				else:
					self.label_sqlite_bool_result.setText(str(self.STOCK_DICTIONARY__code_to_name[stock_code])+ ' - ' + 'not existing yet...')


		except Exception as e:
			print('error in FUNC_STOCK__draw_daily_graph :: ', e)
			self.label_sqlite_bool_result.setText(str(self.STOCK_DICTIONARY__code_to_name[stock_code])+ ' - ' + 'Error...')
			traceback.print_exc()
	

	def FUNC_STOCK__look_up_txt_input(self): # test 탭에서 lookup 수행

		# self.STOCK_DICTIONARY__name_to_code
		try:
			if self.STOCK_IN_ATTENTION.state != "CHECK_3" and self.STOCK_IN_ATTENTION.state != "ERROR_SELL_ALL": #전자동화 수행중
				if self.line_test_text_input.text() in self.STOCK_DICTIONARY__name_to_code:
					self.label_stock_look_up.setText(str(self.STOCK_DICTIONARY__name_to_code[self.line_test_text_input.text()] + ' :: 조회 완료' ))
					self.text_browser_condition_check_print.setText(str(self.STOCK_DICTIONARY_NAMES__basic_info[self.line_test_text_input.text()]))
					if self.line_test_text_input.text() in self.STOCK_DICTIONARY_NAMES__additional_info_tr:
						self.text_browser_additional_result_print.setText(str(self.STOCK_DICTIONARY_NAMES__additional_info_tr[self.line_test_text_input.text()]))
					else:
						self.text_browser_additional_result_print.setText("getting not-yet finished...")
					self.STOCK_IN_ATTENTION.code = str(self.STOCK_DICTIONARY__name_to_code[self.line_test_text_input.text()]) # interest update
					self.STOCK_IN_ATTENTION.name = str(self.line_test_text_input.text())

					self.FUNC_STOCK__draw_daily_graph(str(self.STOCK_DICTIONARY__name_to_code[self.line_test_text_input.text()]))

					self.DISP_FLAG__test_code_look_up_sucess = True
				else:
					self.label_stock_look_up.setText('조회 실패')
					self.DISP_FLAG__test_code_look_up_sucess = False
					self.text_browser_condition_check_print.setText('Blank2...')
					self.text_browser_additional_result_print.setText('Blank2...')
			else:
				self.label_stock_look_up.setText('전자동화 수행중')
				self.text_browser_condition_check_print.setText('Blank3...')
				self.text_browser_additional_result_print.setText('Blank3...')
				#self.DISP_FLAG__test_code_look_up_sucess = False
		except Exception as e:
			print('error in FUNC_STOCK__look_up_txt_input :: ', e)
			self.label_stock_look_up.setText('조회 실패 - 2nd path')
			self.text_browser_condition_check_print.setText('Blank4...')
			self.text_browser_additional_result_print.setText('Blank4...')
			self.DISP_FLAG__test_code_look_up_sucess = False
			traceback.print_exc()
			pass

	def FUNC_STOCK_DATABASE__get_real_time(self): # 실시간으로 종목 받아오는 부분
		"""
		setRealReg 사용?
		실시간 시세 데이터는 하나의 화면번호당, 100개 종목까지 등록 가능합니다!
		SetRealReg 이거같음 : 1.5 키움 개발가이드에서 검색해서 코딩
		:return:
		opt10001 으로 신청할 생각 -> 개장 시간에만 동작!!!!
		
		https://smbyeon.github.io/2019/12/06/kiwoom-graph.html
		https://m.blog.naver.com/PostView.nhn?blogId=jhsgo&logNo=221523044204&targetKeyword=&targetRecommendationCode=1
		https://m.blog.naver.com/PostView.nhn?blogId=jhsgo&logNo=221526307126&proxyReferer=https:%2F%2Fwww.google.com%2F
		"""
		try:
			
			# @ 확인후 실시간 scrno 할당 부분
			if self.SCREEN_NO.scrno_used_lookup_by_stock_code(self.STOCK_IN_ATTENTION.code) == None: 
			# None이어야 사용 안하고 있음, 사용중이면 사용중인 scrno 값 리턴
				tmp_scrno = self.SCREEN_NO.return_scrno_for_realtime(self.STOCK_IN_ATTENTION.code)
			else:
				#self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				return None # 함수 종료
				
			fid_list = "10;15;"
			opt_type = "1"
			"""
			타입 “0”은 항상 마지막에 등록한 종목들만 실시간등록이 됩니다
			타입 “1”은 이전에 실시간 등록한 종목들과 함께 실시간을 받고 싶은 종목을 추가로 등록할 때 사용합니다.
			
			FID 번호는 KOA Studio 에서 실시간 목록을 참고하시기 바랍니다. .
			9001:종목코드
			10 : 현재가
			15 : 거래량
			13 : 누적거래량
			
			종목 , FID 는 각각 한번에 실시간 등록 할 수 있는 개수는 100 개 입니다. .
			"""
			
			# @ set reg 콜
			tmp_return = self.KIWOOM.set_real_register(tmp_scrno, self.STOCK_IN_ATTENTION.code, fid_list, opt_type)
			if tmp_return == 0: # 정상수신일 것이라고 기대 -> 이거 확인해야함사실...
				return True
				
			else: # 미정상수신이면
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				self.SCREEN_NO.disable_scrno_for_realtime(self.STOCK_IN_ATTENTION.code)
			
		except Exception as e:
			print('error in FUNC_STOCK_DATABASE__get_real_time :: ', e)
		
		
	def FUNC_STOCK_DATABASE__disable_real_time(self): # 실시간으로 종목 받아오는 부분 해제
		try:
			# @ 확인후 실시간 scrno 해제 부분
			tmp_scrno_used = self.SCREEN_NO.scrno_used_lookup_by_stock_code(self.STOCK_IN_ATTENTION.code)
			if tmp_scrno_used == None:# 사용하지 않고 있다 -> 잘못됨
				#self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				return None
			else:
				pass
			
			tmp_return = self.KIWOOM.set_real_remove(tmp_scrno_used, self.STOCK_IN_ATTENTION.code)
			if tmp_return == 0: # 정상수신 
				self.SCREEN_NO.disable_scrno_for_realtime(self.STOCK_IN_ATTENTION.code) # 해제 부분
				return True
			else:
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				pass
		
		except Exception as e:
			print('error in FUNC_STOCK_DATABASE__disable_real_time :: ', e)

	def FUNC_STOCK_DATABASE__get_all_codes_api(self): # 전체 종목 코드 가져오는 부분

		try:

			# @ counter update 하는 부분
			#self.counter_stock_api_list_update = self.counter_stock_api_list_update + 1
			# periodic 에서 담당

			print('코스피 - 전 종목 함수 진입...')
			self.STOCK_LIST__all_kospi = copy.deepcopy(self.KIWOOM.get_code_list_by_market(['10']))  # 코스피
			kospi_code_list = copy.deepcopy(self.STOCK_LIST__all_kospi)
			kospi_name_list = []

			print('코스닥 - 전 종목 함수 진입...')
			self.STOCK_LIST__all_kosdq = copy.deepcopy(self.KIWOOM.get_code_list_by_market(['0']))  # 코스피
			kosdoc_code_list = copy.deepcopy(self.STOCK_LIST__all_kosdq)
			kosdoc_name_list = []

			for code in kospi_code_list:
				name = self.KIWOOM.get_master_code_name([code])
				kospi_name_list.append(str(code) + " : " + str(name))

			for code in kosdoc_code_list:
				name = self.KIWOOM.get_master_code_name([code])
				if self.MINIMIZED_STOCK_LIST :
					if "FOCUS" in name or"KOSEF" in name or  "SMART" in name or "KINDEX" in name or "TIGER" in name or "ARIRANG" in name or  "KBSTAR" in name or "KODEX" in name or "HANARO" in name or (int(code[0]) >= 5 ):
						if code in self.MUST_WATCH_LIST:
							kosdoc_name_list.append(str(code) + " : " + (name))
					else:
						kosdoc_name_list.append(str(code) + " : " + (name))
				else:
					kosdoc_name_list.append(str(code) + " : " + (name))

			print(f'kosdoc_name_list : {kosdoc_name_list}')
			print(f'kosdoc_name_list length : {len(kosdoc_name_list)}')

			self.STOCK_LIST__for_total_display = copy.deepcopy(kospi_name_list + kosdoc_name_list) # list 합치기 -> 전체
			self.STOCK_LIST__for_total_display = copy.deepcopy(list(dict.fromkeys(self.STOCK_LIST__for_total_display))) # remove duplicates

			self.STOCK_DICTIONARY__name_to_code = {} # initialize before refresh
			for stock_info in self.STOCK_LIST__for_total_display :
				tmp_code = stock_info.split(' : ')[0].strip()
				tmp_name = stock_info.split(' : ')[1].strip()
				self.STOCK_DICTIONARY__name_to_code[str(tmp_name)] = str(tmp_code)

			self.STOCK_DICTIONARY__code_to_name = {} # initialize before refresh
			for stock_info in self.STOCK_LIST__for_total_display :
				tmp_code = stock_info.split(' : ')[0].strip()
				tmp_name = stock_info.split(' : ')[1].strip()
				self.STOCK_DICTIONARY__code_to_name[str(tmp_code)] = str(tmp_name)

			self.table_stock_list.clear()
			self.table_stock_list.addItems(self.STOCK_LIST__for_total_display) # 리스트를 pyqt5에 feed하기
			self.label_database_stock_list_num.setText('전체 '+ str(len(self.table_stock_list)) + ' 개수의 종목')

			# @ 피클 파일 생성부분
			self.folder_path = os.getcwd()
			self.pickle_folder_path = str(self.folder_path + '\\KIWOOM_API__STOCK_LIST_PICKLE').replace('/', '\\')
			self.pickle_path = str(self.pickle_folder_path + '\\stock_list_pickle.p')
			if os.path.isdir(self.pickle_folder_path):
				pass
			else:
				os.mkdir(self.pickle_folder_path)

			try:
				with open(self.pickle_path, 'wb') as file:
					pickle.dump(self.STOCK_LIST__for_total_display, file)
					print('pickle stock list successfully saved...!')
			except Exception as e:
				print('error in dumping pickle of total stock list : ', e)

			self.CHECK_2_FLAG_STOCK__get_all_stock_codes = True

		except Exception as e:
			print('error in FUNC_STOCK_DATABASE__get_all_codes_api :: ', e)
			traceback.print_exc()
			self.CHECK_2_FLAG_STOCK__get_all_stock_codes = False

	def func_DATABASE__get_latest_data(self, con_top, code, flag_imminent = False):
		"""
		flag_imminent  :  이것이 true이면 당장 fetch해야되는 것임 ( 이전 분봉 데이터 다 필요한 상태이므로! )

		:param con_top:
		:param code:
		:param flag_imminent:
		:return:
		"""

		head_string = 'SELECT * FROM '
		tmp_table_name_sql = "'" + str(code) + "'"
		flag_imminent_func = flag_imminent

		try:
			# df = pd.read_sql(head_string + tmp_table_name_sql, con_top, index_col=None)  # string 자체는 받아낼 수 있는듯
			# tmp_df = df.copy()  # 변하는 것 같아서!!
			tmp_df = pd.read_sql(head_string + tmp_table_name_sql, con_top, index_col=None)  # string 자체는 받아낼 수 있는듯
			# try:
			# 	df = None
			# 	del df
			# 	print('successful df release in func_DATABASE__get_latest_data')
			# except Exception as e:
			# 	print('error in del df func_DATABASE__get_latest_data :: ', e)

			df_date_list = copy.deepcopy(tmp_df['date'].tolist())
			if len(df_date_list) == 0: # 거래 안되는 애인 것 같은데 그냥 가져옴
				convert_date = 'dummy'

			else:
				latest_date_in_df = df_date_list[0] # 가장 최근 날짜를 가져온다  ex) 20200423153000
				convert_date = datetime.datetime.strptime(latest_date_in_df, "%Y%m%d%H%M%S")  # str date :: y/m/d/h/m


			today_date = datetime.datetime.now()  # 현재 날짜
			week_day_num = today_date.weekday()  # 0 : 월, 1 : 화, 2:수, 3:목, 4:금, 5:토, 6:일
			#print('week_day_num :: ', week_day_num)
			
			if self.STATE_TIME.stage == "개장중" and (week_day_num != 5 and week_day_num != 6):
				flag_imminent_func = False
			else:
				pass

			if flag_imminent_func: # 장 open 중일 때 가져오면
				if len(df_date_list) != 0: # sqlite db 데이터 있는 애면
					duration_calc_in_date_secs = (today_date - convert_date).total_seconds()
					min_duration = divmod(duration_calc_in_date_secs, 60)[0]  # min duration calc
				else: # 데이터 없던 애면
					return True
				if min_duration >= 1 :
					return True
				else:
					return False

			else: #3:30 ~ 12:00 / 12:00 ~ 9:00로 구분  -- 장 마감시 + 주말 시간 재설정 + db기록
				flag_change_date = False # 이거 올려서 금요일 개장전 바꿔줌
				if week_day_num == 5: # 토요일
					today_date = today_date - datetime.timedelta(days=1)
					today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0)
					flag_change_date = True
				elif week_day_num == 6: #일요일
					today_date = today_date - datetime.timedelta(days=2)
					today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0)
					flag_change_date = True
				else:
					pass
				# ->  월 00:00 ~ 금 24:00 까지 재조정, 토/일-> 금 15:30

				today_15_30 = self.STATE_TIME.timesec__15_30
				today_24    = self.STATE_TIME.timesec__24
				tomorrow_9 =  self.STATE_TIME.timesec__9

				today_sec = (((today_date.hour) * 60) * 60) + (today_date.minute * 60) + today_date.second
				
				if today_sec < tomorrow_9 : # 개장전
					if str(code) in self.SQLITE_DICTIONARY__db_update_date: # 이거 자정 전과 이후 또 나누어야함
						db_latest_time = datetime.datetime.strptime(self.SQLITE_DICTIONARY__db_update_date[str(code)], "%Y%m%d%H%M%S")
					else: # db에 없으면 -> sqlite db에서 판단
						if len(df_date_list) == 0:  # 빈 데이터프레임 -> 거래 안됨
							return True
						else:
							db_latest_time = convert_date

					if week_day_num == 0: # 월요일 -> 전주 금요일 3시 30분 이상에 포함되는 log있으면 됨
						tmp_today_date = (today_date - datetime.timedelta(days=3)).replace(hour= 15, minute= 30, second=0, microsecond=0) # 금요일
						if db_latest_time >= tmp_today_date:
							return False
						else:
							return True
					else: # 주중 월요일 아닌 다른 요일
						if flag_change_date == True: # 토 / 일은 금요일로 set되었는데 다시 금요일에서 -1 할 수 없음
							back_date = 0
						else:
							back_date = 1
						tmp_today_date = (today_date - datetime.timedelta(days=back_date)).replace(hour=15, minute=30,second=0,microsecond=0)  # 하루전
						if db_latest_time >= tmp_today_date:
							return False
						else:
							return True

				elif (today_sec > today_15_30 and today_sec < today_24) or (flag_change_date == True): 
					# 개장후(토/일은 금요일 15:30으로 변경), 바뀌면 금요일 장이후 취급
					if str(code) in self.SQLITE_DICTIONARY__db_update_date: # 이거 자정 전과 이후 또 나누어야함
						db_latest_time = datetime.datetime.strptime(self.SQLITE_DICTIONARY__db_update_date[str(code)], "%Y%m%d%H%M%S")
					else: # db에 없으면 -> sqlite db에서 판단
						if len(df_date_list) == 0:  # 빈 데이터프레임 -> 거래 안됨
							return True
						else:
							db_latest_time = convert_date

					tmp_today_date = today_date.replace(hour= 15, minute= 30, second=0, microsecond=0) # 금요일
					if db_latest_time >= tmp_today_date:
						return False
					else:
						return True
				
				#elif today_sec <= today_15_30 and today_sec >= timesec__9 : # 토 / 일 개장시간일 시
			try:
				tmp_df = None
				del tmp_df
				print('successful tmp_df release in func_DATABASE__get_latest_data')
			except Exception as e:
				print('error in del tmp_df func_DATABASE__get_latest_data :: ', e)

		except Exception as e:
			print('error in func_DATABASE__get_latest_data - ', e)
			traceback.print_exc()
			return True # 거래가 안되어서 공백 -> 혹시 몰라서 True해서 fetching은 해놓음
	
	# def TH__FUNC_STOCK_DATABASE_SQLITE__create(self): # create_databse 쓰레드 open할 부분
	# 	print('initiate thread - FUNC_STOCK_DATABASE_SQLITE__create')
	# 	if not self.thread_wrapper.is_run :
	# 		self.thread_wrapper.is_run = True
	# 		self.thread_wrapper.thread_set_fucntion(self.FUNC_STOCK_DATABASE_SQLITE__create)
	# 		self.thread_wrapper.start()


	def FUNC_STOCK_DATABASE_SQLITE__create(self):

		"""
		1) 여러개를 신청하여도, request끼리 term을 정해서 날리는 것 필요함
		2) 이미 있는 database와 비교하여야 함
		3) 계속 저장해야됨 -> 전체를 ram에 로드하고 비교한다음에 지속 저장?? 올릴 수나 있는지도 모르겠다.
		:return: 
		"""
		self.SQLITE_FLAG__database_creation_in_progress = True # 같은 함수 사용방식 변경하려고 설정(class 내부에서 작동방식 변경)
		self.SQLITE_FLAG__database_is_stalled = False # 멈춤 flag 다시 사용하기 위해 초기화

		tmp_list_all_codes = []
		tmp_list_db_codes = []
		print('self.SQLITE_FLAG__latest_database_data_checked value :: ', self.SQLITE_FLAG__latest_database_data_checked)
		if not self.SQLITE_FLAG__latest_database_data_checked :
			#self.FUNC_STOCK_DICTIONARY__parse_from_sqlite(string_val='multi')  # 첫 update
			self.tmp_list_db_not_yet_updated = []

		# @ db 업데이트 시점 pickle path 받아오고 ram에 올린다. dictionary 형태
		try:
			self.func_SQLITE_PICKLE__create_db_update_date()
			if os.path.isfile(self.SQLITE_PICKLE__path_for_db_update_date): # 위치 존재하면
				with open(self.SQLITE_PICKLE__path_for_db_update_date,'rb') as file: # pickel path에서 읽어온다
					self.SQLITE_DICTIONARY__db_update_date = copy.deepcopy(pickle.load(file))
					print('successful deep copy!')

				print(' self.SQLITE_DICTIONARY__db_update_date successfully loaded!!')
				print('length of hash in self.SQLITE_DICTIONARY__db_update_date : ', len(list(self.SQLITE_DICTIONARY__db_update_date.keys())))
				self.FUNC_PYQT__rest_timer(1)
			else:
				if len(list(self.SQLITE_DICTIONARY__db_update_date.keys())) == 0:
					self.SQLITE_DICTIONARY__db_update_date = {}  # very initial db dictionary...!

		except Exception as e:
			print("error in FUNC_STOCK_DATABASE_SQLITE__create - loading db pickle, doesn't exist :: ", e)
			print(' self.SQLITE_DICTIONARY__db_update_date not - successfully loaded!!')
			if len(list(self.SQLITE_DICTIONARY__db_update_date.keys())) == 0:
				self.SQLITE_DICTIONARY__db_update_date = {} # very initial db dictionary...!
				self.FUNC_PYQT__rest_timer(1)


		if self.STOCK_LIST__for_total_display: # 전체 리스트 받아온게 있는지 체크
			for stock in self.STOCK_LIST__for_total_display:
				tmp_code = stock.split(':')[0].strip()
				tmp_list_all_codes.append(str(tmp_code))

			### 전체 종목 코드 가지고 있음 현재 이시점에서!
			# 1) check existing database  -> 밑의 함수에서... sq 쓰는법 더 알아봐야 할듯
			# 2) from that database, start retrieving data

			try:
				if len(self.SQLITE_LIST__folder_sub_file_path) != 0 : # 공 len이 아니라면
					print('first path in FUNC_STOCK_DATABASE_SQLITE__create FE')
					self.SQLITE__con_top = sqlite3.connect(self.SQLITE_LIST__folder_sub_file_path[0])
					self.SQLITE__cur_top = self.SQLITE__con_top.cursor()
					tmp_list_db_codes_obj = self.SQLITE__cur_top.execute("SELECT name FROM sqlite_master WHERE type='table';") # fetch all table codes from DB, obj 생성됨
					tmp_tmp_list_db_codes = copy.deepcopy(tmp_list_db_codes_obj.fetchall())

					for code_item in tmp_tmp_list_db_codes:
						if code_item[0] in self.STOCK_DICTIONARY__code_to_name: # 현재 종목 존재하는 경우에만
							tmp_list_db_codes.append(code_item[0])
						else:
							tmp_list_db_codes.append(code_item[0])
					self.STOCK_IN_ATTENTION.stock_list_for_sqlite = copy.deepcopy(tmp_list_db_codes)  # 여기다 copy해서 넣음

					print('length of tmp_list_db_codes :: ', len(tmp_list_db_codes))

					# @ initialize SQLITE__cur_top / con_top
					try:
						self.SQLITE__cur_top.close() #이게 먼저 닫혀야 함
						self.SQLITE__con_top.close()
						self.SQLITE__cur_top = None
						self.SQLITE__con_top = None

						del self.SQLITE__con_top, self.SQLITE__cur_top
						
					except Exception as e:
						print('error in FUNC_STOCK_DATABASE_SQLITE__create - deleting connection sqlite :: ', e)
						traceback.print_exc()
					
					# @  del
					try:
						tmp_list_db_codes_obj = None
						del tmp_list_db_codes_obj
					except Exception as e:
						print('error in del of variables in FUNC_STOCK_DATABASE_SQLITE__create (1) :: ', e)


					# @ DATABASE 에서 전체 table name들 읽어오기
					if len(tmp_list_db_codes) != 0 : # DB에서 table name 존재하는 경우
						print('db already existing... ')
					else:
						print('db is NULL value...')
				else:
					print('second path in FUNC_STOCK_DATABASE_SQLITE__create FE')

			except Exception as e:
				# tmp_list_db_codes is null or DB not existing
				traceback.print_exc()
				print('error in FUNC_STOCK_DATABASE_SQLITE__create - ', e)

			# @ reconnect for self.func_DATABASE__get_latest_data function
			try:
				self.SQLITE__con_top = sqlite3.connect(self.SQLITE_LIST__folder_sub_file_path[0])
			except Exception as e:
				print('self.SQLITE__con_top error _1 :: ', e)

			start_time_fetch_database = time.time()
			if not(self.SQLITE_FLAG__latest_database_data_checked) : # first update was finished already
				for codes in tmp_list_all_codes: # stock list from API
					if len(tmp_list_db_codes) != 0 : # DB list from local
						if codes in tmp_list_db_codes: # 해당 api 코드가 db에 있다면
							print('codes in tmp_list_db_codes :: ', codes)
							if self.func_DATABASE__get_latest_data(self.SQLITE__con_top, codes,flag_imminent = False): # 시간을 판별하여 가져올 여부 선택
								# 여기서 밑에 작업 수행 구현해야 함
								"""
								tmp_list_db_not_yet_updated.append(str(codes))
								"""
								self.tmp_list_db_not_yet_updated.append(str(codes))

							else: # 가져올 필요 없음
								pass
						else: # DB와 비교시 db에는 없는데, api에는 있음
							self.tmp_list_db_not_yet_updated.append(str(codes))

					else: # DB list from local - doesn't exist
						self.tmp_list_db_not_yet_updated.append(str(codes))
			else: # flag for aquiring latest database is True -> no need to check!
				print('self.tmp_list_db_not_yet_updated was already updated... skipping process!')

			# @ remove duplicates
			self.tmp_list_db_not_yet_updated = copy.deepcopy(list(dict.fromkeys(self.tmp_list_db_not_yet_updated)))

			# @ set the flag inside
			if len(self.tmp_list_db_not_yet_updated) > 0:
				self.CHECK_2_FLAG_SQLITE__first_database_create_success = False

			self.SQLITE_FLAG__latest_database_data_checked = True # DB 탐색 끝났으므로 올려줌
			try:
				self.SQLITE__con_top.close()
				self.SQLITE__con_top = None
				del self.SQLITE__con_top
			except Exception as e :
				print('self.SQLITE__con_top error _2 :: ', e)
			total_elapsed_time = time.time() - start_time_fetch_database
			print('func_DATABASE__get_latest_data elapsed time - ', datetime.timedelta(seconds = total_elapsed_time) )


			print('☆'*20)
			print('length of tmp_list_db_not_yet_updated :: ',len(self.tmp_list_db_not_yet_updated))
			print('\n'*2)


			if len(self.tmp_list_db_not_yet_updated) != 0: # fetch할 stock data 남아있다면!
				for codes in self.tmp_list_db_not_yet_updated:
					if self.ERROR_COUNTER_BE__request_num >= self.REQUEST_MAX_NUM:
						print('exiting current creation of database')
						break
					else:
						print('RT code of following : ', codes)
						print('    - process following : ', codes)
						#self.func_STOCK_DATABASE__get_data_from_api(codes, consecutive_request=False)

						"""
						for key, value in kwargs:
						if key == 'code':
							code = value
						elif key == 'kiwoom_class':
							kiwoom_class = value
						elif key == 'scrno':
							scrno = value
						elif key == 'timestage':
							timestage = value
						elif key == 'weekdaynum':
							weekdaynum = value
						elif key == 'child_conn':
							child_conn = value
						elif key == 'file_path':
							file_path = value
						
						return connection (stalled bool, request num)
						->다음 업데이트 해주기
						self.SQLITE_LIST__stocks_already_updated.append(code) # 기록해놓음 -> 이미 업데이트 된 항목
						self.SQLITE_DICTIONARY__db_update_date[str(code)] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

						(fe_be_request_num, max_req_num) = child_conn.recv()
						https://www.kite.com/python/examples/3200/multiprocessing-communicate-with-a-child-process-using-a-%60pipe%60
						https://stackoverflow.com/questions/55110733/python-multiprocessing-pipe-communication-between-processes


						https://stackoverflow.com/questions/2774585/child-processes-created-with-python-multiprocessing-module-wont-print
						-> print 보고싶으면
						"""
						self.func_STOCK_DATABASE__get_data_from_api_MP(codes)

			else:
				print('finished fetching data in self.tmp_list_db_not_yet_updated')
		else:
			print('empty self.STOCK_LIST__for_total_display in FE...')

		if self.SQLITE_FLAG__database_is_stalled == True: # 현재 database 작업 그만두어야 한다는 flag
			# backend 재시작
			#self.func_SQLITE_LIST__for_path_to_db()
			print('update db files that were updated...')
			tmp_updated_list_for_db_not_yet_updated = [] # tmp로 완료항목 제거하고 다시 not_yet_updated list에 copy 해주기 위해
			for codes in self.tmp_list_db_not_yet_updated : # 가져와야 하는 항목중에
				if codes not in self.SQLITE_LIST__stocks_already_updated : # 이미 완료된 아이가 있다면
					tmp_updated_list_for_db_not_yet_updated.append(codes) # tmp list에 담고 그것을 copy해줌
			print('self.SQLITE_LIST__stocks_already_updated :: ', self.SQLITE_LIST__stocks_already_updated)

			# @ remove duplicates
			tmp_updated_list_for_db_not_yet_updated = copy.deepcopy(list(dict.fromkeys(tmp_updated_list_for_db_not_yet_updated)))

			self.tmp_list_db_not_yet_updated = copy.deepcopy(tmp_updated_list_for_db_not_yet_updated)
			print('length of self.tmp_list_db_not_yet_updated after update! :: ', len(self.tmp_list_db_not_yet_updated))
			self.FUNC__restart_api()

		if self.SQLITE_FLAG__database_is_stalled == True: # getting 중지 flag 인지
			print('FUNC_STOCK_DATABASE_SQLITE__create :: ended with stall_true...')
			return None # 함수 종료
		
		else: # 문제없이 완료하였음
			print('FUNC_STOCK_DATABASE_SQLITE__create :: ended without problem...')
			self.SQLITE_FLAG__database_creation_in_progress = False
			self.SQLITE_FLAG__latest_database_data_checked = False
			self.tmp_list_db_not_yet_updated = [] # reset
			self.SQLITE_LIST__stocks_already_updated = [] # reset
			self.CHECK_2_FLAG_SQLITE__first_database_create_success = True
			
			# @ minute DICTIONARY 관리
			#--------------------------------------------------------------------
			self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN = copy.deepcopy({

				'STOCK_MIN': {},
				'SQLITE': {},
				'FILTER': {},
				'BUDGET': {},
				'OWNING': {}
			})  # SQLITE로 boolian True일 때 만 작업함 # SQLITE로 boolian True일 때 만 작업함 -> 여기서 reset해주고 다음 라인에서 다시 작업
			try:
				self.SQLITE__con_top.close()
				self.SQLITE__con_top = None
				del self.SQLITE__con_top
			except Exception as e:
				print('error in FUNC_STOCK_DATABASE_SQLITE__create - failed to close sqlite connection :: ',e)
				traceback.print_exc()

			try:
				print(f'setting up new hard gc threshold')
				print(f'here(1) : {gc.get_count()}')
				gc.set_threshold(*self.GC_THRESHOLD__new)
				print(f'checking for new applied gc threshold : {gc.get_threshold()} \nold threshold : {self.GC_THRESHOLD__old}')
				gc_start = time.time()
				tmp_gc_return = gc.collect()
				gc_total_elapsed_time = datetime.timedelta(seconds=(time.time() - gc_start))
				print(f'here(2) : {gc.get_count()}')
				print('gc garbage collect in FUNC_STOCK_DATABASE_SQLITE__create done...! - with number of unreacables ::',
					  tmp_gc_return)
				print('total elapsed time in gc :: ', str(gc_total_elapsed_time))
			except Exception as e:
				print('error in garbage collection before memory load operation in FUNC_STOCK_DATABASE_SQLITE__create :: ', e)

			# @ Normal 방법
			self.FUNC_STOCK_DICTIONARY__parse_from_sqlite()
			#self.func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup()

			try:
				print(f'(2) setting up new hard gc threshold ')
				print(f'(2) here (1) : {gc.get_count()}')
				gc.set_threshold(*self.GC_THRESHOLD__new)
				print(f'(2) checking for new applied gc threshold  : {gc.get_threshold()} \nold threshold : {self.GC_THRESHOLD__old}')
				gc_start = time.time()
				tmp_gc_return = gc.collect()
				gc_total_elapsed_time = datetime.timedelta(seconds=(time.time() - gc_start))
				print(f'(2) here (2) : {gc.get_count()}')
				print('(2) gc garbage collect in FUNC_STOCK_DATABASE_SQLITE__create done...! - with number of unreacables ::',
					  tmp_gc_return)
				print('(2) total elapsed time in gc :: ', str(gc_total_elapsed_time))
			except Exception as e:
				print('error in garbage collection before memory load operation in FUNC_STOCK_DATABASE_SQLITE__create :: ', e)

			# @ return to old gc
			print(f'returning to old gc threshold')
			gc.set_threshold(*self.GC_THRESHOLD__old)
			print(f'checking for new applied gc threshold : {gc.get_threshold()} \nold threshold : {self.GC_THRESHOLD__old}')

			# @ multiprocessing 다른 방법
			# https://stackoverflow.com/questions/39100971/how-do-i-release-memory-used-by-a-pandas-dataframe

			# @ multiprocess
			# proc_1 = mp.Process(target=self.FUNC_STOCK_DICTIONARY__parse_from_sqlite)
			# proc_1.start()
			# proc_1.join()

			# proc_2 = mp.Process(target=self.func_STOCK_DICTIONARY_MINUTE_DATA__sub_cleanup)
			# proc_2.start()
			# proc_2.join()
			#--------------------------------------------------------------------
			
			try: # save new pickle db date that was updated..
				with open(self.SQLITE_PICKLE__path_for_db_update_date, 'wb') as file:
					pickle.dump(self.SQLITE_DICTIONARY__db_update_date, file)
					print('successfully save new db date pickle file...')
			except Exception as e:
				print('error in FUNC_STOCK_DATABASE_SQLITE__create - failed to save db pickle date data :: ',e)

			pass

	#@return_status_msg_setter
	def func_STOCK_DATABASE__get_data_from_api(self, code, consecutive_request = False):
		try :
			##########################
			"""
			1) 1분봉 데이터 가져오는 부분
			2) consecutive_request :  False로 해서 900 분 가져오기
			"""
			##########################


			"""
			https://wikidocs.net/5756
			"""

			# @ 함수 안 ram 변수 세팅
			tick_range = 1 # 이게 뭐하는건지 모르겠다
			#lookup = 0  # look_type  =>  0 : 단순조회 , 1 : 연속조회
			input_dict ={} # for set input value function in BACKEND


			# @ 종목의 기본 정보들
			base_date = datetime.datetime.today().strftime('%Y%m%d')
			#base_date = '20200419'
			input_dict['종목코드'] = code
			print('1')
			#tmp_stock_name = str(self.KIWOOM.get_master_code_name(code)).strip()
			print('2')
			input_dict['틱범위'] = tick_range
			input_dict['기준일자'] = base_date
			input_dict['수정주가구분'] = 1

			self.KIWOOM.set_input_value(input_dict)
			print('3')
			tmp_db_return = self.KIWOOM.comm_rq_data("opt10080_req", "opt10080", 0, self.SCREEN_NO.opt10080_req, self.STATE_TIME.stage, self.STATE_TIME.weekday_num) # look_type  =>  0 : 단순조회 , 2 : 연속조회



			print('4')
			self.ERROR_COUNTER_BE__request_num = self.ERROR_COUNTER_BE__request_num + 1
			if self.ERROR_COUNTER_BE__request_num >= self.REQUEST_MAX_NUM:
				print('exiting backend current creation of database - 1st')
				self.SQLITE_FLAG__database_is_stalled = True
				return None

			if tmp_db_return != 0: # 에러값, 정상은 0
				#print('tmp_db_return :: ', tmp_db_return)
				#self.FUNC_PYQT__rest_timer(5)
				return None

			ohlcv = copy.deepcopy(self.KIWOOM.latest_tr_data)

			tmp_logic_value_consecutive_data = None # 이걸로 조절할 것임, 900개 이상 추가 데이터 받을지에 대해서
			if consecutive_request == False:
				tmp_logic_value_consecutive_data = False
			else:
				tmp_logic_value_consecutive_data = self.KIWOOM.is_tr_data_remained


			while tmp_logic_value_consecutive_data == True:
				self.KIWOOM.set_input_value(input_dict)
				self.KIWOOM.comm_rq_data("opt10080_req", "opt10080", 2, self.SCREEN_NO.opt10080_req, self.STATE_TIME.stage, self.STATE_TIME.weekday_num)
				self.ERROR_COUNTER_BE__request_num = self.ERROR_COUNTER_BE__request_num + 1
				if self.ERROR_COUNTER_BE__request_num >= self.REQUEST_MAX_NUM:
					print('exiting backend current creation of database - 2nd')
					self.SQLITE_FLAG__database_is_stalled = True
					break
				for key, val in self.KIWOOM.latest_tr_data.items():
					ohlcv[key][-1:] = val

			if self.SQLITE_FLAG__database_is_stalled == False:
				#file_name = str(tmp_stock_name) + '__' + str(code) + '__'
				#file_name = 'SINGLE_DB'
				if type(ohlcv) == type(dict()) and ohlcv : # 비지 않았고 dictionary 타입이다.
					self.SQLITE__con_top = sqlite3.connect(self.SQLITE_LIST__folder_sub_file_path[0])
					# head_string = 'SELECT * FROM '
					# tmp_table_name_sql = "'" + str(code) + "'"

					#df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'],index=ohlcv['date'])
					df = pd.DataFrame(ohlcv, columns=['date','open', 'high', 'low', 'close', 'volume'])
					print(df.head())
					#df.set_index('date', inplace=True)
					#con = sqlite3.connect("./DATABASE/" + str(file_name) +".db")
					#df.to_sql(code, self.SQLITE__con_top, if_exists='replace', index=False, index_label='date') # chunksize 필요한가?? 1000으로 한 경우도 있음
					df.to_sql(code, self.SQLITE__con_top, if_exists='replace', index=False)

					# @ db dictionary update
					self.SQLITE_LIST__stocks_already_updated.append(code) # 기록해놓음 -> 이미 업데이트 된 항목
					self.SQLITE_DICTIONARY__db_update_date[str(code)] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
					self.SQLITE__con_top.close()

					self.SQLITE__con_top = None
					self.KIWOOM.latest_tr_data = None
					try:
						
						ohlcv = None
						df = None
						del ohlcv
						del df
						del self.SQLITE__con_top
						del self.KIWOOM.latest_tr_data

						try:
							gc.collect()
						except Exception as e:
							print('error in gc garbage collect in func_STOCK_DATABASE__get_data_from_api (1) :: ', e)

					except Exception as e:
						print('error in func_STOCK_DATABASE__get_data_from_api del (1) :: ', e)

				else:
					try:
						ohlcv = None
						df = None
						del ohlcv
						del df

						try:
							gc.collect()
						except Exception as e:
							print('error in gc garbage collect in func_STOCK_DATABASE__get_data_from_api (2) :: ', e)

					except Exception as e:
						print('error in func_STOCK_DATABASE__get_data_from_api del (2) :: ', e)
					return None # 그냥 끝내기


		except Exception as e:
			print('    @@**$$backend error - ', e)
			traceback.print_exc()
			self.FUNC_PYQT__rest_timer(5) # 시간초 대기



	def func_STOCK_DATABASE__get_data_from_api_MP(self, codes):
		try :
			##########################
			"""
			1) 1분봉 데이터 가져오는 부분
			2) consecutive_request :  False로 해서 900 분 가져오기
			"""
			##########################


			"""
			https://wikidocs.net/5756
			"""

			"""
			tmp_dict = {'code': codes, 'kiwoom_class': self.KIWOOM, 'scrno': self.SCREEN_NO.opt10080_req,
						'timestage': self.STATE_TIME.stage, 'weekdaynum': self.STATE_TIME.weekday_num,
						'child_conn': child_conn, 'file_path': self.SQLITE_LIST__folder_sub_file_path}
			"""




			# @ 함수 안 ram 변수 세팅
			tick_range = 1 # 이게 뭐하는건지 모르겠다
			#lookup = 0  # look_type  =>  0 : 단순조회 , 1 : 연속조회
			input_dict ={} # for set input value function in BACKEND


			# @ 종목의 기본 정보들
			base_date = datetime.datetime.today().strftime('%Y%m%d')
			#base_date = '20200419'
			input_dict['종목코드'] = codes
			print('1')
			#tmp_stock_name = str(self.KIWOOM.get_master_code_name(code)).strip()
			print('2')
			input_dict['틱범위'] = tick_range
			input_dict['기준일자'] = base_date
			input_dict['수정주가구분'] = 1

			self.KIWOOM.set_input_value(input_dict)
			print('3')
			tmp_db_return = self.KIWOOM.comm_rq_data("opt10080_req", "opt10080", 0, self.SCREEN_NO.opt10080_req, self.STATE_TIME.stage, self.STATE_TIME.weekday_num) # look_type  =>  0 : 단순조회 , 2 : 연속조회

			

			print('4')
			self.ERROR_COUNTER_BE__request_num = self.ERROR_COUNTER_BE__request_num + 1
			if self.ERROR_COUNTER_BE__request_num >= self.REQUEST_MAX_NUM:
				print('exiting backend current creation of database - 1st')
				self.SQLITE_FLAG__database_is_stalled = True
				return None

			if tmp_db_return != 0: # 에러값, 정상은 0
				#print('tmp_db_return :: ', tmp_db_return)
				#self.FUNC_PYQT__rest_timer(5)
				return None

			ohlcv = copy.deepcopy(self.KIWOOM.latest_tr_data)

			if self.SQLITE_FLAG__database_is_stalled == False:
				#file_name = str(tmp_stock_name) + '__' + str(code) + '__'
				#file_name = 'SINGLE_DB'
				if type(ohlcv) == type(dict()) and ohlcv : # 비지 않았고 dictionary 타입이다.
					mp_process = mp.Process(target=func_sub__sqlite, args=[ohlcv, self.SQLITE_LIST__folder_sub_file_path, codes])
					mp_process.start()
					mp_process.join()
					self.SQLITE_LIST__stocks_already_updated.append(codes)
					self.SQLITE_DICTIONARY__db_update_date[str(codes)] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


		except Exception as e:
			print('error in func_STOCK_DATABASE__get_data_from_api del (2) :: ', e)
			traceback.print_exc()
			return None # 그냥 끝내기


	def FUNC_PYQT__rest_timer(self, secs):
		"""
		지정된 시간동안 병렬로 execute로 gui동작 허용하면서 시간 멈춤
		"""
		loop = QEventLoop()
		QTimer.singleShot(secs * 1000, loop.quit) # ms
		loop.exec_()



	def pyq_exec(self, func_name): # 함수 명으로 함수 수행하는 함수
		func = getattr(pyq_object, func_name)
		func(self)
		
	#===============================================================================================
	#===============================================================================================
	"""
	양호준
	used variables from fe
	-----------------------------------------

	self.BALANCE_DICTIONARY__for_normal_update = {'balance':None, 'date':None}

	self.STOCK_IN_ATTENTION

	self.ERROR_COUNTER_BE__request_num = 0
	self.ERROR_DICTIONARY__backend_and_critical={
				'error_backend'  : 0,
				'error_critical' : 0
			}

	self.STOCK_DICTIONARY__name_to_code = {} # 이름에 대한 code값 (사용자 편의)
	self.STOCK_DICTIONARY__code_to_name = {}


	self.STOCK_DICTIONARY_NAMES__owning_stocks = {} # 보유 주식 dictionary
	self.STOCK_DICTIONARY_NAMES__basic_info = {} -> {'result' : tmp_flag_result, 'master_construction' : data_1, 'master_stock_state' : data_2}
	self.STOCK_DICTIONARY_NAMES__additional_info_tr = {} # tr에서 구할 수 있는 더 세밀한 정보
	self.STOCK_DICTIONARY_NAMES__unmet_order = { } # unmet order 나올 때 마다 담아놓을 부분

	self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC = { } # BE 단에서 받아온 realtime 데이터 second :: stock_code -> datetime_data -> price, volume
	self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN = {

				'STOCK_MIN' : {},
				'SQLITE' : {},
				'FILTER' : {},
				'BUDGET' : {},
				'OWNING' : {}
			} # SQLITE로 boolian True일 때 만 작업함
		=> self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML 주소에 dump 뜨면 됨

	self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN = = {'prediction':{ㄱ}, 'trade':{ㄴ}, 'date':''} , ㄱ/ㄴ 내부는 stock_code -> 1depth hash
	self.STOCK_DICTIONARY_FROM_ML__path_for_32bit

	self.STOCK_DICTIONARY__article_dump = {} # 기사 article 엎어치는 부분
	self.STOCK_DICTIONARY_NAME__article_result

	STOCK_AT_MAX_NUM # 맥시멈으로 트레이딩 할 주식 수!

	"""
	"""
	0 : 'WORKING',
	1 : 'P32_OUTPUT',
	2 : 'P32_SEND_READY',
	3 : 'P32_READ_RECIEVE',
	4 : 'P64_READ_READY',
	5 : 'P64_INPUT',
	6 : 'P64_SEND_RECIEVE'
	"""
	def AT_INIT_EVERYTHING(self):
		"""
		AT 용 공통변수 초기화 / 선언 해주는 부분
		"""
		try:
			gc_start = time.time()
			tmp_gc_return = gc.collect()
			gc_total_elapsed_time =  datetime.timedelta(seconds = (time.time() - gc_start))
			print('gc garbage collect in AT_INIT_EVERYTHING done...! - with number of unreacables ::', tmp_gc_return)
			print('total elapsed time in gc :: ', str(gc_total_elapsed_time))
		except Exception as e:
			print('error in gc garbage collect in AT_INIT_EVERYTHING:: ',e)

		print('AT has been initialized...!')
		self.TOTAL_DICT = {}
		self.COMM_32 = Bit_32(self.TEST)
		#self.TARGET_PROFIT = float(self.target_profit_rate)
		self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE = False # 처음 filter stage에서 수행되면 true로 올려서 자동 트레이딩 끝나면 init위한 flag	
		self.AT_FLAG__very_first_init_func_called = True
		self.AT_TUPLE__profit_record_watch = []
		self.AT_TUPLE__profit_record_trans = []

		# @ Single_stock 의 class variable 세팅
		self.AT_FUNC__class_variable_func_assign()

	
	def AT_FUNC__class_variable_func_assign(self):
		"""
		필요한 변수와 func class 자체에 할당
		"""
		Single_stock.SINGLE_CLASS_VAR__tuple_list_watch = self.AT_TUPLE__profit_record_watch
		Single_stock.SINGLE_CLASS_VAR__tuple_list_trans = self.AT_TUPLE__profit_record_trans
		Single_stock.SINGLE_CLASS_VAR__profit_rate_micro = self.STOCK_TARGET_MICRO_PROFIT
		Single_stock.SINGLE_CLASS_VAR__profit_rate_overall = self.STOCK_TARGET_PROFIT
		Single_stock.SINGLE_CLASS_VAR__fe_article = self.STOCK_DICTIONARY_NAME__article_dump 
		# 이거 고쳐져야 한다.(64에서 필요정보만 가져오는 형식)

		Single_stock.SINGLE_CLASS_FUNC_del_scrno = self.FUNC_STOCK_DATABASE__disable_real_time
		Single_stock.SINGLE_CLASS_FUNC_apply_scrno = self.FUNC_STOCK_DATABASE__get_real_time
		#Single_stock.SINGLE_CLASS_FUNC_db_consistancy = self.AT_func__sub_sqlite_to_dict_date_check
		Single_stock.SINGLE_CLASS_POINTER__STOCK_IN_ATTENTION = self.STOCK_IN_ATTENTION
	
	def AT_FUNC_PACKAGE__wake_up(self):
		try:
			# @ set the stage
			self.COMM_32.stage = 0

			tmp_list_comm_return = []
			tmp_flag_comm_return = True
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'WORKING' : self.AT_FUNC__wake_up})) # article 포함되어있음 이부분
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_OUTPUT' :self.AT_FUNC__pickle_dump_for__ML}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_SEND_READY' : None})) # P32_SEND_READY
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_READ_RECIEVE' : None})) # P32_READ_RECIEVE
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_READ_READY' : None}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_INPUT' : self.AT_FUNC__pickle_dump_from__ML}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_SEND_RECIEVE' : None}))
			
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'WORKING+2': self.AT_FUNC__prediction_update__ONGOING, 'WORKING+3':self.AT_FUNC__update_profit_rank__ONGOING, 'WORKING+4': self.AT_FUNC__allocate_budget_for_each, 'WORKING+6':self.AT_FUNC__del_min_stock_data_after_init}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_OUTPUT' :self.AT_FUNC__pickle_dump_for__ML}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_SEND_READY' : None})) # P32_SEND_READY
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_READ_RECIEVE' : None})) # P32_READ_RECIEVE
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_READ_READY' : None}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_INPUT' : self.AT_FUNC__pickle_dump_from__ML})) # sell hold buy 실제로 받아옴
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_SEND_RECIEVE' : None}))

			for returns in tmp_list_comm_return:
				if returns == 'COMM_FAIL':
					tmp_flag_comm_return = False
					break

			if tmp_flag_comm_return == True:
				print('AT_FUNC_PACKAGE__wake_up passed!!!')
				self.CHECK_FILTER_FLAG__at_initialize = True # 완료시 FE 단 flag 올림
				self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE = True # AT 단 첫 init flag 올림
			else:
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
			
		except Exception as e:
			print('error in AT_FUNC_PACKAGE__wake_up :: ', e)
			self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
			self.CHECK_FILTER_FLAG__at_initialize = False
			self.FLAG__FIRST_TIME_REACHED_FILTER_STAGE = False # AT 단 첫 init flag 다시 삭제
			#traceback.print_exc()
			traceback.print_exc()
			

	def AT_FUNC_PACKAGE__on_the_run_PERIODIC(self):
		try:
			pass
			"""
			1) set stage to WORKING
			"""
			# @ set the stage
			self.COMM_32.stage = 0

			tmp_list_comm_return = []
			tmp_flag_comm_return = True

			# 전단계에서 받아온 ML의 주문서 존재
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'WORKING+1':self.AT_FUNC__article_bool__ONGOING, 'WORKING+4': self.AT_FUNC__prediction_update__ONGOING, 'WORKING+5':self.AT_FUNC__update_profit_rank__ONGOING, 'WORKING+6': self.AT_FUNC__allocate_budget_for_each, 'WORKING+9':self.AT_FUNC__automated_transaction, 'WORKING+7':self.AT_FUNC__sell_stocks_with_sell_flag, 'WORKING+3' : self.AT_FUNC__update_TOTAL_DICT__ONGOING})) # article 포함되어있음 이부분
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_OUTPUT' :self.AT_FUNC__pickle_dump_for__ML}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_SEND_READY' : None})) # P32_SEND_READY
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P32_READ_RECIEVE' : None})) # P32_READ_RECIEVE
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_READ_READY' : None}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_INPUT' : self.AT_FUNC__pickle_dump_from__ML}))
			tmp_list_comm_return.append(self.COMM_32.func_proceed(**{'P64_SEND_RECIEVE' : None}))

			for returns in tmp_list_comm_return:
				if returns == 'COMM_FAIL':
					tmp_flag_comm_return = False
					break

			if tmp_flag_comm_return == True:
				print('AT_FUNC_PACKAGE__on_the_run_PERIODIC passed!!!')
			else:
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1

		except Exception as e:
			print('error in AT_FUNC_PACKAGE__on_the_run_PERIODIC :: ', e)
			traceback.print_exc()
	
	def AT_FUNC__del_min_stock_data_after_init(self):
		"""
		init 마지막에 정리하는 부분, ram 위해서
		MIN DATA만 지움
		STOCK_DICTIONARY_FROM_ML__real_time_data_MIN
		 -> 'prediction', 'trade', 'date'

		 self.AT_TUPLE__profit_record_watch 생성이 보장되고 나서
		"""
		try:
			# @ ML단에서 받아온게 있는지 검증
			if self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN: # ML단에서 받아온 데이터 있음
				if 'prediction' in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN and 'trade' in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN:
					if self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'] and self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade']:

						tmp_list_to_preserve = []
						for tuple_items in self.AT_TUPLE__profit_record_watch:
							tmp_list_to_preserve.append(tuple_items[0])
						
						for stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']:
							if stock_code in tmp_list_to_preserve:
								print(f'preserving stock min data in AT_FUNC__del_min_stock_data_after_init : {stock_code}')
							else:
								try:
									print(f'removing stock min data in AT_FUNC__del_min_stock_data_after_init : {stock_code}')
									self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code] = None
									del self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code]
								except Exception as e:
									print(f'error in del stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN["STOCK_MIN"] : {stock_code}, error : {e}')
					else:
						return 'AT_FAIL'
				else:
					return 'AT_FAIL'
			else:
				return 'AT_FAIL'


		except Exception as e:
			print('error in AT_FUNC__del_min_stock_data_after_init :: ', e)
			return 'AT_FAIL'


	
	def AT_FUNC__wake_up(self):
		"""
		시작하면서 새로운 class 생성, 매번 반복할 작업이므로 class로 구현
		# self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code] 이용해서 필요없는 것 생성 안하기로
		"""
		try:
			print('AT_FUNC__wake_up activated...')
			#print(f'self.STOCK_DICTIONARY__code_to_name in AT_FUNC__wake_up : {self.STOCK_DICTIONARY__code_to_name}')
			#print(f'self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN["SQLITE"] in AT_FUNC__wake_up : {self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN["SQLITE"]}')
			
			
			for stock_code in self.STOCK_DICTIONARY__code_to_name : # 첫 생성 부분
			
				if stock_code not in self.TOTAL_DICT :# 존재하지 않으면
					if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']:
						if self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code] == True:
							self.TOTAL_DICT[stock_code] = Single_stock(self.STOCK_DICTIONARY__code_to_name[stock_code], stock_code)
							print(f'creation of Single_stock in AT_FUNC__wake_up : {stock_code}')
						else:
							pass
					else:
						pass
				else:
					pass # 첫 생성은 아니라서 pass
				

			for stock_code in self.TOTAL_DICT:

				# @ owning 여부, unmet 여부, 실제 구입 전체가격
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['number_owned']
					
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__unmet_order:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__unmet_number = self.STOCK_DICTIONARY_NAMES__unmet_order[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['unmet_order_num']
					
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['buy_price'] # 해당 stcok에서 사용중인 금액
					
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_profit = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['profit_rate'] # 해당 stock의 손익률

						
				# 3) article 존재 여부
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAME__article_result:
					if self.STOCK_DICTIONARY_NAME__article_result[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name] >= self.ARTICLE_MIN_NUMBER: # 일단 기사가 존재해야, 초창기 기준
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = True
					else:
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
				else:
					self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
						
					
				# 5) dictionary stock min data 정합성
				self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy = True
				# -> init이 데이터 다 가져오고 생성되었으므로!
				self.TOTAL_DICT[stock_code].SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()
				# if self.AT_func__sub_sqlite_to_dict_date_check(stock_code):
				# 	self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy = True
				# else:
				# 	self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy = False
				# self.TOTAL_DICT[stock_code].SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()


				# 5) 전체 계산
					
					
				# 6) 전체 계산 표시해서, ML 단에서 쓸 수 있도록 함
				# 		self.SINGLE_BOOL__article_exist = False # 결국 이거만 업데이트 되면 된다...
				# 		self.SINGLE_BOOL__sqlite_date_consistancy = False
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['FILTER'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching #self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist * self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['BUDGET'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['OWNING'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number
					

			
			# @ class update
			self.AT_FUNC__class_variable_func_assign()

			# @ MUST WATCH LIST
			for stock_codes in self.TOTAL_DICT:
				if stock_codes in self.MUST_WATCH_LIST:
					self.TOTAL_DICT[stock_codes].SINGLE_VARIABLE__stage = 7

			# @ inheritance update
			for stock_code in self.TOTAL_DICT:
				self.TOTAL_DICT[stock_code].SINGLE_FUNC__calc_stage()

			# @ 전체 개수
			print(f'TOTAL_DICT 개수 :: ', len(list(self.TOTAL_DICT.keys())))

		except Exception as e:
			print('error in AT AT_FUNC__wake_up :: ', e)
			traceback.print_exc()
			#self.CHECK_FILTER_FLAG__at_initialize = False # fail시 FE 단 flag 내림
			return 'AT_FAIL'
	
	def AT_func__sub_sqlite_to_dict_date_check(self, stock_code):
		"""
		sqlite에서 가지고 온 것에서 dictionary 만든 부분에 정합성 체크해서 bool 돌려줌
		#ㅋㅋ

		"""
		try:
			if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE']:
				if self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['SQLITE'][stock_code] == True: # sqlite db -> dict 에서 삭제 안했으면

					if stock_code in self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN']:

						# @ db dictionary에서 작업
						tmp_date_hash = copy.deepcopy(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][stock_code])
						tmp_list_date_keys = copy.deepcopy(list(tmp_date_hash.keys()))
						tmp_list_date_keys.sort()

						tmp_date_hash_latest = tmp_list_date_keys[-1] # latest

						tmp_date_hash_latest__obj = datetime.datetime.strptime(tmp_date_hash_latest, "%Y%m%d%H%M%S").replace(second=0, microsecond=0)
						tmp_datetime_now__obj = datetime.datetime.now()
						
						if tmp_datetime_now__obj - tmp_date_hash_latest__obj >= datetime.timedelta(minutes=2): # avoid
							return False
						else:
							return True

					else:
						return False

					
				else:# dictionary db에서 삭제 됨
					return False
			else: # 일어나지 말아야 함
				return False
		
		except Exception as e:
			print('error in AT_func__sub_sqlite_to_dict_date_check :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'

	def AT_FUNC__update_TOTAL_DICT__ONGOING(self):

		"""
		수행중에 SINGLE_FUNC__calc_stage 계산하는 부분, article 개수 업데이트 한번 돌리고!
		"""
		try:
			print('AT_FUNC__update_TOTAL_DICT__ONGOING activated...')
			for stock_code in self.TOTAL_DICT : # 첫 생성 부분

				# @ owning 여부, unmet 여부, 실제 구입 전체가격
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['number_owned']
				
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__unmet_order:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__unmet_number = self.STOCK_DICTIONARY_NAMES__unmet_order[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['unmet_order_num']
				
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['buy_price']

				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAMES__owning_stocks:
					self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_profit = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['profit_rate'] # 해당 stock의 손익률


				# 3) article 존재 여부
				"""
				# 다른 함수에서 해줌 
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAME__article_result:
					if self.STOCK_DICTIONARY_NAME__article_result[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name] >= self.ARTICLE_MIN_NUMBER_ONGOING and self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist == False:
						# 한번 false 인 경우에
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = True
					else:
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
				else:
					self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
				"""

				# 4) sqlite 부분
				# init에서 고정되어있음

				# 5) dictionary stock min data 정합성
				if self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__db_check_needed == True :
					self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__db_check_needed = False # reset
					self.AT_func__sub_sub_sqlite_get(stock_code) # 가져와서 dictionary에 업데이트
					if self.AT_func__sub_sqlite_to_dict_date_check(stock_code):
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy = True
					else:
						self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy = False
				
					
				
				# 6) 전체 계산 표시해서, ML 단에서 쓸 수 있도록 함
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['FILTER'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching #self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist * self.TOTAL_DICT[stock_code].SINGLE_BOOL__sqlite_date_consistancy # * self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['BUDGET'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed
				self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['OWNING'][stock_code] = self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number			

			# @ class update
			self.AT_FUNC__class_variable_func_assign()

			# @ MUST WATCH LIST
			for stock_codes in self.MUST_WATCH_LIST:
				self.TOTAL_DICT[stock_codes].SINGLE_VARIABLE__stage = 7

			# @ inheritance update
			for stock_code in self.TOTAL_DICT:
				self.TOTAL_DICT[stock_code].SINGLE_FUNC__calc_stage()

		except Exception as e:
			print('error in AT_FUNC__update_TOTAL_DICT__ONGOING :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'

	def AT_func__sub_sub_sqlite_get(self, code):
		"""
		AT_func__sub_sqlite_to_dict_date_check 에서 체크할 수 있도록, 
		SINGLE_CHECK_FLAG__DODO__db_check_needed True인 경우에 데이터 가져옴

		"""
		try :
			# @ 함수 안 ram 변수 세팅
			tick_range = 1 # 이게 뭐하는건지 모르겠다
			#lookup = 0  # look_type  =>  0 : 단순조회 , 1 : 연속조회
			input_dict ={} # for set input value function in BACKEND


			# @ 종목의 기본 정보들
			base_date = datetime.datetime.today().strftime('%Y%m%d')
			input_dict['종목코드'] = code
			input_dict['틱범위'] = tick_range
			input_dict['기준일자'] = base_date
			input_dict['수정주가구분'] = 1

			self.KIWOOM.set_input_value(input_dict)

			tmp_db_return = self.KIWOOM.comm_rq_data("opt10080_req", "opt10080", 0, self.SCREEN_NO.opt10080_req, self.STATE_TIME.stage, self.STATE_TIME.weekday_num) # look_type  =>  0 : 단순조회 , 2 : 연속조회


			self.ERROR_COUNTER_BE__request_num = self.ERROR_COUNTER_BE__request_num + 1
			if self.ERROR_COUNTER_BE__request_num >= self.REQUEST_MAX_NUM_ON_THE_RUN:
				print('exiting backend current creation of database - 1st')
				self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1
				return None

			if tmp_db_return != 0: # 에러값, 정상은 0
				#print('tmp_db_return :: ', tmp_db_return)
				#self.FUNC_PYQT__rest_timer(5)
				return None

			# ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
			ohlcv = copy.deepcopy(self.KIWOOM.latest_tr_data)
			tmp_hash = {}
			if type(ohlcv) == type(dict()) and ohlcv : # 딕셔너리고, 비어있지 않음
				try:
					for i in range(len(ohlcv['date'])):
						tmp_hash[ohlcv['date'][i]] = {'price':ohlcv['open'][i],'volume':ohlcv['volume'][i]}
					self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN['STOCK_MIN'][code] = tmp_hash
				except:
					pass

		except Exception as e:
			print('error in AT_func__sub_sub_sqlite_get :: ', e)
			traceback.print_exc()
			self.FUNC_PYQT__rest_timer(1) # 시간초 대기

	def AT_FUNC__automated_transaction(self):
		"""
		ML 단의 주문 대로 수행하는 부분
		계좌에 10만원 이상 있을 때!

		Total dictionary 안에 포함된
		"""
		"""
		ㅋㅋ
		self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching == True
		self.TOTAL_DICT[stock_data].SINGLE_CHECK_FLAG__DODO__at_active == True 두개 충족중일 때!
		STOCK_DICTIONARY_FROM_ML__real_time_data_MIN
		 -> 'prediction', 'trade', 'date'

		self.SINGLE_VARIABLE__owning_number = 0 # 들고있는 개수
		self.SINGLE_VARIABLE__unmet_number = 0 # 미수 개수
		self.SINGLE_VARIABLE_BUDGET__allowed = 0 # 허용 된 금액 총량
		self.SINGLE_VARIABLE_BUDGET__real_bought = 0 # 실제 구매한 금액 총량
		self.SINGLE_VARIABLE_BUDGET__real_profit = 0 # 실제 수익률 총량 -> 업데이트 FE단에서
		self.SINGLE_VARIABLE_STOCK__expected_profit = 0 # 예상 수익률 : future 30분~ or 그이상 시간동안
		
		1) 계좌에 10만원 이상 남아있을 때 -> FUNC_STOCK__do_action 에 구현되어있음
		2) 한번에 거래 수행할것... logic 넘겨받는 부분이 있어야 함
		3) balance_allowed 보다 perchased 가 낮아야 한다, 높으면 팔던가 hold 하는 것으로, 항상 밑으로 맞춰야 함
		4) SEC 데이터 STOCK_DICTIONARY_FROM_BE__real_time_data_SEC 에서 급락하면 팔아야함, stage로 관리? -> stock_code -> volume/price

		0:"매매금지-기사",
		1:"매매금지-데이터", 
		2:"매매후보", 
		3:"감시수행", 
		4:"자동수행",
		5:"자동보류",
		6:"자동금지"
		7:"모니터링"} -> 모니터링, 코스피 지수 + 인버스 강제 예측 : 나중에 인버스 구매할 수도 있음, 코스피는 계속 예측하자
		"""
		try:
			tmp_tupe_of_order = [] # 파는거 먼저해서 sharing budget 먼저 확보하고 매수 진행하려고
			for stock_code in self.TOTAL_DICT :
				if self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching and self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__at_active and self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__stage == 4:
					# sec데이터로 위험하면 sell
					if stock_code not in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC: # 감시중이어야한다
						self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__stage = 5 # 자동 트레이딩 벗어나서 보류로 넘어감
						continue # 바로 다음 iteration
					else: # in sec data
						tmp_sec_date_hash = list(self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code].keys())
						tmp_sec_date_hash.sort()

						# @ hash값 최신 데이터 10초 통과여부 확인
						if datetime.datetime.now() - datetime.datetime.strptime(tmp_sec_date_hash[-1], "%Y%m%d%H%M%S") <= datetime.timedelta(seconds=10):

							# @ 가장 최신 값
							tmp_latest_price = self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][tmp_sec_date_hash[-1]]['price']

							# @ 가장 highest price
							tmp_highest_price = 0
							for date_stamp in tmp_sec_date_hash:
								if date_stamp in self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code]: # 날짜값 있으면
									if tmp_highest_price <= self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][date_stamp]['price']:
										tmp_highest_price = self.STOCK_DICTIONARY_FROM_BE__real_time_data_SEC[stock_code][date_stamp]['price']
									else:
										pass
								else:
									pass
							if tmp_highest_price != 0: # avoid zero division
								if ((tmp_latest_price - tmp_highest_price) / (tmp_highest_price)) < self.STOCK_TARGET_MINUS_PROFIT : # -3퍼 순간 찍으면
									self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__stage = 5 # 자동 트레이딩 벗어나서 보류로 넘어감
									continue
								else:
									pass
									############ 실제 트레이딩 정상적으로 들어와야 하는 부분 ##################################
									########################################################################################
									# 수익률 감소해서 팔아야 하는 것은 stage 단에서 관리
									# 1) budget 확인
									if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed < self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought : # 만족할 때 까지 매도
										tmp_num_to_sell = 0
										tmp_avg_price = self.STOCK_DICTIONARY_NAMES__owning_stocks[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name]['average_price']
										for i in range(self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number ): # 보유수량 iter
											if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed >= ( -(i+1) * (tmp_avg_price) + self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought ):
												tmp_num_to_sell = (i+1)
												break
										else: # break 없이 진입
											tmp_num_to_sell = self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__owning_number
										#ㅋㅋ
										tmp_tupe_of_order.append( (stock_code, tmp_num_to_sell, "매도", 1) )
										# self.STOCK_IN_ATTENTION.code = stock_code
										# self.STOCK_IN_ATTENTION.order_num = tmp_num_to_sell
										# self.STOCK_IN_ATTENTION.action = "매도"
										# self.FUNC_STOCK__do_action()

									elif (self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed >= self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought) and (self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed < self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought + tmp_latest_price): # 예산 초과는 아닌데 살 금액은 모자른다
										if stock_code in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade']: # 있다면
											# self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code] -> {action : , num :}
											if self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'SELL':
												# self.STOCK_IN_ATTENTION.code = stock_code
												# self.STOCK_IN_ATTENTION.order_num = self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['num']
												# self.STOCK_IN_ATTENTION.action = "매도"
												# self.FUNC_STOCK__do_action()
												tmp_tupe_of_order.append( (stock_code, self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['num'], "매도", 1) )
											elif self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'BUY':
												continue
											elif self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'HOLD':
												continue

										else: # should be unreachable
											self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__stage = 5 # 자동 트레이딩 벗어나서 보류로 넘어감
											continue

									elif self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed >= self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought + tmp_latest_price: # 한개 이상 살 수 있음
										pass
										# ㅋㅋ
										#여기서 몇개 구매 가능한지 계산해봐야됨!
										#홀드, 바이, 셀 각각 대응도 코딩, 큰단위 구매랑 판매는 FUNC_STOCK__do_action 에서 서폿 되니깐 skip!
										if self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'BUY':
											tmp_possible_n = ((self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__allowed - self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_BUDGET__real_bought) / tmp_latest_price)
											tmp_request_n = self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['num']
											tmp_send_n = 0
											if tmp_possible_n >= tmp_request_n:
												tmp_send_n = tmp_request_n
											else:
												tmp_send_n = tmp_possible_n
											# self.STOCK_IN_ATTENTION.code = stock_code
											# self.STOCK_IN_ATTENTION.order_num = tmp_send_n
											# self.STOCK_IN_ATTENTION.action = "매수"
											# self.FUNC_STOCK__do_action()
											tmp_tupe_of_order.append( (stock_code, tmp_send_n, "매수", 0) )
											

										elif self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'SELL':
											# self.STOCK_IN_ATTENTION.code = stock_code
											# self.STOCK_IN_ATTENTION.order_num = self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['num']
											# self.STOCK_IN_ATTENTION.action = "매도"
											# self.FUNC_STOCK__do_action()
											tmp_tupe_of_order.append( (stock_code, self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['num'], "매도", 1) )

										elif self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code]['action'] == 'HOLD':
											pass

									# 3) ML단 정보 이용


							else: # should be unreachable but if it does
								self.ERROR_DICTIONARY__backend_and_critical['error_critical'] = self.ERROR_DICTIONARY__backend_and_critical['error_critical'] + 1

						else: # SEC데이터가 10초를 넘김
							pass

				
				else: # 자동 수행 단계가 아님
					pass
			
			# @ 주문서 작성 FOR loop 끝남
			##############################################

			# 1) 정렬
			# 종목코드, trans 개수, action type, 소팅위한 번호
			tmp_tupe_of_order = copy.deepcopy( sorted(tmp_tupe_of_order, key=lambda x : x[3], reverse=True) )

			# 2) 주문 수행
			for tuple_item in tmp_tupe_of_order:
				self.STOCK_IN_ATTENTION.code = tuple_item[0]
				self.STOCK_IN_ATTENTION.order_num = tuple_item[1]
				self.STOCK_IN_ATTENTION.action = tuple_item[2]
				self.FUNC_STOCK__do_action()

		except Exception as e:
			print('error in AT_FUNC__automated_transaction :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'
		
	def AT_FUNC__update_stock_in_scrno_watch__ONGOING(self):
		
		"""
		# stage 단에서 계산해줌
		자동으로 scrno에 등록해서 getting하는 부분 -> 없던 것이면 database로 게팅 한번 해줘야되는데... 갑자기 하는 애면?
		지워야 하는 애는 지워주어야함
		"""
		try:
			#전체 totaldic으로 봐야될수도
			tmp_list_in_interest = []
			for stock in self.AT_TUPLE__profit_record_watch:
				tmp_list_in_interest.append(stock[0])
			
			for stock_code in self.TOTAL_DICT:
				# 1) SCRNO에 등록 안되어있으면 rank tuple에 있을 때 등록함
				if stock_code in tmp_list_in_interest: # tuple에 있는 항목
					self.STOCK_IN_ATTENTION.code = stock_code
					tmp_return = self.SCREEN_NO.scrno_used_lookup_by_stock_code(stock_code) # None일시 등록 안됨, 아닐시 등록 된 것
					if tmp_return == None:
						self.FUNC_STOCK_DATABASE__get_real_time()
						self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching = True
						
						# @ scrno 재등록/확인 시점 기록
						self.TOTAL_DICT[stock_code].SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()

					else:# 이미 등록되었음
						# @ scrno 재등록/확인 시점 기록
						self.TOTAL_DICT[stock_code].SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()
						pass # SCRNO 등록되어있고 tuple에서도 있음

				else: # tuple에는 없지만 등록은 되어있는 경우
					if self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching == True:
						self.STOCK_IN_ATTENTION.code = stock_code
						self.FUNC_STOCK_DATABASE__disable_real_time()
						self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__start_watching = False

						self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__sell_all = True

					else: # tuple에도 없고 등록도 안되어있음
						pass
					
		except Exception as e:
			print('error in AT_FUNC__update_stock_in_scrno_watch__ONGOING :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'
	
	def AT_FUNC__sell_stocks_with_sell_flag(self):
		"""
		fe 단 함수 사용해서 SINGLE_CHECK_FLAG__DODO__sell_all flag 올라간 것들 대상으로 전부 sell 한다.
		"""
		try:
			pass
			for stock_code in self.TOTAL_DICT:
				if self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG__DODO__sell_all == True: # 다 파는게 true면
					
					# @ FE 단 작업
					self.STOCK_IN_ATTENTION.code = stock_code
					self.FUNC_STOCK__handle_sell_all_single_stock() # 종목코드, 액션, 가격

					# @ single stock init
					#self.TOTAL_DICT[stock_code].SINGLE_FUNC__init() # init

		except Exception as e:
			traceback.print_exc()
			print('error in AT_FUNC__sell_stocks_with_sell_flag :: ', e)
			return 'AT_FAIL'


	def AT_FUNC__allocate_budget_for_each(self):
		"""
		AT_FUNC__update_profit_rank__ONGOING 에서 계산된 결과를 가지고 dynamic하게 자산 배분
		self.AT_TUPLE__profit_record_watch 사용 (종목번호 , 예상 수익률)
		STOCK_AT_MAX_NUM
		ㅋㅋ
		"""
		try:

			# @ update AT list
			self.AT_TUPLE__profit_record_trans.clear()
			if len(self.AT_TUPLE__profit_record_watch) > self.STOCK_AT_MAX_NUM: # 최대 들고있을 개수
				#self.AT_TUPLE__profit_record_trans = copy.deepcopy([]) # initialize
				
				for tuple_item in self.AT_TUPLE__profit_record_watch: # watch list 소팅된것 에서 넣다가
					self.AT_TUPLE__profit_record_trans.append(tuple_item)
					if len(self.AT_TUPLE__profit_record_trans) >= self.STOCK_AT_MAX_NUM:
						break # 지정개수까지 넣고 나옴
			else: # watch 개수가 trans 지정개수보다 작으면
				for tuple_items in self.AT_TUPLE__profit_record_watch:
					self.AT_TUPLE__profit_record_trans.append(tuple_items)
			
			# 이익률 합산 / AT 할 전체 개수 기록
			tmp_every_profit = 0
			tmp_counter_1 = 0
			for tuple_item in self.AT_TUPLE__profit_record_trans:
				tmp_every_profit = tmp_every_profit + tuple_item[1]
				tmp_counter_1 = tmp_counter_1 + 1


			# @ 차후 사용을 위해 hashing
			tmp_dictionary_of_interest = {}
			for items in self.AT_TUPLE__profit_record_trans:
				tmp_dictionary_of_interest[items[0]] = items[1] # stock_code 기록


			# @ 전체 분배할 '총자산' 계산
			tmp_all_budget_summed = copy.deepcopy(self.STOCK_IN_ATTENTION.balance)
			for stock_data in  self.TOTAL_DICT:
				tmp_all_budget_summed = self.TOTAL_DICT[stock_data].SINGLE_VARIABLE_BUDGET__real_bought + tmp_all_budget_summed


			# @ total budget 재분배
			for stock_data in self.TOTAL_DICT:
			
				if stock_data in tmp_dictionary_of_interest:
					self.TOTAL_DICT[stock_data].SINGLE_VARIABLE_BUDGET__allowed = tmp_all_budget_summed * tmp_dictionary_of_interest[stock_data] / tmp_every_profit
					self.TOTAL_DICT[stock_data].SINGLE_CHECK_FLAG__DODO__at_active = True # 자동 투자 올린다.

				else: # autotrade 꺼짐 후보
					if self.TOTAL_DICT[stock_data].SINGLE_CHECK_FLAG__DODO__at_active == True: # 자동트레이딩 상태였다가 내려야함
						self.TOTAL_DICT[stock_data].SINGLE_CHECK_FLAG__DODO__at_active = False # 자동 투자 내린다.
						self.TOTAL_DICT[stock_data].SINGLE_VARIABLE_BUDGET__allowed = 0
						self.TOTAL_DICT[stock_data].SINGLE_CHECK_FLAG__DODO__sell_all = True # 자동 투자 한종목 매도 true
					else: # 원래 자동트레이딩 상태 아니었음
						pass


		except Exception as e:
			print('error in AT_FUNC__allocate_budget_for_each :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'

	def AT_FUNC__update_profit_rank__ONGOING(self):
		try:
			# AT_TUPLE__profit_record_watch
			# (주식번호, 예상수익)
			"""
			self.AT_TUPLE__profit_record_watch = [] # initialize
			for stock_code in self.TOTAL_DICT :
				self.AT_TUPLE__profit_record_watch.append( (stock_code, self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_STOCK__expected_profit) )

			self.AT_TUPLE__profit_record_watch = copy.deepcopy( sorted(self.AT_TUPLE__profit_record_watch,key=lambda x: x[1], reverse=True) ) # 높은게 1위로?

			tmp_list_records_to_watch = []
			for tuple_item in self.AT_TUPLE__profit_record_watch:
				if self.TOTAL_DICT[tuple_item[0]].SINGLE_VARIABLE_STOCK__expected_profit >= self.STOCK_TARGET_MICRO_PROFIT:
					tmp_list_records_to_watch.append(tuple_item)
					if len(tmp_list_records_to_watch) >= self.STOCK_AT_WATCH_MAX_NUM:
						break
					else:
						continue
			
			# @ 엎어 친다
			self.AT_TUPLE__profit_record_watch = copy.deepcopy(tmp_list_records_to_watch) # 이걸로 관리를 해야할 것 같은데?
			"""
			self.AT_TUPLE__profit_record_watch.clear()
			tmp_list_of_record_watch = []
			for stock_code in self.TOTAL_DICT :
				tmp_list_of_record_watch.append( (stock_code, self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_STOCK__expected_profit) )
			tmp_list_of_record_watch = copy.deepcopy(sorted(tmp_list_of_record_watch,key=lambda x: x[1], reverse=True))

			for tuple_items in tmp_list_of_record_watch:
				if self.TOTAL_DICT[tuple_items[0]].SINGLE_VARIABLE_STOCK__expected_profit >= self.STOCK_TARGET_MICRO_PROFIT:
					self.AT_TUPLE__profit_record_watch.append(tuple_items)
					if len(self.AT_TUPLE__profit_record_watch) >= self.STOCK_AT_WATCH_MAX_NUM:
						break
					else:
						continue

			
			
		except Exception as e:
			print('error in AT_FUCN__update_total_dict__ONGOING :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'

	def AT_FUNC__prediction_update__ONGOING(self):
		"""
		ML에서 가져온 lstm 예측 순서대로 list up 하는 부분 + 목표 수익률 이상인 종목 고르는 부분
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN = = {'prediction':{ㄱ}, 'trade':{ㄴ}, 'date':''} , ㄱ/ㄴ 내부는 stock_code -> 1depth hash
		"""
		try:
			# 1) prediction 분봉으로 30분짜리, 10분 delay이면 갖다 치우기
			for stock_code in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'] :
				if self.AT_func__sub_to_find_tradeable_stock_code(stock_code) :	# article 포함 이곳에서 판별해서 true에 도달해야됨
					#for datetime_str in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code] :
					#	tmp_datetime_now__obj = datetime.datetime.now()
					#	datetime_str__obj = datetime.datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
					tmp_datetime_data_list = copy.deepcopy(list(self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code].keys()))
					tmp_datetime_data_list.sort() # 최근 시간이 뒷쪽으로
					
					tmp_datetime_now__obj = datetime.datetime.now().replace(second=0, microsecond = 0)
					

					tmp_list_data_of_future = [] # reset
					for datetime_data_from_ML in tmp_datetime_data_list: # sorted!
						datetime_data_from_ML__obj = datetime.datetime.strptime(datetime_data_from_ML, "%Y%m%d%H%M%S").replace(second=0, microsecond = 0)
						if(      (tmp_datetime_now__obj - datetime_data_from_ML__obj <= datetime.timedelta(minutes=10))
							and (datetime_data_from_ML__obj - tmp_datetime_now__obj <= datetime.timedelta(minutes=30))): # 미래 데이터
							tmp_list_data_of_future.append(datetime_data_from_ML)
					
					try:
						datetime_data_from_ML__obj = None
						datetime_data_from_ML = None

						del datetime_data_from_ML__obj
						del datetime_data_from_ML

					except Exception as e:
						print('error in del of variables in AT_FUNC__prediction_update__ONGOING (1) :: ', e)
					
					if len(tmp_list_data_of_future) > 0:
						tmp_list_data_of_future.sort()
						tmp_future_start_price = self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code][tmp_list_data_of_future[0]]
						tmp_future_end_price = self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code][tmp_list_data_of_future[-1]]
	
						self.TOTAL_DICT[stock_code].SINGLE_VARIABLE_STOCK__expected_profit = (tmp_future_end_price - tmp_future_start_price)/(tmp_future_start_price)

					else:
						pass # 해당 fucntion을 수행하는 의미가 없음
					
				else: # 예측 의미가 없음
					pass
			
			# @ del after for loop (2)
			try:
				tmp_datetime_data_list = None
				tmp_datetime_now__obj = None
				tmp_list_data_of_future = None
				tmp_future_start_price = None
				tmp_future_end_price = None

				del tmp_datetime_data_list
				del tmp_datetime_now__obj
				del tmp_list_data_of_future
				del tmp_future_start_price
				del tmp_future_end_price

			except Exception as e:
				print('error in del of variables in AT_FUNC__prediction_update__ONGOING (2) :: ', e)
		
		except Exception as e:
			print('error in AT - AT_FUNC__prediction_update__ONGOING :: ', e)
			traceback.print_exc()
			return "AT_FAIL"
	
	def AT_func__sub_to_find_tradeable_stock_code(self, stock_code):
		"""
		코드를 넣으면, trade able인지, self.TOTAL_DICT 에 있으면 True 반환 else -> False
		"""
		if stock_code in self.TOTAL_DICT:
			if self.TOTAL_DICT[stock_code].SINGLE_CHECK_FLAG_BOOL__result: #이게 true이면
				return True
			else:
				return False
				
		else:
			return False
	
	def AT_FUNC__article_bool__ONGOING(self):
		"""
		개장 중에 불릴 함수 -> 뉴스만 따로 업데이트 해줌
		"""
		try:
			print('AT_FUNC__article_bool__ONGOING activated...')
			
			for stock_code in self.TOTAL_DICT : # 첫 생성 부분
			
				# 3) article 존재 여부
				if self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name in self.STOCK_DICTIONARY_NAME__article_result:
					if self.self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__stage < 2 : #매매금지-기사 / 매매금지-데이터 :: 0, 1
						if self.STOCK_DICTIONARY_NAME__article_result[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name] > self.ARTICLE_MIN_NUMBER_ONGOING: # 일단 기사가 존재, ONGOING 시점
							self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = True
						else:
							self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
					else: #초창기에 통과했음
						if self.STOCK_DICTIONARY_NAME__article_result[self.TOTAL_DICT[stock_code].SINGLE_VARIABLE__name] >= self.ARTICLE_MIN_NUMBER: # 일단 기사가 존재, ONGOING 시점
							self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = True
						else:
							self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
				else:
					self.TOTAL_DICT[stock_code].SINGLE_BOOL__article_exist = False
				
				self.TOTAL_DICT[stock_code].SINGLE_FUNC__calc_stage() # 감시 여부 bool 업데이트 
				
		except Exception as e:
			print('error in AT_FUNC__article_bool__ONGOING : ', e)
			traceback.print_exc()
			return 'AT_FAIL'
	
	def AT_FUNC__pickle_dump_for__ML(self):
		"""
		Machine learning을 위해 pickel dump 하는 부분
		self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN 을
		self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML 에다 덤프
		"""
		try:
			# if self.TEST == True and False:
			# 	# https://brownbears.tistory.com/249
			# 	print(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN)
			# 	self.tr_mlc__2 = tracemalloc.take_snapshot()
			# 	tmp_stats = self.tr_mlc__2.compare_to(self.tr_mlc__1, 'lineno')
			# 	tmp_stacks = self.tr_mlc__2.compare_to(self.tr_mlc__1, 'traceback')
			#
			# 	print('\n'*3)
			# 	print('※'*60)
			# 	print('※'*60)
			# 	for stats in tmp_stats:
			# 		print(stats)
			# 	for stacks in tmp_stacks:
			# 		print(stacks)
			# 	print('※'*60)
			# 	print('※'*60)

			#print(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN)
			with open(self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML, 'wb') as file:
				pickle.dump(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN, file)
			print('AT_FUNC__pickle_dump_for__ML pickle dump successful!')

			#joblib.dump(self.STOCK_DICTIONARY_FROM_BE__real_time_data_MIN, self.STOCK_DICTIONARY_PICKLE_FROM_BE__path_for_ML)
			#print('AT_FUNC__pickle_dump_for__ML pickle dump successful!')

		except Exception as e:
			print('error in AT_FUNC__pickle_dump_for__ML :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'
		except MemoryError as m_e:
			print('error in AT_FUNC__pickle_dump_for__ML :: ', m_e)
			gc.collect()
			return 'AT_FAIL'
			
					
	
	def AT_FUNC__pickle_dump_from__ML(self):
		"""
		Machine learning dump 받아오는 부분
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN 에 dump로 가져옴 = {'prediction':{ㄱ}, 'trade':{ㄴ}, 'date':''}
		self.STOCK_DICTIONARY_FROM_ML__path_for_32bit 가 ML 제공하는 Package64의 위치

		정합성도 검사해야함!
		
		일단 분별로 다 들고있어서 delta값 예측이랑 많이 벌어지는도 검사해야하니깐 update로 할 것임
		self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN
		"""
		try:
			tmp_dictionary = {}

			if self.TEST == True:
				if os.path.isfile(self.STOCK_DICTIONARY_FROM_ML__path_for_32bit):
					with open(self.STOCK_DICTIONARY_FROM_ML__path_for_32bit, 'rb') as file:
						tmp_dictionary = copy.deepcopy(pickle.load(file))
					print('AT_FUNC__pickle_dump_from__ML pickle dump successful!')
				else:
					pass

			else: # None test
				with open(self.STOCK_DICTIONARY_FROM_ML__path_for_32bit, 'rb') as file:
					tmp_dictionary = copy.deepcopy(pickle.load(file))
				print('AT_FUNC__pickle_dump_from__ML pickle dump successful!')

				#tmp_dictionary = joblib.load(self.STOCK_DICTIONARY_FROM_ML__path_for_32bit)
				#print('AT_FUNC__pickle_dump_from__ML pickle dump successful!')

				if 'prediction' in tmp_dictionary and 'trade' in tmp_dictionary and 'date' in tmp_dictionary:
					if tmp_dictionary['prediction'] and tmp_dictionary['trade']: # 비지 않음
						# datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S")
						if datetime.datetime.now() - datetime.datetime.strptime(tmp_dictionary['date'], "%Y%m%d%H%M%S") <= datetime.timedelta(minutes=1): #1분 차이이면 -> 64bit도 같이 도는 중이니깐 만족할 것임
							for stock_code in tmp_dictionary['prediction']: # prediction은 업데이트로, trade는 엎어치고
								if stock_code in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction']:
									self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['prediction'][stock_code].update( tmp_dictionary['prediction'][stock_code] )

							for stock_code in tmp_dictionary['trade']: # prediction은 업데이트로, trade는 엎어치고
								if stock_code in self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade']:
									self.STOCK_DICTIONARY_FROM_ML__real_time_data_MIN['trade'][stock_code] = copy.deepcopy( tmp_dictionary['trade'][stock_code] )
							
							try:
								tmp_dictionary = None
								del tmp_dictionary
							except Exception as e:
								print('error in del of tmp_dictionary in AT_FUNC__pickle_dump_from__ML :: ', e)

						else:
							return 'AT_FAIL'
					else:
						return 'AT_FAIL'
				else:
					return 'AT_FAIL'

		except Exception as e:
			print('error in AT_FUNC__pickle_dump_from__ML :: ', e)
			traceback.print_exc()
			return 'AT_FAIL'
			
	#===============================================================================================
	#===============================================================================================

class MY_SIG_global(QObject): # 사용자 정의 시그널 포함하는 class
	sig_database = pyqtSignal() # 데이터 베이스 creation 올라와있을 때 쓸 함수
	sig_addtional_info_tr = pyqtSignal() # 추가 정보 만들 때 쓸 함수

	def run_sig_name(self, func_name): # 함수 명으로 함수 수행하는 함수
		func = getattr(pyq_object, func_name)
		func(self)

	@pyqtSlot()
	def database(self): # 데이터베이스 시그널 송출
		self.sig_database.emit()

	def addtional_info(self): # 추가주식정보 가져올 때 시그널 송출
		self.sig_addtional_info_tr.emit()


class Screen_no:
	"""
	스크린 number 지정부분
	CommRqData에 들어감
	200개 이하여야 한다
	-> 한 화면당 몇개 이하로 해야되는데 기억이 안난다 : 이거 image 폴더에 있었던거 같은데
	-----------------------------------------------------------------
	setRealReg 사용?
	실시간 시세 데이터는 하나의 화면번호당, 100개 종목까지 등록 가능합니다!
	:return:
	"""
	untouch_list = ["0101", "0001", "0002", "0003", "0004", "0005", "0006"]
	
	unfilter_list = [] # create when calling the class
	# ----------------------------------------------------
	opt10081_req = "0101"
	opt10080_req = "0101"
	balance_check_normal = "0001"
	balance_check_with_order = "0002"
	send_order = "0003"
	check_owning_stocks = "0004"
	stock_additional_info_tr = "0005"
	check_unmet_order = "0006"
	
	hash_by_scrno_to_stock_code = {}

	def __init__(self, max_num):

		self.max_num = max_num

		self.create_unfilter_list()
	
	def create_unfilter_list(self):
		print('initiate generating scrno for realtime stocks')
		while len(self.unfilter_list) != 200 - len(self.untouch_list) and len(self.unfilter_list) < self.max_num: # 사용하는거 빼고 그만큼 realtime으로 채운다 + 지정 개수만큼
			tmp_val_of_candidate = random.randint(1,999) # 난수 생성
			tmp_str_val_of_candidate = str(tmp_val_of_candidate)
			len_of_zero =  4 - len(tmp_str_val_of_candidate)
			tmp_str_val_of_candidate = '0' * len_of_zero + tmp_str_val_of_candidate
			if not tmp_str_val_of_candidate in self.untouch_list :
				self.unfilter_list.append(tmp_str_val_of_candidate)
		print('done generating scrno for realtime stocks')
		print('length of untouch_list : ', len(self.untouch_list))
		print('length of made unfilter_list : ', len(self.unfilter_list))
		print('created unfilter_list : ', self.unfilter_list)
		
		for scrno_unreserved in self.unfilter_list: # 해쉬에 list로 기입
			self.hash_by_scrno_to_stock_code[scrno_unreserved] = []
	
	def return_scrno_for_realtime(self, stock_code): # scrno 할당, 200여개중에 filter빼고...
		"""
		return None : 불가
		else 값들 : able ( = str scrno인 경우)
		"""
		# 사용 여부 체크
		tmp_flag_found = False
		for scrno_unreserved in self.hash_by_scrno_to_stock_code:
			tmp_list_of_stock_codes = self.hash_by_scrno_to_stock_code[scrno_unreserved]
			if stock_code in tmp_list_of_stock_codes: # stock code가 해당 scrno에 있는지 확인
				tmp_flag_found = True
				break
		if tmp_flag_found == True:
			return None
		else: # 등록이 되어있지 않음 - 원하는 결과 
			tmp_flag_assign_okay = False
			tmp_key_scrno_able = ''
			for scrno_unreserved in self.hash_by_scrno_to_stock_code: 
				tmp_list_of_stock_codes = self.hash_by_scrno_to_stock_code[scrno_unreserved]
				if len(tmp_list_of_stock_codes) == 99: # 한 scrno당 100개 등록이므로.. 넘어간다
					continue
				elif len(tmp_list_of_stock_codes) < 99 : # 100개 미만이면 등록가능 99로 맞추자
					tmp_flag_assign_okay = True
					tmp_key_scrno_able = scrno_unreserved #할당
					break
			if tmp_flag_assign_okay == False: # 등록할 수 없음, 다 찬 상태는
				return None
			else:
				self.hash_by_scrno_to_stock_code[tmp_key_scrno_able].append(stock_code)
				return tmp_key_scrno_able
	
	def scrno_used_lookup_by_stock_code(self, stock_code): # stock code 입력시 사용중일 때 scrno 리턴해줌
		# 사용여부 체크 
		tmp_flag_found = False
		tmp_key_scrno_used = ''
		for scrno_unreserved in self.hash_by_scrno_to_stock_code:
			tmp_list_of_stock_codes = self.hash_by_scrno_to_stock_code[scrno_unreserved]
			if stock_code in tmp_list_of_stock_codes:
				tmp_flag_found = True
				tmp_key_scrno_used = scrno_unreserved # 할당
				break
			else:
				continue
		if tmp_flag_found == False: # 못찾음
			return None
		else: # 찾음
			return tmp_key_scrno_used
	
	def return_list_stock_codes(self):
		"""

		:return:
		"""
		pass
			
	
	def disable_scrno_for_realtime(self, stock_code) : # 사용중인 scrno 제거, unfilter_list에 다시 넣어줌
		"""
		return None - fali
		성공시 stock code 그대로 돌려줌
		"""
		# 사용여부 체크 
		tmp_flag_found = False
		tmp_key_scrno_used = ''
		for scrno_unreserved in self.hash_by_scrno_to_stock_code:
			tmp_list_of_stock_codes = self.hash_by_scrno_to_stock_code[scrno_unreserved]
			if stock_code in tmp_list_of_stock_codes:
				tmp_flag_found = True
				tmp_key_scrno_used = scrno_unreserved # 할당
				break
			else:
				continue
		if tmp_flag_found == False:
			return None
		
		else: #stock code is found in the list
			self.hash_by_scrno_to_stock_code[tmp_key_scrno_used].remove(stock_code)
			return stock_code
			
			


class Message:
	"""
	키움증권 메세지 받을 시 디코딩 하는 부분
	1)계좌 balance 확인
	2)매수 매도시 첫번째로 불림
	"""

	def __init__(self):
		pass

	def decode_message(self, dictionary):
		"""
		EXAMPLE:
		-----------------------------------
		BUY RESULT - fail :: 
		{'screen_no': '0101', 
		'rqname': 'send_sell_order_req', 
		'trcode': 'KOA_NORMAL_SELL_KP_ORD', 
		'message': '[00Z353] 모의투자 주문가능 수량을 확인하세요'} - 모의투자 구분해야되는건가?!

		:return:
		"""
		action, result = '', ''
		if dictionary['rqname'] == "send_sell_order_req":
			action = "매도"
		elif dictionary['rqname'] == "send_buy_order_req":
			action = "매수"
		elif dictionary['rqname'] == "cancle_buy_order_req":
			action = "매수취소"
		elif dictionary['rqname'] == "cancle_sell_order_req":
			action = "매도취소"
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		if "정상처리" in dictionary['message']: #매도 성공
			result = "성공"
		elif "실패" in dictionary['message'] or "확인하세요" in  dictionary['message'] or "않습니다" in dictionary['message']:
			result = "실패"
		"""
		if dictionary['trcode'] == 'KOA_NORMAL_SELL_KP_ORD':
			action = "매도"
		if dictionary['trcode'] == 'KOA_NORMAL_BUY_KP_ORD':
			action = "매수"
		if dictionary['message'] == '[00Z113] 모의투자 정상처리 되었습니다': #매도 성공
			result = "성공"
		if dictionary['message'] == '[00Z112] 모의투자 정상처리 되었습니다': #매수 성공
			result = "성공"
		if dictionary['message'] == '[00Z353] 모의투자 주문가능 수량을 확인하세요':
			result = "실패"
		"""

		print(str(action + result))
		return str(action + result)

class Machine_state(object):

	"""
	https://pypi.org/project/transitions/
	
	STATE 상태는 state로 access하면 된다.
	ex)
	>>> batman = NarcolepticSuperhero("Batman")
	>>> batman.state
	'asleep'
	
	"""
	LIST__states = ['WAKEUP', 'CHECK_1', 'CHECK_2', 'FILTER','CHECK_3', 'ERROR_CANT_RUN', 'ERROR_SELL_ALL']
	def __init__(self):
		
		# @ input list들 설정 -> 이걸 통과해야 check_XX에 도달
		self.LIST__CHECK_1 = []
		self.LIST__CHECK_2 = []
		self.LIST__FILTER = []
		self.LIST__CHECK_3 = []
		self.LIST__ERROR_SELL_ALL = [] # always should be a logical value
		
		# @ state machine model 생성
		self.MACHINE = Machine(model=self, states = self.LIST__states, initial='WAKEUP')

		# @ Transition 설정
		self.MACHINE.add_transition(trigger='run_ERROR_CANT_RUN', source='*', dest='ERROR_CANT_RUN', conditions=['FUNC__to_ERROR_CANT_RUN'])
		self.MACHINE.add_transition(trigger='run_ERROR_CANT_RUN_recover', source='ERROR_CANT_RUN', dest='CHECK_1', conditions=['FUNC__to_CHECK_1'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_CHECK_1', source='WAKEUP', dest='CHECK_1', conditions=['FUNC__to_CHECK_1'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_CHECK_2', source='CHECK_1', dest='CHECK_2', conditions=['FUNC__to_CHECK_2'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_FILTER',  source='CHECK_2', dest='FILTER', conditions=['FUNC__to_FILTER'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_CHECK_3', source='FILTER', dest='CHECK_3',conditions=['FUNC__to_CHECK_3'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_CHECK_3_recover', source='CHECK_3', dest='FILTER',conditions=['FUNC__to_CHECK_3_recover'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_ERROR_SELL_ALL', source='CHECK_3', dest='ERROR_SELL_ALL',
									conditions=['FUNC__to_ERROR_SELL_ALL'])
		self.MACHINE.add_transition(trigger='run_FUNC__to_ERROR_SELL_ALL_recover', source='ERROR_SELL_ALL', dest='CHECK_3',
									conditions=['FUNC__to_ERROR_SELL_ALL_recover'])


	def FUNC__main(self, input_list_1, input_list_2, input_list_filter, input_list_3, error_sell_list):
		"""
		전체 transition wrapping
		:return:
		"""
		self.LIST__CHECK_1 = copy.deepcopy(input_list_1)
		self.LIST__CHECK_2 = copy.deepcopy(input_list_2)
		self.LIST__FILTER = copy.deepcopy(input_list_filter)
		self.LIST__CHECK_3 = copy.deepcopy(input_list_3)
		self.LIST__ERROR_SELL_ALL = copy.deepcopy(error_sell_list)

		try:
			self.run_FUNC__to_CHECK_1()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_CHECK_2()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_FILTER()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_CHECK_3()
		except Exception as e:
			pass
		try:
			self.run_ERROR_CANT_RUN()
		except Exception as e:
			pass
		try:
			self.run_ERROR_CANT_RUN_recover()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_CHECK_3_recover()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_ERROR_SELL_ALL_recover()
		except Exception as e:
			pass
		try:
			self.run_FUNC__to_ERROR_SELL_ALL()
		except Exception as e:
			pass

	def FUNC__to_FILTER(self):
		try:
			if type(self.LIST__FILTER) != type(list()) and len(self.LIST__FILTER) == 0: # 잘못된 체크
				#self.run_ERROR_CANT_RUN()
				return False

			else:
				tmp_flag_filter = True
				for item_filter in self.LIST__FILTER :
					if type(item_filter) == type(int()):
						if item_filter != 0:
							tmp_flag_filter = False
					else:
						if item_filter == False:  # 무조건 return 값들 0 혹은 True로 맞춰야 한다
							tmp_flag_filter = False
				print('tmp_flag_filter :: ', tmp_flag_filter )
				print('self.LIST__FILTER :: ', self.LIST__FILTER)
				if tmp_flag_filter == True : # state change ready
					#self.run_FUNC__to_CHECK_1() # state change
					return True
				else: # 로그인 등 모든 것이므로
					#self.run_ERROR_CANT_RUN
					return False

		except Exception as e:
			print('error in LIST__FILTER :: ', e)
			return False		

	def FUNC__to_ERROR_SELL_ALL_recover(self):
		return not self.FUNC__to_ERROR_SELL_ALL()

	def FUNC__to_ERROR_SELL_ALL(self):
		try:
			if type(self.LIST__ERROR_SELL_ALL) != type(list()) and len(self.LIST__ERROR_SELL_ALL) == 0: # 잘못된 체크
				#self.run_ERROR_CANT_RUN()
				return False

			else:
				tmp_flag_error_sell_all = True
				for item_error in self.LIST__ERROR_SELL_ALL :
					if type(item_error) == type(int()):
						if item_error != 0:
							tmp_flag_error_sell_all = False
					else:
						if item_error == False:  # 무조건 return 값들 0 혹은 True로 맞춰야 한다
							tmp_flag_error_sell_all = False
				print('tmp_flag_1 :: ', tmp_flag_error_sell_all )
				print('self.LIST__CHECK_1 :: ', self.LIST__ERROR_SELL_ALL)
				if tmp_flag_error_sell_all == True : # state change ready
					#self.run_FUNC__to_CHECK_1() # state change
					return True
				else: # 로그인 등 모든 것이므로
					#self.run_ERROR_CANT_RUN
					return False

		except Exception as e:
			print('error in LIST__ERROR_SELL_ALL :: ', e)
			return False

	def FUNC__to_ERROR_CANT_RUN(self):
		return not self.FUNC__to_CHECK_1() # wakeup에서 1로 가는 방향의 반대이니깐

	def FUNC__to_CHECK_1(self):
		try:
			if type(self.LIST__CHECK_1) != type(list()) and len(self.LIST__CHECK_1) == 0: # 잘못된 체크
				#self.run_ERROR_CANT_RUN()
				return False

			else:
				tmp_flag_1 = True
				for item_1 in self.LIST__CHECK_1 :
					if type(item_1) == type(int()):
						if item_1 != 0:
							tmp_flag_1 = False
					else:
						if item_1 == False:  # 무조건 return 값들 0 혹은 True로 맞춰야 한다
							tmp_flag_1 = False
				print('tmp_flag_1 :: ', tmp_flag_1 )
				print('self.LIST__CHECK_1 :: ', self.LIST__CHECK_1)
				if tmp_flag_1 == True : # state change ready
					#self.run_FUNC__to_CHECK_1() # state change
					return True
				else: # 로그인 등 모든 것이므로
					#self.run_ERROR_CANT_RUN
					return False

		except Exception as e:
			print('error in FUNC__to_CHECK_1 :: ', e)
			return False


	def FUNC__to_CHECK_2(self):
		try:
			if type(self.LIST__CHECK_2) != type(list()) and len(self.LIST__CHECK_2) == 0:  # 잘못된 체크
				#self.run_ERROR_CANT_RUN()
				return False

			else:
				tmp_flag_2 = True
				for item_2 in self.LIST__CHECK_2:
					if type(item_2) == type(int()):
						if item_2 != 0:
							tmp_flag_2 = False
					else:
						if item_2 == False:  # 무조건 return 값들 0 혹은 True로 맞춰야 한다
							tmp_flag_2 = False
				print('tmp_flag_2 :: ', tmp_flag_2 )
				print('self.LIST__CHECK_2 :: ', self.LIST__CHECK_2)
				if tmp_flag_2 == True:  # state change ready
					#self.run_FUNC__to_CHECK_2()  # state change
					return True
				else:
					return False
		except Exception as e:
			print('error in FUNC__to_CHECK_2 :: ', e)
			return False

	def FUNC__to_CHECK_3(self):
		try:
			if type(self.LIST__CHECK_3) != type(list()) and len(self.LIST__CHECK_3) == 0:  # 잘못된 체크
				#self.run_ERROR_CANT_RUN()
				return False

			else:
				tmp_flag_3 = True
				for item_3 in self.LIST__CHECK_3:
					if type(item_3) == type(int()):
						if item_3 != 0:
							tmp_flag_3 = False
					else:
						if item_3 == False:  # 무조건 return 값들 0 혹은 True로 맞춰야 한다
							tmp_flag_3 = False
				print('tmp_flag_3 :: ', tmp_flag_3 )
				print('self.LIST__CHECK_3 :: ', self.LIST__CHECK_3)
				if tmp_flag_3 == True:  # state change ready
					#self.run_FUNC__to_CHECK_3()  # state change
					return True
				else:
					return False
		except Exception as e:
			print('error in FUNC__to_CHECK_3 :: ', e)
			return False

	def FUNC__to_CHECK_3_recover(self):
		return not self.FUNC__to_CHECK_3() # 3로 가는 방향의 반대이니깐


#======================================================================================
#======================================================================================
class STOCK_IN_ATTENTION: # api focusing하는 현재 주식
	def __init__(self):
		self.code = None
		self.name = None
		self.accno = None
		self.balance = None
		self.action = None
		self.state = None
		self.order_num = None # 주문 개수
		self.orderno = None # 주문번호
		self.stock_list_for_sqlite = []



#======================================================================================
#======================================================================================
class Time_stage:
	"""
	자동으로 장중 시간 계산하여 return 해주는 부분
	
	"""
	timesec__15_30 = (((12 + 3) * 60) + 30) * 60
	timesec__24 = (((12 + 12) * 60)) * 60
	timesec__9 = (((9) * 60)) * 60

	def __init__(self):
		self.stage = None
		self.weekday_num = None
		
		# @ 자동 계산
		self.func_week_num_stage()
		self.func_today_stage()

	def func_week_num_stage(self):
		today = datetime.datetime.now()
		self.weekday_num = today.weekday()

	def func_today_stage(self):
		today = datetime.datetime.now()
		today_9 = copy.deepcopy(today).replace(hour=9, minute=0, second=0, microsecond = 0)
		today_15_30 = copy.deepcopy(today).replace(hour=15, minute=30, second=0, microsecond = 0)
		#today_24 = copy.deepcopy(today).replcae(hour=24, minute=0, second=0, microsecond = 0)

		if today < today_9 :
			self.stage = "개장전"
		elif today >= today_9 and today <= today_15_30:
			self.stage = "개장중"
		elif today > today_15_30 :
			self.stage = "개장후"

	def func_time_now_to_sec(self):
		today_date = datetime.datetime.now()
		today_sec = (((today_date.hour) * 60) * 60) + (today_date.minute * 60) + today_date.second

		return today_sec

class TimeAxisItem(pg.AxisItem): 
	def __init__(self, *args, **kwargs): 
		super().__init__(*args, **kwargs)
		self.setLabel(text='Time(min)', units=None) 
		self.enableAutoSIPrefix(False)
		#self.setTicks()
		
	def tickStrings(self, values, scale, spacing):
		""" 
		override 하여, tick 옆에 써지는 문자를 원하는대로 수정함. values --> x축 값들 ;
		숫자로 이루어진 Itarable data --> ex) List[int]
		""" # print("--tickStrings valuse ==>", values)
		#return [time.strftime("%m-%d\n%H:%M:%S", time.localtime(local_time)) for local_time in values]
		#datetime.datetime.strptime(datetime_data, "%Y%m%d%H%M%S")
		#return [datetime.datetime.strptime("%m-%d\n%H:%M:%S", datetime.datetime.fromtimestamp(time.localtime(local_time)) ) for local_time in values]

		#return [datetime.datetime.strptime("%m-%d\n%H:%M:%S",pd.to_datetime(local_time).to_pydatetime()) for local_time in values]
		#return [ pd.to_datetime(local_time).to_pydatetime().strftime("%m-%d\n%H:%M:%S") for local_time in values]
		#return [print('type(local_time) :: ',type(local_time)) for local_time in values]
		# datetime.datetime.strptime
		#return [datetime.datetime.fromtimestamp(value).strftime("%m-%d\n%H:%M:%S") for value in values]
		return [time.strftime("%m-%d\n%H:%M:%S", time.localtime(local_time)) for local_time in values]


class Single_stock:

	# @ 그냥 변수할당으로 pointer 수준에서 access 가능하게만 할 것 반드시!!
	SINGLE_CLASS_VAR__tuple_list_watch = None
	SINGLE_CLASS_VAR__tuple_list_trans = None
	SINGLE_CLASS_VAR__profit_rate_micro = None
	SINGLE_CLASS_VAR__profit_rate_overall = None
	SINGLE_CLASS_VAR__fe_article = None

	SINGLE_CLASS_FUNC_del_scrno = None
	SINGLE_CLASS_FUNC_apply_scrno = None
	#SINGLE_CLASS_FUNC_db_consistancy = None

	SINGLE_CLASS_POINTER__STOCK_IN_ATTENTION = None

	__slots__ = ['SINGLE_VARIABLE__name', 'SINGLE_VARIABLE__stock_code', 'SINGLE_VARIABLE__stage', 'SINGLE_VARIBALE_DICTIONARY__stage_name', 'SINGLE_VARIABLE__owning_number', 'SINGLE_VARIABLE__unmet_number', 'SINGLE_VARIABLE_BUDGET__allowed', 'SINGLE_VARIABLE_BUDGET__real_bought', 'SINGLE_VARIABLE_BUDGET__real_profit', 'SINGLE_VARIABLE_STOCK__expected_profit', 'SINGLE_BOOL__article_exist', 'SINGLE_BOOL__sqlite_date_consistancy', 'SINGLE_CHECK_FLAG__DODO__start_watching',  'SINGLE_CHECK_FLAG__DODO__stop_watching', 'SINGLE_CHECK_FLAG__DODO__at_active', 'SINGLE_CHECK_FLAG__DODO__sell_all', 'SINGLE_CHECK_FLAG__DODO__db_check_needed', 'SINGLE_DATETIME_OBJ__last_data_consist_check', 'SINGLE_DATETIME_OBJ__last_AT_trade_on_check' ]

	def __init__(self, name, code):
		self.SINGLE_VARIABLE__name = str(name)
		self.SINGLE_VARIABLE__stock_code = str(code)

		self.SINGLE_VARIABLE__stage = 0
		self.SINGLE_VARIBALE_DICTIONARY__stage_name = 	{
		0:"매매금지-기사",
		1:"매매금지-데이터", 
		2:"매매후보", 
		3:"감시수행", 
		4:"자동수행",
		5:"자동보류",
		6:"자동금지",
		7:"모니터링"}
		
		
		self.SINGLE_VARIABLE__owning_number = 0 # 들고있는 개수
		self.SINGLE_VARIABLE__unmet_number = 0 # 미수 개수
		self.SINGLE_VARIABLE_BUDGET__allowed = 0 # 허용 된 금액 총량
		self.SINGLE_VARIABLE_BUDGET__real_bought = 0 # 실제 구매한 금액 총량
		self.SINGLE_VARIABLE_BUDGET__real_profit = 0 # 실제 수익률 총량 -> 업데이트 FE단에서
		self.SINGLE_VARIABLE_STOCK__expected_profit = 0 # 예상 수익률 : future 30분~ or 그이상 시간동안
		
		

		
		#-------------------------------------------------------------
		# @ AT 시작할 때 사용해야함
		self.SINGLE_BOOL__article_exist = False # 결국 이거만 업데이트 되면 된다...
		self.SINGLE_BOOL__sqlite_date_consistancy = False

		self.SINGLE_CHECK_FLAG__DODO__start_watching = False # scrno 감시 충족하면 True로 올림
		self.SINGLE_CHECK_FLAG__DODO__stop_watching = False 
		self.SINGLE_CHECK_FLAG__DODO__at_active = False # 지금 자동 투자 돌아가고 있는지 여부 
		self.SINGLE_CHECK_FLAG__DODO__sell_all = False # 다팔라는 flag
		self.SINGLE_CHECK_FLAG__DODO__db_check_needed = False # db 체크 해야한다는 의미

		# @ datetime obj for future usage
		self.SINGLE_DATETIME_OBJ__last_data_consist_check = None # db 정합성 체크 부분, 시각
		self.SINGLE_DATETIME_OBJ__last_AT_trade_on_check = None # AT하고있다가 수익률 바닥이어서 뺄 때 시각
	
	def SINGLE_FUNC__get_stage(self):
		return self.SINGLE_VARIBALE_DICTIONARY__stage_name[self.SINGLE_VARIABLE__stage]

	def SINGLE_FUNC__init(self):
		self.__init__(self.SINGLE_VARIABLE__name, self.SINGLE_VARIABLE__stock_code)

	
	def SINGLE_FUNC__calc_stage(self):
		"""
		0:"매매금지-기사",
		1:"매매금지-데이터", 
		2:"매매후보", 
		3:"감시수행", 
		4:"자동수행",
		5:"자동보류",
		6:"자동금지"
		"""
		# @ common condition
		if self.SINGLE_VARIABLE__stage == 7: #반드시 모니터링
			self.SINGLE_CHECK_FLAG__DODO__start_watching = True
		# if self.SINGLE_VARIABLE__stage == 7 and self.SINGLE_VARIABLE__owning_number != 0 : -> 차후 구현


		if self.SINGLE_VARIABLE__stage == 6: # 재감시
			if self.SINGLE_DATETIME_OBJ__last_AT_trade_on_check != None: # 한번 AT 진입했던 것을 체크
				if datetime.datetime.now() - self.SINGLE_DATETIME_OBJ__last_AT_trade_on_check >= datetime.timedelta(minutes=60):
					self.SINGLE_VARIABLE__stage = 0
			else: # unreachable
				pass
		
		#### DB 30분 경과 후 나머지 ok이면 get하는 것도 구현
		if self.SINGLE_BOOL__sqlite_date_consistancy == False: # sqlite false
			if self.SINGLE_VARIABLE__stage == 1 :
				if self.SINGLE_DATETIME_OBJ__last_data_consist_check != None:
					if datetime.datetime.now() - self.SINGLE_DATETIME_OBJ__last_data_consist_check >= datetime.timedelta(minutes=60):
						self.SINGLE_CHECK_FLAG__DODO__db_check_needed = True
						self.SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()
				else:
					self.SINGLE_CHECK_FLAG__DODO__db_check_needed = True
					#self.SINGLE_DATETIME_OBJ__last_data_consist_check = datetime.datetime.now()
			
			elif self.SINGLE_BOOL__article_exist == True :
				self.SINGLE_CHECK_FLAG__DODO__db_check_needed = True
		
		#### SCRNO 업데이트
		tmp_bool = False
		# stock code is in the list_watch
		for tuple_item in Single_stock.SINGLE_CLASS_VAR__tuple_list_watch:
			if tuple_item[0] == self.SINGLE_VARIABLE__stock_code:
				tmp_bool = True
				break
		if self.SINGLE_CHECK_FLAG__DODO__start_watching == False and  tmp_bool == True and self.SINGLE_VARIABLE__stage >= 2:
			self.SINGLE_CHECK_FLAG__DODO__start_watching = True
		elif self.SINGLE_CHECK_FLAG__DODO__start_watching == True and  tmp_bool == False and self.SINGLE_VARIABLE__stage != 7:
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = True
		elif self.SINGLE_CHECK_FLAG__DODO__start_watching == True and self.SINGLE_VARIABLE__stage <= 2:
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = True
			

		if self.SINGLE_BOOL__article_exist == False:
			self.SINGLE_VARIABLE__stage = 0
		else:
			if self.SINGLE_BOOL__sqlite_date_consistancy == False:
				self.SINGLE_VARIABLE__stage = 1
			else:
				if self.SINGLE_CHECK_FLAG__DODO__start_watching == False:
					self.SINGLE_VARIABLE__stage = 2
				else:
					if self.SINGLE_CHECK_FLAG__DODO__at_active == False:
						self.SINGLE_VARIABLE__stage = 3
					else:
						if not(self.SINGLE_VARIABLE__stage == 5 or self.SINGLE_VARIABLE__stage == 6 or self.SINGLE_VARIABLE__stage == 7 ):
							self.SINGLE_VARIABLE__stage = 4
							self.SINGLE_DATETIME_OBJ__last_AT_trade_on_check = datetime.datetime.now()
						else: # 5, 6번은 flag에서 해당 action 수행하고 바꿔줌
							pass
		

		
		# @ special condition changes
		# 1)감시 이하인데 들고있음
		if (self.SINGLE_VARIABLE__stage in [0,1,2,3] )and self.SINGLE_VARIABLE__owning_number != 0:
			self.SINGLE_VARIABLE__stage = 5 # sell all
		
		# 2)감시인데 micro이율보다 작음
		elif self.SINGLE_VARIABLE__stage == 3 and self.SINGLE_VARIABLE__owning_number == 0 and self.SINGLE_VARIABLE_STOCK__expected_profit < Single_stock.SINGLE_CLASS_VAR__profit_rate_micro:
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = True
			self.SINGLE_VARIABLE__stage = 6

		# 3)자동수행이었다가 들고있는게 없으면, 감시수행으로 하향
		elif self.SINGLE_VARIABLE__stage == 4 and self.SINGLE_VARIABLE__owning_number == 0:
			self.SINGLE_VARIABLE__stage = 3
		
		# 4) 자동트레이딩이었는데 들고있다가 종목전체수익이 떨어지면 매도로 전환
		elif self.SINGLE_VARIABLE__stage == 4 and self.SINGLE_VARIABLE__owning_number != 0 and self.SINGLE_VARIABLE_BUDGET__real_profit < Single_stock.SINGLE_CLASS_VAR__profit_rate_overall:
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = True
			self.SINGLE_VARIABLE__stage = 6 # sell all and dont touch for 30 minutes
			self.SINGLE_DATETIME_OBJ__last_AT_trade_on_check = datetime.datetime.now() # 기록
			

		# @ special condition to flag for action
		if self.SINGLE_VARIABLE__stage == 5 and self.SINGLE_VARIABLE__owning_number != 0:
			self.SINGLE_CHECK_FLAG__DODO__sell_all = True
		else:
			if self.SINGLE_VARIABLE__stage == 5 and self.SINGLE_VARIABLE__owning_number == 0:
				self.SINGLE_CHECK_FLAG__DODO__sell_all = False
				self.SINGLE_VARIABLE__stage = 0

		if self.SINGLE_VARIABLE__stage == 6 and self.SINGLE_VARIABLE__owning_number != 0:
			self.SINGLE_CHECK_FLAG__DODO__sell_all = True


		# @ ACTION수행
		if self.SINGLE_CHECK_FLAG__DODO__stop_watching == True:
			Single_stock.SINGLE_CLASS_POINTER__STOCK_IN_ATTENTION.code = self.SINGLE_VARIABLE__stock_code # stock code 할당
			Single_stock.SINGLE_CLASS_FUNC_del_scrno() # 없애는 과정 수행
			self.SINGLE_CHECK_FLAG__DODO__start_watching = False
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = False

		if self.SINGLE_CHECK_FLAG__DODO__start_watching == True:
			Single_stock.SINGLE_CLASS_POINTER__STOCK_IN_ATTENTION.code = self.SINGLE_VARIABLE__stock_code # stock code 할당
			Single_stock.SINGLE_CLASS_FUNC_apply_scrno() # 할당하는 과정 수행
			self.SINGLE_CHECK_FLAG__DODO__stop_watching = False

		# if self.SINGLE_CHECK_FLAG__DODO__db_check_needed == True:
		# 	if Single_stock.SINGLE_CLASS_FUNC_db_consistancy(self.SINGLE_VARIABLE__stock_code):
		# 		self.SINGLE_BOOL__article_exist = True
		# 	else:
		# 		self.SINGLE_BOOL__article_exist = False


		
class Bit_32:

	NUM_OF_MAX_REPEAT_IN_AT_STAGE = 5 # 최대로 반복할 stage 기준 값, 넘으면 error 띄운다.
	NUM_OF_MAX_WAIT_IN_AT_STAGE = 100 # 최대로 대기할 wait

	def __init__(self, test):

		self.TEST = test
	
		self.counter_error = 0
		self.counter_wait = 0
		
		self.stage = 0
		self.dictionary_stage = {
									0 : 'WORKING',
									1 : 'P32_OUTPUT',
									2 : 'P32_SEND_READY',
									3 : 'P32_READ_RECIEVE',
									4 : 'P64_READ_READY',
									5 : 'P64_INPUT',
									6 : 'P64_SEND_RECIEVE'
		}
		
		self.comms_32_dictionary = {'P32_SEND_READY':False , 'P64_SEND_RECIEVE':False} # 32bit 파일 준비완료 알림 / 64에게 받았다고 알림
		self.comms_64_dictionary = {'P64_SEND_READY':False , 'P32_SEND_RECIEVE':False} # 64bit 파일 준비완료 알림 / 32에게 받았다고 알림
		
		self.comms_send_path = None
		self.comms_read_path = None
		self.func_COMM_PICKLE__file_path()
	
	def func_COMM_PICKLE__send_32(self):
		"""
		32 bit 보내는 것!
		self.comms_32_dictionary 를 self.comms_send_path 에다가
		"""
		try:
			
			with open(self.comms_send_path, 'wb') as file:
				pickle.dump(self.comms_32_dictionary, file)
			print('successful pickle save in Bit_32 - func_COMM_PICKLE__send_32')
		
		except Exception as e:
			print('error in Bit_32 - func_COMM_PICKLE__send_32 :: ', e)
			return 'AT_FAIL'
				
	def func_COMM_PICKLE__read_64(self):
		"""
		64 bit 읽는 것!
		self.comms_64_dictionary 를 self.comms_read_path 에서
		"""
		try:
			if self.TEST == True:
				if os.path.isfile(self.comms_read_path):
					with open(self.comms_read_path, 'rb') as file:
						self.comms_64_dictionary = copy.deepcopy(pickle.load(file))
					print('successful pickle read from Bit_64 - func_COMM_PICKLE__read_64')
				else:
					return 'AT_FAIL' ############이부분으로 comms stage이동 확인!
			else:
				with open(self.comms_read_path, 'rb') as file:
					self.comms_64_dictionary = copy.deepcopy(pickle.load(file))
				print('successful pickle read from Bit_64 - func_COMM_PICKLE__read_64')

		except Exception as e:
			print('error in Bit_32 - func_COMM_PICKLE__read_64 :: ', e)
			return 'AT_FAIL'		
		
	
	def func_get_stage(self):
		"""
		stage 값을 return 해주는 부분, error까지 같이 잡음
		"""
		return self.dictionary_stage[self.stage]
	
	
	def func_COMM_PICKLE__file_path(self):
		"""
		communication 용 pickle 파일 경로 지정부분
		"""
		python_path = os.getcwd()
		db_path = str(python_path + '\\KIWOOM_API__ML__COMMON').replace('/', '\\')
		
		if os.path.isdir(db_path): # 경로 존재하는지 확인
			pass
		else:
			os.mkdir(db_path) # 경로 생성
		file_path_send = db_path + '\\' + 'COMM_32.p'
		file_path_read = db_path + '\\' + 'COMM_64.p'
		
		self.comms_send_path = copy.deepcopy(file_path_send)
		self.comms_read_path = copy.deepcopy(file_path_read)
			
	
	def func_proceed(self, **kwargs): # **kwarg로 받아서 매 stage run할 함수들 가변해서 돌림
		"""
		다음 스테이지로 넘어가는 부분
		
		0 : 'WORKING',
		1 : 'P32_OUTPUT',
		2 : 'P32_SEND_READY',
		3 : 'P32_READ_RECIEVE',
		4 : 'P64_READ_READY',
		5 : 'P64_INPUT',
		6 : 'P64_SEND_RECIEVE'
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
			tmp_stage_string = self.func_get_stage()
			tmp_upper_stage = return_upper_wrapping_stage(kwargs)
			tmp_return = []
			print(f'tmp_stage_string : {tmp_stage_string}  \ntmp_upper_stage : {tmp_upper_stage} \nkwargs : {kwargs}')
			
			if 'WORKING' in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string) #pass # working 끝나고 마지막에 부르면 될 듯
			
			elif 'P32_OUTPUT' in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				
			elif "P32_SEND_READY" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				
				self.comms_32_dictionary['P64_SEND_RECIEVE'] = False # P64_SEND_RECIEVE 사용 reset
				self.comms_32_dictionary['P32_SEND_READY'] = True # 보낼 값
				tmp_return.append(self.func_COMM_PICKLE__send_32()) # 보내고 local에 기록하고
				
			
			elif "P32_READ_RECIEVE" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				tmp_return.append(self.func_COMM_PICKLE__read_64())
				if self.TEST == False:
					if self.comms_64_dictionary['P32_SEND_RECIEVE'] == False: # 64에서 받은게 확인되어야 함
						tmp_return.append('WAITING')
					else:
						self.comms_32_dictionary['P32_SEND_READY'] = False	# ram에서 init
						tmp_return.append(self.func_COMM_PICKLE__send_32()) # 보내고 local에서도 init 기록하고
						self.counter_wait = 0 # waiting 카운터 reset

				else:
					pass
			
			elif "P64_READ_READY" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				tmp_return.append(self.func_COMM_PICKLE__read_64())
				if self.TEST == False:
					if self.comms_64_dictionary['P64_SEND_READY'] == False:
						tmp_return.append('WAITING')
					else:
						self.counter_wait = 0 # waiting 카운터 reset
				else:
					pass
			
			elif "P64_INPUT" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)
				
			elif "P64_SEND_RECIEVE" in tmp_stage_string and tmp_stage_string in tmp_upper_stage:
				tmp_return = run_kwargs(kwargs, tmp_stage_string)		
				self.comms_32_dictionary['P64_SEND_RECIEVE'] = True
				tmp_return.append(self.func_COMM_PICKLE__send_32())
			else:
				tmp_return = []
				#tmp_return.append('WAITING')
				
			
			# @ return 값 확인
			tmp_stage_fail = False # fail이 나면 True, 다음 stage로 못 넘어가야 함
			tmp_stage_repeat = False # True 나오면 다시 시작해야됨 stage를
			for return_value in tmp_return : 
				if return_value == "AT_FAIL":
					tmp_stage_fail = True
					#break
				else:
					pass
				
				if return_value == "WAITING":
					tmp_stage_repeat = True
				else:
					pass

			print(f'tmp_return in bit32 comms :: {tmp_return}')
			print('\n' * 2)
				
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

def func_sub__sqlite(ohlcv, path_list, codes):
	SQLITE__con_top = sqlite3.connect(path_list[0])
	df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
	print(df.head())
	df.to_sql(codes, SQLITE__con_top, if_exists='replace', index=False)
	SQLITE__con_top.close()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWindow = App_wrapper(None, None)
	myWindow.show()
	app.exec_()
