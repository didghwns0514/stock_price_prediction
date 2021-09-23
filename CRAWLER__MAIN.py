#-*-coding: utf-8-*-

## my files
##########################################################################
##########################################################################
from lexrankr import LexRank
import CRAWLER__ML_LSTM_lang
import CRAWLER__w2v_interface
import CRAWLER__beautiful_soup
import CRAWLER__Global

"""
lstm 과 에러 포함 예측하는식?
lstm은 참값 보정값 예측하는 애 따로
비지도 분류기 주식상승용으로

"""
##########################################################################

## PY files
##########################################################################
##########################################################################
from konlpy.tag import Komoran
from gensim.models import KeyedVectors
from gensim.summarization.summarizer import summarize
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
import datetime
import time
import re
import kss
import pickle
import threading
import copy
import math
import winsound
from prettytable import PrettyTable
from tabulate import tabulate
from beautifultable import BeautifulTable
import queue
import traceback
##########################################################################

# http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-oop-part-5-%EC%83%81%EC%86%8D%EA%B3%BC-%EC%84%9C%EB%B8%8C-%ED%81%B4%EB%9E%98%EC%8A%A4inheritance-and-subclass/
# ^ 상속 클래스 사용하는 법



class stock_list_get:

	def __init__(self):

		self.company_name_excel = []
		self.company_number_excel = []
		self.raw_stock_list()
		
	def return_stock_lists(self):
		return self.company_name_excel, self.company_number_excel

	def initialize_stock_list(self):
		self.raw_stock_list()
		#self.return_stock_lists()

	def raw_stock_list(self):
		# initialize
		self.company_name_excel = []
		self.company_number_excel = []


		get_from_net = False # 인터넷에서 종목 가져오기
		get_from_csv = False  # True 면 web 주소에서 가져옴, else시 api에서 가져온 정보 사용

		if get_from_net == False : #파일에서 가져오기
			if get_from_csv == True:
				print('fetching stock list from file...')
				base_dir = os.getcwd()
				file_name = '\\CRAWLER__necessary_data\\stocks.xlsx'
				# base_dir = sys.path.append(base_dir)
				# print(base_dir)
				full_dir = str(base_dir + file_name).replace('/', '\\')
				raw_stock_list = pd.read_excel(full_dir, encoding='utf-8')
				# full_dir = full_dir.replace('\\','/')
			else:
				try:  # load from pickle
					# @ pickle file load
					pickle_folder_path = os.getcwd().replace('/', '\\')
					pickle_path = str(pickle_folder_path + '\\KIWOOM_API__STOCK_LIST_PICKLE\\stock_list_pickle.p')
					
					# @ 중복 파일 없애기
					company_namez = ['대상', '청구', '고려', '수성', '모다', '대원', '나노', '도움', '남성', 'DB', '서원', '태양', '배럴', '한창',
									 '강원', '디자인', '타임', '한화', '유성', '카스', '파라다이스', '성화', '원진', '한일', '삼일', '신화', '선진',
									 '카카오M', '카카오엠', '화진', '전방', 'CL', '대호', '대유', '대동', '동선', '일신', '동양', '우진', '진도',
									 '삼신', '호승', '연우', '동신', '청구', '동성', '성화', '우영', '신원', '고영', '대원', '서한', '씨앗', '대망',
									 '삼미', '전방', '대국']
					company_numz = ['003600', '271400', '028260', '002950']  # 이름 중복되는 애들
					bank_name = []
					base_dir = os.getcwd()
					file_name = '\\CRAWLER__necessary_data\\bank_list.txt'
					full_dir = str(base_dir + file_name).replace('/', '\\')
					with open(full_dir, 'r', encoding='utf-8') as f:
						lines = f.readlines()
						for line in lines:
							bank_name.append(line.split('\n')[0])
					print('bank name : ', bank_name)

					company_namez.extend(bank_name)
					print('company_namez : ', company_namez)

					with open(pickle_path, 'rb') as file:
						tmp_stock_data_list = copy.deepcopy(pickle.load(file)) # str(code) + " : " + (name))
						for stock_str in tmp_stock_data_list: # literation하면서
							stock_name_tmp = stock_str.split(' : ')[1].strip()
							stock_code_tmp = stock_str.split(' : ')[0].strip()

							if (stock_name_tmp not in company_namez) and (stock_code_tmp not in company_numz) :
								self.company_name_excel.append(stock_name_tmp)
								self.company_number_excel.append(stock_code_tmp)
					print(self.company_name_excel)
					print(self.company_number_excel)

				except Exception as e:
					print('error in Crawling - get_csv_path else : ', e)
		
		else :
			print('fetching stock list from net...')
			raw_stock_list = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]

		if get_from_csv == True:
		
			raw_stock_list = raw_stock_list[['회사명', '종목코드']].copy()
			raw_stock_list = self.remove_unwanted_companies(raw_stock_list).copy()
			print(raw_stock_list.head(10))

			for i in range(len(raw_stock_list)):
				self.company_name_excel.append(raw_stock_list.iloc[i]['회사명'])
				self.company_number_excel.append(raw_stock_list.iloc[i]['종목코드'])

			#'코스피', '코스닥', '코스피지수', '코스닥지수',
			#additional_info = [['KODEX 200','069500'], ['KODEX 레버리지','122630'], ['KODEX 코스피','226490'], ['KODEX 코스닥 150','229200'], ['KODEX 코스닥150 레버리지','233740'], ['KODEX 코스피100','237350'], ['KODEX 200선물인버스2X','252670'], ['KODEX 코스닥150선물인버스','251340']]

			# for i in range(len(additional_info)):
			# 	self.company_name_excel.append(additional_info[i][0])
			# 	self.company_number_excel.append(additional_info[i][1])

		print('checking duplicate info...')
		duplicates = []
		for item in self.company_name_excel :
			if self.company_name_excel.count(item) > 1:
				duplicates.append(item)
		duplicates = list(set(duplicates))
		print('Duplicate list.... : \n' ,duplicates)
		print('number of duplicate items : ', len(duplicates))

		
	def remove_unwanted_companies(self, df):
		
		df.dropna(how='any', inplace=True)
		
		#회사명 필요없는 것 제거
		company_namez = ['대상', '청구', '고려', '수성', '모다', '대원', '나노', '도움', '남성', 'DB', '서원', '태양', '배럴', '한창','강원', '디자인', '타임', '한화', '유성', '카스', '파라다이스', '성화', '원진', '한일', '삼일', '신화', '선진', '카카오M', '카카오엠', '화진', '전방', 'CL', '대호', '대유', '대동','동선', '일신', '동양', '우진', '진도', '삼신', '호승', '연우', '동신', '청구', '동성', '성화', '우영', '신원', '고영', '대원', '서한','씨앗', '대망', '삼미', '전방', '대국']
		company_numz = ['003600', '271400', '028260', '002950'] # 이름 중복되는 애들
		bank_name = []
		base_dir = os.getcwd()
		file_name = '\\CRAWLER__necessary_data\\bank_list.txt'
		full_dir = str(base_dir + file_name).replace('/', '\\')
		with open(full_dir, 'r', encoding='utf-8') as f :
			lines = f.readlines()
			for line in lines:
				bank_name.append(line.split('\n')[0])
		print('bank name : ', bank_name)
		
		company_namez.extend( bank_name )
		print('company_namez : ', company_namez)
		
		index_name = df[ df['회사명'].isin(company_namez) ].index
		df.drop(index_name, inplace=True)

		index_numb = df[ df['종목코드'].isin(company_numz) ].index
		df.drop(index_numb, inplace=True)

		return df



class sentence_tokener(threading.Thread):  
	# 엑셀 작업은 crawling 단에서 저장용으로 먼저 돌린다.
	folder_path = os.getcwd()
	dic_path = str(folder_path + '\\CRAWLER__necessary_data\\user_dic.txt').replace('/', '\\')

	def __init__(self):
		print('sentence_tokener wake-up success')
		
		# Threading
		threading.Thread.__init__(self)
		
		# queue
		#self.queue_tokener = Global_lang.queue
		self.item_tmp = None

		# 코드 시작시간 표기
		self.start_time = time.time()
		self.start_point = datetime.datetime.now()
		
		self.kmr = Komoran(userdic=self.dic_path)

		self.news_blob = None
		self.company_name = None
		self.company_number = None
		
		self.dictionary = {}
		self.sentence_to_save_list = []
		self.sentence_score_list = []
		self.sentence_topic_list = []
		self.double_topic = True
		
		#데이터 베이스 관리용 class
		print('load database class...')
		self.database = Database() 
		
		# 처음 관심 대상 주식 ram에 load
		self.stock_list_obj = stock_list_get()
		self.company_name, self.company_number = self.stock_list_obj.return_stock_lists()
		print('company list in sentence tokener : ', self.company_name[:10])

		print('total elapsed waking the tokener time :  ', time.time() - self.start_time, '....')
		#self.start_tokener()
		
	def run(self):
		while True :
			self.elapsed_time(self.start_point)
			self.start_tokener()
			time.sleep(10)
	
	def elapsed_time(self, start_point): #시간 측정해서, 주식 리스트 업데이트
		now_time = datetime.datetime.now()
		duration_seconds = (now_time - start_point).total_seconds()
		minutes_duration = divmod(duration_seconds,60)[0]
		
		if minutes_duration > 60 * 1 : # every 1 hours after initial wake up
			# 해당 업데이트 initialize_stock_list
			self.stock_list_obj.initialize_stock_list()
			self.company_name, self.company_number = self.stock_list_obj.return_stock_lists()
			
			# 시간 메모리 때문 다시 업데이트
			self.start_point = now_time
		else:
			pass
	
	def start_tokener(self):
		print('entering get_queue...')
		self.get_queue()
		
		print('entering parser...')
		if self.item_tmp :
			self.parser()
		else:
			print('entering next iteration in lang_train...')
		
		self.database.print_interest()
		##################################
		#initialize variables
		self.item_tmp = copy.deepcopy(None)
		##################################

	def get_queue(self):
		# start = datetime.datetime.now()
		start = time.time()
		
		try:
			print('trying to get a queue...')
			self.item_tmp = CRAWLER__Global.queue.get(timeout=1)
			print('queue get done in lang_train...')
			
		except Exception as e:
			print('no available queue item in global queue : ', e)
			
		try:
			# pickle file local 저장된거 최신으로 부름
			print('pickle normal file load, check for data consistancy when queue is available...')
			self.database.load_pickle_norm()

		except Exception as e:
			print('no available pickle normal item in local : ', e)

		try :
			print('pickle final file load, check for data consistancy when queue is available...')
			self.database.load_pickle_final()
		except Exception as e:
			print('no available pickle final item in local : ', e)

		
		print('total elapsed working with queue file :  ', time.time() - start, '....')

	def parser(self):
		print('entered parser...')
		# sentences = self.kkma.sentences(paragraph) # returns lists
		# https://stackoverflow.com/questions/9542738/python-find-in-list
		# https://github.com/lovit/soynlp
		#################################
		"""
		데이터베이스 load 값과 queue 에서 가져온 것 - article 주소 값으로 새로운 데이터 확인,
		데이터베이스 4일치 지난 것 없는지 갱신
		
		새로운 데이터에 대해서만 작업(새로운 데이터 queue 3번째 항목으로 넣기)
		
		a마지막에 pickle 저장
		"""
		#################################
		
		
		# data consistancy
		# 데이터베이스 load 값과 queue 에서 가져온 것 - article 주소 값으로 새로운 데이터 확인,
		########################################################
		########################################################
		# 큐에 값이 있어서 이미 들어온 것
# 									#  old 포함df    #삭제용    #새것만
# 		self.queue_item_crw = [    None,      None,      None ]
		if self.database.database_dic : # pickle에 관련 파일이 있다면 -> 초기시 dictionary 비어있어서 가능
			###필요 없는 것 지우는 과정
			print('start processing with existing pickle file...')
			if not(self.item_tmp[1].empty) : # 삭제할 내용이 큐에 있었다면
				print('deletion required in queue, processing...')

				redundant_contents = self.parse_contents(self.item_tmp[1])
				tmp_redundant_hash = []
				for single_article_r in redundant_contents:
					# [tmp_title, tmp_time, tmp_content, tmp_article_address]
					tmp_redundant_hash.append(single_article_r[3])

				self.database.remove_redundant(tmp_redundant_hash) # &page= 제거한 부분으로... 그걸 데이터베이스에 넣어줌
			
			###새 dataframe에서 없는 내용 전체 비교해서 추가하고 저장
			# 해쉬(dictionary 형태로 변경하는게 나을 듯)
			if not(self.item_tmp[2].empty) :# 추가할 내용이 큐에 있었다면 -> 새로운 것만 가져오는 것으로 바꿈
											# 새로운것 + 올드한것 => 밑에 아무 것도 존재하지 않을 때 사용
				print('addition required in queue, processing...')
				new_with_old_contents = self.parse_contents(self.item_tmp[2])
# 				tmp_needed_contents = [] # 새로 받는 기사의 address만 저장된 list
# 				tmp_real_needed_contents = [] # 걸러진 기사 기준 받아야 하는 참된 list
# 				for single_article_n in new_with_old_contents:
# 					#get every database address
# 					tmp_needed_contents.append(single_article_n[3])
				
				### dictionary 형태로 hash화 된 database 사용하는 중
				database_hash = self.database.get_all_hash()
				try:
					print('database_hash : ', database_hash[:10])
				except Exception as e:
					print('printing database_hash err - lang_train')
					pass


# 				for single_article in self.news_blob:
# 					self.article_splitter(single_article)
# 				print('article processing finished with existing pickle file...')
				# database hash와 비교해서 해당하는 기사만 작업 들어감
				print('##########@@@@@@@@@@@@@@@!!!!!!!!!!!!!!')
				try:
					print(new_with_old_contents[:10])
				except:
					pass
				print('##########@@@@@@@@@@@@@@@!!!!!!!!!!!!!!')
				tmp_counter = 0
				for single_article in new_with_old_contents:
					# [tmp_title, tmp_time, tmp_content, tmp_article_address]
					tmp_single_article = copy.deepcopy(single_article)
					if not (str(tmp_single_article[3]).split('&page=')[0] in database_hash ) :
						tmp_counter = tmp_counter + 1
				print('total length of articles to add to database, after article splitting : ', int(tmp_counter))

				for single_article in new_with_old_contents:
					# [tmp_title, tmp_time, tmp_content, tmp_article_address]
					tmp_single_article = copy.deepcopy(single_article)
					if not (str(tmp_single_article[3]).split('&page=')[0] in database_hash):
						self.article_splitter(single_article)
				print('article processing finished with existing pickle file...')

				self.database.save_pickle()
				self.database.print_interest()
				print('pickle successfuly saved with filtered existing pickle file...')
			
		
		else: # 비어있다, 전체 처음부터 다시 하기
			print('start processing with empty-non_existing pickle file...')

			parsed_contents = self.parse_contents(self.item_tmp[0])
			print('##########@@@@@@@@@@@@@@@!!!!!!!!!!!!!!')
			try:
				print(parsed_contents[:10])
			except:
				pass
			print('##########@@@@@@@@@@@@@@@!!!!!!!!!!!!!!')
			for single_article in parsed_contents:
				self.article_splitter(single_article)
			print('whole new article processing finished...')

			self.database.save_pickle()
			self.database.print_interest()
			print('whole new pickle successfuly saved...')
		
		
		

	
	def parse_contents(self, df):

		tmp_content_ram = []
		tmp_title = ""
		tmp_content = ""
		tmp_time = ""
		tmp_article_address = ""


		print('length of df is : ', len(df))
		# https://wikidocs.net/4308 : 정규 표현식, 사용법
		for i in range(len(df)):
			"""
			self.content = str(content.split('@')[0].strip())
			self.news_title = str(news_title.split('기사입력')[0].strip())
			self.news_date = str(news_date.strip())
			"""
			tmp_title = str(df.iloc[i]['Title']).split('기사입력')[0].strip()
			tmp_content = str(df.iloc[i]['Content']).split('@')[0].strip()
			tmp_time = str(df.iloc[i]['Date']).split('기사입력')[0].strip()
			tmp_article_address = str(df.iloc[i]['Link']).strip()
			tmp_content_ram.append([tmp_title, tmp_time, tmp_content, tmp_article_address] )

		return tmp_content_ram


	def article_splitter(self, single_article) :

		article_time = single_article[1]
		article = single_article[2]
		article_title = single_article[0]
		article_address = single_article[3]
		#trim article 문장분리 잘 돌아가게
		article = article.split('▶')[0].split('-한국경제TV※')[0].replace('[표]',"").replace('◆',"").replace('[그래프]',"")#.replace('"', ' ').replace('	'," ").replace('\n',' ')
			# ^ 이거 문제 될 수도... !!
		article = article.replace("'", ' ').replace('사진', "").replace('=',"").replace('…'," ").replace('만', ' 만').replace('억', ' 억').replace('조', ' 조')
# 		print(article)
# 		print(type(article))
# 		input('?')
		


		sentences = kss.split_sentences(article)
		tmp_items_in_sentence = []  # reset just in case
		flag_find_first_topic_sentence = 0 # flag for first meaningful sentence

		result_list = []
		tmp_answer_save = None
		flag_tmp_answer = 0
		i = 0
		
		print('length of sentences parsed : ', len(sentences))
		for sentence in sentences:
			print('i - th sentence : ', i + 1)
			i = i + 1
			if len(sentence) > 300: # skip wrongly splitted sentence
				print('sentence skipped... : ', sentence)
				#input('?')
				continue
			else :
				
				#tmp_items_in_sentence = self.kmr.nouns(sentence)
				answer = self.list_compare(self.company_name, sentence, self.double_topic) # list containing items of company
				#print('@@@ answer list!!!!!!!! : ', answer)
				
				if answer == 'Filtered':
					continue
				else:
					pass
				
				if ((answer == 'No_result') and (flag_find_first_topic_sentence == 0)):# no match found, return is false
					#print('no topic of interest was in the sentence yet...')
					continue

				elif answer and flag_find_first_topic_sentence == 0: # 주제 있는 첫 문장일 시
					flag_find_first_topic_sentence = 1

				elif ((answer == 'No_result') and (flag_find_first_topic_sentence == 1 )): # 주제가 없는데 첫문장/이전 지점 주제 파악 됨
					result_list.append([tmp_answer_save, sentence])

				else:
					pass

				if answer and type(answer) == type(list()) and flag_find_first_topic_sentence == 1: # first analysis is able
					result_list.append([answer, sentence])
					tmp_answer_save = answer
				else:
					pass


		# when marking sentences are finished
		# ^ r_list =  [ [ [ , , , ] ,..... ], [ ], [ ] , ....   ]

		print('▶ finished analysis of sentences in article...')

		for i in range(len(result_list)):
			topic_list = result_list[i][0]
			topic_sentence = result_list[i][1]
			for j in range(len(topic_list)):
				str_comp_num = self.comp_num_get(self.company_name, self.company_number, topic_list[j])  # 해당 주식 코드 가져옴

				tokenized_sentence = sentence_to_token(self.kmr.pos(topic_sentence))
				"""
				print("%" * 80)
				print("%" * 80)
				print("%" * 80)
				print('info : ', topic_list[j], '  :::  ', str_comp_num, ' ... article time : ', article_time)
				print('original sentence : ', topic_sentence)
				print('tokened sentence : ', tokenized_sentence)
				print('-' * 80)
				print('kmr pos : ', self.kmr.pos(topic_sentence))
				"""
				tmp_article_address = str(copy.deepcopy(article_address)).split('&page=')[0]
				self.database.add_stock(topic_list[j], str_comp_num, article_time, topic_sentence, tokenized_sentence,tmp_article_address)

				"""
				print('#' * 80)
				print('#' * 80)
				"""

	def comp_num_get(self, list_company_name, list_company_num, found_name):
		# 코스피 : 226490
		# 코스닥 : 229200
		for i in range(len(list_company_name)):
			if not list_company_name[i] == found_name :
				if found_name == '코스피':
					return str('226490')
				elif found_name == '코스닥':
					return str('229200')

			else: # 찾는 이름이 주식 리스트에 포함되어있음
				stock_num_len = len(str(list_company_num[i]))
				fill_num_len = 6 - stock_num_len
				return str('0'*fill_num_len) + str(list_company_num[i])

	def list_compare(self, list_1, sentence, double_topic):
		# self.company_name, tmp_items_in_sentence
		start = time.time()
		result = False
		return_list = []
		si_result_list = []
		
		list_2 = self.kmr.nouns(sentence)
		list_check_ends = self.kmr.pos(sentence)
		flag_end_check = 0
		
		flag_kospi_company_check = 0

		special_interest = ['코스피', '코스닥', '코스피지수', '코스닥지수', '코스피 지수', '코스닥 지수'] #, '주식시장', '주식 시장']
		spam_words = ['섹시', '치어리더', '비키니', '상품권', '▶', '청춘뉘우스',  '△' ,'한화 ', '레이더M', '코인적립',  '장외주식', '장외 주식', '장외종목', '장외 종목', '앵커', '결혼', '노조', '금수저', '자동생성',  '업종별', '신한', '펀드', 'MK라씨로', '투게더앱스', '표', 'MK파운트', '상한가']

		try:
			#2개이상 종목 포함할지 말지 결정하는 swich문 추가해줄 것
			for k in range(len(list_check_ends)):
				if list_check_ends[k][0] == '다':
					flag_end_check = 1
				if list_check_ends[k][0] == '회사' or list_check_ends[k][0] == '종목' or list_check_ends[k][0] == '업체' or list_check_ends[k][0] == '상장': #and list_check_ends[k][1] =='NNG':
					flag_kospi_company_check = 1
			if flag_end_check == 0 :
				#print('   not a complete sentence ...')
				return 'Filtered'
			
			for y in range(len(list_2)):
				if list_2[y] in special_interest :
					si_result_list.append(list_2[y])

			for x in range(len(list_1)):
				for y in range(len(list_2)):
					if ( (list_1[x] in spam_words) or (list_2[y] in spam_words)  ): #filter advertisement word
						#print('   total elapsed comparing list :  ', time.time() - start, '....')
						#print('   spam word has been filtered.. ',list_1[x], ' and ', list_2[y])
						return 'Filtered'

					else: #no spam words filtered

						if list_1[x] == list_2[y]: # 같은 항목 있음
							#result = list_1[x] #끝까지 비교한다 마지막에 카운터 1 일 때 돌려주려고
							return_list.append(list_1[x])
						else:
							pass



			if len(return_list) == 0:
				#print('   found no meaningful result 1st so ever...')
				if len(si_result_list) == 0 or len(si_result_list) >= 2:
					#print('   total elapsed comparing list :  ', time.time() - start, '....')
					#print('   found no meaningful result 2nd what so ever...')
					return 'No_result'

				elif len(si_result_list) == 1: # 지수 관련 1개 찾았다면?!
					if flag_kospi_company_check == 0 : #회사라는 단어가 없어서, 코스피 그대로 코스피 의미 가져감
						#print('   total elapsed comparing list :  ', time.time() - start, '....')
						#print('   found meaningful result in si_ ...', si_result_list)
						#print('\n')

						# return 값 trim하기
						if si_result_list[0] == '코스피지수' or si_result_list[0] == '코스피 지수':
							si_result_list[0] = '코스피'
						elif si_result_list[0] == '코스닥지수'or si_result_list[0] == '코스닥 지수':
							si_result_list[0] = '코스닥'

						return si_result_list
					else: #회사라는 대명사가 쓰임
						#print('   total elapsed comparing list :  ', time.time() - start, '....')
						#print('   found no meaningful result 3rd what so ever...')
						return 'No_result'
					
				else:
					#print('   unexpected result...')
					return 'Filtered'

			else:
				if len(return_list) == 1 and double_topic == False : # 한개 찾았고, 한개만 내보낼 설정일 때
					#print('   total elapsed comparing list :  ', time.time() - start, '....')
					return_list = list(set(return_list)) # make a unique list
					#print('   found ',len(return_list),' number of meaningful results...')
					#print('list : ', return_list)
					#print('\n')

					return return_list
				else: # 2 개 이상
					if double_topic == True : # 가능하다면

						#print('   total elapsed comparing list :  ', time.time() - start, '....')
						return_list = list(set(return_list)) # make a unique list
						#print('   found ',len(return_list),' number of meaningful results...')
						#print('list : ', return_list)
						#print('\n')

						return return_list
					else:
						return 'Filtered'


		except Exception as e:
			"""
			print('   total elapsed comparing list :  ', time.time() - start, '....')
			print('   error in list compare : ',e)
			print('\n')
			"""
			return 'Filtered'

def sentence_to_token(sentence_pos_list):
	# kmr pos 된 것 넣어서... 다시 합치기!!
	result_list = []
	skipped_list = ['한국경제TV', '기자', '라이온봇', '씽크풀', '한국경제신문', '이데일리TV', '서울경제', '연합뉴스','사진','로이터' ,'머니투데이', '이데일리', '흥극증권', '키움증권', '한경닷컴','마켓인사이트', '헤럴드경제','아시아경제' ,'인천공항','뉴시스','베이징신화', '아이뉴스24']

	if isinstance(sentence_pos_list, list) and len(sentence_pos_list) != 0 :
		for tokens  in sentence_pos_list :
			if tokens[0] in ['-', '+', '%', '그러나', '?', '!', '하지만']:
				result_list.append(tokens[0])
			elif tokens[0] in ['코스피지수']:
				result_list.append('코스피')
			elif tokens[0] in ['코스닥지수']:
				result_list.append('코스닥')
			elif tokens[0] in ['코스닥시장', '코스피 시장']:
				result_list.append('코스피')
			elif tokens[0] in ['코스피 지수']:
				result_list.append('코스피')
			elif tokens[0] in ['코스닥 지수']:
				result_list.append('코스닥')
			elif tokens[0] in ['코스닥시장','코스닥 시장']:
				result_list.append('코스닥')
			elif tokens[0] in skipped_list :
				pass


			elif tokens[1] in ['SN', 'NNG', 'NNP', 'NNB', 'VV', 'VX', 'VA', 'XSV', 'EC', 'JKB', 'MAG', 'JX', 'VCN', 'VCP', 'XR', 'SL']:
				result_list.append(tokens[0])

			else:
				pass
		"""
		print('length of tokenized sentence... : ',  len(result_list))
		print('^'*40)
		"""

		return result_list

	else:
		return None


class Database:
	
	# pickle 변수 자체들
	database_dic = {}
	database_final_im = None
	database_final_lg = None

	def __init__(self):
		# @ Test winsound
		print('check window sound...')
		self.window_sound()

		# @ LSTM loaded
		self.LSTM_obj = CRAWLER__ML_LSTM_lang.LSTM(module=True) # lstm 로드 시킴


		# @ dictionary loaded
		self.folder_path = os.getcwd()
		self.dic_path = str(self.folder_path + '\\CRAWLER__necessary_data\\user_dic.txt').replace('/', '\\')
		self.vector_path = str(self.folder_path + "\\CRAWLER__vector" + "\\word_vector_1.1.txt")

		self.loaded_model_obj = KeyedVectors.load(self.vector_path)

		# @ sentence to vector
		self.w2v_obj = CRAWLER__w2v_interface.Meanvector_w2v(self.loaded_model_obj)

		# @ pickle
		self.pickle_folder_path = str(self.folder_path + '\\CRAWLER__pickle').replace('/', '\\')
		self.pickle_path = str(self.pickle_folder_path + '\\pickle.p')
		self.pickle_final_path_im = str(self.pickle_folder_path + '\\pickle_final_im.p')
		self.pickle_final_path_lg = str(self.pickle_folder_path + '\\pickle_final_lg.p')
			# -----------------------
		self.pickle_im_obj = None
		self.pickle_ig_obj = None
			# -----------------------
			# Tracker for the new 15
		self.im_log_15_before = None
		self.im_log_15_now = None
		self.im_log_new_entry = None
		self.lg_log_10_now = None
		self.im_lg_overlap = None
		
		# @ text ranker
		self.lex = LexRank()

		# @ check
		self.flag_database_dic_check = 0 # 0으로 print 한번하고 1로 올리고, 한번만 print 해보게 함
		try:
			self.load_pickle_norm()
			self.remove_redundant_2()
			self.save_pickle()
		except Exception as e:
			print('error in initializing Database : ', e)

	def window_sound(self):
		winsound.Beep(2000, 2300)


	def save_pickle(self):
		try:
			#self.tf_idf() # 요약 추가
			pass
		except Exception as e:
			print('save pickle tf-itf err_2 : ', e)

		with open(self.pickle_path, 'w+b') as file:
			pickle.dump(self.database_dic, file)
			print('pickle successfully saved...!')

		self.database_final_im = sort(dictionary_to_tuple_imminent(self.database_dic))
		self.database_final_lg = sort(dictionary_to_tuple_longterm(self.database_dic))

		with open(self.pickle_final_path_im, 'w+b') as file:
			pickle.dump(self.database_final_im, file)
			print('pickle_final_im successfully saved...!')
		with open(self.pickle_final_path_lg, 'w+b') as file:
			pickle.dump(self.database_final_lg, file)
			print('pickle_final_lg successfully saved...!')

	
	def load_pickle_norm(self):
		with open(self.pickle_path, 'rb') as file:
			self.database_dic = copy.deepcopy(pickle.load(file))
			# if self.flag_database_dic_check == 0 :
			# 	try :
			# 		tmp_keys = []
			# 		for keys in self.database_dic :
			# 			tmp_keys.append(keys)
			# 		print(keys)
			# 		print(self.database_dic)
			# 		self.flag_database_dic_check = 1
			# 	except Exception as e:
			# 		print(e)
			# 		print('something wrong....')
			# 		print('&'*100)

	
	def load_pickle_final(self):
		with open(self.pickle_final_path_im, 'rb') as file_1:
			self.database_final_im = copy.deepcopy(pickle.load(file_1))
		with open(self.pickle_final_path_lg, 'rb') as file_2:
			self.database_final_lg = copy.deepcopy(pickle.load(file_2))
	
	def print_interest(self):
		try:

			self.load_pickle_norm()
			self.save_pickle()
			self.load_pickle_final()
			"""
			self.im_log_15_before = None
			self.im_log_15_now = None
			self.lg_log_10_now = None
			self.im_lg_overlap = None
			"""
			# initialize
			self.im_log_new_entry = []
			self.im_lg_overlap = []
			self.im_log_15_now = self.database_final_im[:15]
			self.lg_log_10_now = self.database_final_lg[:10]
			try:
				tmp_im_log_15_before = []
				for i in range(len(self.im_log_15_before)):
					tmp_im_log_15_before.append(self.im_log_15_before[i][0])
				for i in range(len(self.im_log_15_now)):
					if not self.im_log_15_now[i][0] in tmp_im_log_15_before:
						self.im_log_new_entry.append(self.im_log_15_now[i])
					else:
						pass
				if self.im_log_new_entry : # 새로운 것 존재한다면
					self.window_sound()
			except Exception as e:
				print('error in lang_train save_pickle - not available : ', e)

			for i in range(len(self.im_log_15_now)):
				for j in range(len(self.lg_log_10_now)):
					if self.im_log_15_now[i][0] == self.lg_log_10_now[j][0]:
						tmp_overlap_copy = copy.deepcopy(self.im_log_15_now[i])
						tmp_overlap_copy.append(self.lg_log_10_now[j][2])
						self.im_lg_overlap.append(tmp_overlap_copy)

			tmp_overlap = []
			for i in range(len(self.im_lg_overlap)):
				tmp_overlap.append(self.im_lg_overlap[i][0])

			pure_IM = []
			# tmp_im_log_15_now = []
			# for i in range(len(self.im_log_15_now)):
			# 	tmp_im_log_15_now.append(self.im_log_15_now[i][0])
			for i in range(len(self.im_log_15_now)) :
				if not self.im_log_15_now[i][0] in tmp_overlap :
					pure_IM.append(self.im_log_15_now[i])

			pure_LG = []
			# tmp_lg_log_10_now = []
			# for i in range(len(self.lg_log_10_now)):
			# 	tmp_lg_log_10_now.append(self.lg_log_10_now[i][0])
			for i in range(len(self.lg_log_10_now)) :
				if not self.lg_log_10_now[i][0] in tmp_overlap:
					pure_LG.append(self.lg_log_10_now[i])


			# update im before
			self.im_log_15_before = copy.deepcopy(self.im_log_15_now)
			#----------------------------------------------------------------------------------
			# ----------------------------------------------------------------------------------

			counterz = 0
			for keys in self.database_dic:
				for keys_2 in self.database_dic[keys][1]:
					counterz = counterz + len(list(self.database_dic[keys][1][keys_2]))

			general_list = ['KODEX 200', 'KODEX 레버리지', 'KODEX 코스피', 'KODEX 코스닥 150', 'KODEX 코스닥150 레버리지', 'KODEX 코스피100', 'KODEX 200선물인버스2X', 'KODEX 코스닥150선물인버스', "코스피", "코스닥", "삼성전자", "SK하이닉스"]
			tmp_im_general = []
			sort(self.database_final_im) # 소팅 먼저 하고
			for i in range(len(self.database_final_im)):
				if self.database_final_im[i][0] in general_list :
					tmp_item = copy.deepcopy(self.database_final_im[i])
					tmp_item.append(str(i) + '위')
					tmp_im_general.append(tmp_item)
			tmp_lg_general = []
			sort(self.database_final_lg) # 소팅 먼저 하고
			for i in range(len(self.database_final_lg)):
				if self.database_final_lg[i][0] in general_list :
					tmp_item = copy.deepcopy(self.database_final_lg[i])
					tmp_item.append(str(i) + '위')
					tmp_lg_general.append(tmp_item)

			print("")
			print('#' * 60)
			now_time = datetime.datetime.now()
			print(now_time)
			print('total of [', int(counterz), '] number of sentences...')

			print('#' * 60)
			print('#' * 60)
			print('▶ 새로운 IM 종목')
			self.log_printer(sort(self.im_log_new_entry))
			print('-'*40)
			print("")
			print("▶ IM/LG 공통종목")
			self.log_printer(sort(self.im_lg_overlap))
			print('-' * 40)
			print("")
			print("▶ 순수 IM 종목")
			self.log_printer(sort(pure_IM))
			print('-' * 40)
			print("")
			print("▶ 순수 LG 종목")
			self.log_printer(sort(pure_LG))
			print('-' * 40)

			print("")
			print("▶ 관심종목 순위")
			print('  @ IM')
			self.log_printer(tmp_im_general)
			print('  @ LG')
			self.log_printer(tmp_lg_general)
			print('#' * 60)
			print("")
			"""
			print('#'*60)
			print('#'*60)
			print('IM - 상위 15개 종목!! : \n', self.database_final_im[:15])
			print('\nIM - 하위 15개 종목!! : ')
			low_list = self.database_final_im[- 14 : -1]
			low_list.append(self.database_final_im[len(self.database_final_im) - 1])
			print(low_list )
			print('\n'*1)
			print('LG - 상위 10개 종목!! : \n', self.database_final_lg[:10])
			print('\nLG - 하위 10개 종목!! : ')
			low_list = self.database_final_lg[- 10 : -1]
			low_list.append(self.database_final_lg[len(self.database_final_lg) - 1])
			print(low_list )
			print('#'*60)
			print('#'*60)
			print('\n'*3)
			"""
		except Exception as e:
			print(e)
			#traceback.print_exc()
			print('--- no pickle file avaliable... yet')

	def log_printer(self, lister):
		p = PrettyTable()
		if lister:
			try:
				# table = BeautifulTable()
				# print(table(lister))


				#print(tabulate(lister, colalign=("left","right", "right")))
				#print(tabulate(lister, stralign='center' ))
				try:
					print(tabulate(lister, colalign=("center", "left", "left", "left")))
				except:
					print(tabulate(lister, colalign=("center", "left", "left")))



				# for row in lister:
				# 	p.add_row(row)
				# print(p.get_string(header=False, border=False))

				# for item in lister:
				# 	print(item)
			except:
				print('None')
		else:
			print('None')

	def get_all_hash(self): # address로 설정한 hash 모두 listup해서 돌려준다
		# return a list
		# self.database_dic
		return_list = []
		try:
			if self.database_dic : # 비어있지 않으면
				for keys in self.database_dic :
		# 			for dictionaries in self.database_dic[keys][1] : # self.database_dic[keys][1] 자체가 hash 들 가지고 있는 list
		# # 				for article_address_hash, data in dictionaries :
		# # 					dictionaries[article_address_hash]
		# 				required_key = list(dictionaries)
					return_list.extend( list(self.database_dic[keys][1].keys()) )
				print('len of database_dic_1 : ', len(self.database_dic))
				return return_list
			else:
				print('len of database_dic_2 : ', len(self.database_dic))
				print('empty database_dic...')
				return None
		except Exception as e:
			print(' error in lang_train - get_all_add_hash : ', e)
			return None
		
	def remove_redundant(self, remove_list):
		# page 떼는 작업 해주기
		tmp_remove_list = copy.deepcopy(remove_list)
		for i in range(len(tmp_remove_list)):
			tmp_remove_list[i] = str(tmp_remove_list[i]).split('&page=')[0]



		# remove list -> address of article
		tmp_database_copy = copy.deepcopy(self.database_dic)

		print('deleting database on redundancy from xlsx file ...')
		for keys in tmp_database_copy :
			# hash_list = copy.deepcopy(list(self.database_dic[keys][1].keys()))
			for keys_2 in tmp_database_copy[keys][1] :
				if keys_2 in tmp_remove_list :
					try:
						#del self.database_dic[keys][1][remove_list[i]]
						self.database_dic[keys][1].pop(keys_2, None)
					except:
						print(' error in lang_train, no key found in remove_redundant')

	def remove_redundant_2(self):
		print('deleting database on redundancy from itself_s time or empty ones...')
		now_time = datetime.datetime.now()
		time_window = (now_time - datetime.timedelta(days=4))  # 4일치 window 설정
		# 4일치 설정에다가 자정까지 포함해야된다
		time_window = datetime.datetime.combine(time_window, datetime.datetime.min.time())
		tmp_database_copy = copy.deepcopy(self.database_dic)
		for keys in tmp_database_copy:
			if tmp_database_copy[keys][1]: # 데이터 베이스 안에 기사주소 hash가 존재하면
				for keys_2 in tmp_database_copy[keys][1]:
					article_list =  tmp_database_copy[keys][1][keys_2]
					for i in range(len(article_list)):
						if datetime.datetime.strptime(article_list[i][0], '%Y-%m-%d %H:%M') < time_window :
							self.database_dic[keys][1].pop(keys_2, None)
			else: # hash가 비어있으면
				self.database_dic.pop(keys,None)
		# datetime.datetime.strptime(date_list[i], '%Y-%m-%d %H:%M')


			#for i in range(len(self.database_dic[keys][1])): # address keys in single stock
			#if self.database_dic[keys][1][i].keys() in remove_list :
			# for address in tmp_remove_list:
			# 	try:
			# 		#del self.database_dic[keys][1][remove_list[i]]
			#
			# 		self.database_dic[keys][1].pop(address, None)
			# 	except:
			# 		print(' error in lang_train, no key found in remove_redundant')

		

	def add_stock(self, stock_name, stock_num, article_time, stock_sentence, tokenized_sentence, article_address):
		# score추가해야됨
		# 이름, 넘버 -> 개별 기사시간 + 스코어 + 파싱된 센텐스 intact version으로

		"""
		{stock_name : [stock_num, {article_address : [article_time, article_score, article_sentence] , ... , ....,  ... } ] }
		"""

		# list_stock_name = list(self.database_dic.keys())
		# if stock_name in list_stock_name : # 이미 존재하면 키면

		sentence_score = self.calc_score(stock_name, tokenized_sentence)

		
		try :
			if stock_name in self.database_dic:
				# stock_sentence 추가해야 됨
				if article_address in self.database_dic[stock_name][1]: # 이미 존재하는 기사 hash
					print('already existing address key in database')
					self.database_dic[stock_name][1][article_address].append([article_time, sentence_score, stock_sentence])
				else: # 처음 기사 추가중
					self.database_dic[str(stock_name)][1][article_address] = [[article_time, sentence_score, stock_sentence]]
					print('added to existing database of new article in existing topic...')

			else: # non existing key
				self.database_dic[str(stock_name)] = [
														stock_num,
														{article_address : [[article_time, sentence_score, stock_sentence]]}
													]
				print('added to new database...')
		except Exception as e:
			print('error in lang_train - add_stock... skipping corrupting database')
	def calc_score(self, stock_name, tokenized_sentence):

		######
		# lstm으로 계산해서 return 해주기
		######
		tokenized_sentence_copy = copy.deepcopy(tokenized_sentence)
		tokenized_sentence_copy.insert(0,stock_name)
		

		final_sentence = CRAWLER__w2v_interface.padding( self.w2v_obj.transform(tokenized_sentence_copy) )
		score = self.LSTM_obj.observation_to_predict(final_sentence)
		print('calculated score ... : ', score)

		return score
	
	def tf_idf(self):
		# 문장 중요도에 따른 문장요약
		# parsing 다 하고, database 구축 끝났으면
		tmp_sentence = ""
		summarized = ""
		try:
			for key in self.database_dic :
				tmp_sentence = "" # reset
				summarized = "" # reset

				#if len(blob_of_sentence) > 3:
				blob_of_sentence = self.database_dic[key][1]
				for article_db in blob_of_sentence:
					tmp_sentence = tmp_sentence + article_db[2]
				if len(blob_of_sentence) > 7 :

					self.lex.summarize(tmp_sentence)
					summaries = self.lex.probe(7)
					for i in range(len(summaries)):
						summarized = summarized + summaries[i] + " "

					
					summaries = tmp_sentence
				else:
					summaries = tmp_sentence

				summarized = sum(summaries)

				# iteration 끝났으면, 마지막에 넣는다
				#reduced_text = summarize(tmp_sentence, ratio = 0.1)
				self.database_dic[key].append(summarized)
		except Exception as e :
			print('save pickle tf-itf err_1 : ', e)

def mention_num_mapper_im(minute_dur, num_mentioned):
	#multi = 375.61
	multi = 1.8
	y_2 =  math.exp(-0.017*10)* multi
	num_mentioned = num_mentioned**2
	if minute_dur <= 10 :
		x = minute_dur
		y = (y_2 - 1)*x*(0.1) + 1
		return y * num_mentioned
	else:
		return math.exp(-0.017*minute_dur)*multi* num_mentioned

def mention_num_mapper_lg(minute_dur, num_mentioned):
	#multi = 375.61
	multi = 6
	y_2 =  math.exp(-0.017* 80 )* multi
	num_mentioned = num_mentioned ** 2
	if minute_dur <= 80 :
		x = minute_dur
		#y = (y_2 - 1)*x*(1/80) + 1
		y = (y_2 - 3)*x*(1/80) + 3
		return y * num_mentioned
	else:
		return math.exp(-0.017*minute_dur)*multi* num_mentioned

def sort(lister):
	length = len(lister)
	for i in range(0, length):
		for j in range(0, length - i - 1):
			# 2번이 스코어 계산한거임
			if (lister[j][2] < lister[j + 1][2]):
				tmp = lister[j]
				lister[j] = lister[j + 1]
				lister[j + 1] = tmp
	return lister


def dictionary_to_tuple_imminent(database):
	now_time = datetime.datetime.now()
	# days_back = days_back_num
	# days_filter = now_time - datetime.timedelta(days=days_back)
	# time_filter = datetime.time(hour=hourz, minute=minutez)
	# days_filter = copy.deepcopy(datetime.datetime.combine(days_filter, time_filter))
	# now_time = copy.deepcopy(days_filter)
	# print('im now time : ', now_time)
	tmp_score = None
	data_counter = 0
	return_list = []
	# database[key][1]  ->  hash itself
	for key in database:
		tmp_score = 0
		for key_2 in database[key][1]:  # 1 -a
			data_counter = data_counter + len(list(database[key][1][key_2]))

		# print(key, data_counter)
		# input('?')
		# for i in range(len(database[key][1])):
		for key_2 in database[key][1]:  # 2 -a
			article_data_list = database[key][1][key_2]  # article data list per key_2
			# print(article_data_list)
			k = len(article_data_list)
			for i in range(k):
				data_article_time = datetime.datetime.strptime(article_data_list[i][0], '%Y-%m-%d %H:%M')
				if data_article_time <= now_time:
					duration_seconds = (now_time - data_article_time).total_seconds()
					minutes_duration = divmod(duration_seconds, 60)[0] + 1
					# print(minutes_duration)
					#                 print('score : ',  article_data_list[i][1])
					#                 print('minutes dutation : ',minutes_duration)

					#                 input('?')
					if article_data_list[i][1] < 0.5:
						tmp_score = tmp_score + mention_num_mapper_im(minutes_duration, k) * (
									(((article_data_list[i][1]-0.5)*4) ** 3) * 40) / (
												minutes_duration ** 0.3)  # 7승은 좋은 요인 / 나쁜 요인 차이 서로 벌려놓으려고
					else:
						tmp_score = tmp_score + mention_num_mapper_im(minutes_duration, k) * (
									(((article_data_list[i][1]-0.4)*2) ** 5) )/ (minutes_duration ** 0.3)

		# tmp_score = tmp_score * (data_counter**1.15) #1.5

		return_list.append([key, database[key][0], round(tmp_score,8)])

	return return_list


def dictionary_to_tuple_longterm(database):
	now_time = datetime.datetime.now()
	# days_back = days_back_num
	# days_filter = now_time - datetime.timedelta(days=days_back)
	# time_filter = datetime.time(hour=hourz, minute=minutez)
	# days_filter = copy.deepcopy(datetime.datetime.combine(days_filter, time_filter))
	# now_time = copy.deepcopy(days_filter)
	# print('lg now time : ', now_time)
	tmp_score = None
	data_counter = 0
	return_list = []
	# database[key][1]  ->  hash itself
	for key in database:
		tmp_score = 0
		for key_2 in database[key][1]:  # 1 -b
			data_counter = data_counter + len(list(database[key][1][key_2]))
		for key_2 in database[key][1]:  # 2 -b
			article_data_list = database[key][1][key_2]  # article data list per key_2
			k = len(article_data_list)
			for i in range(k):
				data_article_time = datetime.datetime.strptime(article_data_list[i][0], '%Y-%m-%d %H:%M')
				if data_article_time <= now_time:
					duration_seconds = (now_time - data_article_time).total_seconds()
					minutes_duration = divmod(duration_seconds, 60)[0] + 1
					if article_data_list[i][1] < 0.5:
						tmp_score = tmp_score + mention_num_mapper_lg(minutes_duration, k) * (
									(((article_data_list[i][1]-0.5)*4) ** 3) * 35) / (
											minutes_duration ** 0.1)  # 7승은 좋은 요인 / 나쁜 요인 차이 서로 벌려놓으려고
					else:
						tmp_score = tmp_score + mention_num_mapper_lg(minutes_duration, k) * (
									((article_data_list[i][1]-0.5)*4) ** 3) / (minutes_duration ** 0.1)

		# tmp_score = tmp_score * (data_counter ** 7) # 1.4

		return_list.append([key, database[key][0], round(tmp_score, 8)])

	return return_list



def main_sub():
	t1 = sentence_tokener()
	t2 = CRAWLER__beautiful_soup.NaverFinanceNewsCrawler(module=True)
	t1.start()
	t2.start()

if __name__ == '__main__':
	main_sub()
	







