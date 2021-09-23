#-*-coding: utf-8-*-

## my files
##########################################################################
##########################################################################
import CRAWLER__Global

##########################################################################

## PY files
##########################################################################
##########################################################################
#import json
import time
import os
from bs4 import BeautifulSoup
import pandas as pd
import datetime
#from newspaper import Article
import re
import requests
import queue
from webob.compat import urlparse
from collections import Counter
#from konlpy.tag import Twitter
import threading
import copy
import traceback
##########################################################################


def get_bs_obj(url):
	# from URL_NAVER_NEWS_FLASH
	from fake_useragent import UserAgent  # 매번 header randomize
	import random
	counter = 0
	#     try:
	ua = UserAgent()
	ua_list = []
	ua_list.append(ua.chrome)
	ua_list.append(ua.firefox)
	ua_list.append(ua['Internet Explorer'])


	header = {'User-Agent': str(random.choice(ua_list))}
	# header = {'User-Agent':str(ua.random)}

	try:

		result = requests.get(url, headers=header)
		time.sleep(0.1)
		#bs_obj = BeautifulSoup(result.content, "html.parser")
		bs_obj = BeautifulSoup(result.content, "lxml") #훨씬 더 빠른 방식
		if bs_obj == None :
			print('url error!')
			print('url directed : ', url)
			raise ValueError('wrong url')
		try:
			del result
		except Exception as e:
			print(e)

		return bs_obj

	except:
		return None




class NaverFinanceNewsCrawler(threading.Thread):
	URL_NAVER_FINANCE = "http://finance.naver.com"
	URL_NAVER_FINANCE_NEWS_QUERY = "http://finance.naver.com/news/news_search.nhn?q=%s&x=0&y=0"  # params: query
	URL_NAVER_FINANCE_NEWS_CODE = "http://finance.naver.com/item/news_news.nhn?code=%s&page=%s"  # params: (code, page)
	
	#-------------------------------------------------------------------------------------------
	URL_NAVER_NEWS_FLASH = "http://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258"
	# ^ 실시간 속보
	#URL_NAVER_NEWS_FLASH = "https://finance.naver.com/news/news_list.nhn?mode=LSS3D&section_id=101&section_id2=258&section_id3=401"
	# ^ 시황/전망
	#-------------------------------------------------------------------------------------------
	
	URL_NAVER_STOCK_NOTICE = "http://finance.naver.com/item/news_notice.nhn?code=%s&page=%s"  # params: (code, page)

	def __init__(self, module = True):
		print('NaverFinanceNewsCrawler wake-up success...')
		
		# Threading
		threading.Thread.__init__(self)
		
		# queue
		#self.queue_crw = queue.Queue(1) # 한개짜리로 설정
		#self.queue_crw = Global_lang.queue # 한개짜리로 설정
							      #  old 포함df        #삭제용           #새것만
		self.queue_item_crw = [    pd.DataFrame(),    pd.DataFrame(),    pd.DataFrame() ]
		
		# 날짜 뒤로 볼 만큼 설정
		self.days_back = int(4)

		# 오늘날짜 crawling 용 counter
		self.counter_today_only = 0
		
		# crawling 모듈 자체 쓰기 위함
		self.dictionary_ = []
		self.module = module # 자동화 위해
		self.article_address = [] # 계속 관리를 위해
		self.tmp_list = [] # page안 link crawling 결과 담는 곳
		
		self.df_redundant = None # 데이터프레임 불만족 부분 갱신용
		self.df_adpt = None # 데이터프레임 만족 부분 갱신용
		
		# csv load용
		self.folder_path_crw = os.getcwd()
		self.article_path_crw = str(self.folder_path_crw + '\\CRAWLER__article_result_dynamic').replace('/', '\\')
		if not os.path.isdir(self.article_path_crw):  # doesn't exist
			os.mkdir(self.article_path_crw)
			print('created self.article_path_crw directory...')
		else:
			pass
		self.sub_files_list_crw = os.listdir(os.path.abspath(self.article_path_crw))
		self.full_path_crw = []

	def get_csv_path(self):

		# csv load용
		self.folder_path_crw = os.getcwd()
		self.article_path_crw = str(self.folder_path_crw + '\\CRAWLER__article_result_dynamic').replace('/', '\\')
		if not os.path.isdir(self.article_path_crw):  # doesn't exist
			os.mkdir(self.article_path_crw)
			print('created self.article_path_crw directory...')
		else:
			pass
		self.sub_files_list_crw = os.listdir(os.path.abspath(self.article_path_crw))
		self.full_path_crw = []

		try:
			if len(self.sub_files_list_crw) >= 1:  # 파일이 하나 이상 있다면
				for i in range(len(self.sub_files_list_crw)):
					base = os.path.abspath(self.article_path_crw)
					full_path = str(
						re.escape(base).replace('\.', '.') + re.escape("\\" + str(self.sub_files_list_crw[i])).replace(
							'\.', '.'))
					self.full_path_crw.append(full_path)
				if self.full_path_crw:
					print('in Crawling - successfuly parsed xlsx files...')
					if len(self.full_path_crw) > 0:
						for i in range(len(self.full_path_crw)):
							print('PATH - NO.', i + 1, ' is : ', self.full_path_crw[i])
			else:
				pass

		except Exception as e:
			print('error in Crawling - _craw_get_all : ', e)



	def run(self):
		self.crawl()
		
	def crawl(self):

		
		if self.module == False: # module안에서 쓰이는 경우 자동화
			num_days = int(input('how many days do you want to crawl...?'))
			page_num = int(input('how many pages do you want to craw...?'))
		else:
			num_days = int(self.days_back)
			page_num = int(700)
			
		return self._craw_get_all(num_days, page_num)

	def _craw_get_all(self, num_days=None, page_num=None):
		import datetime
		num_days_ = int(num_days)
		page_num_ = int(page_num)
		
		# ram에 고정하기 위해 미리 선언
		self.dictionary_ = []

		# Threading으로 계속 돌려야 할 부분
		################################################################################
		################################################################################
		
		while True:
			time.sleep(10) # 10초 쉬고...

			# reset할 variables
			self.full_path_crw = []
			self.dictionary_ = [] # 처음 날짜별(date에 대한) 기사 위치
			self.article_address = []
			self.df_redundant = pd.DataFrame() # 데이터프레임 불만족 부분 갱신용, 완전 빈 것
			self.df_adpt = pd.DataFrame() # 데이터프레임 만족 부분 갱신용, 완전 빈 것
			self.tmp_list = []
			self.queue_item_crw = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
			self.counter_new_article = 0


			#첫 csv address load
			#이거 진행되려면, csv자체를 save 할 때 한 파일로 해야됨
			self.get_csv_path() # update the path every time

			start = datetime.datetime.now()

			# load CSV address
			try :
				time_window = (start - datetime.timedelta(days=self.days_back) ) # 4일치 window 설정
				#4일치 설정에다가 자정까지 포함해야된다
				time_window = datetime.datetime.combine(time_window, datetime.datetime.min.time())
				print('made time window : ', time_window)
				
				if self.full_path_crw : # csv 파일 경로가 존재한다면
					for paths in self.full_path_crw : 
						df = pd.read_excel(paths,  sheet_name='Sheet1', encoding='utf-8', index_col=0)
						#print(df.head(3))
						print(df.info())
						######
						# 4일 초과 한 부분 삭제 들어가야 됨
						# datetime.datetime.strptime(database[key][1][i][0], '%Y-%m-%d %H:%M')
						#for i in range(len(df)): # iteration on df
						df_for_index = df.copy()
						#print(df_for_index.head(3))
						print(df_for_index.info())
						df_for_index['Date'] = pd.to_datetime(df_for_index['Date'])
						print(df_for_index.info())
							#-----------------------------
						index_adpt = df_for_index[df_for_index['Date'] >= time_window ].index
						index_redund = df_for_index[df_for_index['Date'] < time_window ].index
						df_adpt = df.iloc[index_adpt].copy() # 기준 만족하는 부분
						df_redundant = df.iloc[index_redund].copy() # 기준 불만족 하는 부분
						try: # ram 할당된 것 제거
							del df_for_index
							del index_adpt
							del index_redund
							del df
						except Exception as e:
							print('error in Crawling - deleting df and indexs')
						
						# 이 프로세스는 담겨있는 모둔 xlsx 파일에 대해 iter중이라서
						if self.df_redundant.empty: # 첫 variable assign
							self.df_redundant = df_redundant.copy()
						else:
							self.df_redundant = pd.concat([self.df_redundant, df_redundant],axis=0, ignore_index=True).copy() #index_col=0

						if self.df_adpt.empty:# 첫 variable assign
							self.df_adpt = df_adpt.copy()
						else:
							self.df_adpt = pd.concat([self.df_adpt, df_adpt] ,axis=0, ignore_index=True).copy()
							
# 					if not self.df_redundant.empty:
# 						self.df_redundant.reset_index(drop=True)
# 					if not self.df_adpt.empty:
# 						self.df_adpt.reset_index(drop=True)
					print('^'*60)
					print('^'*60)
					print('^'*60)
					print('total df_adpt info : ', self.df_adpt.info())
					print('-'*60)
					print('total df_redundant info : ', self.df_redundant.info())
					#ans = input('?')

				else:
					print('no CSV file exist in the directory...')
					pass

			except Exception as e:
				print('error in Crawlig - loading cvs address : ', e)
				traceback.print_exc()

			# @ self.counter_today_only :: reset 하는 부분 -> 오늘치만 crawling 하기 위함
			if self.counter_today_only % 100 == 0 :
				print('self.counter_today_only has been initialized...')
				self.counter_today_only = 0
			else:
				pass
			
			# @ Crawling 실제로 수행하는 부분
			now = datetime.datetime.now()
			for i in range(0, num_days_, 1): #정해진 날짜 만큼 crawling
				target = now - datetime.timedelta(days=1) * i
				target_url = self.URL_NAVER_NEWS_FLASH + "&date=" + str(target.strftime("%Y%m%d")) # 현재 페이지

				print('target url is : ', target_url)
	# 			self.dictionary_.append(webpage_wrapper('web_' + str(i + 1), target_url, page_num_))
				tmp_web_wrapper = webpage_wrapper('web_' + str(i + 1), target_url, page_num_)
				tmp_web_wrapper_return = copy.deepcopy(tmp_web_wrapper.return_webpage_wrapper_result())
				self.dictionary_.extend(tmp_web_wrapper_return) # 기사 전체 링크 가지고 있는 부분
				try:
					del tmp_web_wrapper
					del tmp_web_wrapper_return
				except Exception as e:
					print('error in Crawling - deletion of tmp_web_wrapper : ', e)

				if self.counter_today_only != 0:
					print("   - geting only today's article -   ")
					break
				else: # self.counter_today_only == 0
					print("   - crawl 4 days of article -   ")

			# @ counter update
			self.counter_today_only = self.counter_today_only + 1

			# 새롭게 추가되어야 하는 부분 고르고, 삭제되어야하는 부분 넘겨줘야한다
			# CSV가 있고 없고로 구분
			################################################################################
			################################################################################
			# 1) 새로운 것은 검사해서 추가
			# 2) 제거 항목은 넘겨주기
			# 3) 둘다 queue 써서 넘겨주기 -> 둘중 하나라도 해당하는게 있으면 돌려야됨

			self.tmp_list = []
			print('$$$')
			if self.df_adpt.empty:
				print('empty df_adpt')
			else:
				print('df_adpt length : ', len(self.df_adpt))
				
			if self.df_redundant.empty:
				print('empty df_adpt')
			else:
				print('df_redundant length : ', len(self.df_redundant))
			
			if (self.full_path_crw and ( not(self.df_adpt.empty) or not(self.df_redundant.empty) )): #존재한다면!
				if not(self.df_adpt.empty) :
					print('new articles are available in Crawling - into queue...')
					col_old_adpt_address_list = copy.deepcopy(self.df_adpt['Link'].tolist()) # 전체 old+new csv 보관 링크
					for i in range(len(col_old_adpt_address_list)): # &page= 부분 잘라서 보관하기위함
						col_old_adpt_address_list[i] = str(col_old_adpt_address_list[i]).split('&page=')[0] # 앞부분만 보관
					tmp_link_list_for_web_sub = []

					try:
						print('length of parsed list from web.. : ', len(self.dictionary_))
						print('length of parsed list from web..beggining : ', self.dictionary_[:10])
						print('length of parsed list from web..end : ', self.dictionary_[-11:-1])
					except Exception as e:
						print('printing self.dictionary_ : ',e)
					
					dictionary_crop_list = copy.deepcopy(self.dictionary_) # 카피 떠서 page 앞부분만 보관
					for i in range(len(dictionary_crop_list)):
						dictionary_crop_list[i] = str(dictionary_crop_list[i]).split('&page=')[0]
					
					for i in range(len(dictionary_crop_list)): # 새로 구한 crawling list에 대해
						if not( dictionary_crop_list[i] in col_old_adpt_address_list) : # 새것이 old list에 없다면
							self.counter_new_article = self.counter_new_article + 1
							#필요한 애들만 담는다
							tmp_link_list_for_web_sub.append(self.dictionary_[i]) # 실제 page 포함된 link를 다시 보관
# 							tmp_webpage_sub = webpage_sub('link_' + str(i), self.dictionary_[i])
# 							tmp_webpage_sub_return = tmp_webpage_sub.return_webpage_sub_result()
# 							self.tmp_list.append(tmp_webpage_sub_return)
					try:
						print('required link lists beginning : ', tmp_link_list_for_web_sub[:10])
						print('required link lists ends : ', tmp_link_list_for_web_sub[-11:-1])
					except Exception as e:
						print('printing tmp_links : ', e)

							
					# dataframe 새로 저장하기
					##################################################
					##################################################
					
					print('newly added data length : ', self.counter_new_article)
					for i in range(len(tmp_link_list_for_web_sub)):
						tmp_webpage_sub = webpage_sub('link_' + str(i), tmp_link_list_for_web_sub[i])
						if tmp_webpage_sub.flag_article_get_safe == 1:  # article 잘 가져온거라면!
							# return [self.news_title, self.news_date, self.content, self.url_low]
							tmp_webpage_sub_return = copy.deepcopy(tmp_webpage_sub.return_webpage_sub_result())
							self.tmp_list.append(tmp_webpage_sub_return)
							try:
								del tmp_webpage_sub
								del tmp_webpage_sub_return
							except Exception as e:
								print('error in Crawling - deletion of tmp_webpage_sub : ', e)
						else: # flag_article_get_safe ==0
							pass
					

					# delet and clean the cvs directory
					self.clean_csv(switch=False)

					tmp_new_df = self.save_as_csv(self.df_adpt, concat=True).copy()
					tmp_new_only_df = self.save_as_csv(self.df_adpt, concat=False).copy()
					#저장하고, self.tmp_list에서 크롤링한 부분을 넘겨주기 위해 가져옴
					#for i in range(len(self.tmp_list)):
					self.queue_item_crw[0] = tmp_new_df.copy()
					self.queue_item_crw[2] = tmp_new_only_df.copy()


				else: # old + new에 해당하는 새로운 업데이트 부분 자체가 없음
					pass

				if not(self.df_redundant.empty): #제거 대상 dataframe
					###
					print('remove articles are available in Crawling - into queue...')
					self.queue_item_crw[1] = self.df_redundant.copy() # 4일보다 지난 것
				else:
					pass
					#삭제할 부분이 없음
				
				#if not(self.df_redundant.empty) or not()
				if not(self.queue_item_crw[0].empty) or not(self.queue_item_crw[1].emtpy) or not(self.queue_item_crw[2].empty): #None value 세서 3보다 작으면
					try:
						print('whole input is available in Crawling - into queue...')
						CRAWLER__Global.queue.put(self.queue_item_crw, timeout=1)
						print('queue updated in Crawling...')
					except Exception as e:
						print('error in queue operation_1 : ', e)
						try:
							Global_lang.queue.get(timeout=0.01)
							Global_lang.queue.put(self.queue_item_crw, timeout=1)
						except:
							print('failure while handling exception in operation_1')

				
				else:
					print('unexpected behavior in Crawling py')

			elif not self.full_path_crw: # empty csv

				for i in range(len(self.dictionary_)): #iterate over all links
					tmp_webpage_sub_2 = webpage_sub('link_' + str(i), self.dictionary_[i])
					if tmp_webpage_sub_2.flag_article_get_safe == 1:
						tmp_webpage_sub_return_2 = copy.deepcopy(tmp_webpage_sub_2.return_webpage_sub_result())
						self.tmp_list.append(tmp_webpage_sub_return_2)
					try:
						del tmp_webpage_sub_2
						del tmp_webpage_sub_return_2
					except Exception as e:
						print('error in Crawling - deletion of tmp_webpage_sub : ', e)

					else: # webpage_sub.flag_article_get_safe == 0
						pass


				# delet and clean the cvs directory
				self.clean_csv(switch=False)

				tmp_new_df = self.save_as_csv(self.df_adpt, concat=True).copy() #empty df here , self.df_adpt
				#저장하고, self.tmp_list에서 크롤링한 부분을 넘겨주기 위해 가져옴
				#for i in range(len(self.tmp_list)):
				self.queue_item_crw[0] = tmp_new_df.copy()

				if not(self.queue_item_crw[0].empty) or not(self.queue_item_crw[1].emtpy) or not(self.queue_item_crw[2].empty): #None value 세서 3보다 작으면:  # None value 세서 3보다 작으면
					try:
						print('whole input is available in Crawling - into queue...')
						CRAWLER__Global.queue.put(self.queue_item_crw, timeout=1)
						print('queue updated in Crawling...')
					except Exception as e:
						print('error in queue operation_2 : ', e)
						try:
							CRAWLER__Global.queue.get(timeout=0.01)
							CRAWLER__Global.queue.put(self.queue_item_crw, timeout=1)
						except:
							print('failure while handling exception in operation_2')
				else:
					print('unexpected behavior in Crawling py')

			else:
				print('no queue updated in Crawling...')
				pass
				# queue 업데이트 안함, csv 업데이트 안함


			print('total elapsed time :  ', datetime.datetime.now() - start, '....')
			print('periodic task done ...')
	def clean_csv(self, switch):
		if switch:
			import shutil
			# delet and clean the cvs directory
			print('cleaning the csv directory...')
			try:
				target_dir = os.getcwd() + "\\CRAWLER__article_result_dynamic"
				shutil.rmtree(target_dir)
			except Exception as e:
				print('shutil cleaning directory failed...', e)

		else:
			print('cleaning csv disabled')

		
	def save_as_csv(self, old_df, concat = None):
		import pandas as pd
		import os
		import numpy as np

		dictionary = {}
		title_list = []
		date_list = []
		content_list = []
		link_list = []
		# [self.news_title, self.news_date, self.content, self.url_low]
		for i in range(len(self.tmp_list)):
			webpage_sub = self.tmp_list[i]
			title_list.append(webpage_sub[0])
			date_list.append(webpage_sub[1])
			content_list.append(webpage_sub[2])
			link_list.append(webpage_sub[3])

		dictionary.update({'Title': np.array(title_list).tolist()})
		dictionary.update({'Date': np.array(date_list).tolist()})
		dictionary.update({'Content': np.array(content_list).tolist()})
		dictionary.update({'Link': np.array(link_list).tolist()})

	
		df_new = pd.DataFrame.from_dict(dictionary)
		df_new.reset_index(drop=True)
		if concat == True :
			df = pd.concat([df_new, old_df] ,axis=0, ignore_index=True).copy()
			try:
				target_dir = os.getcwd() + "\\CRAWLER__article_result_dynamic"
				if not os.path.isdir(target_dir):  # doesn't exist
					os.mkdir(target_dir)
					print('created directory...')
				else:
					pass

				# data frame save

				df.to_excel(target_dir + "\\" + "dynamic_article_csv.xlsx", encoding='utf-8')
				print('csv file has been saved... !')
			except Exception as e:
				print('error in Crawling - save as csv : ', e)

			return df

		else:
			df = df_new.copy()
			return df





class webpage_wrapper:
	URL_NAVER_FINANCE = "http://finance.naver.com"
	URL_NAVER_FINANCE_NEWS_QUERY = "http://finance.naver.com/news/news_search.nhn?q=%s&x=0&y=0"  # params: query
	URL_NAVER_FINANCE_NEWS_CODE = "http://finance.naver.com/item/news_news.nhn?code=%s&page=%s"  # params: (code, page)
	URL_NAVER_NEWS_FLASH = "http://finance.naver.com/news/news_list.nhn?mode=LSS2D&section_id=101&section_id2=258"
	URL_NAVER_STOCK_NOTICE = "http://finance.naver.com/item/news_notice.nhn?code=%s&page=%s"  # params: (code, page)

	def __init__(self, name, url_top, page_num):
		self.name = name
		self.url_top = url_top
		self.page_num = page_num

		self.list_of_pages_url = []  # 한 날짜 안의 page의 url 가져옴
		self.get_webpage()

		self.list_of_link_in_pages = []  # page url 별 안의 기사 가져옴, list item is class
		self.get_title_link_pages()


	def get_webpage(self): # date기반해서 가져오는 부분
		bs = get_bs_obj(self.url_top)
		if bs != None:
			try:
				day_page_list = bs.find_all('table', class_='Nnavi')[0].find_all('a')

				for i in range(len(day_page_list)):
					day_page_list[i] = self.URL_NAVER_FINANCE + str(day_page_list[i].get('href')).replace('§ion_id', '&sect').replace('&amp;', '&')


				page_queue = [day_page_list[0], day_page_list[-1]]  # 처음과 마지막 소스
				page_url_base = str(page_queue[0]).split("page=")[0]
				page_first_num = int(str(page_queue[0]).split("page=")[1])
				page_last_num = int(str(page_queue[1]).split("page=")[1])

				if self.page_num < 0 :
					raise ValueError('Wrong page num crawling...')
				else:
					if self.page_num <= 1:
						self.page_num = 1
					else:
						if self.page_num >= page_last_num :
							self.page_num = page_last_num
						else:
							pass

				for i in range(page_first_num, self.page_num + 1, 1):  # iteration
					self.list_of_pages_url.append(page_url_base + "page=" + str(i))

				self.list_of_pages_url = list(dict.fromkeys(self.list_of_pages_url))  # 중복점 없앤다

				try :
					del bs
				except Exception as e :
					print(e)

				print('+' * 15)
			except Exception as e:
				print(e)
				pass


		else:
			pass


	def get_title_link_pages(self):
		print('@@ length of list of pages url : ', len(self.list_of_pages_url))
		print('check list of pages url..')
		print('+' * 15)
		for i in range(len(self.list_of_pages_url)):
			print(self.list_of_pages_url[i])
		print('+' * 15)

		for i in range(len(self.list_of_pages_url)):
			bs = get_bs_obj(self.list_of_pages_url[i])  # overwrite bs
			if bs != None :
				#time.sleep(0.05)

				# --------------------------------------------------------------------------
				tmp_dd = bs.find_all('ul', class_='realtimeNewsList')
				tmp_dd = tmp_dd[0].find_all('dd', class_='articleSubject')

				for j in range(len(tmp_dd)):
					wanted_url = (self.URL_NAVER_FINANCE + str(tmp_dd[j].a.get('href'))).replace('§ion_id', '&sect').replace(
						'&amp;', '&')
					#self.list_of_link_in_pages.append(webpage_sub('link_' + str(j), wanted_url))
					self.list_of_link_in_pages.append(str(wanted_url))

				tmp_dt = bs.find_all('ul', class_='realtimeNewsList')[0].find_all( 'dt', class_='articleSubject')
				for j in range(len(tmp_dt)):
					wanted_url_ = (self.URL_NAVER_FINANCE + str(tmp_dt[j].a.get('href'))).replace('§ion_id', '&sect').replace(
						'&amp;', '&')
					#self.list_of_link_in_pages.append(webpage_sub('link_' + str(j), wanted_url_))
					self.list_of_link_in_pages.append(str(wanted_url_))

			else:
				pass
		
		

		print('number of list_of_link_in_pages of ', self.name, ' is : ', len(self.list_of_link_in_pages))
		print('-' * 10)
		print('\n' * 2)

		# save memory by del method
		try:
			del bs
			del tmp_dt
			del tmp_dd
		except Exception as e :
			print(e)
	
	def return_webpage_wrapper_result(self):
		return self.list_of_link_in_pages # 모든 페이지 안의 기사 link 돌려줌
						   
						   


class webpage_sub:

	def __init__(self, name, url_low):
		self.name = name

		self.url_low = url_low
		print('############################ : ', url_low)
		self.code = []  # 종목 코드
		self.content = None  # parsing 된 전체 text
		self.news_date = None
		self.news_title = None

		self.flag_article_get_safe = 0

		self.get_article_content()

	def get_article_content(self):
		bs = get_bs_obj(self.url_low)
		if bs != None:
			try:

				content = bs.find_all('div', class_='articleCont')
				content = content[0].get_text()
				news_date = bs.find('span', class_='article_date').get_text()
				news_title = bs.find('div', class_='article_info').get_text()

				content = re.sub('<.+?>', '', content, 0).strip().replace('\n','').replace('\t','').replace('\r','')
				news_date = re.sub('<.+?>', '', news_date, 0).strip().replace('\n','').replace('\t','').replace('\r','')
				news_title = re.sub('<.+?>', '', news_title, 0).strip().replace('\n','').replace('\t','').replace('\r','')

				"""
				https://dojang.io/mod/page/view.php?id=2462
				https://redscreen.tistory.com/163
				"""

				self.content = str(content.split('@')[0].strip())
				self.news_title = str(news_title.split('기사입력')[0].strip())
				self.news_date = str(news_date.strip())
				self.content = self.news_title + " " + self.content

				self.flag_article_get_safe = 1
				# check data condsistancy
				if self.url_low == None or self.content == None or self.news_date == None or self.news_title == None :
					#raise ValueError('wrong data consistancy')
					self.flag_article_get_safe = 0
				else:
					pass

				# memory saveing by del method
				try:
					del bs
				except Exception as e :
					print(e)

			except Exception as e:
				print(e)
				self.flag_article_get_safe = 0
				bs = None

		else:
			self.flag_article_get_safe = 0
			pass
		
	
	def return_webpage_sub_result(self):
		#return [self.content, self.news_title, self.news_date]
		return [self.news_title, self.news_date, self.content, self.url_low]



    
if __name__ == "__main__":
	crawler = NaverFinanceNewsCrawler(module=False)
	docs = crawler.crawl()
