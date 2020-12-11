# coding=utf-8

"""
이 모듈은 Kiwoom.py의 Kiwoom 클래스 내의 _on_receive_tr_data 메소드에서만 사용하도록 구현됨.
TR마다 각각의 메소드를 일일이 작성해야하기 때문에 기존 클래스에서 분리하였음
"""

# cyclic import 를 피하며 type annotation 을 하기 위해 TYPE_CHECKING 을 이용함
# (https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports)
import datetime
import traceback

from typing import TYPE_CHECKING


if TYPE_CHECKING:
	#from KIWOOM_API__MAIN import KiwoomAPI
	from KIWOOM_API__BACKEND import KiwoomAPI


def on_receive_opt10080(kw: 'KiwoomAPI', rqname, trcode):
	"""주식분봉차트조회요청 완료 후 서버에서 보내준 데이터를 받는 메소드"""

	data_cnt = kw.get_repeat_cnt(trcode, rqname)
	ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

	for i in range(data_cnt):
		date = kw.comm_get_data(trcode, "", rqname, i, "체결시간")
		open = kw.comm_get_data(trcode, "", rqname, i, "시가")
		high = kw.comm_get_data(trcode, "", rqname, i, "고가")
		low = kw.comm_get_data(trcode, "", rqname, i, "저가")
		close = kw.comm_get_data(trcode, "", rqname, i, "현재가")
		volume = kw.comm_get_data(trcode, "", rqname, i, "거래량")

		ohlcv['date'].append(date)
		ohlcv['open'].append(abs(int(open)))
		ohlcv['high'].append(abs(int(high)))
		ohlcv['low'].append(abs(int(low)))
		ohlcv['close'].append(abs(int(close)))
		ohlcv['volume'].append(int(volume))

	return ohlcv


def on_receive_opt10081(kw: 'KiwoomAPI', rqname, trcode):
	"""주식일봉차트조회요청 완료 후 서버에서 보내준 데이터를 받는 메소드"""

	data_cnt = kw.get_repeat_cnt(trcode, rqname)
	ohlcv = {'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

	for i in range(data_cnt):
		date = kw.comm_get_data(trcode, "", rqname, i, "일자")
		open = kw.comm_get_data(trcode, "", rqname, i, "시가")
		high = kw.comm_get_data(trcode, "", rqname, i, "고가")
		low = kw.comm_get_data(trcode, "", rqname, i, "저가")
		close = kw.comm_get_data(trcode, "", rqname, i, "현재가")
		volume = kw.comm_get_data(trcode, "", rqname, i, "거래량")

		ohlcv['date'].append(date.strip())
		ohlcv['open'].append(int(open.strip()))
		ohlcv['high'].append(int(high.strip()))
		ohlcv['low'].append(int(low.strip()))
		ohlcv['close'].append(int(close.strip()))
		ohlcv['volume'].append(int(volume.strip()))

	return ohlcv

def on_receive_balance_check_normal(kw: 'KiwoomAPI', rqname, trcode):
	ohlcv = {'balance': [], 'time':[]}
	#balance = kw.get_comm_data(trcode, rqname, 0, "주문가능금액")
	balance = kw.comm_get_data(trcode, "", rqname, 0, "주문가능금액")

	ohlcv['balance'].append(float(balance.strip()))
	ohlcv['time'].append(str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))

	return ohlcv

def on_receive_balance_check_with_order(kw: 'KiwoomAPI', rqname, trcode):
	"""
	>>>
	이부분 수정해야됨, 맞게
	+ kw.comm_get_data 수정!!! 1.5V 문서 확인해서 작업
	# 서버에서 100% 증거금 조회
	"""
	ohlcv = {
			 'balance_1': [],
			 'balance_2': [],
			 'balance_3': []
			}
	#balance = kw.get_comm_data(trcode, rqname, 0, "주문가능금액")
	balance_1 = kw.comm_get_data(trcode, "", rqname, 0, "주문가능현금")
	balance_2 = kw.comm_get_data(trcode, "", rqname, 0, "미수불가주문가능금액")
	balance_3 = kw.comm_get_data(trcode, "", rqname, 0, "증거금100주문가능금액")

	ohlcv['balance_1'].append(float(balance_1))
	ohlcv['balance_2'].append(float(balance_2))
	ohlcv['balance_3'].append(float(balance_3))
	
	# ohlcv['balance'].append(float(balance.strip()))
	# ohlcv['time'].append(str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
	#print('!!!!!',balance_1, balance_2, balance_3)

	return ohlcv

def on_receive_owning_stocks(kw: 'KiwoomAPI', rqname, trcode):
	"""
	>>>
	이부분 수정해야됨, 맞게
	+ kw.comm_get_data 수정!!! 1.5V 문서 확인해서 작업
	# 서버에서 100% 증거금 조회
	
	          KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"종목명").Trim(),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"종목코드").TrimEnd('0'),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"보유수량").TrimEnd('0'),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"평균단가").TrimEnd('0'),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"매입금액").Trim(),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"평가금액").Trim(),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"손익금액").Trim(),
              KiAPI.CommGetData(e.sTrCode,"",e.sRQName,nLoop,"손익율").Trim()


출처: https://psps.tistory.com/19 [PowerStock]
	"""
	data_cnt = kw.get_repeat_cnt(trcode, rqname)
	print('on_receive_owning_stocks data_cnt :: ', data_cnt)
	ohlcv = {
			}
	#balance = kw.get_comm_data(trcode, rqname, 0, "주문가능금액")



	for i in range(data_cnt):
		stock_name = kw.comm_get_data(trcode, "", rqname, i, "종목명")
		stock_code = kw.comm_get_data(trcode, "", rqname, i, "종목코드")
		number_owned = kw.comm_get_data(trcode, "", rqname, i, "보유수량")
		average_price = kw.comm_get_data(trcode, "", rqname, i, "평균단가")
		buy_price = kw.comm_get_data(trcode, "", rqname, i, "매입금액") # 이거 합산 다되어서 계산됨(진입시점마다의 가격)
		access_price = kw.comm_get_data(trcode, "", rqname, i, "평가금액")
		profit_price = kw.comm_get_data(trcode, "", rqname, i, "손익금액")
		#profit_rate = kw.comm_get_data(trcode, "", rqname, i, "손익율")
		
		ohlcv[stock_name] = {} # double hash

		ohlcv[stock_name]['stock_name']=str(stock_name.strip())
		ohlcv[stock_name]['stock_code']=str(stock_code.strip())
		ohlcv[stock_name]['number_owned']=int(number_owned.strip())
		ohlcv[stock_name]['average_price']=float(average_price.strip())
		ohlcv[stock_name]['buy_price']=float(buy_price.strip())
		ohlcv[stock_name]['access_price']=float(access_price.strip())
		ohlcv[stock_name]['profit_price']=float(profit_price.strip())
		#ohlcv[stock_name]['profit_rate']=float(profit_rate.strip())
		ohlcv[stock_name]['profit_rate'] = float(profit_price.strip())/float(buy_price.strip())

	

	return ohlcv

def on_receive_additional_info_tr(kw: 'KiwoomAPI', rqname, trcode):
	try:
		ohlcv = { }
		stock_name = kw.comm_get_data(trcode, "", rqname, 0, "종목명")
		stock_code = kw.comm_get_data(trcode, "", rqname, 0, "종목코드")
		payed_month = kw.comm_get_data(trcode, "", rqname, 0, "결산월")
		straight_price = kw.comm_get_data(trcode, "", rqname, 0, "액면가")
		cash_own = kw.comm_get_data(trcode, "", rqname, 0, "자본금")

		all_stocks = kw.comm_get_data(trcode, "", rqname, 0, "상장주식")
		credit_ratio = kw.comm_get_data(trcode, "", rqname, 0, "신용비율")
		year_high = kw.comm_get_data(trcode, "", rqname, 0, "연중최고")
		year_low = kw.comm_get_data(trcode, "", rqname, 0, "연중최저")
		all_stocks_cash = kw.comm_get_data(trcode, "", rqname, 0, "시가총액")
		all_stocks_cash_ratio = kw.comm_get_data(trcode, "", rqname, 0, "시가총액비중")
		outside_own = kw.comm_get_data(trcode, "", rqname, 0, "외인소진률")
		per = kw.comm_get_data(trcode, "", rqname, 0, "PER")
		eps = kw.comm_get_data(trcode, "", rqname, 0, "EPS")
		roe = kw.comm_get_data(trcode, "", rqname, 0, "ROE")
		pbr = kw.comm_get_data(trcode, "", rqname, 0, "PBR")
		ev = kw.comm_get_data(trcode, "", rqname, 0, "EV")
		bps = kw.comm_get_data(trcode, "", rqname, 0, "BPS")

		profit_price = kw.comm_get_data(trcode, "", rqname, 0, "매출액")
		profit_by_run = kw.comm_get_data(trcode, "", rqname, 0, "영업이익")
		tr_number = kw.comm_get_data(trcode, "", rqname, 0, "거래량")
		tr_number_compare = kw.comm_get_data(trcode, "", rqname, 0, "거래대비")

		ohlcv[stock_name] = {} # 선언부

		ohlcv[stock_name]['stock_name'] = str(stock_name.strip())
		ohlcv[stock_name]['stock_code'] = str(stock_code.strip())
		ohlcv[stock_name]['payed_month'] = str(payed_month.strip())
		
		if straight_price.strip() != '':
			ohlcv[stock_name]['straight_price'] = float(straight_price.strip())
		else:
			ohlcv[stock_name]['straight_price'] = str(straight_price.strip())
		
		if cash_own.strip() != '':
			ohlcv[stock_name]['cash_own'] = float(cash_own.strip())
		else:
			ohlcv[stock_name]['cash_own'] = str(cash_own.strip())
		
		if all_stocks.strip() != '':
			ohlcv[stock_name]['all_stocks'] = int(all_stocks.strip())
		else:
			ohlcv[stock_name]['all_stocks'] = str(all_stocks.strip())
			
		ohlcv[stock_name]['credit_ratio'] = str(credit_ratio.strip())
		
		if year_high.strip() != '':
			ohlcv[stock_name]['year_high'] = float(year_high.strip())
		else:
			ohlcv[stock_name]['year_high'] = str(year_high.strip())
		
		if year_low.strip() != '':
			ohlcv[stock_name]['year_low'] = float(year_low.strip()) # 현재시점보다 낮으면 - 붙어서 나오는 듯
		else:
			ohlcv[stock_name]['year_low'] = str(year_low.strip()) # 현재시점보다 낮으면 - 붙어서 나오는 듯
		
		if all_stocks_cash.strip() != '':
			ohlcv[stock_name]['all_stocks_cash'] = float(all_stocks_cash.strip())
		else:
			ohlcv[stock_name]['all_stocks_cash'] = str(all_stocks_cash.strip())
			
		ohlcv[stock_name]['all_stocks_cash_ratio'] = str(all_stocks_cash_ratio.strip())
		
		if outside_own.strip() != '':
			ohlcv[stock_name]['outside_own'] = float(outside_own.strip())
		else:
			ohlcv[stock_name]['outside_own'] = str(outside_own.strip())

		if per.strip() != '':
			ohlcv[stock_name]['per'] = float(per.strip())
		else:
			ohlcv[stock_name]['per'] = str(per.strip())

		if eps.strip() != '':
			ohlcv[stock_name]['eps'] = float(eps.strip())
		else:
			ohlcv[stock_name]['eps'] = str(eps.strip())

		if roe.strip() != '':
			ohlcv[stock_name]['roe'] = float(roe.strip())
		else:
			ohlcv[stock_name]['roe'] = str(roe.strip())

		if pbr.strip() != '':
			ohlcv[stock_name]['pbr'] = float(pbr.strip())
		else:
			ohlcv[stock_name]['pbr'] = str(pbr.strip())

		if ev.strip() != '':
			ohlcv[stock_name]['ev'] = float(ev.strip())
		else:
			ohlcv[stock_name]['ev'] = str(ev.strip())

		if bps.strip() != '':
			ohlcv[stock_name]['bps'] = float(bps.strip())
		else:
			ohlcv[stock_name]['bps'] = str(bps.strip())

		if profit_price.strip() != '':
			ohlcv[stock_name]['profit_price'] = float(profit_price.strip())
		else:
			ohlcv[stock_name]['profit_price'] = str(profit_price.strip())
			
		if profit_by_run.strip() != '':
			ohlcv[stock_name]['profit_by_run'] = float(profit_by_run.strip())
		else:
			ohlcv[stock_name]['profit_by_run'] = str(profit_by_run.strip())

		if tr_number.strip() != '':
			ohlcv[stock_name]['tr_number'] = float(tr_number.strip())
		else:
			ohlcv[stock_name]['tr_number'] = str(tr_number.strip())

		if tr_number_compare.strip() != '':
			ohlcv[stock_name]['tr_number_compare'] = float(tr_number_compare.strip())
		else:
			ohlcv[stock_name]['tr_number_compare'] = str(tr_number_compare.strip())

		ohlcv[stock_name]['date'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

		return ohlcv

	except Exception as e:
		print('error in tr_recieve handler : ', e )
		print('per : ', per)
		print('eps : ', eps)
		print('eps : ', eps)
		print('pbr ', pbr)
		print('ev : ', ev)
		print('bps : ', bps)

		traceback.print_exc()

def on_receive_realtime_data(kw: 'KiwoomAPI', stock_code, realtype):
	ohlcv = {}

	if realtype == "순간체결량" :
		print('enter real time data handler (1)')
		price = kw.get_comm_real_data(stock_code, 10)
		volume = kw.get_comm_real_data(stock_code, 15)

		ohlcv['price'] = abs(float(price.strip()))
		ohlcv['volume'] = abs(float(volume.strip()))

		return ohlcv

	elif realtype == "주식체결" : # 실제 모투 장중 돌아가던 부분
		print('enter real time data handler (2)')
		price = kw.get_comm_real_data(stock_code, 10)
		volume = kw.get_comm_real_data(stock_code, 15)

		ohlcv['price'] = abs(float(price.strip()))
		ohlcv['volume'] = abs(float(volume.strip()))

		return ohlcv

	elif realtype == "주식예상체결" : # 장마감 가까워지고 작동 부분
		print('enter real time data handler (2)')
		price = kw.get_comm_real_data(stock_code, 10)
		volume = kw.get_comm_real_data(stock_code, 15)

		ohlcv['price'] = abs(float(price.strip()))
		ohlcv['volume'] = abs(float(volume.strip()))

		return ohlcv

def on_recieve_unmet_order(kw: 'KiwoomAPI', rqname, trcode):
	"""
	double hash :
	종목이름 -> 주문번호 -> 나머지 순서
	"""
	ohlcv = {}
	
	data_cnt = kw.get_repeat_cnt(trcode, rqname)
	print('on_recieve_unmet_order data_cnt :: ', data_cnt)
	
	for i in range(data_cnt):
		order_num = str(kw.comm_get_data(trcode, "", rqname, 0, "주문번호")).strip() # order send 와 동일한 이름들 적용해야되는 부분 있음, 평단가로 계산되는 듯
		stock_code = str(kw.comm_get_data(trcode, "", rqname, 0, "종목코드")).strip()
		stock_name = kw.comm_get_data(trcode, "", rqname, 0, "종목명")
		order_num_count = kw.comm_get_data(trcode, "", rqname, 0, "주문수량")
		order_price = kw.comm_get_data(trcode, "", rqname, 0, "주문가격")
		unmet_order_num = kw.comm_get_data(trcode, "", rqname, 0, "미체결수량")
		order_state = kw.comm_get_data(trcode, "", rqname, 0, "주문구분")
		order_time = kw.comm_get_data(trcode, "", rqname, 0, "시간")
		price = kw.comm_get_data(trcode, "", rqname, 0, "현재가")
		
		if stock_name not in ohlcv:
			ohlcv[stock_name] = { } # 선언
		else: # stock_name 있음
			if order_num not in ohlcv[stock_name]:
				ohlcv[stock_name][order_num] = {}
			else:
				pass
			
		ohlcv[stock_name][order_num]['stock_name'] = str(stock_name.strip())
		ohlcv[stock_name][order_num]['order_num'] = str(order_num.strip())
		ohlcv[stock_name][order_num]['stock_code'] = str(stock_code.strip())
		ohlcv[stock_name][order_num]['order_num_count'] = int(order_num_count.strip()) # 원 주문수량인듯
		ohlcv[stock_name][order_num]['order_price'] = float(order_price.strip())
		ohlcv[stock_name][order_num]['unmet_order_num'] = int(unmet_order_num.strip())
		ohlcv[stock_name][order_num]['order_state'] = str(order_state.strip())
		ohlcv[stock_name][order_num]['order_time'] = str(order_time.strip())
		ohlcv[stock_name][order_num]['price'] = float(price.strip())
	
	return ohlcv

def on_receive_high_low_data(kw: 'KiwoomAPI', rqname, trcode):
	ohlcv = {}
	data_cnt = kw.get_repeat_cnt(trcode, rqname)

	print('on_receive_high_low_data data_cnt :: ', data_cnt)

	for i in range(data_cnt):
		stock_code = str(kw.comm_get_data(trcode, "", rqname, 0, "종목코드")).strip()
		stock_name = str(kw.comm_get_data(trcode, "", rqname, 0, "종목명")).strip()

		ohlcv[stock_code] = stock_name

	return ohlcv



