from utils import *

import time
import numpy as np
import pandas as pd
import os



from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
import datetime
import seaborn as sns

import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.decomposition import PCA

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

def parser(x):
	return datetime.datetime.strptime(x,'%Y-%m-%d')

def ARAMIS(dataset):
	"""
	https://byeongkijeong.github.io/ARIMA-with-Python/
	https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
	https://www.google.com/search?rlz=1C1SQJL_koKR876KR876&sxsrf=ACYBGNSu5YMws6t1NwgfV1WtFb6ajsm11Q%3A1581861387281&ei=C0pJXoLpEM_ZhwPV4a6YDg&q=python+arima+order+selection&oq=python+arima+order&gs_l=psy-ab.3.1.0j0i30j0i8i30l3.399892.409370..410754...6.0..0.181.3878.1j28......0....1..gws-wiz.....0..0i131j0i67j33i160.vai200KrJF0
	http://alkaline-ml.com/pmdarima/
	https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python

	"""

	#c:\Users\82102\Anaconda3\envs\chicken36\Lib\site-packages\pyramid\arima\auto.py

	global test_begin_date
	from statsmodels.tsa.arima_model import ARIMA
	#from pyramid.arima import ARIMA
	from pyramid.arima import auto_arima


	date_index = dataset.index
	np_dataset = np.asarray(dataset['Close'].tolist())

	# filter them removing nan, for ARAMIS
	xi = np.arange(len(np_dataset))
	mask = np.isfinite(np_dataset)
	ARMIS_dataset_filtered = np.interp(xi, xi[mask], np_dataset[mask])

	X = ARMIS_dataset_filtered  # X = series.values
	size = test_begin_date
	train, test = X[0:size], X[size:len(X)]

	# fitting a stepwise model:
	stepwise_fit = auto_arima(train, start_p=5, start_q=5, max_p=10, max_q=10, m=1,
								 start_P=0, seasonal=False, d=0, D=0, trace=True,
								 max_D = 3,
								 error_action='ignore',
								 suppress_warnings=True,
								 stepwise=True, scoring='mse',
							  	 n_jobs= -1,
							  	 maxiter=1000)  # set to stepwise

	print(type(stepwise_fit))
	print(stepwise_fit)
	print(stepwise_fit.scoring)
	print(stepwise_fit.order)
	print(stepwise_fit.summary())

	selected_order = stepwise_fit.order

	# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
	# plot_acf(ARMIS_dataset_filtered)
	# plot_pacf(ARMIS_dataset_filtered)
	# plt.show()
	#
	#
	# X = ARMIS_dataset_filtered #X = series.values
	# size = test_begin_date
	# train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()

	for t in range(len(test)):

		model = ARIMA(history, order=selected_order)
		#from statsmodels.tsa.stattools import _safe_arma_fit
		#model = _safe_arma_fit(history, order=selected_order)
		#model_fit = ARIMA(order = selected_order, start_params=selected_order).fit(y=history)
		#model._
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		#output = model_fit.predict(n_periods=1)
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)

	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)

	print('There are {} number of days in the dataset.'.format(dataset.shape[0]))
	print('There are {} number of days in the predict set.'.format(len(predictions)))
	print('There are {} number of days in the history set.'.format(len(history)))

	predictions = np.array(predictions)

	previous_nan_array = np.empty((len(train),1))
	previous_nan_array[:] = np.NaN
	#full_array = np.transpose(np.concatenate([previous_nan_array, predictions]))
	full_array = (np.concatenate([previous_nan_array, predictions]))
	full_df = pd.DataFrame(full_array, index=date_index)

	return full_df

	# model = ARIMA(series, order=(5, 1, 0))
	# model_fit = model.fit(disp=0)
	# print(model_fit.summary())
	# input('?')
def feature_XGB(dataset_techinical):
	global test_begin_date
	dataset_techinical_ = dataset_techinical.copy()

	def get_feature_importance_data(data_income):
		global test_begin_date

		data = data_income.copy()
		data.dropna(axis=0, how='any', inplace=True)
		data.drop(columns=['Open','Volume','Low','High'], inplace =True)

		print(data.head(10))

		y = data['Close']
		print('y data \n', y.head(10))
		X = data.iloc[:, 1:]
		print('X : \n',X.head(10))
		X.drop(columns=['Close'], inplace=True)
		print('X after drop : \n',X.head(10))

		train_samples = int(X.shape[0] * 0.75)

		X_train = X.iloc[:train_samples]
		X_test = X.iloc[train_samples:]

		y_train = y.iloc[:train_samples]
		y_test = y.iloc[train_samples:]

		return (X_train, y_train), (X_test, y_test)

	(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_techinical_)
	regressor = xgb.XGBRegressor(gamma=0.0, n_estimators=150, base_score=0.7, colsample_bytree=1, learning_rate=0.05)
	xgbModel = regressor.fit(X_train_FI, y_train_FI, \
							 eval_set=[(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
							 verbose=False)
	eval_result = regressor.evals_result()
	print('eval result : ', eval_result)
	print('type of eval reasult : ', type(eval_result))

	training_rounds = range(len(eval_result['validation_0']['rmse']))

	plt.figure(figsize=(16, 10), dpi=100)
	plt.subplot(2,1,1)
	plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
	plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
	plt.xlabel('Iterations')
	plt.ylabel('RMSE')
	plt.title('Training Vs Validation Error')
	plt.legend()
	#plt.show()

	plt.subplot(2, 1, 2)
	plt.xticks(rotation='vertical')
	plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(),
			tick_label=X_test_FI.columns)
	plt.title('Figure 6: Feature importance of the technical indicators.')
	plt.show()



def get_technical_indicators(dataset):
	# Create 7 and 21 days Moving Average
	dataset['ma5'] = dataset['Close'].rolling(window=5).mean()
	dataset['ma10'] = dataset['Close'].rolling(window=10).mean()
	dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
	dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
	dataset['ma33'] = dataset['Close'].rolling(window=33).mean() ####
	dataset['ma60'] = dataset['Close'].rolling(window=60).mean() ####
	dataset['ma120'] = dataset['Close'].rolling(window=120).mean()  ####

	# Create MACD
	dataset['26ema'] = dataset.Close.ewm(span=26).mean()
	dataset['12ema'] = dataset.Close.ewm(span=12).mean()
	dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
	dataset['MACDS'] = dataset.MACD.ewm(span=9).mean()
	dataset['MACDO'] = (dataset['MACD'] - dataset['MACDS'])

	# Disparity
	dataset['disparity5'] = (dataset['Close'])/(dataset['ma5'])*100
	dataset['disparity33'] = (dataset['Close']) / (dataset['ma33'])*100

	#RSI
	"""
	https://github.com/mrjbq7/ta-lib
	"""
	def calc_rsi(dataset) :
		date_index = dataset.index
		period = 14
		U = np.where(dataset['Close'].diff(1) > 0, dataset['Close'].diff(1), 0)
		D = np.where(dataset['Close'].diff(1) <0 , dataset['Close'].diff(1)*(-1), 0)
		AU = pd.DataFrame(U,index=date_index).rolling(window=period).mean()
		AD = pd.DataFrame(D,index=date_index).rolling(window=period).mean()
		RSI = (AU*100) / (AD + AU)

		return RSI
	dataset['RSI'] = calc_rsi(dataset).copy()


	#CCI
	def calc_cci(dataset) :
		period = 20
		Mt = (dataset['High'] + dataset['Low'] + dataset['Close'])/3
		Ma = Mt.rolling(window=period).mean()
		D = abs(Mt-Ma).rolling(window=period).mean()
		CCI = (Mt - Ma)/(D*0.015)

		return CCI
	dataset['CCI'] = calc_cci(dataset).copy()


	# Create Bollinger Bands
	dataset['20sd'] = dataset['Close'].rolling(window=20).std()
	dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
	dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

	# Create Exponential moving average
	dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()


	#KDJ stochastic
	def calc_KDJ(dataset):
		"""
		https://excelsior-cjh.tistory.com/111
		"""
		n = 15
		m = 5
		t = 3
		ndays_high = dataset['High'].rolling(window=n, min_periods=1).max()
		ndays_low = dataset['Low'].rolling(window=n, min_periods=1).min()

		#Fast k => not normally used
		kdj_k = ((dataset['Close'] - ndays_low) / (ndays_high - ndays_low))*100
		slow_k = kdj_k.ewm(span=m).mean()
		slow_d = slow_k.ewm(span=t).mean()

		return slow_k, slow_d

	dataset['slow_k'], dataset['slow_d'] = calc_KDJ(dataset)

	#Fourier transforms
	def calc_fourier(dataset):
		date_index = dataset.index
		np_dataset = np.asarray(dataset['Close'].tolist())

		#filter them removing nan, for FT
		xi = np.arange(len(np_dataset))
		mask = np.isfinite(np_dataset)
		np_dataset_filtered = np.interp(xi, xi[mask], np_dataset[mask])

		close_fft = np.fft.fft(np_dataset_filtered) # 종가 퓨리에 변환
		# print(close_fft)
		# print(np.count_nonzero(~np.isnan(close_fft)))

		fft_df_3 = np.copy(close_fft)
		fft_df_3[3:-3]=0
		fft_df_3 = np.copy(np.absolute(np.fft.ifft(fft_df_3)))
		#print(np.count_nonzero(~np.isnan(fft_df_3)))
		data3 = pd.DataFrame(fft_df_3, index=date_index)

		fft_df_6 = np.copy(close_fft)
		fft_df_6[6:-6] = 0
		fft_df_6 = np.copy(np.absolute(np.fft.ifft(fft_df_6)))
		#print(np.count_nonzero(~np.isnan(fft_df_6)))
		#data6 = pd.DataFrame({'fft_6': fft_df_6})
		data6 = pd.DataFrame(fft_df_6, index=date_index)

		fft_df_9 = np.copy(close_fft)
		fft_df_9[9:-9] = 0
		fft_df_9 = np.copy(np.absolute(np.fft.ifft(fft_df_9)))
		#print(np.count_nonzero(~np.isnan(fft_df_9)))
		data9 = pd.DataFrame(fft_df_9, index=date_index)

		k = 100
		fft_df_k = np.copy(close_fft)
		fft_df_k[k:-k] = 0
		fft_df_k = np.copy(np.absolute(np.fft.ifft(fft_df_k)))
		#print(np.count_nonzero(~np.isnan(fft_df_k)))
		datak = pd.DataFrame(fft_df_k, index=date_index)

		# print(data3.head(4))
		# print(data6.head(4))
		# print(data9.head(4))

		# fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
		# fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

		return data3, data6, data9, datak

	dataset['ft_3'], dataset['ft_6'], dataset['ft_9'], dataset['ft_k'] = calc_fourier(dataset)



	# Create Momentum
	def calc_momentum(closes):
		from scipy.stats import linregress
		returns = np.log(closes)
		x = np.arange(len(returns))
		slope, _, rvalue, _, _ = linregress(x, returns)
		return ((1 + slope) ** 252) * (rvalue ** 2)


	dataset['log_momentum'] = dataset['Close'].rolling(90).apply(calc_momentum, raw=False)
	dataset['momentum'] = dataset['Close'] - 1

	csv_check_dir = str(os.getcwd() + "\\csv_check")
	if not os.path.isdir(csv_check_dir):
		os.mkdir(csv_check_dir)
		dataset.to_csv(os.getcwd() + "\\csv_check\\parsed_data.csv")
		print('csv_checker is saved...')
	else:
		dataset.to_csv(os.getcwd() + "\\csv_check\\parsed_data.csv")
		print('csv_checker is saved...')

	dataset['ARAMIS'] = ARAMIS(dataset)

	return dataset


def plot_technical_indicators(dataset, last_days):
	global test_begin_date

	plt.figure(figsize=(16, 10), dpi=100)
	shape_0 = dataset.shape[0]
	xmacd_ = shape_0 - last_days

	dataset = dataset.iloc[-last_days:, :]
	x_ = range(3, dataset.shape[0])
	x_ = list(dataset.index)
	"""
	https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
	"""
	# Plot first subplot
	plt.subplot(2, 1, 1)
	plt.plot(dataset['ma5'], label='MA 5', color='b', linestyle='--')
	plt.plot(dataset['ma33'], label='MA 33', color='k', linestyle='--')
	plt.plot(dataset['Close'], label='Closing Price', color='b')
	plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
	plt.plot(dataset['26ema'], label='EMA 26', color='y', linestyle='-.')
	plt.plot(dataset['12ema'], label='EMA 12', color='k', linestyle='-.')
	plt.plot(dataset['ft_3'], label='FT 3', color='m', linestyle='-.')
	plt.plot(dataset['ft_6'], label='FT 6', color='c', linestyle='-.')
	plt.plot(dataset['ft_9'], label='FT 9', color='k', linestyle='-.')
	plt.plot(dataset['ft_k'], label='FT k', color='r', linestyle='-.')
	plt.plot(dataset['ARAMIS'], label='ARAMIS', color='r', linestyle = '-')
	plt.plot(dataset['upper_band'], label='Upper Band', color='c')
	plt.plot(dataset['lower_band'], label='Lower Band', color='c')
	max_y_value = dataset['Close'].max()
	plt.vlines(test_begin_date, 0, max_y_value, linestyles='--', colors='gray', label='Test data cut-off')
	plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
	plt.title('Technical indicators for Microsoft - last {} days.'.format(last_days))
	plt.ylabel('USD')
	plt.legend()

	# Plot second subplot
	plt.subplot(2, 1, 2)
	plt.title('MACD')
	plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
	plt.plot(dataset['MACDO'], label='MACDO', linestyle='-.')
	plt.plot(dataset['log_momentum'], label='Momentum', color='b', linestyle='-')
	plt.plot(dataset['RSI'], label='RSI14', color='y', linestyle='solid')
	plt.plot(dataset['CCI'], label='CCI', color='g', linestyle='-')
	plt.fill_between(x_, dataset['RSI'], 0, alpha=0.25)

	plt.plot(dataset['slow_k'], label='slow_k', color='b', linestyle='--')
	plt.plot(dataset['slow_d'], label='slow_d', color='g', linestyle='--')

	plt.plot(dataset['disparity5'] , label='Disp5', linestyle='-')
	plt.plot(dataset['disparity33'], label='Disp33', linestyle='-')

	plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
	plt.hlines(0, xmacd_, shape_0, colors='b', linestyles='-', alpha=0.35)
	plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')



	plt.legend()
	plt.show()

test_begin_date = None

def main():
	global test_begin_date

	dir_to_folder = os.getcwd() + str('\\csv_original')

	#set1 = pd.read_csv(dir_to_folder + '\\' + 'bitcoin_daily_12.25.csv').drop(columns=['Volume (BTC)','Weighted Price'])
	set2 = pd.read_csv(dir_to_folder + '\\' + 'continental.csv').drop(columns=['Adj Close'])
	#set3 = pd.read_csv(dir_to_folder + '\\' + 'hyundai_motor.csv').drop(columns=['Adj Close'])
	#set4 = pd.read_csv(dir_to_folder + '\\' + 'inno.csv', header=0, parse_dates=[0],).drop(columns=['Adj Close'])
	#set5 = pd.read_csv(dir_to_folder + '\\' + 'microsoft.csv', header=0, parse_dates=[0], date_parser=parser).drop(columns=['Adj Close'])
	#set6 = pd.read_csv(dir_to_folder + '\\' + 'samsung_electronics.csv').drop(columns=['Adj Close'])

	#dataset_ex_df = pd.read_csv('data/panel_data_close.csv', header=0, parse_dates=[0], date_parser=parser)
	dataset_ex_df = set2.copy()
	print(dataset_ex_df[['Date', 'Close']].head(3))
	print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

	# plt.figure(figsize=(14, 5), dpi=90)
	# plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label='Microsoft stock')

	test_begin_date = int(dataset_ex_df.shape[0]*.75)
	axis_date = dataset_ex_df['Date'].iloc[test_begin_date]
	max_y_value = dataset_ex_df['Close'].max()
	plt.vlines(axis_date, 0, max_y_value, linestyles='--', colors='gray', label='Train/Test data cut-off')
	plt.xlabel('Date')
	plt.ylabel('USD')
	plt.title('Figure 1: Microsoft stock price')
	plt.legend()
	plt.show()


	print('Number of training days: {}. Number of test days: {}.'.format(test_begin_date, dataset_ex_df.shape[0]-test_begin_date))

	#dataset_TI_df = get_technical_indicators(dataset_ex_df[['Close']])
	dataset_TI_df = get_technical_indicators(dataset_ex_df).copy()
	dataset_TI_df.head()

	plot_technical_indicators(dataset_TI_df, dataset_TI_df.shape[0])
	#feature_XGB(dataset_TI_df) # -> 이상하다

	print('Total dataset has {} samples, and {} features.'.format(dataset_TI_df.shape[0],dataset_TI_df.shape[1]))

class Options :
	def __init__(self, env):

		self.INPUT_DATA_DIM = env[0] # 입력 데이터 variable 갯수
		self.RNN_HIDDEN_CELL_DIM = env[1] # 각 셀의 (hidden)출력 크기
		self.RESULT_DATA_DIM = env[2] # 결과데이터의 컬럼 개수 : many to one
		self.N_EMBEDDING = env[3] # stacked LSTM layers 개수
		self.SEQ_LENGTH = env[4] # window, for time series length

		self.FORGET_BIAS = env[5] # 망각편향(기본값 1.0)
		self.MAX_EPISODE = env[6]  # max number of episodes iteration
		self.LR = env[7]  # learning rate # 학습률
		self.DROP_OUT = env[8]

class LSTM :

	""""
	https://teddylee777.github.io/machine-learning/sklearn%EC%99%80-pandas%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%84%EB%8B%A8-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D
	https://m.blog.naver.com/wideeyed/221160038616
	"""
	# 일단위 학습 모델에 분을 추가해도 되려나...????
	# many to one
	# 다음날 종가 / open가
	#       0,   1,   2,   3,    4,  5,     6,     7
	envs = [35, 30,   1,  32, 31*3,  1,  1000,  0.01]
	
	def __init__(self, dataset, objective):
		from sklearn.preprocessing import MinMaxScaler
		from sklearn.preprocessing import StandardScaler
		import tensorflow as tf

		self.sess = tf.compat.v1.InteractiveSession()

		self.options = Options(self.envs)
		self.dataset = self.clear_nan(dataset).copy()
		self.objective = objective # close or open as string value

		self.hypothesis, self.answer, self.targets, self.predictions = self.add_value_net(self.options)
		self.loss = tf.reduce_sum(tf.square(self.hypothesis - self.answer))
		self.optimizer = tf.train.AdamOptimizer(self.options.LR)
		self.train = self.optimizer.minimize(self.loss)

		self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.targets, self.predictions)))

		self.sess.run(tf.global_variables_initializer())

	def dataset_to_train(self, dataset):
		pass

	def add_value_net(self, options): # creating lstm celss
		import tensorflow as tf

		def lstm_cell(options): #single lstm celss
			cell = tf.contrib.rnn.BasicLSTMCell(num_units=options.RNN_HIDDEN_CELL_DIM,
												forget_bias=options.FORGET_BIAS, state_is_tuple=True,activation=tf.nn.softsign)
			if options.DROP_OUT < 1.0:
				cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=options.DROP_OUT)
			else:
				pass
			return cell
		# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
		targets = tf.placeholder(tf.float32, [None, 1])
		predictions = tf.placeholder(tf.float32, [None, 1])

		observation = tf.compat.v1(tf.float32, [None, options.SEQ_LENGTH, options.INPUT_DATA_DIM])
		answer = tf.placeholder(tf.float32, [None, 1])
		stackedRNNs = [lstm_cell(options) for _ in range(options.N_EMBEDDING)]
		multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if options.N_EMBEDDING > 1 else lstm_cell(options)

		H_, _states = tf.nn.dynamic_rnn(multi_cells, observation, dtype=tf.float32)
		H = tf.contrib.layer.fully_connected(H_[:-1], options.RESULT_DATA_DIM, activation_fn=tf.identity)

		return H, answer, targets, predictions


	def clear_nan(self, dataset):
		dataset.dropna(axis=0, how='any', inplace=True)
		return dataset


	# def data_standardization(self, data): # 정규화 하기
	# 	st_data = np.asarray(data)
	# 	return (st_data - st_data.mean())/st_data.std()
	# 	# 데이터가 너무 크거나 작을까봐
	#
	# def data_min_max_scaling(self, data):
	#
	# 	st_data = np.asarray(data)
	# 	return (st_data - st_data.min()) / (st_data.max() - st_data.min() + 1e-7)
	#
	# def inv_min_max_scaling(self,org_data_, data_ ):
	# 	# org_data : 정규화 하기 이전의 데이터
	# 	org_data = np.asarray(org_data_)
	# 	data = np.asarray(data_)
	# 	return(data * ( org_data.max()-org_data.min() + 1e-7 )) + org_data.min()



if __name__ == '__main__':
	main()