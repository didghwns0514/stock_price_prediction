# -*-coding: utf-8-*-

# @ normal library
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
import traceback
import joblib
import pickle
import codecs

# @ keras
import keras
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, UpSampling1D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras import backend as K
import h5py
from keras.backend.tensorflow_backend import set_session

# @ xgboos
import xgboost

# @ tensorflow
import tensorflow as tf


# @ outside module
import ENCODER__ML_MAIN as EN
import DENOISER__ML_MAIN as DE
import sub_DATETIME_function as SUB_F



class NN_wrapper:
	def __init__(self):
		pass

class Options:
	def __init__(self, env):
		self.INPUT_DATA_DIM = env[0]  # 입력 데이터 variable 갯수
		self.RNN_HIDDEN_CELL_DIM = env[1]  # 각 셀의 (hidden)출력 크기
		self.RESULT_DATA_DIM = env[2]  # 결과데이터의 컬럼 개수 : many to one
		self.N_EMBEDDING = env[3]  # stacked LSTM layers 개수
		self.SEQ_LENGTH = env[4]  # window, for time series length

		self.FORGET_BIAS = env[5]  # 망각편향(기본값 1.0)
		self.MAX_EPISODE = env[6]  # max number of episodes iteration
		self.LR = env[7]  # learning rate # 학습률
		self.DROP_OUT = env[8]

class Regression_stock_prediction:
	NAME = 'XGBOOST_stock_prediction_'
	LENGTH__MINUTE_DATA = int(60 * 4) # 3 data used, stock / kospi / dollar-mearchant
	LENGTH__NEWS_ENCODED = int(20)
	LENGTH__ALL_INPUT = int(LENGTH__MINUTE_DATA * 3) \
						+ int(LENGTH__NEWS_ENCODED) \
						+ int(1200)
	#                 0,  1,   2,   3,              4,    5,      6,       7,     8
	envs = [None, 90,  30,   2, None,  1.0,   500,  0.0005,  0.6]  # 0.6 / 0.72 # 180
	# 7 -> 0.0006


	def __init__(self, module=True):

		# @ previous declarations
		self.AGENT_SUB__encoder = EN.Autoencoder(module=True)
		self.AGENT_SUB__denoiser = DE.Denoiser(module=True)

		self.options = Options(self.envs)
		self.module = module


		# @ locations
		self.AT_SAVE_PATH__folder = str(os.getcwd() + "\\PREDICTER__REGRESSION_SAVE")
		if os.path.isdir(self.AT_SAVE_PATH__folder):
			pass
		else:
			os.mkdir(self.AT_SAVE_PATH__folder)




	def FUNC_FOLDER__per_stock(self, stock_code):
		"""

		:param stock_code:
		:return:  if check point exists, returns check point.
		"""
		pass


