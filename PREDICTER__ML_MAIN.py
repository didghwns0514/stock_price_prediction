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


# @ tensorflow
import tensorflow as tf


# @ outside module
import ENCODER__ML_MAIN as EN
import DENOISER__ML_MAIN as DE
import sub_DATETIME_function as SUB_F
import PREDICTER__ML_CLASS as PCLS
from LOGGER_FOR_MAIN import pushLog as pl




class Stock_prediction:
	NAME = 'stock_prediction_'
	LENGTH__MINUTE_DATA = int(60 * 4) # 3 data used, stock / kospi / dollar-mearchant
	LENGTH__NEWS_ENCODED = int(20)
	LENGTH__ALL_INPUT = int(LENGTH__MINUTE_DATA * 3) \
						+ int(LENGTH__NEWS_ENCODED) \
						+ int(1200)
	LENGTH__ALL_OUTPUT = int(30)



	def __init__(self, module=True):

		# @ previous declarations
		self.AGENT_SUB__encoder = EN.Autoencoder(module=True)
		self.AGENT_SUB__denoiser = DE.Denoiser(module=True)
		self.nestgraph = PCLS.NestedGraph(shape_input=Stock_prediction.LENGTH__ALL_INPUT,
		shape_output=Stock_prediction.LENGTH__ALL_OUTPUT)

		#self.options = Options(self.envs)
		self.module = module


	def _getDay(self):

		return datetime.datetime.now().replace(hour=0,
										minute=0,
										second=0,
										mirosecond=0)


	




