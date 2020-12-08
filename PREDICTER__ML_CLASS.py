
import os
import tensorflow as tf


class PrivTensorWrapper:

	LODADING_BOOL = None
	PT_CREATION_DIC_CNT = {}

	def __init__(self,
				stock_code,
				code_folder_location,
				save_file_location,
				loading_bool = False):

		if stock_code not in self.PT_CREATION_DIC_CNT:
			self.PT_CREATION_DIC_CNT[stock_code] = 0
		self.PT_CREATION_DIC_CNT[stock_code] += 1

		self.PT__GlobalCnt = 0

		self.PT__code_folder_location = code_folder_location
		self.PT__save_file_location = save_file_location

		self.MAIN_GRAPH = tf.Graph()
		self.MAIN_SESS = tf.Session()

	def PT__handle_mode(self):
		"""
		param : 
		return : Action - load if model save file exist
		"""
		if os.path.isfile(self.PT__save_file_location):
			pass #

	

class NestedGraph:
	
	LOOKUP = {}
	MAX_NUM_OF_ENSEM = 2

	def __init__(self):

		# @ locations
		self.AT_SAVE_PATH__folder = str(os.getcwd() + "\\PREDICTER__MODEL_SAVE")
		if os.path.isdir(self.AT_SAVE_PATH__folder):
			pass
		else:
			os.mkdir(self.AT_SAVE_PATH__folder)		

	def NG__wrapper(self):
		"""
		
		"""
		pass


	def NG__allocater(self, stock_code):
		"""
		:param stock_code
		:return: Action - create stock code folder, returns private tesnsor class
		"""

		# create stock folder location
		tmp_stock_code_folder = self.AT_SAVE_PATH__folder + '\\' + str(stock_code)
		if os.path.isdir(tmp_stock_code_folder):
			pass
		else:
			os.mkdir(tmp_stock_code_folder)
		
		# get save file location
		tmp_stock_code_checkpoint_file_location = tmp_stock_code_folder \
			  + '\\' + 'saved_model.h5'

		# create instance and return
		rtn = PrivTensorWrapper(
			 stock_code=stock_code,
			 code_folder_location=tmp_stock_code_folder,
		save_file_location=tmp_stock_code_checkpoint_file_location)

		return rtn
	
	def NG__check_graph(self, stock_code, day):

		if day not in NestedGraph.LOOKUP:
			NestedGraph.LOOKUP[day] = {}
		
		if stock_code not in NestedGraph.LOOKUP[day]:
			NestedGraph.LOOKUP[day][stock_code] = \
				self.NG__allocater(stock_code=stock_code)
