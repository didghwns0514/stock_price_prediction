
class ReturnWrap:

	CHECK_LIST_PREDICTER = ['PREDICTER', 'PREDICTER_TEST']

	CHECK_TOTAL_LIST = CHECK_LIST_PREDICTER

	@staticmethod
	def _type(_type, rtn_val):
		"""

		:param _type: type where it is used (ex module)
		:param rtn_val: return value of the module
		:return:
		"""

		assert _type in ReturnWrap.CHECK_TOTAL_LIST

		if _type == 'PREDICTER':
			ReturnWrap._predicter(rtn_val)

		elif _type == 'PREDICTER_TEST':
			ReturnWrap._predicter_test(rtn_val)

	@staticmethod
	def _predicter(rtn_val):
		pass

	@staticmethod

	def _predicter_test(rtn_val):
		"""

		:param rtn_val: return value to decode to action in the test session
		:return: action
		"""

		assert isinstance(rtn_val, list) and len(rtn_val) == 2

		log, data = rtn_val

		assert log in ['Predictable', 'No-prediction_set', 'Not-traininable', 'No-article']

		if log == 'Predictable':
			return 'Predictable'

		else:
			return 'No-predictable'