import datetime
from sub_function_configuration import *


def SESS_parse_answer_data_from_sqlite(tmp_whole_df, datetime_obj_now, min_duration_forward):
    # import sub_DATETIME_function as SUB_F
    return_dict = {}

    # import pandas as pd
    # head_string = 'SELECT * FROM '
    # tmp_table_name_sql = "'" + str(stock_code) + "'"
    # tmp_whole_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top_connection, index_col=None)
    # tmp_whole_df['date'] = pd.to_datetime(tmp_whole_df['date'], format="%Y%m%d%H%M%S")

    if datetime_obj_now + datetime.timedelta(minutes=min_duration_forward) <= datetime_obj_now.replace(hour=15,
                                                                                                       minute=30):
        df_target = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_obj_now) & (
                tmp_whole_df.date < datetime_obj_now + datetime.timedelta(minutes=min_duration_forward))]

        dict_target = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_target), datetime_obj_now,
                                                      datetime_obj_now + datetime.timedelta(
                                                          minutes=min_duration_forward))

        return_dict.update(dict_target)

    else:
        tmp_delta_time_forward_now = datetime_obj_now.replace(hour=15, minute=30, second=0,
                                                              microsecond=0) - datetime_obj_now
        tmp_delta_time_forward_in_minutes = divmod(tmp_delta_time_forward_now.total_seconds(), 60)[0]
        tmp_calc_minutes = min_duration_forward - tmp_delta_time_forward_in_minutes
        datetime_target = None
        if datetime_obj_now.weekday() == 4:  # 금요일
            datetime_target = datetime_obj_now + datetime.timedelta(days=3)

        else:
            datetime_target = datetime_obj_now + datetime.timedelta(days=1)

        df_target_start_time = datetime_target.replace(hour=9, minute=0, second=0, microsecond=0)
        df_target_end_time = df_target_start_time + datetime.timedelta(minutes=tmp_calc_minutes)
        df_target = tmp_whole_df.loc[
            (tmp_whole_df.date >= df_target_start_time) & (tmp_whole_df.date <= df_target_end_time)]

        df_now = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_obj_now) & (
                tmp_whole_df.date < datetime_obj_now.replace(hour=15, minute=30, second=0, microsecond=0))]

        # dictionary
        dict_df_now = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_now), datetime_obj_now,
                                                      datetime_obj_now.replace(hour=15, minute=30, second=0,
                                                                               microsecond=0))
        dict_df_target = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(df_target),
                                                         df_target_start_time, df_target_end_time)

        # @ update
        return_dict.update(dict_df_now)
        return_dict.update(dict_df_target)

    return return_dict


# @decorator_function
def SESS_parse_data_from_sqlite(tmp_whole_df, datetime_obj_now, hours_duration_back):
	return_dict = {}

	# import pandas as pd
	# head_string = 'SELECT * FROM '
	# tmp_table_name_sql = "'" + str(stock_code) + "'"
	# tmp_whole_df = pd.read_sql(head_string + tmp_table_name_sql, sqlite_con_top_connection, index_col=None)
	# tmp_whole_df['date'] = pd.to_datetime(tmp_whole_df['date'], format="%Y%m%d%H%M%S")

	# print(f'at the start datetime_obj_now : {datetime_obj_now}')

	tmp_datetime_list__obj_to_parse = SUB_F.FUNC_return_datetime_obj__backward(datetime_obj_now, hours_duration_back)

	for i in range(len(tmp_datetime_list__obj_to_parse)):
		tmp_df = None
		if i == 0:
			tmp_df = tmp_whole_df.loc[(tmp_whole_df.date >= tmp_datetime_list__obj_to_parse[i][0]) & (
						tmp_whole_df.date <= tmp_datetime_list__obj_to_parse[i][1])]
		else:
			tmp_df = tmp_whole_df.loc[(tmp_whole_df.date >= tmp_datetime_list__obj_to_parse[i][0]) & (
						tmp_whole_df.date <= tmp_datetime_list__obj_to_parse[i][1])]

		if tmp_df.empty:
			pass
		else:
			tmp_dict = SESS__fill_missing_data_in_dict(SESS__convert_dataframe_to_dic(tmp_df),
													   tmp_datetime_list__obj_to_parse[i][0],
													   tmp_datetime_list__obj_to_parse[i][1])
			return_dict.update(tmp_dict)

	return return_dict


# tmp_single_day_df = tmp_whole_df.loc[(tmp_whole_df.date >= datetime_single_start__fix_obj) & (tmp_whole_df.date < datetime_single_start__end_obj)]

def SESS__convert_dataframe_to_dic(dataframe):
	# datetime.datetime.now().strftime('%Y%m%d%H%M%S') : obj to string
	# print(f'★★★ convert_dataframe_to_dic len of dataframe : {len(dataframe)}')
	tmp_dictionary_return = {}

	for row_tuple in dataframe.itertuples():
		tmp_dictionary_return[row_tuple.date.strftime('%Y%m%d%H%M%S')] = {'price': row_tuple.open,
																		  'volume': row_tuple.volume}

	# print(f'☆☆☆ convert_dataframe_to_dic len of tmp_dictionary_return : {len(list(tmp_dictionary_return.keys()))}')
	# print(f'dataframe, tmp_dictionary_return : {dataframe, tmp_dictionary_return}')
	return tmp_dictionary_return


def SESS__fill_missing_data_in_dict(dictionary, start_time_obj, end_time_obj):
	####여기서 missing 나온다
	# try:
	tmp_return_dictionary = copy.deepcopy(dictionary)

	tmp_list_of_missing_datastamp = []

	tmp_datetime_stamp_list = list(dictionary.keys())
	# print(f'dictionary : {dictionary}')
	# print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')
	tmp_datetime_stamp_list.sort()

	# print(f'tmp_datetime_stamp_list : {tmp_datetime_stamp_list}')

	#	if tmp_return_dictionary : # not empty

	tmp_start_datetime_stamp = tmp_datetime_stamp_list[0]  # 첫 데이터
	tmp_end_datetime_stamp = tmp_datetime_stamp_list[-1]  # 마지막 데이터
	tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)

	if tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj:
		before_price = None
		before_volume = None
		while tmp_start_datetime_stamp_obj <= tmp_end_datetime_stamp_obj:  # datetime obj끼리 비교 while 문이라 위험??
			# @ 처음은 list에서 뽑아왔으므로 있다
			tmp_start_datetime_stamp_obj_convert = tmp_start_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
			if tmp_start_datetime_stamp_obj_convert in dictionary:
				before_price = dictionary[tmp_start_datetime_stamp_obj_convert]['price']
				before_volume = dictionary[tmp_start_datetime_stamp_obj_convert]['volume']
			else:
				tmp_list_of_missing_datastamp.append(tmp_start_datetime_stamp_obj_convert)
				tmp_return_dictionary[tmp_start_datetime_stamp_obj_convert] = {'price': before_price,
																			   'volume': 0}  # 'volume': before_volume

			tmp_start_datetime_stamp_obj = tmp_start_datetime_stamp_obj + datetime.timedelta(minutes=1)

	# 1) 뒤쪽에서 값이 missing된 경우
	tmp_end_datetime_stamp_obj = datetime.datetime.strptime(tmp_end_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_end_stub_price = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['price']
	tmp_end_stub_volume = dictionary[tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')]['volume']
	while tmp_end_datetime_stamp_obj < end_time_obj:
		tmp_end_datetime_stamp_obj_convert = tmp_end_datetime_stamp_obj.strftime('%Y%m%d%H%M%S')
		if tmp_end_datetime_stamp_obj_convert in tmp_return_dictionary:
			pass
		else:
			tmp_return_dictionary[tmp_end_datetime_stamp_obj_convert] = {'price': tmp_end_stub_price,
																		 'volume': 0}  # 'volume': tmp_end_stub_volume
		tmp_end_datetime_stamp_obj = tmp_end_datetime_stamp_obj + datetime.timedelta(minutes=1)

	# 2) 앞쪽에서 값이 missing된 경우
	tmp_start_datetime_stamp_obj = datetime.datetime.strptime(tmp_start_datetime_stamp, "%Y%m%d%H%M%S").replace(
		second=0, microsecond=0)
	tmp_start_stub_price = dictionary[tmp_start_datetime_stamp]['price']
	tmp_start_stub_volume = dictionary[tmp_start_datetime_stamp]['volume']
	tmp_end_time_obj = start_time_obj
	while tmp_end_time_obj <= tmp_start_datetime_stamp_obj:
		tmp_end_time_obj_convert = tmp_end_time_obj.strftime('%Y%m%d%H%M%S')
		if tmp_end_time_obj_convert in tmp_return_dictionary:
			pass
		else:
			tmp_return_dictionary[tmp_end_time_obj_convert] = {'price': tmp_start_stub_price,
															   'volume': 0}  # 'volume': tmp_start_stub_volume

		tmp_end_time_obj = tmp_end_time_obj + datetime.timedelta(minutes=1)

	# print(f'tmp_return_dictionary in  SESS__fill_missing_data_in_dict : \n{tmp_return_dictionary}')

	return tmp_return_dictionary