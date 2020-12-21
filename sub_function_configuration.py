import datetime
#import copy


def FUNC_dtSwtich(datetime_item, string_method='%Y%m%d%H%M%S'):
    """

    :param datetime_item: datetime either string or datetime object
    :return: converts to each cases and returns
    """
    if isinstance(datetime_item, str):
        return datetime.datetime.strptime(datetime_item, string_method)

    elif isinstance(datetime_item, datetime.date):
        return datetime_item.strftime(string_method)


def FUNC_dtRect(datetime_obj, string_time=None):
    """

    :param datetime_obj: datetime object to strip second and microsecond
    :return: returns rectified time object in datetime format
    """
    if string_time != None:
        assert isinstance(datetime_obj, datetime.date)
        assert isinstance(string_time, str)

        h, m = string_time.split(':')

        return datetime_obj.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
    else:
        return datetime_obj.replace(second=0, microsecond=0)






def FUNC_datetime_backward(datetime_now__obj_, hours_back):
    """
    dont include time, 'NOW' in the return
    """
    
    tmp_list_for_return = []
    tmp_list_for_return_ = None # for ram!
    
    tmp_total_minutes_back = 60 * hours_back
    datetime_now__obj = FUNC_dtRect(datetime_now__obj_)
    datetime_today_fix_start__obj = FUNC_dtRect(datetime_now__obj_, "9:00")
    datetime_today_fix_end__obj = FUNC_dtRect(datetime_now__obj_,"15:30")
    
    # @ adjust today's date
    if datetime_now__obj < datetime_today_fix_start__obj:
        datetime_now__obj = datetime_today_fix_start__obj
    elif datetime_now__obj > datetime_today_fix_end__obj:
        datetime_now__obj = datetime_today_fix_end__obj
    
    # @ tmp_minutes to calculate
    tmp_datetime_for_secs_today = datetime_now__obj - datetime_today_fix_start__obj
    tmp_datetime_for_min_today = divmod(tmp_datetime_for_secs_today.total_seconds(), 60)
    tmp_minutes_to_goback = tmp_total_minutes_back - tmp_datetime_for_min_today[0]
    
    # @ already inside today's window coverage
    if tmp_minutes_to_goback <= 0:
        tmp_list_for_return.append([datetime_now__obj - datetime.timedelta(minutes=tmp_total_minutes_back), datetime_now__obj])
        return tmp_list_for_return
    
    else:
        tmp_list_for_return.append( [FUNC_dtRect(datetime_now__obj,"9:00"),
                                     datetime_now__obj])
        tmp_list_for_return = func_sub_iteratior(tmp_minutes_to_goback, datetime_today_fix_start__obj, tmp_list_for_return)
        return tmp_list_for_return


def func_sub_iteratior(miutes_left, before_datetime_start__obj, return_list):
    TOTAL_DAY_MINUTE_NUMBER = 391 # stock day's window in minutes
    
    tmp_before_datetime__obj = FUNC_dtRect(before_datetime_start__obj, "9:00")
    tmp_target_datetime__obj = tmp_before_datetime__obj
    while (tmp_target_datetime__obj.weekday()) in [5, 6] or \
          (tmp_before_datetime__obj.weekday() == tmp_target_datetime__obj.weekday()):
        # 월, 화, 수, 목, 금, 토, 일
        # 0,  1,  2,  3, 4, 5,  6
        tmp_target_datetime__obj = tmp_target_datetime__obj - datetime.timedelta(days=1)
    
    if miutes_left <= TOTAL_DAY_MINUTE_NUMBER:
        return_list.append( [ FUNC_dtRect(tmp_target_datetime__obj,"15:30") - datetime.timedelta(minutes=miutes_left) ,
                              FUNC_dtRect(tmp_target_datetime__obj,"15:30")] )
        return return_list
    
    else:
        tmp_minutes_left = miutes_left - TOTAL_DAY_MINUTE_NUMBER
        return_list.append( [ FUNC_dtRect(tmp_target_datetime__obj,"9:00") ,
                              FUNC_dtRect(tmp_target_datetime__obj, "15:30") ] )
        
        return func_sub_iteratior(tmp_minutes_left, FUNC_dtRect(tmp_target_datetime__obj,"9:00"), return_list )

        
if __name__ == '__main__':
    import random
    iter_num = 1
    #for i in range(iter_num):
    datetime_tmp = datetime.datetime.now()
    #year = now.year
    months = int(input('month you want? : '))
    days = int(input('day you want? : '))
    hours = int(input('hour you want? : '))
    hours_back = int(input('hours you want to go back? : '))
    #minutes = int(random.randrange(0, 60))
    minutes = int(0)

    datetime_tmp = datetime_tmp.replace(month=months, day=days, hour=hours, minute=minutes)
    print(f'created date stamp : {datetime_tmp}')

    tmp_return = FUNC_datetime_backward(datetime_tmp, hours_back)
    print(f'returned list : {tmp_return}')
    print(f'\n')
    print(f'hours back in total minutes : {hours_back*60}')
    tmp_counter = 0
    for items in tmp_return:
        # start, end
        start = items[0]
        end = items[1]
        secs = end - start
        mins = divmod(secs.total_seconds(), 60)
        print(f'mins : {mins}')
        tmp_counter = tmp_counter + int(mins[0])
        print('-'*30)
        print('\n')
    print(f'total time accumulated in minutes : {tmp_counter}')

#     input(';;;;;;')
#     print('\n'*2)
    
