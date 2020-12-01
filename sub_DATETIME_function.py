import datetime
#import copy


def FUNC_return_datetime_obj__backward(datetime_now__obj_, hours_back):
    """
    dont include time, 'NOW' in the return
    """
    
    tmp_list_for_return = []
    tmp_list_for_return_ = None # for ram!
    
    tmp_total_minutes_back = 60 * hours_back
    datetime_now__obj = datetime_now__obj_.replace(second=0, microsecond=0) #- datetime.timedelta(minutes=1)
    datetime_today_fix_start__obj = datetime_now__obj_.replace(hour=9, minute=0, second=0, microsecond=0)
    datetime_today_fix_end__obj = datetime_now__obj_.replace(hour=15, minute=30, second=0, microsecond=0)
    
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
        #print('FUNC_return_datetime_obj - 1')
        # [start, end] for each day
        tmp_list_for_return.append([datetime_now__obj - datetime.timedelta(minutes=tmp_total_minutes_back), datetime_now__obj])
        return tmp_list_for_return
    
    else:
        #print('FUNC_return_datetime_obj - 2')
        tmp_list_for_return.append([datetime_now__obj.replace(hour=9, minute=0, second=0, microsecond=0), datetime_now__obj])
        
        tmp_list_for_return = func_sub_iteratior(tmp_minutes_to_goback, datetime_today_fix_start__obj, tmp_list_for_return)
        #print(f'tmp_list_for_return_ in FUNC_return_datetime_obj : {tmp_list_for_return}')
    
        #print(f'retrieved return list of datetimes : {tmp_list_for_return}')
        return tmp_list_for_return

def func_sub_iteratior(miutes_left, before_datetime_start__obj, return_list):
    TOTAL_DAY_MINUTE_NUMBER = 391 # stock day's window in minutes
    #return_list = copy.deepcopy(return_list_)
    
    tmp_before_datetime__obj = before_datetime_start__obj.replace(hour=9, minute=0, second=0, microsecond=0)
    tmp_target_datetime__obj = tmp_before_datetime__obj
    while (tmp_target_datetime__obj.weekday()) in [5, 6] or (tmp_before_datetime__obj.weekday() == tmp_target_datetime__obj.weekday()):
        # 월, 화, 수, 목, 금, 토, 일
        # 0,  1,  2,  3, 4, 5,  6
        tmp_target_datetime__obj = tmp_target_datetime__obj - datetime.timedelta(days=1)
    
    if miutes_left <= TOTAL_DAY_MINUTE_NUMBER:
        #print('func_sub_iteratior - 1')
        #print(f'minutes left in the last : {miutes_left}')
        return_list.append( [ tmp_target_datetime__obj.replace(hour=15, minute=30, second=0, microsecond=0) - datetime.timedelta(minutes=miutes_left) , tmp_target_datetime__obj.replace(hour=15, minute=30, second=0, microsecond=0) ] )
        #print(f'return_list in func_sub_iteratior : {return_list}')
        return return_list
    
    else:
        #print('func_sub_iteratior - 2')
        tmp_minutes_left = miutes_left - TOTAL_DAY_MINUTE_NUMBER
        return_list.append( [ tmp_target_datetime__obj.replace(hour=9, minute=0, second=0, microsecond=0) , tmp_target_datetime__obj.replace(hour=15, minute=30, second=0, microsecond=0) ] )
        
        return func_sub_iteratior(tmp_minutes_left, tmp_target_datetime__obj.replace(hour=9, minute=0, second=0, microsecond=0), return_list )

        
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

    tmp_return = FUNC_return_datetime_obj(datetime_tmp, hours_back)
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
    
