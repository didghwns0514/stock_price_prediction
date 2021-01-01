import logging
from logging.handlers import RotatingFileHandler
import os
from functools import wraps

"""
https://towardsdatascience.com/using-wrappers-to-log-in-python-ccffe4c46b54
https://dojang.io/mod/page/view.php?id=2454
http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%8D%BC%EC%8A%A4%ED%8A%B8%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%95%A8%EC%88%98-first-class-function/
"""
class _Log(object): # old styple
    # https://hashcode.co.kr/questions/487/object%EB%8A%94-%EC%99%9C-%EC%83%81%EC%86%8D%EB%B0%9B%EB%8A%94-%EA%B1%B4%EA%B0%80%EC%9A%94

    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR = os.getcwd().replace('/','\\') + '\\' + 'LOGGER_FOLDER'
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FILE_DIR = LOG_DIR + '\\' 

    ## root looger
    #ROOT_LOGGER = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s \n')
    #ROOT_LOGGER.setLevel(logging.INFO)

    ## sub logger
    DICT_SUB_LOGGER = {}
    

    @staticmethod
    def write_config(dest, lv='', module=''):
        """
        logging 하는 class의 기본 인자들 설정
        """

       ## get targ dir and file location
        targ_dir = _Log.LOG_FILE_DIR + dest
        if not os.path.isdir(targ_dir):
            os.mkdir(targ_dir)
        
        ## get key of the dict
        targ_file = targ_dir + '\\' + 'file' + '.log'
        if targ_dir not in _Log.DICT_SUB_LOGGER:
            ## add new logger
            _Log.DICT_SUB_LOGGER[targ_dir] = logging.getLogger(str(dest))

            ## set level of the logger
            _Log.setLevel(logger_obj=_Log.DICT_SUB_LOGGER[targ_dir],lv=lv)
            
            ## put file hander inside dict
            #rtn = logging.FileHandler(targ_file)
            rtn = RotatingFileHandler(maxBytes=10*1024*1024,
                                      filename=targ_file,
                                      backupCount=10)
            #rtn.terminator = '\n'
            rtn.setFormatter(_Log.formatter)

            ## connect the handler
            _Log.DICT_SUB_LOGGER[targ_dir].addHandler(rtn)
        else:
            _Log.setLevel(logger_obj=_Log.DICT_SUB_LOGGER[targ_dir],lv=lv)

        return targ_dir


    @staticmethod
    def write_normal(dest, lv='', module='', normal_memo=None):
        ## output
        out_str = ''
        out_str += f'running Function : {str(module)}'

        if not normal_memo:
            out_str += "\n" + f'normal memo : {str(normal_memo)}'

        ## leave log
        targ_dir = _Log.write_config(dest=dest, lv=lv, module=module)
        _Log.DICT_SUB_LOGGER[targ_dir].info(out_str)


    @staticmethod
    def write_error(dest, lv='', module='', error_msg=None):
        ## output
        out_str = ''
        out_str += f'error in Function : {str(module)}'

        ## leave log
        targ_dir = _Log.write_config(dest=dest, lv='ERROR', module=module)
        if not error_msg:
            pass
            
        else:
            out_str += "\n" + f'error messag : {str(error_msg)}'
        
        #_Log.DICT_SUB_LOGGER[targ_dir].error(out_str, exc_info=bool( True * error_msg))
        _Log.DICT_SUB_LOGGER[targ_dir].error(out_str)
        

    
    @staticmethod
    def write_excption(dest, lv='', module='', exception_msg=None, excpt_memo=None):

        ## output
        out_str = ''
        out_str += f'wrapped by Exception in Function : {module}'

        ## leave log
        targ_dir = _Log.write_config(dest=dest, lv='WARNING', module=module)

        if exception_msg:
            out_str += '\n' + f'exception message : {str(exception_msg)}'
        if excpt_memo:
            out_str += '\n' + f'exception memo : {str(excpt_memo)}'

        _Log.DICT_SUB_LOGGER[targ_dir].warning(out_str)

        

    @staticmethod
    def setLevel(logger_obj, lv='INFO'):


        if lv == 'DEBUG':
            logger_obj.setLevel(logging.DEBUG)
        elif lv == 'INFO':
            logger_obj.setLevel(logging.INFO)       
        elif lv == 'WARNING':
            logger_obj.setLevel(logging.WARNING)
        elif lv == 'ERROR':
            logger_obj.setLevel(logging.ERROR)
        elif lv == 'CRITICAL':
            logger_obj.setLevel(logging.CRITICAL)
        else: 
            logger_obj.setLevel(logging.INFO)       



def pushLog(dst_folder, lv='', module='', exception=False, exception_msg=None, memo=None):
    """
    :param dst_folder: top folder that will be used to save files 
    :param lv: level of logger, [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    :param module: function name, if "exception True", needs function name as input
    :param exception: to log under exception wrapping
    :param exception_msg: exception message to be saved
    :param memo: if "exception True", additional message / else message is saved under 'INFO' level
    :return: if "exception True", log is saved / else returns wrapper function -> used as decorator  
    """

    # print(f'in pushLog')
    # print(f'in pushLog parms : {dst_folder}')

    if not exception:

        def wrap1(function):
            # print(f'in wrap1')
            # print(f'in wrap1 function : {function}')

            @wraps(function)
            def wrap2(*args, **kwargs):
                try:
                    #print('path_1')
                    ret = function(*args, **kwargs)
                    _Log.write_normal(dest=dst_folder,lv=lv,module=function.__name__, normal_memo=memo)


                    return ret

                except Exception as e:
                    _Log.write_error(dest=dst_folder,lv='ERROR',module=function.__name__, error_msg=e)

            return wrap2
        return wrap1    

    else:
        #print('path_2')
        _Log.write_excption(dest=dst_folder,
                            lv='WARNING',
                            module=module,
                            exception_msg=exception_msg,
                            excpt_memo=memo)







if __name__ == '__main__':

    class Test:
        NAME = 'TEST'

        def __init__(self):
            self.a1_result = None
            

        @pushLog(dst_folder='classNN')
        def sub_test(self, a1):
            print(f'a1  :  {a1}')
            self.a1_result = a1

    #logger = LogWrap()

    @pushLog(dst_folder='func_noArg')
    def orig_func():
        pass

    #@pushLog(dst_folder='func_withArg')
    def ori_func_2(str1, str2):
        try:
            print(f'this is ori_func_2')
            print(f'str1 : {str1}')
            print(f'str2 : {str2}')
            print(f'{1/0}')
            
        except Exception as e:
            print('call push log for exception wrapping')
            pushLog(dst_folder='func_withArg',module='ori_func_2',exception=True, exception_msg=e)
            print('error happended!')

    for i in range(100):
        orig_func()
    ori_func_2('My name', 'is hojune')

    test = Test()
    test.sub_test('hihihi')

    print(test.a1_result)
    