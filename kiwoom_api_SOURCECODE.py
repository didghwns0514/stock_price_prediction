#-*-coding: utf-8-*-


"""
http://webschool.kr/page.php?bbs=news_stock&bbs_idx=21&pg=1
 2018-05-03 23:36:30
"""
import sys
import requests
import json
import webbrowser
import re
import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QGridLayout, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QPushButton, QLineEdit
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QEventLoop
from PyQt5.QAxContainer import *

from functools import partial
from collections import OrderedDict

# ui = uic.loadUiType("auto_pay.ui")[0]
# order_ui = uic.loadUiType("order_setup.ui")[0]
# keyword_ui = uic.loadUiType("keyword_setup.ui")[0]
import auto_pay
import keyword_setup
import order_setup

# class MyWindow(QMainWindow, ui):
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.setupUi(self)
        self.show()

        self.main_ui = auto_pay.Ui_MainWindow2()
        self.main_ui.setupUi(self)

        # print(ReturnCode.CAUSE[-100])
        # self.testLoop = QEventLoop()
        # self.testLoop.exec_()

        test = False
        #test = True

        # 매수목록 년월일
        '''
        d = datetime.date.today()
        #print(d)
        ymd_split = re.split("-", str(d))
        start_year = int(ymd_split[0]) - 3
        end_year = int(ymd_split[0]) + 1
        for i in range(start_year, end_year):
            self.main_ui.yearbox.addItem(str(i))

        for i in range(1, 13):
            self.main_ui.monthbox.addItem(str(i))

        for i in range(1, 32):
            self.main_ui.daybox.addItem(str(i))
        '''

        #self.testLoop = QEventLoop()
        #self.testLoop.exec_()

        if test == True:
            # 테스트 모드
            self.ki = kiwoom()
            self.ki.user_id = "dev84"
        else:
            # 실사모드
            self.ki = kiwoom()
            self.ki.commConnect()
            # 요기서 loop로 멈춤

            self.server = self.ki.get_login_info("GetServerGubun")

            if len(self.server) == 0 or self.server != "1":
                self.serverGubun = "실제운영서버"
            else:
                self.serverGubun = "모의투자서버"

            self.set_logbox(self.serverGubun + " 접속성공")

            # 계좌콤보박스
            account_array = re.split(";", self.ki.account_number)
            for account in account_array:
                if account:
                    self.main_ui.account_box.addItem(account)

        self.main_ui.btn_buylist.clicked.connect(self.btn_buylist_proc2)
        # 키움에서 매수 리스트 가져오기
        # self.btn_buylist_proc2()
        self.mybuylist = {} # 나의 매수리스트
        self.mybuylist_cnt = 0 # 나의 매수 row 카운트(종목카운트)
        self.mybuylist_total_money = 0 # 나의 종목 토탈 머니(총 매입금액)
        #self.mybuylist_row_cnt = 50 # row 카운터를 미리 알수가 없기 때문에 50개로 박아둠

        # 실시간 뉴스 버튼
        self.main_ui.notice_btn.clicked.connect(self.notice_btn_proc)

        # 주문설정 버튼
        self.main_ui.order_btn.clicked.connect(self.order_btn_proc)

        # 나의매수리스트 가져오기 타이머
        self.slist_timer = None
        self.slist_timer_interval = 1000
        self.slist_timer_cnt = 0

        # 공시뉴스 크롤링 타이머
        self.current_timer = None
        self.current_timer_interval = 3000
        self.current_timer_cnt = 0

        # 공시뉴스 데이터
        self.news_list = None

        # 공시뉴스 긁어오기 시작
        self.start_timer()

        # 뉴스 크롤링 및 매칭 서버
        self.api_url = "비밀"
        self.news_open = "NO"

        # 키워드설정 버튼
        self.main_ui.keyword_btn.clicked.connect(self.keyword_btn_proc)

        # 자동구매시작
        self.start_buy = "NO"
        # self.is_call = "OK"

        # 자동매수실행 버튼
        self.main_ui.autobuy_btn.clicked.connect(self.autobuy_start)

        # 종료버튼
        self.main_ui.exit_btn.clicked.connect(self.news_exit)

        # 전체히스토리
        self.main_ui.btn_all_history.clicked.connect(self.all_history)

        # 주식종목 업데이트 할일이 있을때만 호출하자
        # self.stock_update()

        #테스트
        #self.ki.call_stock_before("100", "200", "000660", "300")
        #self.ki.call_stock_before("매수요청", "opt10001", "035250", "21014") #시가총액 걸리는 예시
        #self.ki.call_stock_before("매수요청", "opt10001", "001680", "22612") #이미 구매한 종목 테스트(일주일 이내)

    def all_history(self):
        url = self.api_url + "?type=news_history&stock_id=" + self.ki.user_id
        webbrowser.open(url)

    # 키움 API를 통해 가져온 내역
    def btn_buylist_proc2(self):
        '''
        /********************************************************************/
        /// ########## Open API 함수를 이용한 전문처리 C++용 샘플코드 예제입니다.

        [ OPW00004 : 계좌평가현황요청 ]

        1. Open API 조회 함수 입력값을 설정합니다.
        계좌번호 = 전문 조회할 보유계좌번호
        SetInputValue("계좌번호"   ,  "입력값 1");

        비밀번호 = 사용안함(공백)
        SetInputValue("비밀번호"   ,  "입력값 2");

        상장폐지조회구분 = 0:전체, 1:상장폐지종목제외
        SetInputValue("상장폐지조회구분"   ,  "입력값 3");

        비밀번호입력매체구분 = 00
        SetInputValue("비밀번호입력매체구분" ,  "입력값 4");


        2. Open API 조회 함수를 호출해서 전문을 서버로 전송합니다.
        CommRqData( "RQName"   ,  "OPW00004"  ,  "0" ,  "화면번호");

        /********************************************************************/
        '''
        try:
            # 나의 매수리스트 초기화
            self.mybuylist = {}
            #self.main_ui.buy_table.setRowCount(self.mybuylist_row_cnt)

            self.ki.btn_buylist_proc2()
        except Exception as error:
            self.set_logbox("btn_buylist_proc2 Error = " + error)

    # 매수내역 리스트
    # DB에서 가져온 내용
    def btn_buylist_proc(self):
        #print("btn_buylist_proc 호출됨")
        y = self.main_ui.yearbox.currentText()
        m = self.main_ui.monthbox.currentText()
        d = self.main_ui.daybox.currentText()

        post_data = {"type": "get_buylist", "stock_id": self.ki.user_id, "y": y, "m": m, "d": d}
        # print("post_data = " + str(post_data))

        return_data = requests.post(self.api_url, data=post_data)
        if return_data.status_code == 200:
            return_data = return_data.text
            api_ok = None
            try:
                print("return_data = " + return_data)
                api_ok = True
            except Exception as error:
                print(str(error))
                self.set_logbox(error)
                api_ok = False

            #print("return_data = " + str(return_data))

            if api_ok == True:
                r_data = json.loads(return_data)
                # print(r_data['result'])
                if r_data['result'] == "OK":
                    cnt = 0
                    #print("list = " + str(r_data['list']))
                    self.main_ui.buy_table.setRowCount(int(r_data['cnt']))
                    for row in r_data['list']:
                        #print("row = " + str(row))
                        #print("buy_time = " + row['buy_time'])
                        self.main_ui.buy_table.setItem(cnt, 0, QTableWidgetItem(str(row['buy_time'])))
                        self.main_ui.buy_table.setItem(cnt, 1, QTableWidgetItem(str(row['buy_name'])))
                        self.main_ui.buy_table.setItem(cnt, 2, QTableWidgetItem(str(row['buy_cnt'])))
                        self.main_ui.buy_table.setItem(cnt, 3, QTableWidgetItem(str(row['buy_price'])))
                        self.main_ui.buy_table.setItem(cnt, 4, QTableWidgetItem(str(row['total_price'])))

                        #  데이터 갱신을 위해 포커스를 주자
                        self.main_ui.buy_table.setCurrentCell(cnt, 0)
                        self.main_ui.buy_table.setCurrentCell(cnt, 1)
                        self.main_ui.buy_table.setCurrentCell(cnt, 2)
                        self.main_ui.buy_table.setCurrentCell(cnt, 3)
                        self.main_ui.buy_table.setCurrentCell(cnt, 4)
                        cnt = cnt + 1
                elif r_data['result'] == "NO":
                    QMessageBox.information(self, "알림", "매수목록 가져오기에 실패했습니다.")
        else:
            QMessageBox.information(self, "알림", "매수목록 가져오기에 실패했습니다.(API 호출실패)")

    def stock_update(self):
        '''
        [GetCodeListByMarket() 함수]

        GetCodeListByMarket(
            BSTR sMarket    // 시장구분값
        )

        국내 주식 시장별 종목코드를 ';'로 구분해서 전달합니다. 만일 시장구분값이 NULL이면 전체 시장코드를 전달합니다.
        로그인 한 후에 사용할 수 있는 함수입니다.

        [시장구분값]
        0 : 장내
        10 : 코스닥
        3 : ELW
        8 : ETF
        50 : KONEX
        4 :  뮤추얼펀드
        5 : 신주인수권
        6 : 리츠
        9 : 하이얼펀드
        30 : K-OTC
        '''
        all_json = OrderedDict()
        stock_gubun = ["0", "10"]
        for gubun in stock_gubun:
            stock_list = self.ki.getStockList(gubun)
            stock_array = re.split(";", stock_list)
            for code in stock_array:
                # print(row)
                '''
                [GetMasterCodeName() 함수]

                GetMasterCodeName(
                BSTR strCode    // 종목코드
                )

                종목코드에 해당하는 종목명을 전달합니다.
                로그인 한 후에 사용할 수 있는 함수입니다.                        
                '''
                if code :
                    name = self.ki.getCodeName(code)
                    all_json[code] = name
                    #print(gubun + " : " + code + " = " + name)

        data = json.dumps(all_json)

        post_data = {"type": "stock_update", "stock_id": self.ki.user_id, "data": data}
        # print("post_data = " + str(post_data))

        return_data = requests.post(self.api_url, data=post_data)
        if return_data.status_code == 200:
            return_data = return_data.text
            api_ok = None
            try:
                print("return_data = " + return_data)
                api_ok = True
            except Exception as error:
                print(str(error))
                self.set_logbox(error)
                api_ok = False
    
            if api_ok == True:
                # print("return_data = " + str(return_data))
                r_data = json.loads(return_data)
                # print(r_data['result'])
                if r_data['result'] == "OK":
                    QMessageBox.information(self, "알림", "종목 업데이트에 성공했습니다.")
                elif r_data['result'] == "NO":
                    QMessageBox.information(self, "알림", "종목 업데이트에 실패했습니다.")
        else:
            QMessageBox.information(self, "알림", "종목 업데이트에 실패했습니다.(API 호출실패)")

    # 자동매수시작
    def autobuy_start(self):
        if self.start_buy == "OK":
            # QMessageBox.warning(self, "이미 눌렀음", "이미 자동매수 실행중입니다.")
            choice = QMessageBox.question(self, "확인", "자동매수를 정지하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                self.start_buy = "NO"
                self.main_ui.now_stats.setText("자동매수 대기중")
                self.main_ui.autobuy_btn.setText("자동매수실행")
            else:
                pass
        else:
            choice = QMessageBox.question(self, "매수비밀번호 넣었나열?", "우측하단의 매수비밀번호를 꼭 입력해야만 자동매수가 정상작동합니다.\n자동매수를 시작하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
            if choice == QMessageBox.Yes:
                self.start_buy = "OK"
                self.main_ui.now_stats.setText("자동매수 실행중")
                self.main_ui.autobuy_btn.setText("자동매수정지")

                # 매수리스트 불러오기
                self.btn_buylist_proc2()
            else:
                pass

    def keyword_btn_proc(self):
        # 키워드설정 팝업
        self.keyworddlg = KeywordSetupDialog()
        self.keyworddlg.show()

    def news_exit(self):
        # myWindow.close()
        self.close()

    def order_btn_proc(self):
        # 주문설정 팝업
        self.osdlg = OrderSetupDialog()
        self.osdlg.show()

    def notice_btn_proc(self):
        '''
        self.news_open = "OK"

        # 공시뉴스 팝업
        self.dlg = NoticeDialog()
        # self.dlg.exec_()
        self.dlg.show()
        '''
        url = self.api_url + "?type=news_list&stock_id=" + self.ki.user_id
        webbrowser.open(url)

    def OnCustomWinClosed(self):
        print("OnCustomWinClosed 호출됨")

    def start_timer(self):
        if self.current_timer:
            self.current_timer.stop()
            self.current_timer.deleteLater()

        self.current_timer = QTimer()
        self.current_timer.timeout.connect(self.start_news)
        self.current_timer.setSingleShot(True)
        self.current_timer.start(self.current_timer_interval)

    def start_news(self):
        #타이머 안에서는 에러가 안보이게 된다 ㅠㅠ
        '''
        try:
            r = {}
            r['test'] = "ttt1"
            print(str(r['test2']))
        except Exception as error:
            self.set_logbox("timer error = " + str(error))
        '''

        self.current_timer_cnt = self.current_timer_cnt + 1
        print("뉴스 타이머 (" + str(self.current_timer_cnt) + ")")

        try:
            self.get_set_news()
        except Exception as error:
            msg = "\n=============================\n타이머1 에러가 발생했습니다!\n에러내용 => " + str(error) + "\n재연결하겠습니다.\n=============================\n"
            self.set_logbox(msg)
            self.start_timer()
            #QMessageBox.information(self, "타이머1 에러!", msg)

    def start_timer_slist(self):
        if self.slist_timer:
            self.slist_timer.stop()
            self.slist_timer.deleteLater()

        self.slist_timer = QTimer()
        self.slist_timer.timeout.connect(self.start_news_slist)
        self.slist_timer.setSingleShot(True)
        self.slist_timer.start(self.slist_timer_interval)

    def start_news_slist(self):
        if self.start_buy == "OK":
            self.slist_timer_cnt = self.slist_timer_cnt + 1
            if self.slist_timer_cnt % 10 == 0:
                print("주문리스트 타이머 (" + str(self.slist_timer_cnt) + ")")
                try:
                    self.btn_buylist_proc2()
                except Exception as error:
                    #msg = "\n=============================\n타이머2 에러가 발생했습니다!\n개발자에게 바로 알려주세요!\n에러내용 => " + str(error) + "\n=============================\n"
                    msg = "\n=============================\n타이머2 에러가 발생했습니다!\n에러내용 => " + str(error) + "\n재연결하겠습니다.\n=============================\n"
                    self.set_logbox(msg)
                    self.btn_buylist_proc2()
                    #QMessageBox.information(self, "타이머2 에러!", msg)

    # API 서버에서 뉴스크롤링 실행시키고, 바로 가져오기
    def get_set_news(self):
        # 히안하게 여기선 에러코드가 붉은색으로 안나옴... 원인을 알아보장.
        # 디버그
        '''
        r = {}
        r['test'] = "ttt1"
        print(str(r['test2']))
        '''

        # print("get_set_news 호출, stock_id = " + self.ki.user_id)
        post_data = {
            "type": "getset2",
            "stock_id": self.ki.user_id,
            "start_buy": self.start_buy,
            "mybuylist_cnt": self.mybuylist_cnt,
            "mybuylist_total_money": self.mybuylist_total_money,

            "news_open": self.news_open
        }
        #print(str(post_data))

        #print("post_data = " + str(post_data))
        '''
        api_ok = None
        try:
            r = requests.post(self.api_url, data=post_data)
            result_json = r.text  # {"result":"OK"}
            api_ok = True
            print("r.status_code = " + str(r.status_code))
            print("r.raise_for_status = " + str(r.raise_for_status()))
            #r.status_code = 200
            #r.raise_for_status = None                
        except Exception as error:
            print("\n========== get_set_news error ==========\n" + str(error) + "\n====================\n")
            self.set_logbox(error)
            api_ok = False
        #print("r = " + str(r))
        print("api_ok = " + str(api_ok))
        '''
        is_ok = False
        try:
            r = requests.post(self.api_url, data=post_data)
            print("r = " + str(r))
            if r.status_code == 200:
                is_ok = True
        except Exception as error:
            myWindow.set_logbox("news_loading Error = " + error)

        if is_ok == True:
            result_json = r.text  # {"result":"OK"}
            data = json.loads(result_json)
            # print("get_set_news = " + dict['result'])

            # print("log_text = " + data['log_text'])
            # print("buy_code = " + str(data['buy_code']))
            # print("buy_name = " + str(data['buy_name']))
            # print("ㅇㅇㅇ")

            # result_json = {"log_text":null,"buy_option":{"idx":"37","stock_id":"ALL_MEMBER","order_money":"500000","order_money_cnt":"1","order_limit_cnt":"5","order_stock_limit_money":"1500000","order_stop_time_09":"N","order_stop_time_10":"N","order_stop_time_11":"N","order_stop_time_12":"N","order_stop_time_13":"N","order_stop_time_14":"N","order_stop_time_15":"N"},"buy_code":[],"buy_name":[],"buy_newsidx":[],"no_buy_text":"","buy_able_money":1500000,"time_over":"NO","result":"OK"}

            # 장 종료됨
            if data['time_over'] == "OK":
                # self.is_call = "NO"
                self.main_ui.now_stats.setText("주식장 종료됨")
            else:
                self.start_timer_slist()
                #print("1")
                # self.is_call = "OK"
                if self.start_buy == "OK":
                    self.main_ui.now_stats.setText("자동매수 실행중")

                    if data['result'] != "OK":
                        QMessageBox.information(self, "뉴스크롤링 실패ㅠㅠ", "뉴스크롤링에 실패했습니다.\n개발자에게 문의 부탁드려요~")

                    # 뉴스 갱신
                    # print("data.list = " + data['list'])
                    # self.news_list = data['list']
                    #print("3")
                    if self.start_buy == "OK":
                        # 여기에 매수로직 넣으면 됨
                        # print(data['log_text'])
                        # print("log_text 진입전")

                        # 왜 구매 못하는지에 대한 텍스트
                        if data['no_buy_text']:
                            self.set_logbox(data['no_buy_text'])

                        if data['log_text']:
                            # print(str(data))
                            self.set_logbox(data['log_text'])

                            self.ki.order_money = int(data['buy_option']['order_money'])
                            self.ki.order_money_cnt = int(data['buy_option']['order_money_cnt'])
                            self.ki.order_limit_cnt = int(data['buy_option']['order_limit_cnt'])
                            self.ki.order_stock_limit_money = int(data['buy_option']['order_stock_limit_money'])

                            self.ki.buy_able_money = int(data['buy_able_money'])

                            tmp_cnt = 0
                            for code in data['buy_code']:
                                # print("buy_code = " + code)
                                # self.ki.call_stock("매수요청", "opt10001", code, data['buy_newsidx'][tmp_cnt])
                                self.ki.call_stock_before("매수요청", "opt10001", code, data['buy_newsidx'][tmp_cnt])
                                tmp_cnt = tmp_cnt + 1

                        #print("log_text 아웃")
                else:
                    self.main_ui.now_stats.setText("자동매수 대기중")
                    #print("2")
        else:
            self.set_logbox("get_set_news API 호출 실패(status != 200)")

        self.start_timer()

    def set_logbox(self, msg):
        content = self.main_ui.logbox.toPlainText()
        self.main_ui.logbox.setText(str(msg) + "\n" + content)

class NoticeDialog(QDialog):
    def __init__(self):
        super().__init__()

        print("NoticeDialog 오픈됨.")

        self.setGeometry(700, 200, 1070, 650)
        self.setWindowTitle("실시간 공시뉴스")

        self.notice_table = QTableWidget(self)
        self.notice_table.setRowCount(20)
        self.notice_table.setColumnCount(4)
        self.notice_table.setColumnWidth(0, 70)
        self.notice_table.setColumnWidth(1, 720)
        self.notice_table.setColumnWidth(2, 140)
        self.notice_table.setColumnWidth(3, 90)
        self.notice_table.setHorizontalHeaderLabels(['뉴스index', '제목', '시간', '상세보기'])

        self.layout = QGridLayout()
        self.layout.addWidget(self.notice_table, 0, 0)

        self.setLayout(self.layout)

        self.current_news_timer = None
        self.current_news_start = "YES"

        # 공시뉴스 마지막 제목
        self.last_title = None

        self.news_timer()

        # self.close()

    def closeEvent(self, event):
        print("closeEvent 호출됨")

        myWindow.news_open = "NO"

        event.ignore()
        self.hide()
        self.current_news_start = "NO"

    def news_loading(self):
        # print("news_list = " + str(myWindow.news_list))
        try:
            if myWindow.news_list != None:
                cnt = 0
                for row in myWindow.news_list:
                    if cnt == 0:
                        if self.last_title == row['title']:
                            break
                        else:
                            self.last_title = row['title']

                    # title = row['title'][0:30]
                    title = row['title']

                    try:
                        is_buy = row['BUY_OK']
                        if is_buy == "OK":
                            #print("매수한 뉴스 : " + str(row['idx']))
                            title = "[ 매수완료 ] " + title
                    except Exception as error:
                        #myWindow.set_logbox(error)
                        print(str(error))

                    self.notice_table.setItem(cnt, 0, QTableWidgetItem(row['idx']))
                    self.notice_table.setItem(cnt, 1, QTableWidgetItem(title))
                    self.notice_table.setItem(cnt, 2, QTableWidgetItem(row['time']))
                    detailbtn = QPushButton("보기")
                    detailbtn.clicked.connect(partial(self.detailbtn_clicked, row['idx']))
                    self.notice_table.setCellWidget(cnt, 3, detailbtn)

                    #  데이터 갱신을 위해 포커스를 주자
                    self.notice_table.setCurrentCell(cnt, 0)
                    self.notice_table.setCurrentCell(cnt, 1)
                    self.notice_table.setCurrentCell(cnt, 2)
                    self.notice_table.setCurrentCell(cnt, 3)
                    cnt = cnt + 1
        except Exception as error:
            myWindow.set_logbox("news_loading Error = " + error)

        if self.current_news_start == "YES":
            self.news_timer()

    # 기사 상세보기 띄우기
    def detailbtn_clicked(self, idx):
        url = myWindow.api_url + "?type=news_detail&idx=" + idx + "&stock_id=" + myWindow.ki.user_id
        webbrowser.open(url)

    def news_timer(self):
        # print("news_timer 호출, self.current_news_timer = " + str(self.current_news_timer))
        if self.current_news_timer:
            self.current_news_timer.stop()
            self.current_news_timer.deleteLater()

        self.current_news_timer = QTimer()
        self.current_news_timer.timeout.connect(self.news_loading)
        self.current_news_timer.setSingleShot(True)
        self.current_news_timer.start(1000)

#class OrderSetupDialog(QDialog, order_ui):
class OrderSetupDialog(QDialog):
    def __init__(self):
        super().__init__()

        # print("OrderSetupDialog 오픈됨." + str(order_ui))
        try:
            # self.setupUi(self)
            self.order_ui = order_setup.Ui_Form()
            self.order_ui.setupUi(self)
        except Exception as error:
            myWindow.set_logbox("OrderSetupDialog Error = " + error)

        # self.order_ui.btn_order_add.clicked.connect(self.order_table_add)
        self.order_ui.btn_order_save.clicked.connect(self.order_save_proc)
        self.vars = {}

        post_data = {"type": "order_get", "stock_id": myWindow.ki.user_id}
        # print("post_data = " + str(post_data))

        return_data = requests.post(myWindow.api_url, data=post_data)
        return_data = return_data.text

        api_ok = None
        try:
            print("return_data = " + return_data)
            api_ok = True
        except Exception as error:
            print(str(error))
            self.set_logbox(error)
            api_ok = False

        #print("return_data = " + str(return_data))

        if api_ok == True:
            r_data = json.loads(return_data)
            order_list_number = 0
            if r_data['result'] == "OK":
                self.order_ui.order_money.setText(r_data['opt']['order_money'])
                self.order_ui.order_money_cnt.setText(r_data['opt']['order_money_cnt'])
                self.order_ui.order_limit_cnt.setText(r_data['opt']['order_limit_cnt'])
                self.order_ui.order_stock_limit_money.setText(r_data['opt']['order_stock_limit_money'])

                if r_data['opt']['order_stop_time_09'] == "Y":
                    self.order_ui.order_stop_time_09.setChecked(True)
                else:
                    self.order_ui.order_stop_time_09.setChecked(False)

                if r_data['opt']['order_stop_time_10'] == "Y":
                    self.order_ui.order_stop_time_10.setChecked(True)
                else:
                    self.order_ui.order_stop_time_10.setChecked(False)

                if r_data['opt']['order_stop_time_11'] == "Y":
                    self.order_ui.order_stop_time_11.setChecked(True)
                else:
                    self.order_ui.order_stop_time_11.setChecked(False)

                if r_data['opt']['order_stop_time_12'] == "Y":
                    self.order_ui.order_stop_time_12.setChecked(True)
                else:
                    self.order_ui.order_stop_time_12.setChecked(False)

                if r_data['opt']['order_stop_time_13'] == "Y":
                    self.order_ui.order_stop_time_13.setChecked(True)
                else:
                    self.order_ui.order_stop_time_13.setChecked(False)

                if r_data['opt']['order_stop_time_14'] == "Y":
                    self.order_ui.order_stop_time_14.setChecked(True)
                else:
                    self.order_ui.order_stop_time_14.setChecked(False)

                if r_data['opt']['order_stop_time_15'] == "Y":
                    self.order_ui.order_stop_time_15.setChecked(True)
                else:
                    self.order_ui.order_stop_time_15.setChecked(False)

                '''
                for row in r_data['data']:
                    # print(row['name'])
                    self.btn_order_add_proc(order_list_number)
                    self.vars['edit_name_%d' % order_list_number].setText(row['name'])
                    self.vars['edit_code_%d' % order_list_number].setText(row['code'])
                    order_list_number = order_list_number + 1
                '''
            else:
                QMessageBox.information(self, "알림", "주문설정 가져오기에 실패했습니다.")

    def order_save_proc(self):
        '''
        cnt = self.order_ui.order_table.rowCount()
        all_json = OrderedDict()
        all_json_cnt = 0
        for i in range(cnt):
            row_json = OrderedDict()
            row_json['name'] = self.vars['edit_name_%d' % i].text()
            row_json['code'] = self.vars['edit_code_%d' % i].text()
            all_json[str(all_json_cnt)] = row_json
            all_json_cnt = all_json_cnt + 1

        # print("all_json = " + str(all_json))
        data = json.dumps(all_json)
        '''
        #print("order_stop_time_09 = " + str(self.order_ui.order_stop_time_09.isChecked()))
        #self.testLoop = QEventLoop()
        #self.testLoop.exec_()

        post_data = {
            "type": "order_save",
            "stock_id": myWindow.ki.user_id,
            #"data": data,

            "order_money": self.order_ui.order_money.text(),
            "order_money_cnt": self.order_ui.order_money_cnt.text(),
            "order_limit_cnt": self.order_ui.order_limit_cnt.text(),
            "order_stock_limit_money": self.order_ui.order_stock_limit_money.text()
        }
        if self.order_ui.order_stop_time_09.isChecked():
            post_data['order_stop_time_09'] = "Y"
        else:
            post_data['order_stop_time_09'] = "N"

        if self.order_ui.order_stop_time_10.isChecked():
            post_data['order_stop_time_10'] = "Y"
        else:
            post_data['order_stop_time_10'] = "N"

        if self.order_ui.order_stop_time_11.isChecked():
            post_data['order_stop_time_11'] = "Y"
        else:
            post_data['order_stop_time_11'] = "N"

        if self.order_ui.order_stop_time_12.isChecked():
            post_data['order_stop_time_12'] = "Y"
        else:
            post_data['order_stop_time_12'] = "N"

        if self.order_ui.order_stop_time_13.isChecked():
            post_data['order_stop_time_13'] = "Y"
        else:
            post_data['order_stop_time_13'] = "N"

        if self.order_ui.order_stop_time_14.isChecked():
            post_data['order_stop_time_14'] = "Y"
        else:
            post_data['order_stop_time_14'] = "N"

        if self.order_ui.order_stop_time_15.isChecked():
            post_data['order_stop_time_15'] = "Y"
        else:
            post_data['order_stop_time_15'] = "N"

        #print("post_data = " + str(post_data))
        #self.testLoop = QEventLoop()
        #self.testLoop.exec_()

        return_data = requests.post(myWindow.api_url, data=post_data)
        return_data = return_data.text

        api_ok = None
        try:
            print("return_data = " + return_data)
            api_ok = True
        except Exception as error:
            print(str(error))
            self.set_logbox(error)
            api_ok = False

        if api_ok == True:
            #print("return_data = " + str(return_data))
            r_data = json.loads(return_data)
            #print(r_data['result'])
            if r_data['result'] == "OK":
                QMessageBox.information(self, "알림", "저장성공")
            elif r_data['result'] == "NO":
                QMessageBox.information(self, "알림", r_data['msg'])

    def order_table_add(self):
        cnt = self.order_ui.order_table.rowCount()
        self.btn_order_add_proc(cnt)

    def btn_order_add_proc(self, rowCnt):
        # print("btn_order_add_proc 호출됨")
        self.order_ui.order_table.insertRow(rowCnt)

        self.vars['edit_name_%d' % rowCnt] = QLineEdit()
        self.order_ui.order_table.setCellWidget(rowCnt, 0, self.vars['edit_name_%d' % rowCnt])

        self.vars['edit_code_%d' % rowCnt] = QLineEdit()
        self.order_ui.order_table.setCellWidget(rowCnt, 1, self.vars['edit_code_%d' % rowCnt])

        self.vars['btn_remove_%d' % rowCnt] = QPushButton("삭제" + str(rowCnt))
        self.vars['btn_remove_%d' % rowCnt].clicked.connect(partial(self.btn_remove_clicked, rowCnt))
        self.order_ui.order_table.setCellWidget(rowCnt, 2, self.vars['btn_remove_%d' % rowCnt])

    def btn_remove_clicked(self, index):
        # print("btn_remove_clicked 호출, index = " + str(index))
        rowCnt = self.order_ui.order_table.rowCount()
        for i in range(rowCnt):
            if index > i:
                continue
            elif index == i:
                self.order_ui.order_table.removeRow(index)
            else:
                edit_name_value = self.vars['edit_name_%d' % i].text()
                edit_code_value = self.vars['edit_code_%d' % i].text()

                upper_number = i - 1
                self.btn_order_add_proc(upper_number)
                self.vars['edit_name_%d' % upper_number].setText(edit_name_value)
                self.vars['edit_code_%d' % upper_number].setText(edit_code_value)

                self.order_ui.order_table.removeRow(i)

#class KeywordSetupDialog(QDialog, keyword_ui):
class KeywordSetupDialog(QDialog):
    def __init__(self):
        super().__init__()

        # print("KeywordSetupDialog 오픈됨." + str(order_ui))
        try:
            #self.setupUi(self)
            self.keyword_ui = keyword_setup.Ui_keyword_window()
            self.keyword_ui.setupUi(self)
        except Exception as error:
            print("KeywordSetupDialog Error = " + error)

        self.keyword_ui.btn_keyword_add.clicked.connect(self.keyword_table_add)
        self.keyword_ui.btn_keyword_add_limit.clicked.connect(self.keyword_table_add_limit)

        self.keyword_ui.btn_keyword_save.clicked.connect(self.keyword_save_proc)
        self.keyword_ui.btn_keyword_close.clicked.connect(self.keyword_close_proc)

        self.vars = {}
        self.vars_limit = {}

        post_data = {"type": "keyword2_get", "stock_id": myWindow.ki.user_id}
        # print("post_data = " + str(post_data))

        return_data = requests.post(myWindow.api_url, data=post_data)
        return_data = return_data.text

        api_ok = None
        try:
            print("return_data = " + return_data)
            api_ok = True
        except Exception as error:
            print(str(error))
            self.set_logbox(error)
            api_ok = False

        if api_ok == True:
            # print("return_data = " + str(return_data))
            r_data = json.loads(return_data)
            if r_data['result'] == "OK":
                order_list_number = 0
                for row in r_data['data']:
                    # print(row['name'])
                    self.btn_keyword_add_proc(order_list_number)
                    self.vars['keyword1_%d' % order_list_number].setText(row['keyword1'])
                    self.vars['keyword2_%d' % order_list_number].setText(row['keyword2'])
                    self.vars['keyword3_%d' % order_list_number].setText(row['keyword3'])
                    self.vars['keyword4_%d' % order_list_number].setText(row['keyword4'])
                    self.vars['keyword5_%d' % order_list_number].setText(row['keyword5'])
                    self.vars['keyword6_%d' % order_list_number].setText(row['keyword6'])
                    self.vars['keyword7_%d' % order_list_number].setText(row['keyword7'])
                    self.vars['keyword8_%d' % order_list_number].setText(row['keyword8'])
                    self.vars['keyword9_%d' % order_list_number].setText(row['keyword9'])
                    self.vars['keyword10_%d' % order_list_number].setText(row['keyword10'])
                    order_list_number = order_list_number + 1

                order_list_number = 0
                for row in r_data['data_limit']:
                    # print(row['name'])
                    self.btn_keyword_add_proc_limit(order_list_number)
                    self.vars_limit['keyword1_%d' % order_list_number].setText(row['keyword1'])
                    self.vars_limit['keyword2_%d' % order_list_number].setText(row['keyword2'])
                    self.vars_limit['keyword3_%d' % order_list_number].setText(row['keyword3'])
                    self.vars_limit['keyword4_%d' % order_list_number].setText(row['keyword4'])
                    self.vars_limit['keyword5_%d' % order_list_number].setText(row['keyword5'])
                    self.vars_limit['keyword6_%d' % order_list_number].setText(row['keyword6'])
                    self.vars_limit['keyword7_%d' % order_list_number].setText(row['keyword7'])
                    self.vars_limit['keyword8_%d' % order_list_number].setText(row['keyword8'])
                    self.vars_limit['keyword9_%d' % order_list_number].setText(row['keyword9'])
                    self.vars_limit['keyword10_%d' % order_list_number].setText(row['keyword10'])
                    order_list_number = order_list_number + 1
            else:
                QMessageBox.information(self, "알림", "키워드설정 가져오기에 실패했습니다.")

    def keyword_close_proc(self):
        self.close()

    def keyword_save_proc(self):
        cnt = self.keyword_ui.keyword_table.rowCount()
        all_json = OrderedDict()
        all_json_cnt = 0
        for i in range(cnt):
            row_json = OrderedDict()
            row_json['keyword1'] = self.vars['keyword1_%d' % i].text()
            row_json['keyword2'] = self.vars['keyword2_%d' % i].text()
            row_json['keyword3'] = self.vars['keyword3_%d' % i].text()
            row_json['keyword4'] = self.vars['keyword4_%d' % i].text()
            row_json['keyword5'] = self.vars['keyword5_%d' % i].text()
            row_json['keyword6'] = self.vars['keyword6_%d' % i].text()
            row_json['keyword7'] = self.vars['keyword7_%d' % i].text()
            row_json['keyword8'] = self.vars['keyword8_%d' % i].text()
            row_json['keyword9'] = self.vars['keyword9_%d' % i].text()
            row_json['keyword10'] = self.vars['keyword10_%d' % i].text()
            all_json[str(all_json_cnt)] = row_json
            all_json_cnt = all_json_cnt + 1

        # print("all_json = " + str(all_json))
        data = json.dumps(all_json)

        cnt = self.keyword_ui.keyword_table_limit.rowCount()
        all_json = OrderedDict()
        all_json_cnt = 0
        for i in range(cnt):
            row_json = OrderedDict()
            row_json['keyword1'] = self.vars_limit['keyword1_%d' % i].text()
            row_json['keyword2'] = self.vars_limit['keyword2_%d' % i].text()
            row_json['keyword3'] = self.vars_limit['keyword3_%d' % i].text()
            row_json['keyword4'] = self.vars_limit['keyword4_%d' % i].text()
            row_json['keyword5'] = self.vars_limit['keyword5_%d' % i].text()
            row_json['keyword6'] = self.vars_limit['keyword6_%d' % i].text()
            row_json['keyword7'] = self.vars_limit['keyword7_%d' % i].text()
            row_json['keyword8'] = self.vars_limit['keyword8_%d' % i].text()
            row_json['keyword9'] = self.vars_limit['keyword9_%d' % i].text()
            row_json['keyword10'] = self.vars_limit['keyword10_%d' % i].text()
            all_json[str(all_json_cnt)] = row_json
            all_json_cnt = all_json_cnt + 1

        # print("all_json = " + str(all_json))
        data_limit = json.dumps(all_json)

        post_data = {"type": "keyword2_save", "stock_id": myWindow.ki.user_id, "data": data, "data_limit": data_limit}
        # print("post_data = " + str(post_data))

        return_data = requests.post(myWindow.api_url, data=post_data)
        return_data = return_data.text

        api_ok = None
        try:
            print("return_data = " + return_data)
            api_ok = True
        except Exception as error:
            print(str(error))
            self.set_logbox(error)
            api_ok = False

        if api_ok == True:
            #print("return_data = " + str(return_data))
            r_data = json.loads(return_data)
            #print(r_data['result'])
            if r_data['result'] == "OK":
                QMessageBox.information(self, "알림", "저장성공")
            elif r_data['result'] == "NO":
                QMessageBox.information(self, "알림", r_data['msg'])

    def keyword_table_add(self):
        cnt = self.keyword_ui.keyword_table.rowCount()
        #print("keyword_table_add 호출됨, cnt = " + str(cnt))
        self.btn_keyword_add_proc(cnt)

    def keyword_table_add_limit(self):
        cnt = self.keyword_ui.keyword_table_limit.rowCount()
        #print("keyword_table_add 호출됨, cnt = " + str(cnt))
        self.btn_keyword_add_proc_limit(cnt)

    def btn_keyword_add_proc(self, rowCnt):
        # print("btn_keyword_add_proc 호출됨")
        self.keyword_ui.keyword_table.insertRow(rowCnt)

        self.vars['keyword1_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 0, self.vars['keyword1_%d' % rowCnt])
        self.vars['keyword2_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 1, self.vars['keyword2_%d' % rowCnt])
        self.vars['keyword3_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 2, self.vars['keyword3_%d' % rowCnt])
        self.vars['keyword4_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 3, self.vars['keyword4_%d' % rowCnt])
        self.vars['keyword5_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 4, self.vars['keyword5_%d' % rowCnt])
        self.vars['keyword6_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 5, self.vars['keyword6_%d' % rowCnt])
        self.vars['keyword7_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 6, self.vars['keyword7_%d' % rowCnt])
        self.vars['keyword8_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 7, self.vars['keyword8_%d' % rowCnt])
        self.vars['keyword9_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 8, self.vars['keyword9_%d' % rowCnt])
        self.vars['keyword10_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 9, self.vars['keyword10_%d' % rowCnt])

        self.vars['btn_remove_%d' % rowCnt] = QPushButton("삭제" + str(rowCnt))
        self.vars['btn_remove_%d' % rowCnt].clicked.connect(partial(self.btn_remove_clicked, rowCnt))
        self.keyword_ui.keyword_table.setCellWidget(rowCnt, 10, self.vars['btn_remove_%d' % rowCnt])

    def btn_keyword_add_proc_limit(self, rowCnt):
        # print("btn_keyword_add_proc 호출됨")
        self.keyword_ui.keyword_table_limit.insertRow(rowCnt)
        self.vars_limit['keyword1_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 0, self.vars_limit['keyword1_%d' % rowCnt])
        self.vars_limit['keyword2_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 1, self.vars_limit['keyword2_%d' % rowCnt])
        self.vars_limit['keyword3_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 2, self.vars_limit['keyword3_%d' % rowCnt])
        self.vars_limit['keyword4_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 3, self.vars_limit['keyword4_%d' % rowCnt])
        self.vars_limit['keyword5_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 4, self.vars_limit['keyword5_%d' % rowCnt])
        self.vars_limit['keyword6_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 5, self.vars_limit['keyword6_%d' % rowCnt])
        self.vars_limit['keyword7_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 6, self.vars_limit['keyword7_%d' % rowCnt])
        self.vars_limit['keyword8_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 7, self.vars_limit['keyword8_%d' % rowCnt])
        self.vars_limit['keyword9_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 8, self.vars_limit['keyword9_%d' % rowCnt])
        self.vars_limit['keyword10_%d' % rowCnt] = QLineEdit()
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 9, self.vars_limit['keyword10_%d' % rowCnt])

        self.vars_limit['btn_remove_%d' % rowCnt] = QPushButton("삭제" + str(rowCnt))
        self.vars_limit['btn_remove_%d' % rowCnt].clicked.connect(partial(self.btn_remove_clicked_limit, rowCnt))
        self.keyword_ui.keyword_table_limit.setCellWidget(rowCnt, 10, self.vars_limit['btn_remove_%d' % rowCnt])

    def btn_remove_clicked(self, index):
        # print("btn_remove_clicked 호출, index = " + str(index))
        rowCnt = self.keyword_ui.keyword_table.rowCount()
        for i in range(rowCnt):
            if index > i:
                continue
            elif index == i:
                self.keyword_ui.keyword_table.removeRow(index)
            else:
                keyword1_value = self.vars['keyword1_%d' % i].text()
                keyword2_value = self.vars['keyword2_%d' % i].text()
                keyword3_value = self.vars['keyword3_%d' % i].text()
                keyword4_value = self.vars['keyword4_%d' % i].text()
                keyword5_value = self.vars['keyword5_%d' % i].text()
                keyword6_value = self.vars['keyword6_%d' % i].text()
                keyword7_value = self.vars['keyword7_%d' % i].text()
                keyword8_value = self.vars['keyword8_%d' % i].text()
                keyword9_value = self.vars['keyword9_%d' % i].text()
                keyword10_value = self.vars['keyword10_%d' % i].text()

                upper_number = i - 1
                self.btn_keyword_add_proc(upper_number)
                self.vars['keyword1_%d' % upper_number].setText(keyword1_value)
                self.vars['keyword2_%d' % upper_number].setText(keyword2_value)
                self.vars['keyword3_%d' % upper_number].setText(keyword3_value)
                self.vars['keyword4_%d' % upper_number].setText(keyword4_value)
                self.vars['keyword5_%d' % upper_number].setText(keyword5_value)
                self.vars['keyword6_%d' % upper_number].setText(keyword6_value)
                self.vars['keyword7_%d' % upper_number].setText(keyword7_value)
                self.vars['keyword8_%d' % upper_number].setText(keyword8_value)
                self.vars['keyword9_%d' % upper_number].setText(keyword9_value)
                self.vars['keyword10_%d' % upper_number].setText(keyword10_value)

                self.keyword_ui.keyword_table.removeRow(i)

    def btn_remove_clicked_limit(self, index):
        # print("btn_remove_clicked 호출, index = " + str(index))
        rowCnt = self.keyword_ui.keyword_table_limit.rowCount()
        for i in range(rowCnt):
            if index > i:
                continue
            elif index == i:
                self.keyword_ui.keyword_table_limit.removeRow(index)
            else:
                keyword1_value = self.vars_limit['keyword1_%d' % i].text()
                keyword2_value = self.vars_limit['keyword2_%d' % i].text()
                keyword3_value = self.vars_limit['keyword3_%d' % i].text()
                keyword4_value = self.vars_limit['keyword4_%d' % i].text()
                keyword5_value = self.vars_limit['keyword5_%d' % i].text()
                keyword6_value = self.vars_limit['keyword6_%d' % i].text()
                keyword7_value = self.vars_limit['keyword7_%d' % i].text()
                keyword8_value = self.vars_limit['keyword8_%d' % i].text()
                keyword9_value = self.vars_limit['keyword9_%d' % i].text()
                keyword10_value = self.vars_limit['keyword10_%d' % i].text()

                upper_number = i - 1
                self.btn_keyword_add_proc_limit(upper_number)
                self.vars_limit['keyword1_%d' % upper_number].setText(keyword1_value)
                self.vars_limit['keyword2_%d' % upper_number].setText(keyword2_value)
                self.vars_limit['keyword3_%d' % upper_number].setText(keyword3_value)
                self.vars_limit['keyword4_%d' % upper_number].setText(keyword4_value)
                self.vars_limit['keyword5_%d' % upper_number].setText(keyword5_value)
                self.vars_limit['keyword6_%d' % upper_number].setText(keyword6_value)
                self.vars_limit['keyword7_%d' % upper_number].setText(keyword7_value)
                self.vars_limit['keyword8_%d' % upper_number].setText(keyword8_value)
                self.vars_limit['keyword9_%d' % upper_number].setText(keyword9_value)
                self.vars_limit['keyword10_%d' % upper_number].setText(keyword10_value)

                self.keyword_ui.keyword_table_limit.removeRow(i)


class kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()

        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        # 키움 이벤트 등록
        self.OnEventConnect.connect(self.event_connect)
        self.OnReceiveTrData.connect(self.receiveTrData)
        self.OnReceiveChejanData.connect(self.receiveChejanData)
        self.OnReceiveMsg.connect(self.receiveMsg)

        # self.OnReceiveRealData.connect(self.receiveRealData) # 말그대로 리얼 데이터임
        # self.OnReceiveConditionVer.connect(self.receiveConditionVer)
        # self.OnReceiveTrCondition.connect(self.receiveTrCondition)
        # self.OnReceiveRealCondition.connect(self.receiveRealCondition)

        self.server = None
        self.user_id = None

        # 구매임시변수
        # self.buy_stock_price = None #구매종목단가
        # self.now_code = None # 구매종목코드
        self.order_money = None  # 단일매수 주문금액
        self.order_money_cnt = None  # 단일매수가능 주문수량(최소)
        self.order_limit_cnt = None  # 매수종목 수량제한
        self.order_stock_limit_money = None  # 매수종목 총액제한
        self.buy_able_money = None # 구매가능한 머니

        # Loop 변수
        # 비동기 방식으로 동작되는 이벤트를 동기화(순서대로 동작) 시킬 때, 즉 루프가 종료되기 전까지 다음 프로세스를 진행하지 않는다.
        self.loginLoop = None


        self.buy_temp = {}
        self.upper_percent = 6

    def commConnect(self):
        """
        로그인을 시도합니다.

        수동 로그인일 경우, 로그인창을 출력해서 로그인을 시도.
        자동 로그인일 경우, 로그인창 출력없이 로그인 시도.
        """
        self.dynamicCall("CommConnect()")
        self.loginLoop = QEventLoop()
        self.loginLoop.exec_()

    def event_connect(self, returnCode):
        print("event_connect 호출, returnCode = " + str(returnCode))
        """
        통신 연결 상태 변경시 이벤트

        returnCode가 0이면 로그인 성공
        그 외에는 ReturnCode 클래스 참조.

        :param returnCode: int
        """
        msg = None
        try:
            if returnCode == ReturnCode.OP_ERR_NONE:
                self.server = self.get_login_info("GetServerGubun", True)
                self.user_id = self.get_login_info("USER_ID", True)
                self.account_number = self.get_login_info("ACCNO", True)
                if len(self.server) == 0 or self.server != "1":
                    msg = "실서버 연결 성공" + "\n"
                else:
                    msg = "모의투자서버 연결 성공" + "\n"

                #QMessageBox.information(self, "[필독!] 꼭 읽어야 해욤!", "우측하단 키움 트레이에서 계좌 비밀번호 저장을 꼭 해주세요!\n\n그래야만 매수리스트 불러오기 및 자동매수가 가능합니다.")
            else:
                msg = "연결 끊김: 원인 - " + ReturnCode.CAUSE[returnCode] + "\n"
            print(msg)
        except Exception as error:
            # self.log.error('eventConnect {}'.format(error))
            print("event_connect 에러 : " + str(error))

        # 다음 프로세스 진행
        self.loginLoop.exit()

    def getStockList(self, gubun):
        #self.dynamicCall("SetInputValue(QString, QString)", "종목조건", "1")
        stock_list = self.dynamicCall("GetCodeListByMarket(QString)", gubun)
        return stock_list

    def getCodeName(self, code):
        name = self.dynamicCall("GetMasterCodeName(QString)", code)
        return name

    def get_login_info(self, tag, isConnectState=False):
        """
        사용자의 tag에 해당하는 정보를 반환한다.

        tag에 올 수 있는 값은 아래와 같다.
        ACCOUNT_CNT: 전체 계좌의 개수를 반환한다.
        ACCNO: 전체 계좌 목록을 반환한다. 계좌별 구분은 ;(세미콜론) 이다.
        USER_ID: 사용자 ID를 반환한다.
        USER_NAME: 사용자명을 반환한다.
        GetServerGubun: 접속서버 구분을 반환합니다.("1": 모의투자, 그외(빈 문자열포함): 실서버)

        :param tag: string
        :param isConnectState: bool - 접속상태을 확인할 필요가 없는 경우 True로 설정.
        :return: string
        """

        if not isConnectState:
            if not self.getConnectState():
                raise KiwoomConnectError()

        if not isinstance(tag, str):
            raise ParameterTypeError()

        if tag not in ['ACCOUNT_CNT', 'ACCNO', 'USER_ID', 'USER_NAME', 'GetServerGubun']:
            raise ParameterValueError()

        if tag == "GetServerGubun":
            info = self.getServerGubun()
        else:
            cmd = 'GetLoginInfo("%s")' % tag
            info = self.dynamicCall(cmd)

        return info

    def getConnectState(self):
        """
        현재 접속상태를 반환합니다.

        반환되는 접속상태는 아래와 같습니다.
        0: 미연결, 1: 연결

        :return: int
        """

        state = self.dynamicCall("GetConnectState()")
        return state

    def getServerGubun(self):
        """
        서버구분 정보를 반환한다.
        리턴값이 "1"이면 모의투자 서버이고, 그 외에는 실서버(빈 문자열포함).

        :return: string
        """

        ret = self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")
        return ret

    # 매수리스트 로딩
    def btn_buylist_proc2(self):
        account_number = myWindow.main_ui.account_box.currentText()
        #print("account_number = " + account_number)
        self.dynamicCall("SetInputValue(QString, QString)", "계좌번호", account_number)
        self.dynamicCall("SetInputValue(QString, QString)", "비밀번호", "1234")
        self.dynamicCall("SetInputValue(QString, QString)", "상장폐지조회구분", "0")
        self.dynamicCall("SetInputValue(QString, QString)", "비밀번호입력매체구분", "00")

        sRQName = "나의매수리스트_WOOK_"
        self.dynamicCall("CommRqData(QString, QString, int, QString)", sRQName, "OPW00004", 1, "0102")

    #self.call_stock_before("100","200","000660","300")
    def call_stock_before(self, sRQName, sTrCode, code, news_idx):
        # 미리 담아두기
        self.buy_temp[code] = {}
        self.buy_temp[code]['sRQName'] = sRQName
        self.buy_temp[code]['sTrCode'] = sTrCode
        self.buy_temp[code]['code'] = code
        self.buy_temp[code]['news_idx'] = news_idx
        '''
        /********************************************************************/
        /// ########## Open API 함수를 이용한 전문처리 C++용 샘플코드 예제입니다.

         [ opt10015 : 일별거래상세요청 ]

         1. Open API 조회 함수 입력값을 설정합니다.
            종목코드 = 전문 조회할 종목코드
            SetInputValue("종목코드"   ,  "입력값 1");

            시작일자 = YYYYMMDD (20160101 연도4자리, 월 2자리, 일 2자리 형식)
            SetInputValue("시작일자"   ,  "입력값 2");


         2. Open API 조회 함수를 호출해서 전문을 서버로 전송합니다.
            CommRqData( "RQName"   ,  "opt10015"  ,  "0" ,  "화면번호");

        /********************************************************************/
        '''
        self.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)

        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(weeks=-1)
        ttt = re.split(" ", str(tomorrow))
        week_date = ttt[0].replace("-", "")
        #print(week_date)
        self.dynamicCall("SetInputValue(QString, QString)", "시작일자", week_date)
        self.dynamicCall("CommRqData(QString, QString, int, QString)", "일주일전가격호출_WOOK_" + code, "opt10015", 0, "0100")

    def call_stock(self, sRQName, sTrCode, code, news_idx):
        print("call_stock 호출됨, sRQNAME = " + sRQName + ". sTrCode = " + sTrCode + ", code = " + code + ", news_idx = " + news_idx)
        #self.now_code = code

        self.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)

        # 종목조건 = 0:전체조회, 1: 관리종목제외, 5: 증100제외, 6: 증100만보기, 7: 증40만보기, 8: 증30만보기, 9: 증20만보기
        # self.dynamicCall("SetInputValue(QString, QString)", "종목조건", "1")

        '''
        [CommRqData() 함수]

        CommRqData(
        BSTR sRQName,    // 사용자 구분명
        BSTR sTrCode,    // 조회하려는 TR이름
        long nPrevNext,  // 연속조회여부
        BSTR sScreenNo  // 화면번호
        )

        조회요청함수이며 빈번하게 조회요청하면 시세과부하 에러값으로 -200이 전달됩니다.
        리턴값
        0이면 조회요청 정상 나머지는 에러
        -200 시세과부하
        -201 조회전문작성 에러        
        '''
        # self.dynamicCall("CommRqData(QString, QString, int, QString)", opt_name, "wook", 0, "0101")
        sRQNameToCode = sRQName + "_WOOK_" + code + "_WOOK_" + news_idx
        self.dynamicCall("CommRqData(QString, QString, int, QString)", sRQNameToCode, sTrCode, 0, "0101")

    def receiveTrData(self, screenNo, requestName, trcode, recordName, inquiry,
                      deprecated1, deprecated2, deprecated3, deprecated4):
        """
        TR 수신 이벤트

        조회요청 응답을 받거나 조회데이터를 수신했을 때 호출됩니다.
        requestName과 trCode는 commRqData()메소드의 매개변수와 매핑되는 값 입니다.
        조회데이터는 이 이벤트 메서드 내부에서 getCommData() 메서드를 이용해서 얻을 수 있습니다.

        :param screenNo: string - 화면번호(4자리)
        :param requestName: string - TR 요청명(commRqData() 메소드 호출시 사용된 requestName)
        :param trcode: string
        :param recordName: string
        :param inquiry: string - 조회('0': 남은 데이터 없음, '2': 남은 데이터 있음)
        """
        print("====================================")
        print("receiveTrData 실행, screenNo = " + screenNo + ", requestName = " + requestName + ", trcode = " + trcode)
        '''
        if requestName == "관심종목정보요청":
            data = self.getCommDataEx(trCode, "관심종목정보")
        '''
        rr = requestName

        temp = re.split("_WOOK_", requestName)
        requestName = temp[0]
        code = temp[1]

        if requestName == "매수요청":
            #cc = self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, 0, "종목코드"]) # CommGetData 보다는 GetCommData 를 추천함
            #print("code = " + code + ", cc = " + cc.strip())

            codeName = self.getCodeName(code)

            is_gwansim = self.dynamicCall("GetMasterStockState(QString)", code)
            #print(code + "관리종목이냐? = " + is_gwansim) #140910관리종목이냐? = 증거금100%|관리종목
            if "관리종목" in is_gwansim:
                msg = ""
                msg += "==============================================\n"
                msg += codeName + "(" + code + ")는 관리종목이라서 매수하지 않음."
                msg += "\n"
                myWindow.set_logbox(msg)
            else:
                '''
                name = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "종목명")
                volume = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "거래량")
                '''
                now_price = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "현재가")
                n_price = abs(int(now_price.strip()))
                print("단가 = " + str(n_price))

                시가총액 = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "시가총액")
                시가총액 = abs(int(시가총액.strip()))
                print("시가총액 = " + str(시가총액)) # 이수화학 시가총액 = 2032(단위가 억임)

                print(str("code = " + code))
                print(str(self.buy_temp[code]))

                if 시가총액 >= 10000:
                    print("시가총액 start1")
                    msg = ""
                    msg += "==============================================\n"
                    msg += codeName + "(" + code + ")는 시가총액을 1조넘김, "+codeName+" 시가총액 = " + str(시가총액) + "(단위:억)"
                    msg += "\n"
                    myWindow.set_logbox(msg)
                    buy_okok1 = False
                    print("시가총액 start1 end")
                else:
                    print("시가총액 start2")
                    buy_okok1 = True

                print("buy_okok1 통과")

                # 일주일전 가격이랑 비교해서 6% 올랐으면 매수를 하지말자.
                if n_price >= self.buy_temp[code]['week_jonga']:
                    print("upper_per start")
                    #upper_per = n_price * (self.buy_temp[code]['week_jonga'] / 100) / 100
                    upper_per = (n_price * 100 / self.buy_temp[code]['week_jonga']) - 100
                    print("upper_per = " + str(upper_per))
                    print("self.upper_percent = " + str(self.upper_percent))
                    if upper_per >= self.upper_percent:
                        msg = ""
                        msg += "==============================================\n"
                        msg += codeName + "(" + code + ")는 일주일전 매수가 "+str(self.buy_temp[code]['week_jonga'])+" 보다 "+str(self.upper_percent)+"% 더 올라서 매수하지 않음.(현재가 : "+str(n_price)+")"
                        msg += "\n"
                        myWindow.set_logbox(msg)
                        buy_okok2 = False
                    else:
                        buy_okok2 = True
                else:
                    buy_okok2 = True

                print("buy_okok2 통과")

                if buy_okok1 == True and buy_okok2 == True:
                    '''
                    [GetCommData() 함수]
                    
                    GetCommData(
                    BSTR strTrCode,   // TR 이름
                    BSTR strRecordName,   // 레코드이름
                    long nIndex,      // TR반복부
                    BSTR strItemName) // TR에서 얻어오려는 출력항목이름
                    
                    OnReceiveTRData()이벤트 함수가 호출될때 조회데이터를 얻어오는 함수입니다.
                    이 함수는 반드시 OnReceiveTRData()이벤트 함수가 호출될때 그 안에서 사용해야 합니다.            
                    '''
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("종목코드"));   strData.Trim();
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("거래량"));   strData.Trim();
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("시가"));   strData.Trim();
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("고가"));   strData.Trim();
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("저가"));   strData.Trim();
                    # strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("현재가"));   strData.Trim();

                    '''
                    [SendOrder() 함수]
                    
                    SendOrder(
                    BSTR sRQName, // 사용자 구분명
                    BSTR sScreenNo, // 화면번호
                    BSTR sAccNo,  // 계좌번호 10자리
                    LONG nOrderType,  // 주문유형 1:신규매수, 2:신규매도 3:매수취소, 4:매도취소, 5:매수정정, 6:매도정정
                    BSTR sCode, // 종목코드
                    LONG nQty,  // 주문수량
                    LONG nPrice, // 주문가격
                    BSTR sHogaGb,   // 거래구분(혹은 호가구분)은 아래 참고
                    BSTR sOrgOrderNo  // 원주문번호입니다. 신규주문에는 공백, 정정(취소)주문할 원주문번호를 입력합니다.
                    )
                    
                    9개 인자값을 가진 국내 주식주문 함수이며 리턴값이 0이면 성공이며 나머지는 에러입니다.
                    1초에 5회만 주문가능하며 그 이상 주문요청하면 에러 -308을 리턴합니다.
                    
                    [거래구분]
                    모의투자에서는 지정가 주문과 시장가 주문만 가능합니다.
                    
                    00 : 지정가
                    03 : 시장가
                    05 : 조건부지정가
                    06 : 최유리지정가
                    07 : 최우선지정가
                    10 : 지정가IOC
                    13 : 시장가IOC
                    16 : 최유리IOC
                    20 : 지정가FOK            
                    '''

                    # msg = ""
                    # msg += "==============================================\n"
                    # msg += "종목코드 : " + code + "\n"
                    # msg += "종목명 : " + name.strip() + "\n"
                    # msg += "거래량 : " + volume.strip() + "\n"
                    # msg += "현재가 : " + str(n_price) + "\n"

                    '''
                    order_money     => 매수 주문 금액
                    buy_able_money  => 실제주문가능한 금액 최대치(나의 설정에서 매수 종목 총액 - 현재 매수 총액) 
                    '''

                    buy_money = self.order_money
                    buy_end = "NO"
                    print("order_money start, self.order_money = " + str(self.order_money) + ", self.buy_able_money = " + str(self.buy_able_money))
                    if self.order_money > self.buy_able_money:
                        # 실제 주문 가능한 금액보다 클 순 없다.
                        buy_money = self.buy_able_money
                        buy_end = "OK"

                    print("buy_count start")
                    buy_count = buy_money / int(n_price)
                    buy_count = int(buy_count)
                    if buy_end == "OK":
                        # 끝에 1개 더 사줘야 [매수 종목 총액제한] 을 초과하기 때문에, 다음부터 구매하지 않게됨.
                        buy_count = buy_count + 1
                    else:
                        # 최소 이정도는 구매해줘야함.(단일 매수 주문수량(최소 여기 카운트만큼은 사야된다는 거임))
                        # order_money_cnt 이거 사용안할 듯함. 일단 무조건 설정값은 '1' 이 들어갈듯
                        if self.order_money_cnt > buy_count:
                            buy_count = self.order_money_cnt
                    print("buy_count end")

                    buy_account_number = myWindow.main_ui.account_box.currentText()
                    print("buy_account_number = " + buy_account_number)

                    #  여기서 매수하자
                    returnCode = self.dynamicCall(
                        "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                        ["매수시도_WOOK_" + code, "4989", buy_account_number, 1, code, buy_count, "", "03", ""])

                    msg = "==============================================\n"
                    if returnCode != 0:
                        print("매수실패! returnCode = " + str(returnCode))
                        msg += "매수실패 : " + ReturnCode.CAUSE[returnCode] + "\n"
                        msg += "실패종목코드 : " + code + "(" + codeName + ")\n"
                    else:
                        print("매수성공!")
                        # self.start_news = "NO"
                        # msg = "=============================================\n"
                        # msg = msg + "매수성공, 종목코드 = " + code + ", 계좌번호 = " + buy_account_number
                        # print(msg)
                        msg += "매수성공!\n"
                        msg += "구매종목코드 : " + code + "(" + codeName + ")\n"
                        msg += "구매가능금액 : " + str(self.order_money) + "\n"
                        msg += "구매가능갯수 : " + str(buy_count) + "\n"
                        msg += "구매계좌번호 : " + str(buy_account_number) + "\n"
                        # print("매수성공2!")
                        post_data = {"type": "stock_buy_ok", "stock_id": self.user_id, "buy_code": code, "buy_cnt": str(buy_count), "buy_account": buy_account_number, "news_idx": temp[2], "buy_price": n_price}
                        # print("post_data = " + str(post_data))

                        # print("매수성공3!")

                        return_data = requests.post(myWindow.api_url, data=post_data)
                        return_data = return_data.text

                        api_ok = None
                        try:
                            print("return_data = " + return_data)
                            api_ok = True
                        except Exception as error:
                            print(str(error))
                            self.set_logbox(error)
                            api_ok = False

                        if api_ok == True:
                            # print("return_data = " + str(return_data))
                            r_data = json.loads(return_data)
                            print("r_data = " + str(r_data['result']))
                            if r_data['result'] == "OK":
                                msg += "매수기록 데이터베이스 저장성공!"
                                print("매수기록 디비 저장성공!")

                                # 매수리스트 리플래쉬
                                #myWindow.btn_buylist_proc()
                                #myWindow.btn_buylist_proc2()

                            elif r_data['result'] == "NO":
                                msg += "매수기록 데이터베이스 저장실패!"
                                print("매수기록 디비 저장실패~")

                    msg += "\n"
                    # 로그남기기
                    myWindow.set_logbox(msg)
        elif requestName == "나의매수리스트":
            # 반복되서 들어오는지 테스트 해봐야 함.
            # 테스트결과 반복해서 receiveTrData 가 호출되지는 않고, 아래소스처럼 반복문 돌리면 됨.
            '''
            int nCnt = OpenAPI.GetRepeatCnt(sTrcode, strRQName);
            for (int nIdx = 0; nIdx < nCnt; nIdx++)
            {
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("종목코드"));   strData.Trim();
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("거래량"));   strData.Trim();
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("시가"));   strData.Trim();
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("고가"));   strData.Trim();
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("저가"));   strData.Trim();
                strData = OpenAPI.GetCommData(sTrcode, strRQName, nIdx, _T("현재가"));   strData.Trim();
            }            
            '''
            cnt = int(self.dynamicCall("GetRepeatCnt(QString, QString)", [trcode, rr]))
            myWindow.mybuylist_cnt = cnt
            #print("나의 매수 카운트 = " + str(cnt))
            myWindow.main_ui.buy_table.setRowCount(cnt)
            try:
                myWindow.mybuylist_total_money = 0
                for key in range(cnt):
                    #계좌평가현황요청
                    #종목코드, 종목명, 보유수량, 현재가, 평가금액, 손익금액, 손익율, 대출일, 매입금액, 결제잔고, 전일매수수량, 전일매도수량, 금일매수수량, 금일매도수량
                    종목코드 = self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "종목코드"]).strip()
                    종목명 = self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "종목명"]).strip()
                    보유수량 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "보유수량"]).strip())
                    현재가 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "현재가"]).strip())
                    평가금액 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "평가금액"]).strip())
                    손익금액 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "손익금액"]).strip())
                    손익율 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "손익율"]).strip())
                    매입금액 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "매입금액"]).strip())
                    # 결제잔고 = float(self.dynamicCall("GetCommData(QString, QString, int, QString)", [trcode, rr, key, "결제잔고"]).strip())

                    myWindow.mybuylist_total_money += 매입금액

                    mybuylist = {
                        "종목코드": 종목코드,
                        "종목명": 종목명,
                        "보유수량": 보유수량,
                        "현재가": 현재가,
                        "평가금액": 평가금액,
                        "손익금액": 손익금액,
                        "손익율": 손익율,
                        "매입금액": 매입금액
                        # "결제잔고": 결제잔고
                    }
                    '''
                    myWindow.mybuylist[myWindow.mybuylist_cnt] = mybuylist
                    myWindow.mybuylist_cnt = myWindow.mybuylist_cnt + 1
                    print("receiveTrData mybuylist = " + str(myWindow.mybuylist))
                    '''
                    # print("매수정보 row = " + str(mybuylist))

                    myWindow.main_ui.buy_table.setItem(key, 0, QTableWidgetItem(str(종목명)))
                    myWindow.main_ui.buy_table.setItem(key, 1, QTableWidgetItem(str(보유수량)))
                    myWindow.main_ui.buy_table.setItem(key, 2, QTableWidgetItem(str(현재가)))
                    myWindow.main_ui.buy_table.setItem(key, 3, QTableWidgetItem(str(평가금액)))
                    myWindow.main_ui.buy_table.setItem(key, 4, QTableWidgetItem(str(손익금액)))
                    myWindow.main_ui.buy_table.setItem(key, 5, QTableWidgetItem(str(손익율)))
                    myWindow.main_ui.buy_table.setItem(key, 6, QTableWidgetItem(str(매입금액)))
                    #myWindow.main_ui.buy_table.setItem(key, 7, QTableWidgetItem(str(결제잔고)))

                    # 데이터 갱신을 위해 포커스를 주자
                    myWindow.main_ui.buy_table.setCurrentCell(key, 0)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 1)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 2)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 3)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 4)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 5)
                    myWindow.main_ui.buy_table.setCurrentCell(key, 6)
                    #myWindow.main_ui.buy_table.setCurrentCell(key, 7)
            except Exception as error:
                myWindow.set_logbox("receiveTrError = " + error)

        elif requestName == "일주일전가격호출":
            print("requestName = " + requestName)
            일자 = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "일자")
            print("일자 = " + str(일자))
            종가 = self.dynamicCall("CommGetData(QString, QString, QString, int, QString)", trcode, "", requestName, 0, "종가")
            #print("종가 = " + 종가)
            종가 = abs(int(종가.strip()))
            print("종가 = " + str(종가))
            #print(str(self.buy_temp))
            self.buy_temp[code]['week_jonga'] = 종가
            print(str(self.buy_temp))
            self.call_stock(self.buy_temp[code]['sRQName'], self.buy_temp[code]['sTrCode'], code, self.buy_temp[code]['news_idx'])
            '''
            print("일자 = " + 일자)
            print("종가 = " + 종가)
            print("buy_temp = " + str(self.buy_temp))
            
            SK하이닉스임
            일자 =             20180420
            종가 =               -84400
            buy_temp = {'000660': {'sRQName': '100', 'sTrCode': '200', 'code': '000660', 'news_idx': '300'}}            
            '''

    def receiveChejanData(self):
        print("receiveChejanData 호출")
        '''
        [GetChejanData() 함수]
        
        GetChejanData(
        long nFid   // 실시간 타입에 포함된FID
        )
        
        OnReceiveChejan()이벤트 함수가 호출될때 체결정보나 잔고정보를 얻어오는 함수입니다.
        이 함수는 반드시 OnReceiveChejan()이벤트 함수가 호출될때 그 안에서 사용해야 합니다.        
        '''

    def receiveRealData(self, sGubun, nItemCnt, sFIdList):
        print("receiveRealData 호출됨, sGubun = " + str(sGubun) + ", nItemCnt = " + str(nItemCnt) + ", sFIdList = " + str(sFIdList))

    # 이게 ReceiveTrData 보다 먼저 호출되드라...
    def receiveMsg(self, scr_no, rq_name, tr_code, msg):
        print("receiveMsg 호출됨, scr_no = " + scr_no + ", rq_name = " + rq_name + ", tr_code = " + tr_code + ", msg = " + msg)

        rq_name_split = re.split("_WOOK_", rq_name)
        code = rq_name_split[1]
        codeName = self.getCodeName(code)

        sRQName = rq_name_split[0]

        msg2 = ""
        if sRQName == "매수시도":
            msg2 += "===== 키움 API 에서 받은 매수체결 수신 메세지 =====\n"
            msg2 += "종목명 : " + codeName + " (" + code + ")\n"
        '''
        elif sRQName == "나의매수리스트":
            msg2 += "===== 키움 API 에서 받은 나의매수 수신 메세지 =====\n"
            print("receiveMsg : mybuylist = " + str(myWindow.mybuylist))
        '''

        if msg2 != "":
            msg2 += "메세지 내용 : " + msg + "\n"
            msg2 += "\n"
            myWindow.set_logbox(msg2)

class ParameterTypeError(Exception):
    """ 파라미터 타입이 일치하지 않을 경우 발생하는 예외 """

    def __init__(self, msg="파라미터 타입이 일치하지 않습니다."):
        self.msg = msg

    def __str__(self):
        return self.msg

class KiwoomProcessingError(Exception):
    """ 키움에서 처리실패에 관련된 리턴코드를 받았을 경우 발생하는 예외 """

    def __init__(self, msg="처리 실패"):
        self.msg = msg

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg

class KiwoomConnectError(Exception):
    """ 키움서버에 로그인 상태가 아닐 경우 발생하는 예외 """

    def __init__(self, msg="로그인 여부를 확인하십시오"):
        self.msg = msg

    def __str__(self):
        return self.msg


class ParameterValueError(Exception):
    """ 파라미터로 사용할 수 없는 값을 사용할 경우 발생하는 예외 """

    def __init__(self, msg="파라미터로 사용할 수 없는 값 입니다."):
        self.msg = msg

    def __str__(self):
        return self.msg


class ReturnCode(object):
    """ 키움 OpenApi+ 함수들이 반환하는 값 """

    OP_ERR_NONE = 0  # 정상처리
    OP_ERR_FAIL = -10  # 실패
    OP_ERR_LOGIN = -100  # 사용자정보교환실패
    OP_ERR_CONNECT = -101  # 서버접속실패
    OP_ERR_VERSION = -102  # 버전처리실패
    OP_ERR_FIREWALL = -103  # 개인방화벽실패
    OP_ERR_MEMORY = -104  # 메모리보호실패
    OP_ERR_INPUT = -105  # 함수입력값오류
    OP_ERR_SOCKET_CLOSED = -106  # 통신연결종료
    OP_ERR_SISE_OVERFLOW = -200  # 시세조회과부하
    OP_ERR_RQ_STRUCT_FAIL = -201  # 전문작성초기화실패
    OP_ERR_RQ_STRING_FAIL = -202  # 전문작성입력값오류
    OP_ERR_NO_DATA = -203  # 데이터없음
    OP_ERR_OVER_MAX_DATA = -204  # 조회가능한종목수초과
    OP_ERR_DATA_RCV_FAIL = -205  # 데이터수신실패
    OP_ERR_OVER_MAX_FID = -206  # 조회가능한FID수초과
    OP_ERR_REAL_CANCEL = -207  # 실시간해제오류
    OP_ERR_ORD_WRONG_INPUT = -300  # 입력값오류
    OP_ERR_ORD_WRONG_ACCTNO = -301  # 계좌비밀번호없음
    OP_ERR_OTHER_ACC_USE = -302  # 타인계좌사용오류
    OP_ERR_MIS_2BILL_EXC = -303  # 주문가격이20억원을초과
    OP_ERR_MIS_5BILL_EXC = -304  # 주문가격이50억원을초과
    OP_ERR_MIS_1PER_EXC = -305  # 주문수량이총발행주수의1%초과오류
    OP_ERR_MIS_3PER_EXC = -306  # 주문수량이총발행주수의3%초과오류
    OP_ERR_SEND_FAIL = -307  # 주문전송실패
    OP_ERR_ORD_OVERFLOW = -308  # 주문전송과부하
    OP_ERR_MIS_300CNT_EXC = -309  # 주문수량300계약초과
    OP_ERR_MIS_500CNT_EXC = -310  # 주문수량500계약초과
    OP_ERR_ORD_WRONG_ACCTINFO = -340  # 계좌정보없음
    OP_ERR_ORD_SYMCODE_EMPTY = -500  # 종목코드없음

    CAUSE = {
        0: '정상처리',
        -10: '실패',
        -100: '사용자정보교환실패',
        -102: '버전처리실패',
        -103: '개인방화벽실패',
        -104: '메모리보호실패',
        -105: '함수입력값오류',
        -106: '통신연결종료',
        -200: '시세조회과부하',
        -201: '전문작성초기화실패',
        -202: '전문작성입력값오류',
        -203: '데이터없음',
        -204: '조회가능한종목수초과',
        -205: '데이터수신실패',
        -206: '조회가능한FID수초과',
        -207: '실시간해제오류',
        -300: '입력값오류',
        -301: '계좌비밀번호없음',
        -302: '타인계좌사용오류',
        -303: '주문가격이20억원을초과',
        -304: '주문가격이50억원을초과',
        -305: '주문수량이총발행주수의1%초과오류',
        -306: '주문수량이총발행주수의3%초과오류',
        -307: '주문전송실패',
        -308: '주문전송과부하',
        -309: '주문수량300계약초과',
        -310: '주문수량500계약초과',
        -340: '계좌정보없음',
        -500: '종목코드없음'
    }


class FidList(object):
    """ receiveChejanData() 이벤트 메서드로 전달되는 FID 목록 """

    CHEJAN = {
        9201: '계좌번호',
        9203: '주문번호',
        9205: '관리자사번',
        9001: '종목코드',
        912: '주문업무분류',
        913: '주문상태',
        302: '종목명',
        900: '주문수량',
        901: '주문가격',
        902: '미체결수량',
        903: '체결누계금액',
        904: '원주문번호',
        905: '주문구분',
        906: '매매구분',
        907: '매도수구분',
        908: '주문/체결시간',
        909: '체결번호',
        910: '체결가',
        911: '체결량',
        10: '현재가',
        27: '(최우선)매도호가',
        28: '(최우선)매수호가',
        914: '단위체결가',
        915: '단위체결량',
        938: '당일매매수수료',
        939: '당일매매세금',
        919: '거부사유',
        920: '화면번호',
        921: '921',
        922: '922',
        923: '923',
        949: '949',
        10010: '10010',
        917: '신용구분',
        916: '대출일',
        930: '보유수량',
        931: '매입단가',
        932: '총매입가',
        933: '주문가능수량',
        945: '당일순매수수량',
        946: '매도/매수구분',
        950: '당일총매도손일',
        951: '예수금',
        307: '기준가',
        8019: '손익율',
        957: '신용금액',
        958: '신용이자',
        959: '담보대출수량',
        924: '924',
        918: '만기일',
        990: '당일실현손익(유가)',
        991: '당일신현손익률(유가)',
        992: '당일실현손익(신용)',
        993: '당일실현손익률(신용)',
        397: '파생상품거래단위',
        305: '상한가',
        306: '하한가'
    }


class RealType(object):
    REALTYPE = {
        '주식시세': {
            10: '현재가',
            11: '전일대비',
            12: '등락율',
            27: '최우선매도호가',
            28: '최우선매수호가',
            13: '누적거래량',
            14: '누적거래대금',
            16: '시가',
            17: '고가',
            18: '저가',
            25: '전일대비기호',
            26: '전일거래량대비',
            29: '거래대금증감',
            30: '거일거래량대비',
            31: '거래회전율',
            32: '거래비용',
            311: '시가총액(억)'
        },

        '주식체결': {
            20: '체결시간(HHMMSS)',
            10: '체결가',
            11: '전일대비',
            12: '등락율',
            27: '최우선매도호가',
            28: '최우선매수호가',
            15: '체결량',
            13: '누적체결량',
            14: '누적거래대금',
            16: '시가',
            17: '고가',
            18: '저가',
            25: '전일대비기호',
            26: '전일거래량대비',
            29: '거래대금증감',
            30: '전일거래량대비',
            31: '거래회전율',
            32: '거래비용',
            228: '체결강도',
            311: '시가총액(억)',
            290: '장구분',
            691: 'KO접근도'
        },

        '주식호가잔량': {
            21: '호가시간',
            41: '매도호가1',
            61: '매도호가수량1',
            81: '매도호가직전대비1',
            51: '매수호가1',
            71: '매수호가수량1',
            91: '매수호가직전대비1',
            42: '매도호가2',
            62: '매도호가수량2',
            82: '매도호가직전대비2',
            52: '매수호가2',
            72: '매수호가수량2',
            92: '매수호가직전대비2',
            43: '매도호가3',
            63: '매도호가수량3',
            83: '매도호가직전대비3',
            53: '매수호가3',
            73: '매수호가수량3',
            93: '매수호가직전대비3',
            44: '매도호가4',
            64: '매도호가수량4',
            84: '매도호가직전대비4',
            54: '매수호가4',
            74: '매수호가수량4',
            94: '매수호가직전대비4',
            45: '매도호가5',
            65: '매도호가수량5',
            85: '매도호가직전대비5',
            55: '매수호가5',
            75: '매수호가수량5',
            95: '매수호가직전대비5',
            46: '매도호가6',
            66: '매도호가수량6',
            86: '매도호가직전대비6',
            56: '매수호가6',
            76: '매수호가수량6',
            96: '매수호가직전대비6',
            47: '매도호가7',
            67: '매도호가수량7',
            87: '매도호가직전대비7',
            57: '매수호가7',
            77: '매수호가수량7',
            97: '매수호가직전대비7',
            48: '매도호가8',
            68: '매도호가수량8',
            88: '매도호가직전대비8',
            58: '매수호가8',
            78: '매수호가수량8',
            98: '매수호가직전대비8',
            49: '매도호가9',
            69: '매도호가수량9',
            89: '매도호가직전대비9',
            59: '매수호가9',
            79: '매수호가수량9',
            99: '매수호가직전대비9',
            50: '매도호가10',
            70: '매도호가수량10',
            90: '매도호가직전대비10',
            60: '매수호가10',
            80: '매수호가수량10',
            100: '매수호가직전대비10',
            121: '매도호가총잔량',
            122: '매도호가총잔량직전대비',
            125: '매수호가총잔량',
            126: '매수호가총잔량직전대비',
            23: '예상체결가',
            24: '예상체결수량',
            128: '순매수잔량(총매수잔량-총매도잔량)',
            129: '매수비율',
            138: '순매도잔량(총매도잔량-총매수잔량)',
            139: '매도비율',
            200: '예상체결가전일종가대비',
            201: '예상체결가전일종가대비등락율',
            238: '예상체결가전일종가대비기호',
            291: '예상체결가',
            292: '예상체결량',
            293: '예상체결가전일대비기호',
            294: '예상체결가전일대비',
            295: '예상체결가전일대비등락율',
            13: '누적거래량',
            299: '전일거래량대비예상체결률',
            215: '장운영구분'
        },

        '장시작시간': {
            215: '장운영구분(0:장시작전, 2:장종료전, 3:장시작, 4,8:장종료, 9:장마감)',
            20: '시간(HHMMSS)',
            214: '장시작예상잔여시간'
        },

        '업종지수': {
            20: '체결시간',
            10: '현재가',
            11: '전일대비',
            12: '등락율',
            15: '거래량',
            13: '누적거래량',
            14: '누적거래대금',
            16: '시가',
            17: '고가',
            18: '저가',
            25: '전일대비기호',
            26: '전일거래량대비(계약,주)'
        },

        '업종등락': {
            20: '체결시간',
            252: '상승종목수',
            251: '상한종목수',
            253: '보합종목수',
            255: '하락종목수',
            254: '하한종목수',
            13: '누적거래량',
            14: '누적거래대금',
            10: '현재가',
            11: '전일대비',
            12: '등락율',
            256: '거래형성종목수',
            257: '거래형성비율',
            25: '전일대비기호'
        },

        '주문체결': {
            9201: '계좌번호',
            9203: '주문번호',
            9205: '관리자사번',
            9001: '종목코드',
            912: '주문분류(jj:주식주문)',
            913: '주문상태(10:원주문, 11:정정주문, 12:취소주문, 20:주문확인, 21:정정확인, 22:취소확인, 90,92:주문거부)',
            302: '종목명',
            900: '주문수량',
            901: '주문가격',
            902: '미체결수량',
            903: '체결누계금액',
            904: '원주문번호',
            905: '주문구분(+:현금매수, -:현금매도)',
            906: '매매구분(보통, 시장가등)',
            907: '매도수구분(1:매도, 2:매수)',
            908: '체결시간(HHMMSS)',
            909: '체결번호',
            910: '체결가',
            911: '체결량',
            10: '체결가',
            27: '최우선매도호가',
            28: '최우선매수호가',
            914: '단위체결가',
            915: '단위체결량',
            938: '당일매매수수료',
            939: '당일매매세금'
        },

        '잔고': {
            9201: '계좌번호',
            9001: '종목코드',
            302: '종목명',
            10: '현재가',
            930: '보유수량',
            931: '매입단가',
            932: '총매입가',
            933: '주문가능수량',
            945: '당일순매수량',
            946: '매도매수구분',
            950: '당일총매도손익',
            951: '예수금',
            27: '최우선매도호가',
            28: '최우선매수호가',
            307: '기준가',
            8019: '손익율'
        },

        '주식시간외호가': {
            21: '호가시간(HHMMSS)',
            131: '시간외매도호가총잔량',
            132: '시간외매도호가총잔량직전대비',
            135: '시간외매수호가총잔량',
            136: '시간외매수호가총잔량직전대비'
        }
    }

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    # myWindow.btn_buylist_proc2()
    app.exec_()