import time
import threading
from threading import Lock,Thread
import time,os 
def hh(i = 1):
    while(1):
        print(i)
        if(i == 1):
            time.sleep(5)


t1 = threading.Thread(target=hh,args=(1,))     # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
t2 = threading.Thread(target=hh,args=(2,))
t1.start()
t2.start()