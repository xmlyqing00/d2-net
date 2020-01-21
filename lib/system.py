import time

def gct(f='l'):
    '''
    get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    '''
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%m_%d_%H_%M', time.localtime(time.time()))
