import sys
import time


class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

#
# for i in range(3):
#     logger = Logger('temp%s.txt' % i)
#     a = (0, 1, 2, 3)
#     print(a, i)
#     logger.reset()
