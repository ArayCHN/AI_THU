import sys
def func(a):
    def fun(a):
        if a == 1: return 0
        return fun(a - 1)
    a = a - 1
    return fun(a + 5)

if __name__=="__main__":
    print func(int(sys.argv[1]))