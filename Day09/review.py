def fun(**kwargs):
    print(type(kwargs))
    for value in kwargs.values():
        print(value)

# fun(a=1,b=2,c=3)
fun(**{'a':1,'b':2,'c':3})

def fun(a,b,c):
    print(f'a={a}')
    print(f'b={b}')
    print(f'c={c}')
fun(**{'a':1,'b':2,'c':3})