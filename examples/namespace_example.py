from torchrunner.utils import Namespace

def foo(x, y):
    return x * y

def before_foo(ns):
    print(f"before_foo ns: {ns}")
    ns.x += 1
    ns.y += 1

def after_foo(ns):
    print(f"after_foo ns: {ns}")
    ns.ret -= 1

def bar(**kwargs):
    ns = Namespace(**kwargs)
    before_foo(ns)
    ns.ret = foo(**ns)
    after_foo(ns)
    return ns.ret

def main():
    z = bar(x=5, y=7)
    print(f"z: {z}")

if __name__ == '__main__':
    main()
