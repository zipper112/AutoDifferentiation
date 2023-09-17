import math
from typing import *

class Tensor:
    def __init__(self, x: float, grad: float) -> None:
        self.x = x
        self.grad = grad
    
    def __str__(self) -> str:
        return 'value: ' + str(self.x) + ' grad: ' + str(self.grad)

    def __add__(self, other):
        x, grad = None, None
        if isinstance(other, Tensor):
            x = self.x + other.x
            grad = self.grad + other.grad
        elif isinstance(other, float) or isinstance(other, int):
            x = self.x + other
            grad = self.grad
        else:
            raise NotImplementedError('不支持的运算操作：{}与{}做加法'.format(str(self), str(other)))
        return Tensor(x, grad)
    
    def __sub__(self, other):
        x, grad = None, None
        if isinstance(other, Tensor):
            x = self.x - other.x
            grad = self.grad - other.grad
        elif isinstance(other, float) or isinstance(other, int):
            x = self.x - other
            grad = self.grad
        else:
            raise NotImplementedError('不支持的运算操作：{}与{}做减法'.format(str(self), str(other)))
        return Tensor(x, grad)
    
    def __mul__(self, other):
        x, grad = None, None
        if isinstance(other, Tensor):
            x = self.x * other.x
            grad = self.grad * other.x + other.grad * self.x
        elif isinstance(other, float) or isinstance(other, int):
            x = self.x * other
            grad = self.grad * other
        else:
            raise NotImplementedError('不支持的运算操作：{}与{}做乘法'.format(str(self), str(other)))
        return Tensor(x, grad)


def log(x: Tensor):
    res, grad = math.log(x.x), x.grad / x.x
    return Tensor(res, grad)

def sin(x: Tensor):
    res, grad = math.sin(x.x), math.cos(x.x) * x.grad
    return Tensor(res, grad)

def AutoDifferentiation(func: Callable[[Tensor], Tensor], inputs: List[float]) -> List[Tensor]:
    """
    @params: func
        输入若干个Tensor，返回一个Tensor的函数
    @inputs: List[float]
        一个float的列表，表示上述函数的输入

    return: List[Tensor]
        所有输入变量的偏导数
    """
    res = []
    for i in range(len(inputs)):
        inputs_ = [Tensor(x, 0) for x in inputs]
        inputs_[i].grad = 1
        res.append(Tensor(func(*inputs_).grad, 0))
    return res

def fun(x1: Tensor, x2: Tensor):
    f = log(x1) + x1 * x2 - sin(x2)
    return f

print(AutoDifferentiation(fun, [2., 5.]))
