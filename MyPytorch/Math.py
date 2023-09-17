import math
from MyPytorch.TensorUnit import TensorUnit, Tape
from MyPytorch import Matrix
from typing import *

def __single_log(x: TensorUnit) -> TensorUnit:
    res = math.log(x.data)
    def back(df_dv):
        return df_dv * 1 / x.data,
    tape = Tape(priors=(x, ), call_fun=back, oper='log')
    return TensorUnit(data=res, tape=tape)

def __single_exp(x: TensorUnit) -> TensorUnit:
    res = math.exp(x.data)
    def back(df_dv):
        return math.exp(x.data) * df_dv,
    tape = Tape(priors=(x, ), call_fun=back, oper='exp')
    return TensorUnit(data=res, tape=tape)

def __single_sigmoid(x: TensorUnit) -> TensorUnit:
    return 1.0 / (1.0 + __single_exp(-x))

def log(x: Union[TensorUnit, Matrix]) -> Union[TensorUnit, Matrix]:
    """
    @param: x
        输入TensorUnit或者Matrix, 对其取自然底的对数
    return: 输入TensorUnit或者Matrix
    """
    if isinstance(x, TensorUnit):
        return __single_log(x)
    elif isinstance(x, Matrix):
        return x.apply_single(__single_exp)

def exp(x: Union[TensorUnit, Matrix]) -> Union[TensorUnit, Matrix]:
    """
    @param: x
        输入TensorUnit或者Matrix, 对其取自然底的指数
    return: 输入TensorUnit或者Matrix
    """
    if isinstance(x, TensorUnit):
        return __single_exp(x)
    elif isinstance(x, Matrix):
        return x.apply_single(__single_exp)

def sigmoid(x: Union[TensorUnit, Matrix]) -> Union[TensorUnit, Matrix]:
    """
    @param: x
        输入TensorUnit或者Matrix, 对其计算sigmoid函数(f(x) = 1 / (1 + e^{-x}))
    return: 输入TensorUnit或者Matrix
    """
    if isinstance(x, TensorUnit):
        return __single_sigmoid(x)
    elif isinstance(x, Matrix):
        return x.apply_single(__single_sigmoid)