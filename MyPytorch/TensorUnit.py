from __future__ import annotations
from pydantic import BaseModel
from typing import *
import math

class Tape(BaseModel):
    call_fun: Callable[[float], Tuple[float]]
    oper: str
    priors: list

class TensorUnit:
    """
        基础的浮点可微分变量
    """
    tape: Tape = None
    grad: float = None
    def __init__(self, data: float, tape: Tape=None) -> None:
        self.data = data
        self.tape = tape

    def __str__(self) -> str:
        return 'value: ' + str(self.data) + ' grad: ' + str(self.grad)
    
    #==============================加法运算===============================
    def __add__(self, other):
        return self.add_ops(other)
    
    def __radd__(self, other):
        return self + other
        
    def __iadd__(self, other):
        return self.add_ops(other)

    def add_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, TensorUnit):
            res = other.data + self.data
            def back(df_dv):
                return df_dv, df_dv
            tape = Tape(call_fun=back, oper='add', priors=(self, other))
            return TensorUnit(res, tape)
        elif isinstance(other, float) or isinstance(other, int):
            res = other + self.data
            def back(df_dv):
                return df_dv, 
            tape = Tape(call_fun=back, oper='add', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做加法运算'.format(type(self), type(other)))
        
    #==============================减法运算===============================
    def __sub__(self, other):
        return self.sub_ops(other)

    def __rsub__(self, other):
        return self.rsub_ops(other)

    def __isub__(self, other):
        return self.sub_ops(other)

    def sub_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, TensorUnit):
            res = self.data - other.data
            def back(df_dv):
                return df_dv, -df_dv
            tape = Tape(call_fun=back, oper='sub', priors=(self, other))
            return TensorUnit(res, tape)
        elif isinstance(other, float) or isinstance(other, int):
            res = self.data - other
            def back(df_dv):
                return df_dv, 
            tape = Tape(call_fun=back, oper='sub', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做减法运算'.format(type(self), type(other)))

    def rsub_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, float) or isinstance(other, int):
            res = other - self.data
            def back(df_dv):
                return -df_dv, 
            tape = Tape(call_fun=back, oper='sub', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做减法运算'.format(type(self), type(other)))
    #==============================乘法运算===============================
    def __mul__(self, other):
        return self.mul_ops(other)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self.mul_ops(other)
    
    def mul_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, TensorUnit):
            res = self.data * other.data
            def back(df_dv):
                return df_dv * other.data, df_dv * self.data
            tape = Tape(call_fun=back, oper='mul', priors=(self, other))
            return TensorUnit(res, tape)
        elif isinstance(other, float) or isinstance(other, int):
            res = self.data * other
            def back(df_dv):
                return df_dv * other, 
            tape = Tape(call_fun=back, oper='mul', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做乘法运算'.format(type(self), type(other)))
    #==============================开方&负运算===============================
    def __neg__(self):
        res = None
        def back(df_dv):
            return -df_dv, 
        tape = Tape(call_fun=back, oper='neg', priors=(self, ))
        return TensorUnit(res, tape)
    
    def __pow__(self, other):
        return self.pow_ops(other)

    def __rpow__(self, other):
        return self.rpow_ops(other)
    
    def __ipow__(self, other):
        return self.pow_ops(other)
    
    def pow_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, TensorUnit):
            res = self.data ** other.data
            def back(df_dv):
                df_self = df_dv * self.data ** (other.data - 1) * other.data
                df_other = df_dv * math.log(self.data) * self.data ** other.data
                return df_self, df_other
            tape = Tape(call_fun=back, oper='pow', priors=(self, other))
            return TensorUnit(res, tape)
        
        elif isinstance(other, float) or isinstance(other, int):
            res = self.data ** other
            def back(df_dv):
                return df_dv * self.data ** (other - 1) * other, 
            tape = Tape(call_fun=back, oper='pow', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做乘方运算'.format(type(self), type(other)))

    def rpow_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, float) or isinstance(other, int):
            res = other ** self.data
            def back(df_dv):
                return math.log(self.data) * other ** self.data * df_dv, 
            tape = Tape(call_fun=back, oper='pow', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做乘方运算'.format(type(self), type(other)))

    #==============================除法运算===============================
    def __truediv__(self, other):
        return self.truediv_ops(other)

    def __rtruediv__(self, other):
        return self.rtruediv_ops(other)

    def __itruediv__(self, other):
        return self.truediv_ops(other)
    
    def truediv_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, TensorUnit):
            res = self.data / other.data
            def back(df_dv):
                df_self = df_dv / other.data
                df_other = -df_dv * self.data / other.data ** 2
                return df_self, df_other
            tape = Tape(call_fun=back, oper='div', priors=(self, other))
            return TensorUnit(res, tape)
        
        elif isinstance(other, float) or isinstance(other, int):
            res = self.data / other
            def back(df_dv):
                return df_dv / other,  
            tape = Tape(call_fun=back, oper='div', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做除法运算'.format(type(self), type(other)))

    def rtruediv_ops(self, other) -> TensorUnit:
        res = None
        if isinstance(other, float) or isinstance(other, int):
            res = other / self.data
            def back(df_dv):
                return -other * df_dv / self.data ** 2,  
            tape = Tape(call_fun=back, oper='div', priors=(self,))
            return TensorUnit(res, tape)
        else:
            raise NotImplementedError('Error: TensorUnit不支持{}与{}做除法运算'.format(type(self), type(other)))

#   =====================================其他功能=====================================

    def backward(self, prior:TensorUnit=None, df_dv: float=1.) -> TensorUnit:
        # 递归地反向自动微分
        if prior == None:
            prior = self

        if prior.grad == None:
            prior.grad = 0.
        prior.grad += df_dv
        if prior.tape is None:
            return
        back_pairs = zip(prior.tape.call_fun(df_dv), prior.tape.priors)
        for n_df_dv, n_prior in back_pairs:
            self.backward(n_prior, n_df_dv)