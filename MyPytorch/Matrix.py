from __future__ import annotations
from numpy import ndarray
from MyPytorch.TensorUnit import TensorUnit
from typing import *
import numpy


class Matrix:
    """
    Matrix:是一个二维矩阵，维度是且仅是2，当维度不是2时会出现异常错误
    """
    data = None
    shape = None
    def __init__(self, data: Union[list, ndarray]) -> None:
        if isinstance(data, list):
            tmp = numpy.array(data)
            self.data, self.shape = self.NptoMatrix(tmp)
        elif isinstance(data, ndarray):
            self.data, self.shape = self.NptoMatrix(data)
        else:
            raise TypeError('Error: 只支持输入numpy矩阵，或者多维列表矩阵')

    #============================类型转换区域===========================

    def NptoMatrix(self, data: ndarray):
        if isinstance(data, ndarray):
            shape = list(data.shape)
            data = data.tolist()
            for i in range(shape[0]):
                for j in range(shape[1]):
                    data[i][j] = TensorUnit(float(data[i][j]))
            return data, shape
        else:
            raise TypeError('Error: 只能转化Ndarray，但却接收到{}'.format(type(data)))

    #==============================运算区域=============================

    def dot(self, other) -> Matrix:
        if not isinstance(other, Matrix):
            raise TypeError('Error: 只支持Matrix与Matrix的dot乘积')
        if self.shape[1] != other.shape[0]:
            raise NotImplementedError('Error: 矩阵形状不匹配，你正在尝试{}与{}做乘积'\
                                .format(str(self.shape), str(other.shape)))
        res = [[0 for _ in range(other.shape[1])] for _ in range(self.shape[0])]
        res = Matrix(res)
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    res[i, j] += self[i, k] * other[k, j]
        return res
    
    def __add__(self, other):
        return self.apply_double(lambda x, y: x + y, other)
    
    def __mul__(self, other):
        return self.apply_double(lambda x, y: x * y, other)

    def __sub__(self, other):
        return self.apply_double(lambda x, y: x - y, other)

    def apply_single(self, fun: Callable[[Matrix, Any], Matrix], *args) -> Matrix:
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i, j] = fun(self[i, j], *args)
        return self
    
    def apply_double(self, fun: Callable[[Matrix, Matrix, Any], Matrix], other: Matrix, *args) -> Matrix:
        if not isinstance(other, Matrix):
            raise NotImplementedError('Error: 错误，Matrix不能与{}做运算'.format(type(other)))
        if other.shape[0] != self.shape[0] or other.shape[1] != self.shape[1]:
            raise TypeError('Error: 矩阵形状不一致{}与{}无法进行加法运算'.format(str(self.shape), str(other.shape)))
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i, j] = fun(self[i, j], other[i, j], *args)
        return self
    #==================================特殊功能区域=======================================

    @classmethod
    def zeros(cls, shape: List(int)):
        return Matrix(numpy.zeros(shape=shape))

    @classmethod
    def ones(cls, shape: List(int)):
        return Matrix(numpy.ones(shape=shape))
    
    @classmethod
    def randn(cls, shape: List(int)):
        return Matrix(numpy.random.rand(*shape))
    
    def transpose(self):
        new_shape = self.shape[1], self.shape[0]
        res = self.zeros(shape=new_shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[j, i] = self[i, j]
        return res

    def __getitem__(self, idx):
        return self.data[idx[0]][idx[1]]

    def __setitem__(self, key, value):
        self.data[key[0]][key[1]] = value
    
    def __str__(self):
        res = ['[' + ' '.join([str(v.data) for v in h]) + ']' for h in self.data]
        return '[' + ',\n'.join(res) + ']'
    