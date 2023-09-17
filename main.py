# from MyPytorch import TensorUnit, Matrix


# x = Matrix([
#     [1., 2.]
# ])
# y = Matrix([
#     [3.],
#     [4.]
# ])
# z = x.dot(y)[0, 0]
# z.backward()
# print(x[0, 0])

from __future__ import annotations
# import typing

# class Singleton:
#     def __new__(cls, *args, **keywords) -> Singleton:
#         if not hasattr(cls, '_instance'):
#             cls._instance = super(Singleton, cls).__new__(cls)
#         return cls._instance

# class Test(Singleton):
#     def __init__(self, x) -> None:
#         self.x = x

# from abc import ABCMeta, abstractmethod


# class Module(metaclass=ABCMeta):
#     @abstractmethod
#     def test(self):...
# Module()

from dataclasses import dataclass, field
@dataclass(order=True)
class MyData:
    age: int
    name: str = 9
    other_info: list = field(default=None, init=False, repr=True, hash=True)

    def __post_init__(self):
        if self.age >= 18:
            self.other_info = "this person has matured"
        else:
            self.other_info = "this person has not matured"


x = MyData(10, 'asd')
y = MyData(20, 'asssd')
print(sorted([y, x]))


