from abc import ABCMeta, abstractmethod


class MyClass(ABCMeta):
    @abstractmethod
    def test(self): ...


obj = MyClass()