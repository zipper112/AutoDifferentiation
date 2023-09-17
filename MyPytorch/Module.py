from abc import ABCMeta, abstractclassmethod
from typing import Any

class Module(ABCMeta):

    def parameters(self):
        pass

    @abstractclassmethod
    def forward(self, *args, **keywords) -> Any: ...

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    