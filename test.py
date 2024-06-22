import torch
# print(torch.cuda.is_available())

class MyClass:
    @staticmethod
    def my_static_method(x, y):
        return x + y
    def __call__(self, x):
        print("Calling with",x)

result = MyClass.my_static_method(1, 2)
print(result)
obj = MyClass()
obj(110)
