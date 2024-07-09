import torch
# print(torch.cuda.is_available())

# class MyClass:
#     @staticmethod
#     def my_static_method(x, y):
#         return x + y
#     def __call__(self, x):
#         print("Calling with",x)
#
# result = MyClass.my_static_method(1, 2)
# print(result)
# obj = MyClass()
# obj(110)

# 读取文件内容
def read_data(filename):
    with open(filename, 'r') as file:
        # data = file.read()
        # print("Data read from file:")
        # print(data)
        first_line = file.readline().strip()

    # 将第一行分割成列表，以空格为分隔符
    numbers = first_line.split()

    # 获取第一个数字，即列表中的第一个元素
    first_x = numbers[0]

    # 将字符串转换为整数或浮点数
    # 如果数字是整数，使用int转换
    x_value = int(first_x)

    # 如果数字是浮点数，使用float转换
    # x_value = float(first_x)

    print("The first x value is:", x_value)


# 调用函数
read_data('output.txt')
