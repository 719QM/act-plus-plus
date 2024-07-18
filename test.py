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
# def read_data(filename):
#     with open(filename, 'r') as file:
#         # 初始化行号计数器
#         line_number = 0
#         # 逐行读取文件
#         while True:
#             line = file.readline()
#             if not line:  # 如果读取到文件末尾，结束循环
#                 break
#             line_number += 1  # 每读取一行，行号加1
#
#             # 去除行尾的换行符并分割行
#             numbers = line.strip().split()
#
#             # 检查是否有数字存在
#             if numbers:
#                 # 获取第一个数字，即列表中的第一个元素
#                 first_x = numbers[0]
#
#                 # 将字符串转换为整数或浮点数
#                 try:
#                     x_value = int(first_x)  # 尝试转换为整数
#                 except ValueError:
#                     x_value = float(first_x)  # 如果失败，转换为浮点数
#
#                 print(f"Line {line_number}: The first x value is: {x_value}")
#             else:
#                 print(f"Line {line_number}: No numbers found in line.")
#
#
# # 调用函数
# read_data('output_0.txt')

# def make_conditional_counter(increment_interval):
#     count = 0
#     result = 0
#
#     def conditional_counter():
#         nonlocal count, result
#         if count % increment_interval == 0:
#             result += 1
#             count = 0  # 重置计数器
#         count += 1
#         return result
#
#     return conditional_counter
#
# # 创建一个每三次调用增加一次的计数器
# my_conditional_counter = make_conditional_counter(3)
#
# # 使用计数器
# print(my_conditional_counter())  # 输出: 1 (因为初始调用时result为0)
# print(my_conditional_counter())  # 输出: 1
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2

# def increment_every_third_call():
#     if not hasattr(increment_every_third_call, 'counter'):
#         increment_every_third_call.counter = 0  # 初始化计数器
#         increment_every_third_call.call_count = 0  # 初始化调用次数计数器
#
#
#     # 检查是否达到增加数值的条件（每三次）
#     if increment_every_third_call.call_count >= 2:
#         increment_every_third_call.counter += 1  # 增加数值
#         increment_every_third_call.call_count = 0  # 重置调用次数计数器
#
#     increment_every_third_call.call_count += 1  # 每次调用函数时，调用计数加1
#
#     return increment_every_third_call.counter  # 返回当前的数值
#
# # 使用函数
# print(increment_every_third_call())  # 输出: 1 (第一次调用，counter从0变为1)
# print(increment_every_third_call())  # 输出: 1 (第二次调用，不满足条件，counter不变)
# episode_number = increment_every_third_call()
# print(f"episode_number: ", episode_number)
# print(f"episode_number: ", episode_number)
# print(f"episode_number: ", episode_number)
# print(increment_every_third_call())  # 输出: 2 (第三次调用，满足条件，counter增加1)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)

def incrementing_function():
    if not hasattr(incrementing_function, 'counter'):
        incrementing_function.counter = -1  # 初始化计数器
    incrementing_function.counter += 1
    return incrementing_function.counter

# 使用函数
print(incrementing_function())  # 输出: 1
print(incrementing_function())  # 输出: 2



