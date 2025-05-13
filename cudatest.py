import torch
print("CUDA 可用：", torch.cuda.is_available())
print("CUDA 版本：", torch.version.cuda)
print("设备名：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")
