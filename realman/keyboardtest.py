from pynput import keyboard
import threading
import time
import os

def on_press(key):
    print(f'按下: {key}')
    if key == keyboard.Key.space:  # 按 ESC 退出
        os._exit(0)

def on_release(key):
    print(f'释放: {key}')

# 运行监听器的线程
def start_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# **开启一个新线程运行监听器**
listener_thread = threading.Thread(target=start_listener, daemon=True)
listener_thread.start()

# **主线程仍然可以运行其他任务**
while True:
    print("主线程运行中...")
    time.sleep(2)
