import numpy as np
from scipy.interpolate import CubicSpline
import time
import threading
from queue import Queue


class SmoothInterpolator:
    def __init__(self, dim=7, policy_freq=20, control_freq=100, v_max=50.0, a_max=None):
        self.dim = dim
        self.control_freq = control_freq
        self.v_max = np.ones(dim) * v_max  # max velocity per joint (deg/s)
        self.a_max = np.ones(dim) * a_max if a_max is not None else None

        self.trajectory = np.zeros((1, dim))
        self.current_index = 0
        self.last_vel = np.zeros(dim)
        self.last_pos = np.zeros(dim)

        self.lock = threading.Lock()
        self.new_target_queue = Queue()

    def compute_interp_time(self, start_pos, target_pos):
        start_pos = np.array(start_pos)
        target_pos = np.array(target_pos)

        delta = np.abs(target_pos - start_pos)
        T_joint = delta / self.v_max
        T_min = np.max(T_joint)

        if self.a_max is not None:
            T_acc = np.sqrt(2 * delta / self.a_max)
            T_min = max(T_min, np.max(T_acc))

        N = max(2, int(np.ceil(T_min * self.control_freq)))
        T = N / self.control_freq
        print(f"Interpolation time: {T:.3f}s, Steps: {N}, Delta: {delta}, Start pos: {start_pos}, Target pos: {target_pos}")

        return T, N

    def update_target(self, new_target):
        # 这个函数用于计算新的目标点的插值轨迹，并将轨迹存储在self.trajectory中
        with self.lock:
            # Determine current position and velocity
            if self.current_index == 0 and np.allclose(self.trajectory[0], 0.0):
                print("First interpolation, using last_pos and last_vel")
                current_pos = self.last_pos
                current_vel = self.last_vel
            elif self.current_index < len(self.trajectory):
                current_pos = self.trajectory[self.current_index]
                if self.current_index > 0:
                    prev_pos = self.trajectory[self.current_index - 1]
                    dt = 1.0 / self.control_freq
                    current_vel = (current_pos - prev_pos) / dt
                else:
                    current_vel = self.last_vel
            else:
                current_pos = self.last_pos
                current_vel = self.last_vel

            # Compute interpolation time and steps
            T, N = self.compute_interp_time(current_pos, new_target)
            t = np.linspace(0, T, N)

            trajectory = []
            for i in range(self.dim):
                cs = CubicSpline([0, T], [current_pos[i], new_target[i]],
                                 bc_type=((1, current_vel[i]), (1, 0.0)))
                traj_i = cs(t)
                self.last_vel[i] = cs(T, 1)
                trajectory.append(traj_i)

            self.trajectory = np.stack(trajectory, axis=1)
            self.current_index = 0
            self.last_pos = new_target

    def get_next_point(self):
        # 这个函数用于获取下一个插值点，从self.trajectory中返回
        with self.lock:
            if self.current_index < len(self.trajectory):
                point = self.trajectory[self.current_index]
                self.current_index += 1
                return point
            else:
                return self.last_pos

    def controller_loop(self):
        # 这个函数用于控制循环，定期从插值轨迹中获取下一个点并发送给机器人
        rate = 1.0 / self.control_freq
        while True:
            if not self.new_target_queue.empty():
                new_target = self.new_target_queue.get()
                self.update_target(new_target)

            point = self.get_next_point()
            self.send_to_robot(point)
            time.sleep(rate)

    def send_to_robot(self, point):
        print("Sending to robot:", point)
        # 真实控制逻辑放这里

    def receive_new_target(self, target):
        self.new_target_queue.put(target)


# 示例代码
if __name__ == "__main__":
    interpolator = SmoothInterpolator(dim=16, policy_freq=20, control_freq=100)

    # 启动控制线程
    control_thread = threading.Thread(target=interpolator.controller_loop, daemon=True)
    control_thread.start()

    # 模拟策略不断给出目标点
    for step in range(10):
        next_target = np.random.uniform(-1.0, 1.0, size=16) * 45  # 关节角范围
        print(f"Step {step}: new target {next_target}")
        interpolator.receive_new_target(next_target)
        time.sleep(1.0 / interpolator.policy_freq)

    time.sleep(2)  # 等待线程跑完剩余插值