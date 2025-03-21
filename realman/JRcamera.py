import cv2


def count_cameras(max_test=10):
    """检测电脑上可用的摄像头索引"""
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


def capture_camera(camera_index=0):
    """打开指定索引的摄像头"""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return

    print(f"正在使用摄像头 {camera_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        cv2.imshow(f"Camera {camera_index}", frame)  # 显示图像

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cameras = count_cameras()

    if not cameras:
        print("未检测到摄像头")
    else:
        print(f"检测到的摄像头索引: {cameras}")

        # 如果有多个摄像头，用户选择一个
        if len(cameras) > 1:
            camera_index = int(input(f"请输入要打开的摄像头索引 {cameras}: "))
        else:
            camera_index = cameras[0]  # 只有一个摄像头时，自动选择

        capture_camera(camera_index)
