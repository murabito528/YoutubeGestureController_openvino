#model_path = "_internal/models/common-sign-language-0002.xml"
model_path = "models/common-sign-language-0002.xml"

import cv2
import numpy as np
from openvino.runtime import Core

import configparser
import sys
import tkinter.messagebox as messagebox

core = Core()

# configの読み取り
config = configparser.ConfigParser()
#config.read('_internal/config.ini')
config.read('config.ini')
available_devices = core.available_devices
print("available_devices:\n", available_devices)
req_device = config.get("CONFIG", "use_device")
def_device = config.get("CONFIG", "default_device")
device = req_device if req_device in available_devices else def_device

# OpenVINOのセットアップ
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device)

if not device in available_devices:
    messagebox.showerror('Error', 'Need ' + req_device)
    sys.exit()

# カメラからの入力をキャプチャ
cap = cv2.VideoCapture(0)

gesture = "none"

acc = 0.8
gesture_ex = ["zero", "one", "two", "three", "four", "five", "good", "bad", "down", "up", "left", "right"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームのリサイズと前処理
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = resized_frame.transpose((2, 0, 1))  # HWC -> CHW

    # 3次元の入力形式に変換
    input_tensor = np.expand_dims(resized_frame, axis=0)  # [1, 3, 224, 224]に変換

    # モデルに入力して結果を取得
    results = compiled_model([input_tensor])[compiled_model.output(0)]

    # 結果の処理（例: ジェスチャー認識結果の表示）
    gesture = "none"
    for i in range(12):
        if results[0][i] > acc:
            gesture = gesture_ex[i]
            break

    cv2.putText(frame, "device:" + device, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "gesture:" + gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Camera Feed', frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
