#model_path = "./intel/common-sign-language-0002/FP16-INT8/common-sign-language-0002.xml"
model_path = "_internal/models/common-sign-language-0002.xml"
device = "CPU"

import cv2
import numpy as np
from openvino.runtime import Core

import configparser
import sys
import tkinter.messagebox as messagebox

import pyautogui as pag
import re
import win32gui # type: ignore

def is_powerpoint_in_slideshow_mode(title):
    pattern = r'^PowerPoint .*? - .*?'
    return bool(re.match(pattern, title))

def match_title_format(title):
    pattern = r'^.*? - YouTube.*? - .*?$'
    return bool(re.match(pattern, title))

def get_active_window_title():
    activeWindowTitle = win32gui.GetWindowText(win32gui.GetForegroundWindow())
    return activeWindowTitle

class YoutubeController:
    def __init__(self):
        self.pausing = False

    def pause(self):
        active_window = get_active_window_title()
        print("現在のアクティブウィンドウのタイトル: ", active_window)
        if(match_title_format(active_window)):
            print("now -youtube")
            pag.hotkey('k')

    def skip(self):
        active_window = get_active_window_title()
        print("現在のアクティブウィンドウのタイトル: ", active_window)
        if(match_title_format(active_window)):
            print("now -youtube")
            pag.hotkey('l')
        return

    def rewind(self):
        active_window = get_active_window_title()
        print("現在のアクティブウィンドウのタイトル: ", active_window)
        if(match_title_format(active_window)):
            print("now -youtube")
            pag.hotkey('j')
        return
    
    def next_slide(self):
        active_window = get_active_window_title()
        #print("現在のアクティブウィンドウのタイトル: ", active_window)
        if is_powerpoint_in_slideshow_mode(active_window):
            print("now -powerpoint")
            pag.hotkey('down')
        return
    
    def back_slide(self):
        active_window = get_active_window_title()
        print("現在のアクティブウィンドウのタイトル: ", active_window)
        if is_powerpoint_in_slideshow_mode(active_window):
            print("now -powerpoint")
            pag.hotkey('up')
        return


core = Core()

ctrl = YoutubeController()

#configの読み取り
config = configparser.ConfigParser()
#config.read('config.ini')
config.read('_internal/config.ini')
available_devices = core.available_devices
print("available_devices:\n",available_devices)
req_device = config.get("CONFIG","use_device")
def_device = config.get("CONFIG","default_device")
device = req_device if req_device in available_devices else def_device

#OpenVINOのセットアップ
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device)

if not device in available_devices:
    messagebox.showerror('Error', 'Need '+ req_device)
    sys.exit()

# カメラからの入力をキャプチャ
cap = cv2.VideoCapture(0)

# フレームを保持するためのリスト
frames = []

gesture = "none"
last_gesture = "none"
acc=0.8
gesture_ex=["zero","one","two","three","four","five","good","bad","down","up","left","right"]

tm = cv2.TickMeter()
tm.start()
frame_count = 0
max_count = 10
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        frame_count = 0

    # フレームのリサイズと前処理
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = resized_frame.transpose((2, 0, 1))  # HWC -> CHW
    frames.append(resized_frame)

    # 8フレーム集まったらモデルに入力
    if len(frames) == 8:
        input_tensor = np.expand_dims(np.stack(frames, axis=1), axis=0)  # [1, 3, 8, 224, 224]に変換

        # モデルに入力して結果を取得
        results = compiled_model([input_tensor])[compiled_model.output(0)]
        
        # 結果の処理（例: ジェスチャー認識結果の表示）
        #print(results)

        # フレームリストをリセット
        frames = []
        gesture = "none"
        for i in range(12):
            if results[0][i]>acc:
                gesture = gesture_ex[i]
                break

        if gesture != last_gesture:
            if gesture == "five":
                print("pause")
                ctrl.pause()
                ctrl.next_slide()
            elif gesture == "two":
                print("rewind")
                ctrl.rewind()
            elif gesture == "good":
                print("skip")
                ctrl.skip()
            elif gesture == "three":
                print("back_slide")
                ctrl.back_slide()
            last_gesture = gesture
            if gesture == "two" or gesture == "good":
                last_gesture = "none"

        

    #cv2.putText(frame, "device:"+device, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "gesture:"+gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('GestureController', frame)
    frame_count+=1
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or cv2.getWindowProperty("GestureController", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
