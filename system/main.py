import cv2
import threading
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.predict_darts import predict_image, visualize
from realtime_infer import paste_patch_on_dartboard

THIS_FOLDER_DIR = os.path.abspath(os.path.dirname(__file__))
img_source = cv2.imread(os.path.abspath(os.path.join(THIS_FOLDER_DIR, "result_labeled.jpg"))) 

# ========== 全局状态 ==========
frame = None                    # 当前帧图像
frame_lock = threading.Lock()  # 用于多线程安全访问帧
capture_thread = None          # 摄像头线程
running = True                 # 主线程运行状态
roi_box = None                 # 用户选定的 ROI 区域
drawing = False                # 是否正在拖动鼠标框选
ref_pt = []                    # 框选的两个点 [左上, 右下]

# ========== 摄像头帧采集线程 ==========
def capture_frames(cap):
    global frame, running
    while running:
        ret, new_frame = cap.read()
        if not ret:
            print("⚠️ 摄像头读取失败")
            break
        with frame_lock:
            frame = new_frame.copy()
        time.sleep(0.01)

# ========== 鼠标事件回调 ==========
def mouse_callback(event, x, y, flags, param):
    global drawing, ref_pt, roi_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ref_pt = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if len(ref_pt) == 1:
            ref_pt.append((x, y))  # 首次添加第二点
        else:
            ref_pt[1] = (x, y)     # 实时更新第二点

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(ref_pt) == 2:
            x1, y1 = ref_pt[0]
            x2, y2 = ref_pt[1]
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # 简单过滤无效框
                roi_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                print(f"✅ ROI 选定成功：{roi_box}")
            else:
                print("⚠️ ROI 框太小，已忽略")

# ========== 主逻辑入口 ==========
def run_camera():
    global running, roi_box
 
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 设置后端为 cv2.CAP_DSHOW（DirectShow），更稳定些
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("🎯 启动成功：按 [r] 框选，按 [p] 预测，按 [q] 退出")

    # 启动摄像头线程
    global capture_thread
    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    capture_thread.start()

    cv2.namedWindow("Live")
    cv2.setMouseCallback("Live", mouse_callback)  # 选取ROI
    

    while True:
        with frame_lock:
            if frame is None:
                continue
            disp = frame.copy()

        # 绘制提示文字
        cv2.putText(disp, "Press [r]=ROI  [p]=Predict  [q]=Quit  [s]=Save",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 动态绘制 ROI 框（拖动或已选定）
        if drawing and len(ref_pt) == 2:
            cv2.rectangle(disp, ref_pt[0], ref_pt[1], (0, 255, 0), 2)
        elif roi_box:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Live", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("🛑 程序已退出")
            running = False
            break

        elif key == ord("r"):
            print("🔍 请输入鼠标拖动 ROI 框选区域")
            roi_box = None
            ref_pt.clear()

        elif key == ord("p"):
            with frame_lock:
                cur_frame = frame.copy()

            if roi_box:
                x1, y1, x2, y2 = roi_box
                crop = cur_frame[y1:y2, x1:x2]
                print(f"🧠 正在预测 ROI 区域：{roi_box}")
            else:
                crop = cur_frame
                print("🧠 正在预测整张图像")

            try:
                # crop = paste_patch_on_dartboard(crop.copy())
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop = cv2.merge([crop, crop, crop])
                xy, processed_img = predict_image(crop.copy())
                print("📍 预测坐标：", xy)
                vis = visualize(processed_img.copy(), xy)
                cv2.imshow("Result", vis)
            except Exception as e:
                print("❌ 预测失败：", e)

        elif key == ord("s"):
          filename = time.strftime("predict_%Y%m%d_%H%M%S.png")
          if 'vis' in locals():
              save_path = os.path.abspath(os.path.join(THIS_FOLDER_DIR, filename))
              cv2.imwrite(save_path, vis)
              print(f"💾 已保存预测结果图：{filename}")
          else:
              print("⚠️ 当前还没有可保存的预测图")

        elif key == ord("c"):
            with frame_lock:
                cur_frame = frame.copy()
            
            # filename 
            filename = time.strftime("data_%Y%m%d_%H%M%S.png")
            save_path = os.path.abspath(os.path.join(THIS_FOLDER_DIR, filename))
            if roi_box:
                x1, y1, x2, y2 = roi_box
                crop = cur_frame[y1:y2, x1:x2]
                cv2.imwrite(save_path, crop)
                print(f"💾 正在保存 ROI 区域：{roi_box}")
            else:
                crop = cur_frame
                cv2.imwrite(save_path, crop)
                print("💾 正在保存 整张图像")


    cap.release()
    cv2.destroyAllWindows()
    capture_thread.join()

# ========== 程序入口 ==========
if __name__ == "__main__":
    run_camera()
