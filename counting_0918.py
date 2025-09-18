pip install ultralytics

import cv2
from ultralytics import solutions
import os

# 1. MP4動画ファイルのパスを指定
input_mp4_path = "正面入口.mp4"

# 2. 動画ファイルの読み込み
cap = cv2.VideoCapture(input_mp4_path)
if not cap.isOpened():
    print("動画ファイルを読み込めませんでした。パスを確認してください。")
    exit()

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 3. 処理済み動画の出力設定
filename_without_ext = os.path.splitext(input_mp4_path)[0]
output_mp4_path = f"{filename_without_ext}_processed.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (w, h))

# 4. カウント領域の定義（Line）
region_points = [(600, 460), (300, 1670)]#[(始点のX座標600, 始点のY座標460), (終点のX座標300, 終点のY座標1670)]

# 5. ObjectCounterクラスの初期化
counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="yolo11n.pt",#yolov8nとyolo11nというモデルがあるが、yolo11nの方が人の検出に特化している（処理中も歩幅に合わせて長方形が変形する）
    classes=[0],
    show_in=True,
    show_out=True,
)

# 6. 動画フレームの処理
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("動画の処理が完了しました。")
        break
    
    _ = counter(im0)
    im_bgr = counter.annotator.im

    video_writer.write(im_bgr)
    
    resized_im_bgr = cv2.resize(im_bgr, (int(w / 2), int(h / 2)))
    cv2.imshow("Object Counting", resized_im_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()