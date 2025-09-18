pip install ultralytics

import cv2
from ultralytics import YOLO, solutions
import os
import csv

# 1. MP4動画ファイルのパスを指定
input_mp4_path = "正面入口.mp4"

# 2. 動画ファイルの読み込み
cap = cv2.VideoCapture(input_mp4_path)
if not cap.isOpened():
    print("動画ファイルを読み込めませんでした。パスを確認してください。")
    exit()

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# 3. 処理済み動画の出力設定
filename_without_ext = os.path.splitext(input_mp4_path)[0]
output_mp4_path = f"{filename_without_ext}_processed.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (w, h))

# 4. カウント領域の定義（Line）
region_points = [(600, 460), (300, 1670)]

# 5. ObjectCounterクラスの初期化（表示用）
counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="yolo11n.pt",  # 人検出に特化
    classes=[0],
    show_in=True,
    show_out=True,
)

# 6. YOLOモデルの初期化（CSV用）
model = YOLO("yolo11n.pt")

# 7. CSVファイルの準備
csv_path = f"{filename_without_ext}_detections.csv"
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["frame", "detections"])  # detections列にまとめて書く

# 8. 動画フレームの処理
frame_idx = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("動画の処理が完了しました。")
        break

    # --- カウントライン処理 ---
    _ = counter(im0)
    im_bgr = counter.annotator.im

    # --- YOLOによる検出（CSV用） ---
    results = model.predict(im0, classes=[0], verbose=False)
    detections = []

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            cx = x1 + width / 2
            cy = y1 + height / 2
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append(f"{cls}:{conf:.2f}:{cx:.1f}:{cy:.1f}:{width:.1f}:{height:.1f}")

    # --- CSVに書き込み ---
    writer.writerow([frame_idx, ";".join(detections)])

    # --- 出力動画保存と表示 ---
    video_writer.write(im_bgr)
    resized_im_bgr = cv2.resize(im_bgr, (int(w / 2), int(h / 2)))
    cv2.imshow("Object Counting", resized_im_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# 9. リソース解放
cap.release()
video_writer.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"検出結果を {csv_path} に保存しました。")
