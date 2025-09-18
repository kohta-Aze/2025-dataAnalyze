#pip install ultralytics

# save as combined_count_and_csv.py
import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO, solutions

# ------------------ 設定 ------------------
INPUT_VIDEO = "正面入口.mp4"
MODEL_PATH = "yolo11n.pt"

# 検出／トラッキングパラメータ
CONF_TH = 0.30
IOU_SUPPRESS = 0.6
IMGSZ = 640
MAX_TRACK_LOST = 15
MAX_MATCH_DIST = 140

# カウントライン（従来と同じ）
region_points = [(600, 460), (300, 1670)]  # (x1,y1), (x2,y2)

# クロス判定の安定化
MIN_SIDE_DIST = 2.0            # ライン上（距離が小さい）は無視
MIN_FRAMES_BETWEEN_CROSSES = 5 # 同一トラックの連続カウント抑止

# 出力ファイル
OUT_VIDEO_SUFFIX = "_combined_out.mp4"
CSV_NAME_SUFFIX = "_combined_detections.csv"
# -------------------------------------------

# ----- ユーティリティ関数 -----
def signed_cross(pt, a, b):
    """2D 線分 (a->b) に対する点 pt の符号付き交差量（クロス積）。
       正負で左右が判る。長さ正規化は不要（符号だけ使う）。"""
    return (pt[0] - a[0]) * (b[1] - a[1]) - (pt[1] - a[1]) * (b[0] - a[0])

def side_of_line(pt, a, b, eps=MIN_SIDE_DIST):
    """点がラインのどちら側か。ラインに近ければ 0 を返す."""
    s = signed_cross(pt, a, b)
    if abs(s) < eps:
        return 0
    return 1 if s > 0 else -1

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_simple(dets, iou_thresh=0.6):
    if len(dets) == 0:
        return []
    dets_sorted = sorted(dets, key=lambda d: d['conf'], reverse=True)
    keep = []
    for d in dets_sorted:
        skip = False
        for k in keep:
            if iou_xyxy(d['xyxy'], k['xyxy']) > iou_thresh:
                skip = True
                break
        if not skip:
            keep.append(d)
    return keep

# ---- 簡易トラッカー（重心距離でマッチ） ----
class SimpleTracker:
    def __init__(self, max_lost=15, max_dist=140):
        self.next_id = 1
        self.tracks = {}  # id -> {center, bbox, last_seen, lost, conf}
        self.max_lost = max_lost
        self.max_dist = max_dist

    def update(self, detections, frame_idx):
        assigned = []
        # 新規トラッキング開始（最初のフレーム等）
        if len(self.tracks) == 0:
            for d in detections:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    'center': (d['cx'], d['cy']),
                    'bbox': d['xyxy'],
                    'last_seen': frame_idx,
                    'lost': 0,
                    'conf': d['conf']
                }
                assigned.append((tid, d))
            return assigned

        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[t]['center'] for t in track_ids])
        det_centers = np.array([[d['cx'], d['cy']] for d in detections]) if len(detections) > 0 else np.zeros((0,2))

        if det_centers.shape[0] == 0:
            # 検出0：全トラックを lost+
            for tid in track_ids:
                self.tracks[tid]['lost'] += 1
            remove = [tid for tid,v in self.tracks.items() if v['lost'] > self.max_lost]
            for tid in remove: del self.tracks[tid]
            return []

        dists = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=2)  # (T,D)
        assigned_tracks = set(); assigned_dets = set()
        # greedy 最小距離マッチ
        while True:
            ti, di = np.unravel_index(np.argmin(dists), dists.shape)
            minval = dists[ti, di]
            if minval > self.max_dist or minval == 1e6:
                break
            tid = track_ids[ti]
            if tid in assigned_tracks or di in assigned_dets:
                dists[ti, di] = 1e6
                if np.min(dists) > self.max_dist:
                    break
                continue
            d = detections[di]
            assigned.append((tid, d))
            # update
            self.tracks[tid]['center'] = (d['cx'], d['cy'])
            self.tracks[tid]['bbox'] = d['xyxy']
            self.tracks[tid]['last_seen'] = frame_idx
            self.tracks[tid]['lost'] = 0
            self.tracks[tid]['conf'] = d['conf']
            assigned_tracks.add(tid); assigned_dets.add(di)
            dists[ti, :] = 1e6; dists[:, di] = 1e6
            if np.min(dists) > self.max_dist:
                break

        # unmatched detections -> new tracks
        for di, d in enumerate(detections):
            if di in assigned_dets: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {
                'center': (d['cx'], d['cy']),
                'bbox': d['xyxy'],
                'last_seen': frame_idx,
                'lost': 0,
                'conf': d['conf']
            }
            assigned.append((tid, d))

        # unmatched tracks -> lost++
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['last_seen'] != frame_idx:
                self.tracks[tid]['lost'] += 1
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]

        return assigned

# -------------------- メイン --------------------
if not os.path.exists(INPUT_VIDEO):
    raise SystemExit("入力動画が見つかりません: " + INPUT_VIDEO)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit("動画が開けません")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

base = os.path.splitext(INPUT_VIDEO)[0]
out_video_path = base + OUT_VIDEO_SUFFIX
csv_path = base + CSV_NAME_SUFFIX

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_video_path, fourcc, FPS, (W, H))

# モデルとObjectCounter（描画・従来ラインはObjectCounterに任せる）
model = YOLO(MODEL_PATH)
counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model=MODEL_PATH,
    classes=[0],
    show_in=True, show_out=True
)

# CSV作成
csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "id", "class", "conf", "cx", "cy", "w", "h"])

# tracker と tracks_info（クロス判定用）
tracker = SimpleTracker(max_lost=MAX_TRACK_LOST, max_dist=MAX_MATCH_DIST)
tracks_info = {}  # tid -> {'prev_center':(x,y), 'prev_side': int, 'last_cross_frame': int}

# カウント
counts = {'in': 0, 'out': 0}

frame_idx = 0
print("処理開始:", INPUT_VIDEO)
while True:
    ret, im0 = cap.read()
    if not ret:
        break

    # 1) ObjectCounter による描画（従来の見た目）
    _ = counter(im0)
    im_bgr = counter.annotator.im  # 描画済み画像

    # 2) YOLOによる検出（CSV・トラッキング用）
    results = model.predict(im0, imgsz=IMGSZ, conf=CONF_TH, classes=[0], verbose=False)
    res0 = results[0]
    detections = []
    if hasattr(res0, 'boxes') and len(res0.boxes) > 0:
        boxes_xyxy = res0.boxes.xyxy.cpu().numpy()
        boxes_conf = res0.boxes.conf.cpu().numpy()
        boxes_cls = res0.boxes.cls.cpu().numpy()
        for i in range(len(boxes_xyxy)):
            x1,y1,x2,y2 = boxes_xyxy[i].tolist()
            conf = float(boxes_conf[i]); cls = int(boxes_cls[i])
            if conf < CONF_TH: continue
            w_box = x2 - x1; h_box = y2 - y1
            cx = x1 + w_box / 2.0; cy = y1 + h_box / 2.0
            detections.append({
                'cx': float(cx), 'cy': float(cy), 'w': float(w_box), 'h': float(h_box),
                'conf': float(conf), 'cls': int(cls), 'xyxy': (float(x1),float(y1),float(x2),float(y2))
            })

    # 3) NMS
    detections = nms_simple(detections, iou_thresh=IOU_SUPPRESS)

    # 4) トラッカー更新（ID付与）
    assigned = tracker.update(detections, frame_idx)

    # 5) CSV出力 & 自前のカウント判定（ライン交差）
    #    とともに、自前で描画（ObjectCounterの描画は残すが、
    #    CSV と ID ラベルの確実性を保証するためこちらでラベルも描画）
    vis = im_bgr.copy()
    # ライン（可視化）
    cv2.line(vis, region_points[0], region_points[1], (255,0,0), 2)

    # line vector and normal
    a = region_points[0]; b = region_points[1]
    dvec = (b[0]-a[0], b[1]-a[1])
    normal = (-dvec[1], dvec[0])  # 法線ベクトル

    for tid, d in assigned:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        cx, cy = d['cx'], d['cy']
        conf = d['conf']

        # CSV に1行
        csv_writer.writerow([frame_idx, tid, "person", f"{conf:.2f}",
                             f"{cx:.1f}", f"{cy:.1f}", f"{d['w']:.1f}", f"{d['h']:.1f}"])

        # 描画（自分のID/信頼度）
        label = f"{tid} person {conf:.2f}"
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # クロス判定
        prev = tracks_info.get(tid, {})
        prev_center = prev.get('prev_center', (cx, cy))
        prev_side = prev.get('prev_side', side_of_line(prev_center, a, b))
        curr_side = side_of_line((cx, cy), a, b)
        last_cross = prev.get('last_cross_frame', -9999)

        # side==0 はライン近傍（無視）
        if prev_side == 0:
            prev_side = side_of_line(prev_center, a, b)  # recalc just in case

        if prev_side != 0 and curr_side != 0 and prev_side != curr_side:
            # 交差（符号変化）が起きた
            # ただし短時間での多重カウントを避ける
            if frame_idx - last_cross >= MIN_FRAMES_BETWEEN_CROSSES:
                # movement vector
                mv = (cx - prev_center[0], cy - prev_center[1])
                # dot with normal
                dot = mv[0]*normal[0] + mv[1]*normal[1]
                if dot > 0:
                    counts['in'] += 1
                    cross_label = "IN"
                else:
                    counts['out'] += 1
                    cross_label = "OUT"
                # 更新
                prev['last_cross_frame'] = frame_idx
                # optional: draw arrow indicating crossing direction
                cv2.putText(vis, f"{cross_label}#{counts[cross_label.lower()]}", (int(cx+10), int(cy+10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # update tracks_info
        tracks_info[tid] = {'prev_center': (cx, cy), 'prev_side': curr_side,
                            'last_cross_frame': prev.get('last_cross_frame', -9999)}

    # 6) オーバーレイで自前カウントを表示
    cv2.putText(vis, f"MyCount IN:{counts['in']} OUT:{counts['out']}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    # 7) 書き出しと表示
    writer.write(vis)
    if frame_idx % 100 == 0:
        print(f"frame {frame_idx}, detections {len(assigned)}, my IN {counts['in']}, OUT {counts['out']}")
    # cv2.imshow("combined", cv2.resize(vis, (int(W/2), int(H/2))))
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

    frame_idx += 1

# cleanup
csv_file.close()
writer.release()
cap.release()
print("完了。CSV:", csv_path, "動画:", out_video_path)