import os
import cv2
import numpy as np


def _as_numpy(value):
    """Convert NumPy arrays or Torch tensors to NumPy without requiring Torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def normalized_cxcywh_to_xyxy(boxes, width, height):
    """Convert normalized ``cx, cy, w, h`` boxes into pixel ``x1, y1, x2, y2``."""
    boxes = _as_numpy(boxes)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, box_width, box_height = boxes.T
    return np.column_stack((
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    ))


def _clip_xyxy(box, width, height):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return (
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, 0, width - 1)),
        int(np.clip(y2, 0, height - 1)),
    )


def save_detection_visualization(image_path, image_id, prediction, gt_boxes,
                                 gt_labels, save_dir, score_thresh=0.3) -> bool:
    """Save separate ground-truth and prediction annotations for one image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Could not read image_id {image_id} at path: {image_path}")
        return False

    gt_dir = os.path.join(save_dir, "gt")
    pred_dir = os.path.join(save_dir, "pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    height, width = image.shape[:2]
    gt_image = image.copy()
    for box, label in zip(normalized_cxcywh_to_xyxy(gt_boxes, width, height),
                          _as_numpy(gt_labels).reshape(-1)):
        x1, y1, x2, y2 = _clip_xyxy(box, width, height)
        color = get_color_by_label(label)
        cv2.rectangle(gt_image, (x1, y1), (x2, y2), color, 2)
        text_y = min(height - 1, max(12, y1 - 5))
        cv2.putText(gt_image, f"GT:{int(label)}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    pred_image = image.copy()
    boxes = _as_numpy(prediction.get("boxes", [])).reshape(-1, 4)
    scores = _as_numpy(prediction.get("scores", [])).reshape(-1)
    labels = _as_numpy(prediction.get("labels", [])).reshape(-1)
    for box, score, label in zip(boxes, scores, labels):
        score = float(score)
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = _clip_xyxy(box, width, height)
        color = get_color_by_label(label)
        cv2.rectangle(pred_image, (x1, y1), (x2, y2), color, 2)
        text_y = min(height - 1, max(12, y1 + 15))
        cv2.putText(pred_image, f"{int(label)}:{score:.2f}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    filename = f"{image_id}_{os.path.basename(image_path)}"
    gt_path = os.path.join(gt_dir, filename)
    pred_path = os.path.join(pred_dir, filename)
    gt_written = cv2.imwrite(gt_path, gt_image)
    pred_written = cv2.imwrite(pred_path, pred_image)
    if not gt_written or not pred_written:
        print(f"[WARN] Failed to write visualization for image_id {image_id} at path: {image_path}")
        return False
    return True


def compute_per_image_ap(coco_eval):
    """
    从 COCOeval 中提取每张图的 AP（近似）
    """
    img_ap = {}

    eval_imgs = coco_eval.evalImgs
    if eval_imgs is None:
        return img_ap

    for e in eval_imgs:
        if e is None:
            continue

        img_id = e['image_id']
        dtMatches = e['dtMatches']  # [TxD]
        dtScores = e['dtScores']

        if len(dtScores) == 0:
            continue

        # 简化 precision 计算
        matches = (dtMatches > 0).sum()
        precision = matches / len(dtScores)

        if img_id not in img_ap:
            img_ap[img_id] = []

        img_ap[img_id].append(precision)

    # 求平均
    img_ap = {k: np.mean(v) for k, v in img_ap.items()}
    return img_ap


import os
import cv2
import numpy as np


def get_color_by_label(label):
    """根据类别返回颜色"""
    if int(label) == 0:
        return (0, 0, 255)   # 红
    elif int(label) == 1:
        return (255, 0, 0)   # 蓝
    else:
        return (0, 255, 0)   # 其他类别 → 绿


def visualize_topk_split_gt_pred(dataset, predictions, img_ap, save_dir,
                                topk=5, score_thresh=0.3):

    gt_dir = os.path.join(save_dir, "gt")
    pred_dir = os.path.join(save_dir, "pred")

    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    topk_imgs = sorted(img_ap.items(), key=lambda x: x[1], reverse=True)[:topk]

    for img_id, ap in topk_imgs:

        pred_entry = predictions[img_id]

        img_path = pred_entry.get("img_path", None)
        if img_path is None:
            print(f"[WARN] No image path for img_id {img_id}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] image not found: {img_path}")
            continue

        h, w = img.shape[:2]

        # =========================
        # 1️⃣ GT 图
        # =========================
        img_gt = img.copy()

        gt_boxes = pred_entry.get("gt_boxes", [])
        gt_labels = pred_entry.get("gt_labels", [])

        for box, label in zip(gt_boxes, gt_labels):
            box = np.array(box)

            # 🔥 关键：如果是归一化 → 转像素
            if box.max() <= 1.0:
                cx, cy, cw, ch = box
                cx = cx * w
                cy = cy * h
                cw = cw * w
                ch = ch * h
                x1 = cx - cw / 2
                y1 = cy - ch / 2
                x2 = cx + cw / 2
                y2 = cy + ch / 2
            else:
                x1, y1, x2, y2 = box

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            color = get_color_by_label(label)

            cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_gt,
                        f"GT:{int(label)}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1)

        gt_save_path = os.path.join(gt_dir, f"{img_id}_ap{ap:.3f}_gt.jpg")
        cv2.imwrite(gt_save_path, img_gt)

        # =========================
        # 2️⃣ Pred 图
        # =========================
        img_pred = img.copy()

        boxes = pred_entry['boxes']
        scores = pred_entry['scores']
        labels = pred_entry['labels']

        # 排序
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

        for box, score, label in zip(boxes, scores, labels):
            if score < score_thresh:
                continue

            x1, y1, x2, y2 = map(int, box)

            color = get_color_by_label(label)

            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred,
                        f"{int(label)}:{score:.2f}",
                        (x1, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1)

        pred_save_path = os.path.join(pred_dir, f"{img_id}_ap{ap:.3f}_pred.jpg")
        cv2.imwrite(pred_save_path, img_pred)

    print(f"[INFO] Done!")
    print(f"  GT   → {gt_dir}")
    print(f"  Pred → {pred_dir}")
