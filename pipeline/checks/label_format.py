"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

NAME = "label_format"


def _overflow(cx, cy, w, h):
    """Mức tràn lớn nhất của 4 mép box ra ngoài [0, 1] (0 nếu nằm trong ảnh)."""
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    return max(0.0, -x1, x2 - 1.0, -y1, y2 - 1.0)


def check_file(ds, stem, cfg):
    p = cfg["label_format"]
    n_cls = len(ds.classes)
    lines = ds.read_label_lines(stem)
    if lines is None:                       # thiếu file nhãn: việc của missing_labels
        return []
    issues = []
    size = None
    seen = []                               # (số dòng, box) các dòng hợp lệ phía trước, để tìm DUP_BOX

    for ln, text in lines:
        try:
            c, cx, cy, w, h = parse_yolo_line(text)
        except ValueError as e:
            issues.append(Issue(NAME, ERROR, "BAD_FORMAT", stem, f"dòng sai định dạng ({e}): {text.strip()!r}", ln))
            continue
        if not 0 <= c < n_cls:
            issues.append(Issue(NAME, ERROR, "BAD_CLASS_ID", stem, f"id lớp {c} ngoài 0..{n_cls - 1}", ln, {"class_id": c}))
            continue
        if w <= 0 or h <= 0:
            issues.append(Issue(NAME, ERROR, "BOX_ZERO", stem, f"box không có diện tích (w={w}, h={h})", ln))
            continue

        over = _overflow(cx, cy, w, h)
        if over > p["overflow_tol"]:
            issues.append(Issue(NAME, ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn ra ngoài ảnh {over:.4f}", ln,
                                {"overflow": round(over, 6)}))

        box = (cx, cy, w, h)
        for prev_ln, (pc, pbox) in seen:
            if pc == c:
                iou = iou_xywh(box, pbox)
                if iou > p["dup_box_iou"]:
                    issues.append(Issue(NAME, ERROR, "DUP_BOX", stem, f"trùng box dòng {prev_ln} (IoU {iou:.3f})", ln,
                                        {"iou": round(iou, 4), "same_as": prev_ln}))
                    break
        seen.append((ln, (c, box)))

        if size is None:
            size = ds.image_size(stem) or ()
        if size:
            short = min(w * size[0], h * size[1])
            if short < p["min_box_px"]:
                issues.append(Issue(NAME, WARN, "BOX_TINY", stem, f"cạnh ngắn {short:.1f}px < {p['min_box_px']}px", ln,
                                    {"short_px": round(short, 2)}))
        if w * h > p["max_box_frac"]:
            issues.append(Issue(NAME, WARN, "BOX_HUGE", stem, f"box chiếm {w * h:.0%} ảnh", ln, {"area_frac": round(w * h, 4)}))
    return issues


def run(ds, cfg):
    issues = []
    for img in ds.image_files():
        issues.extend(check_file(ds, img.stem, cfg))
    return issues
