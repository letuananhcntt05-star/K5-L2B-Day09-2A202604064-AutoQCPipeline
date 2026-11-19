"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401

NAME = "second_opinion"


def _label_boxes(ds, stem):
    """[(số dòng gốc, id lớp, (cx, cy, w, h))] cho các dòng parse được, id lớp hợp lệ và có diện tích."""
    out = []
    for ln, text in ds.read_label_lines(stem) or []:
        try:
            c, cx, cy, w, h = parse_yolo_line(text)
        except ValueError:
            continue
        if 0 <= c < len(ds.classes) and w > 0 and h > 0:
            out.append((ln, c, (cx, cy, w, h)))
    return out


def _best(pairs):
    """pairs = [(iou, nhãn)] -> (iou, nhãn) có IoU lớn nhất, hoặc None."""
    return max(pairs, key=lambda t: t[0]) if pairs else None


def check_image(ds, stem, rows, p):
    labels = _label_boxes(ds, stem)
    size = ds.image_size(stem)
    W, H = size if size else (None, None)
    name = ds.classes
    issues = []

    # --- (A) từng dự đoán đủ tin cậy và đủ lớn: thiếu nhãn / box lỏng / nhầm lớp
    for r in rows:
        pc, conf, box = int(r[0]), float(r[1]), tuple(r[2:6])
        if conf < p["conf"]:
            continue
        if W is not None and box[2] * W * box[3] * H < p["min_area_px"]:
            continue
        cands = [(iou_xywh(box, lb), (ln, lc)) for ln, lc, lb in labels]
        cands = [t for t in cands if t[0] >= p["iou_missing"]]
        pname = name[pc] if 0 <= pc < len(name) else str(pc)
        if not cands:
            issues.append(Issue(NAME, WARN, "POSSIBLE_MISSING", stem,
                                f"model thấy {pname} (conf {conf:.2f}) nhưng nhãn không có box nào ở đó",
                                extra={"conf": conf, "pred_class": pc, "box": [round(v, 4) for v in box]}))
            continue
        same = _best([t for t in cands if t[1][1] == pc])
        if same is not None:
            if same[0] < p["iou_loose"]:
                issues.append(Issue(NAME, WARN, "POSSIBLE_LOOSE_BOX", stem,
                                    f"box {pname} dòng {same[1][0]} lệch/lỏng so với model (IoU {same[0]:.2f})",
                                    extra={"conf": conf, "iou": round(same[0], 4), "label_line": same[1][0]}))
            continue
        other = _best(cands)
        if other[0] >= p["iou_match"]:
            lc = other[1][1]
            issues.append(Issue(NAME, WARN, "POSSIBLE_WRONG_CLASS", stem,
                                f"dòng {other[1][0]} gán {name[lc]} nhưng model nói {pname} (conf {conf:.2f}, IoU {other[0]:.2f})",
                                extra={"conf": conf, "iou": round(other[0], 4), "label_line": other[1][0],
                                       "label_class": lc, "pred_class": pc}))

    # --- (B) từng box nhãn: không khớp dự đoán nào (kể cả conf thấp, kể cả vật nhỏ) -> có thể là nhãn thừa
    loose = [tuple(r[2:6]) for r in rows if float(r[1]) >= p["fp_conf"]]
    for ln, lc, lb in labels:
        best = max((iou_xywh(lb, b) for b in loose), default=0.0)
        if best < p["iou_missing"]:
            issues.append(Issue(NAME, WARN, "POSSIBLE_FALSE_POS", stem,
                                f"box {name[lc]} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})", ln,
                                {"iou": round(best, 4)}))
    return issues


def run(ds, cfg, preds=None):
    p = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    issues = []
    for img in ds.image_files():
        stem = img.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue
        issues.extend(check_image(ds, stem, preds[stem], p))
    return issues
