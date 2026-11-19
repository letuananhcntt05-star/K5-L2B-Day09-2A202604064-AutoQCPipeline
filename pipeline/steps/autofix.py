"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}          # + mọi mã bắt đầu bằng DUP_IMAGE_
PREC = 6                                                           # số chữ số thập phân khi ghi box đã cắt


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _rec(image, line, action, before, after):
    return dict(image=image, line="" if line is None else line, action=action, before=before, after=after)


def _clip(text):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu sau khi cắt box không còn diện tích."""
    c, cx, cy, w, h = parse_yolo_line(text)
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    nw, nh = round(x2 - x1, PREC), round(y2 - y1, PREC)
    if nw <= 0 or nh <= 0:
        return None
    ncx, ncy = round((x1 + x2) / 2, PREC), round((y1 + y2) / 2, PREC)
    return f"{c} {ncx:.{PREC}f} {ncy:.{PREC}f} {nw:.{PREC}f} {nh:.{PREC}f}"


def _remove_image(ds, stem):
    p = ds.image_path(stem)
    if p is None:                       # đã xoá (nhiều mã cùng trỏ một ảnh)
        return None
    name = p.name
    p.unlink()
    lp = ds.label_path(stem)
    had_label = lp.exists()
    if had_label:
        lp.unlink()
    ds._size_cache.pop(stem, None)
    return _rec(stem, None, "remove_image", name, "(đã xoá ảnh" + (" + nhãn)" if had_label else ")"))


def _rename_classes(ds, cfg):
    data = {}
    if ds.yaml_path.exists():
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
    before = ds.yaml_names()
    want = dict(enumerate(cfg["classes"]))
    if before == want:
        return None
    data["names"] = want
    ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _rec("data.yaml", None, "rename_classes", str([before[k] for k in sorted(before)]), str(cfg["classes"]))


def _fix_lines(ds, stem, codes_by_line):
    """Sửa một file nhãn: gom mọi dòng lỗi, quyết định theo số dòng GỐC, ghi lại một lần."""
    lp = ds.label_path(stem)
    if ds.image_path(stem) is None or not lp.exists():
        return []
    raw = lp.read_text(encoding="utf-8", errors="replace").splitlines()
    out, fixes = [], []
    for ln, text in enumerate(raw, 1):
        codes = codes_by_line.get(ln)
        if not codes or not text.strip():
            out.append(text)
            continue
        if codes & DROP_CODES:                              # xoá thắng cắt
            fixes.append(_rec(stem, ln, "drop_line", text.strip(), "(xoá dòng: " + ",".join(sorted(codes & DROP_CODES)) + ")"))
            continue
        if codes & CLIP_CODES:
            try:
                new = _clip(text)
            except ValueError:                              # dòng không parse được: không thể cắt, xoá
                new = None
            if new is None:
                fixes.append(_rec(stem, ln, "drop_line", text.strip(), "(xoá dòng: box nằm hẳn ngoài ảnh)"))
                continue
            fixes.append(_rec(stem, ln, "clip_box", text.strip(), new))
            out.append(new)
            continue
        out.append(text)
    if fixes:
        lp.write_text("".join(l + "\n" for l in out), encoding="utf-8")
    return fixes


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]
    fixes = []

    # 1) xoá ảnh trước: issue dòng của ảnh đã xoá sẽ tự bị bỏ qua ở bước 3
    for stem in sorted({i.image for i in errors if _is_remove(i.code)}):
        r = _remove_image(ds, stem)
        if r:
            fixes.append(r)

    # 2) tên lớp trong data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        r = _rename_classes(ds, cfg)
        if r:
            fixes.append(r)

    # 3) lỗi theo dòng: gom theo file -> {số dòng gốc: {mã}}
    by_file = defaultdict(lambda: defaultdict(set))
    for i in errors:
        if i.line is not None and i.code in DROP_CODES | CLIP_CODES:
            by_file[i.image][int(i.line)].add(i.code)
    for stem in sorted(by_file):
        fixes.extend(_fix_lines(ds, stem, by_file[stem]))

    # 4) ảnh chưa có nhãn: không bịa nhãn, chuyển cho người
    for stem in sorted({i.image for i in errors if i.code == "MISSING_LABEL_FILE"}):
        if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
            fixes.append(_rec(stem, None, "needs_label", "(không có file)", "chuyển người gán"))
    return fixes
