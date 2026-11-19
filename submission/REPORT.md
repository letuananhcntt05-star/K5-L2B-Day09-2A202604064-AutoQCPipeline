# Báo cáo Day 09 — Lê Tuấn Anh, 2A202604064

## 1. Kết luận trước
Pipeline kết luận **Hold** (mã thoát 10). Vào 224 ảnh; cửa nhận loại 22 ảnh (hỏng/nhỏ/trùng/rò rỉ test), còn **202 ảnh** qua QC.
Sau sửa tự động + người duyệt (80/80 ảnh ngân sách), **không còn lỗi bắt buộc** (0 ERROR trong `07_error_list_v3`), nhưng
**38/202 ảnh (18,8%)** vẫn còn cảnh báo chưa được người xem, ≥ ngưỡng 10% → Hold. Dataset 202 ảnh được xuất ở `submission/export/clean_v1/`
dưới dạng **bản nháp** (manifest ghi `decision=Hold`), chưa phải bản phát hành.

## 2. Lỗi và phân việc
Nguồn: `01_intake` (mẫu số 224 ảnh) và `02_error_list_v1` (mẫu số 202 ảnh). Máy sửa 35 thay đổi (`03_fixes.csv`: 20 `drop_line`, 8 `clip_box`, 1 `rename_classes`, 6 `needs_label`).

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR | 8 / 4 ảnh (trên 224) | máy (loại ở cửa nhận) | trùng md5 / phash ≤ 12: giữ lại làm lệch phân bố, tăng giả số ảnh |
| LEAK_TEST | 5 ảnh (trên 224) | máy (loại) | trùng ảnh test/gold_val → làm đẹp giả mAP trên test |
| CORRUPT_IMAGE / BAD_SIZE | 2 / 3 ảnh (trên 224) | máy (loại) | không mở được / cạnh ngắn < 250 px |
| BAD_CLASS_NAME | 1 file (2 tên sai: `Person`, `Bus`) | máy (`rename_classes`) | quy tắc: names phải đúng `cfg.classes` |
| BAD_FORMAT / BAD_CLASS_ID / BOX_ZERO / DUP_BOX | 3 / 6 / 4 / 7 dòng | máy (`drop_line`) | dòng không dùng được (nan, id 93, w=0, box lặp IoU>0,9): xoá an toàn |
| BOX_OUT_OF_IMAGE | 8 dòng / 8 ảnh | máy (`clip_box`) | cắt về [0,1], giữ lớp; mọi thay đổi ghi trước → sau |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | máy không bịa nhãn; không tạo file rỗng (rỗng = ảnh nền) |
| POSSIBLE_MISSING | 53 / 52 ảnh | người | model lớn thấy vật mà nhãn không có — chỉ cảnh báo |
| POSSIBLE_WRONG_CLASS | 18 / 18 ảnh | người | cùng vị trí (IoU ≥ 0,5) nhưng khác lớp |
| POSSIBLE_LOOSE_BOX | 28 / 24 ảnh | người | đúng lớp nhưng IoU < 0,5 |
| POSSIBLE_FALSE_POS | 25 / 24 ảnh | người | box nhãn không khớp dự đoán nào (conf ≥ 0,1) |
| EMPTY_LABEL / BOX_TINY / BOX_HUGE / CLASS_RARE | 12 / 6 dòng / 3 / 4 | người | ý nghĩa (ảnh nền? vật quá nhỏ?) — máy không quyết được |

**Hàng đợi** (`05_review_queue.csv`, 125 ảnh): điểm = Σ trọng số × (1 + conf). Đầu hàng là `img_0020` (11,52: vừa bị xoá dòng id 93,
vừa có MISSING + WRONG_CLASS), tiếp theo 6 ảnh `NEEDS_LABEL` (10,0) — không có người thì 6 ảnh này chắc chắn làm Rework. Sau đó là các ảnh WRONG_CLASS
(trọng số 3) và MISSING (2) với conf cao. Ảnh thứ 80 có điểm 3,34. Với ngân sách 80 ảnh người xem được: 6/6 ảnh thiếu nhãn, 18/18 ảnh WRONG_CLASS,
50/52 ảnh MISSING, 12/23 ảnh FALSE_POS, nhưng chỉ 8/24 ảnh LOOSE_BOX (trọng số thấp nhất) → phần còn lại của hàng (45 ảnh, chủ yếu 16 LOOSE_BOX, 11 FALSE_POS)
chính là 38 ảnh làm cổng ra Hold.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | NEAR: 1,00/0,50 → 1,00/0,67 · EXACT giữ 1,00/1,00 | Recall NEAR phẳng 0,67 trên cả dải 4–14, precision luôn 1,00 trên calib. Chọn theo **raw**: cặp trùng thật xa nhất tìm thấy là `img_0186`≈`img_0061` (dist 10), cặp sai gần nhất ở dist 16 → 12 nằm giữa, chừa biên cả hai phía. Không chọn 16 dù F1 calib cao nhất (0,91), xem dưới. |
| `second_opinion.conf` | 0.25 | **0.6** | WRONG_CLASS 0,56/0,83 → 0,62/0,83 · MISSING 0,50/1,00 → 0,57/1,00 · LOOSE 0,29/0,40 → 0,20/0,20 · FALSE_POS 0,36/1,00 (không đổi, dùng `fp_conf`) | Từ 0,25 lên 0,6 precision WRONG_CLASS/MISSING tăng mà **không mất recall**; ngân sách duyệt cố định nên bớt cờ giả = đẩy lỗi thật lên trong 80 ảnh. Đánh đổi: LOOSE_BOX mất 1/5 (vốn đã yếu). Không chọn 0,8: MISSING recall rơi 1,00 → 0,75 (mất 2/8 box thiếu — loại lỗi hại nhất khi train), và 0,8 là mép sweep. |

**Chỗ calibration đánh lừa:** (1) Trên calib (39 ảnh, chỉ 6 ca NEAR) `phash_dist=16` là tốt nhất (NEAR 1,00/0,83), nhưng chạy trên raw thì cửa nhận loại **31 thay vì 22 ảnh**;
mở xem 9 ảnh bị loại thêm thì **8 là sai**: `img_0011` (tennis) bị coi là trùng `img_0008` (trượt tuyết), `img_0177` (lướt sóng) ≈ `img_0113` (người ngồi bàn), và cả 6 `LEAK_TEST` mới
(`img_0013`≈`test_087`, `img_0017`≈`test_071`, …) đều là cảnh khác hẳn. Precision 1,00 trên calib chỉ vì calib quá nhỏ để có cặp khác-nhưng-giống. (2) Ngưỡng `conf` điều khiển trực tiếp
số cảnh báo: thử `conf=0.8` thì cổng ra **Release** (16/203 = 7,9%) chỉ vì máy im lặng hơn, không phải dữ liệu sạch hơn — vì thế tôi không chọn ngưỡng theo kết luận gate.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn thật:** `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` → kết luận **Rework**, **mã thoát 20**,
lý do "còn 6 lỗi bắt buộc (MISSING_LABEL_FILE: 6); đồng thời 112/202 ảnh (55,5%) còn cảnh báo chưa duyệt", log in "KHÔNG xuất", và thư mục `work/exp_nb` **không được tạo**.
Lỗi bắt buộc không thể bị che bằng `reviewed` (có test `test_error_cannot_be_hidden_by_reviewed`). Tái lập: chạy `make run` 2 lần, 8 file báo cáo CSV + nhãn xuất giống hệt từng byte.

**(b) Điểm mù (có bằng chứng):**
1. **Ảnh trùng biến đổi mạnh lọt qua:** `img_0084` gần như trùng `img_0074` (cắt/đổi sáng nhẹ) nhưng phash dist = 16 > 12 nên vẫn vào dataset; trên calib, `calib_037`/`calib_038` (mức biến đổi 3–4) cũng không bị bắt. Muốn bắt phải dùng đặc trưng khác (embedding), vì nâng phash lên 16 gây 8 ca sai.
2. **Box lỏng gần như không bắt được:** LOOSE_BOX recall 0,20 trên calib. `calib_012` (box phóng ×1,45) và `calib_024` (×1,54) không có cảnh báo nào; `calib_006` lại bị gán thành FALSE_POS, `calib_002` thành MISSING + FALSE_POS (người duyệt sẽ được báo sai loại lỗi).
3. **Nhầm lớp mà model lớn cũng nhầm:** `calib_009` truck→bus không bị cờ (yolo26l không phân biệt chắc truck/bus); 5/6 ca nhầm còn lại đều là person→car, dễ hơn nhiều.
4. **Sửa tự động tạo lỗi mới:** `clip_box` ở `img_0030` dòng 9 (box gần như nằm ngoài ảnh) để lại box rộng 9,8 px → BOX_TINY mới ở v2 (6 → 7 dòng). Máy không biết đó là vật thật bị cắt hay box rác.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
**Chưa chạy.** Lý do: phần train cần GPU (Colab T4) và repo public đã push `submission/export`; lần nộp này dừng ở phần lõi (`make check-core`).
Khi chạy, tôi sẽ so chênh mAP50 (clean − raw) với độ lệch chuẩn giữa 2 seed và chỉ kết luận "tốt hơn" nếu chênh lớn hơn nhiễu.
Lưu ý trước: `raw` còn 5 ảnh rò rỉ test nên mAP test của raw có thể bị đẩy lên giả; test chỉ 100 ảnh nên nhiễu sẽ lớn.

## 6. Quyết định và ghi chú AI
**(a)** Quyết định cuối: **Hold** — không phát hành. Để thành Release cần số ảnh còn cảnh báo chưa duyệt < 10% × 202, tức **≤ 20 ảnh**: duyệt thêm **≥ 18 ảnh**
trong 38 ảnh còn lại (hàng 81 trở đi của `05_review_queue.csv`, chủ yếu LOOSE_BOX/FALSE_POS), hoặc mở thêm một phiên duyệt. Không hạ cảnh báo bằng cách nâng `conf` (xem §3).
Nếu phiên duyệt sau để lại lỗi bắt buộc (ví dụ ảnh vẫn chưa có nhãn) thì kết luận sẽ là Rework.

**(b) Ghi chú AI:** Có dùng AI (Claude Code) để hỗ trợ cách viết 4 module và soạn nháp báo cáo.
