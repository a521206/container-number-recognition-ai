# Recommendations: Container Number Recognition AI

*Based on code review of uncommitted changes — 2026-04-02*

---

## 🔴 Fix These Before Committing

### 1. Shape Index Swap in `get_ctnr_color_from_byte` (Line 111–115)

NumPy arrays are indexed as `[rows, cols]` = `[y, x]`. The current bounds are swapped.

**Current (broken):**
```python
cropped_img: np.ndarray = img_np[
    max(0, crop_zone[1] - 100): min(crop_zone[3] + 100, img_np.shape[1]),  # wrong: shape[1] is width
    max(0, crop_zone[0] - 100): min(crop_zone[2] + 100, img_np.shape[0])]  # wrong: shape[0] is height
```

**Fix:**
```python
cropped_img: np.ndarray = img_np[
    max(0, crop_zone[1] - 100): min(crop_zone[3] + 100, img_np.shape[0]),  # y capped by height
    max(0, crop_zone[0] - 100): min(crop_zone[2] + 100, img_np.shape[1])]  # x capped by width
```

---

### 2. `IndexError` on Filenames Without an Extension (Line 352)

`rsplit('.', 1)[1]` raises `IndexError` if the filename contains no dot.

**Current (broken):**
```python
if file and file.filename.rsplit('.', 1)[1].lower() not in ["jpg", "jpeg", "bmp", "png"]:
```

**Fix:**
```python
parts = file.filename.rsplit('.', 1)
if len(parts) < 2 or parts[1].lower() not in ["jpg", "jpeg", "bmp", "png"]:
    return jsonify({"message": "Wrong file type"}), 415, headers
```

---

### 3. `from flask import jsonify` Placed Before the Docstring (Line 322–324)

Placing an import before the docstring means `http_request.__doc__` is `None`. Move it to top-level imports.

**Fix:** Add `from flask import jsonify` at the top of the file with the other imports and remove the in-function import.

---

## 🟡 Logic Issues to Address

### 4. `break` Only Exits the Innermost Loop in `extract_ctnr_location` (Line 151)

When a complete container number is found, only the word loop exits. The region and line loops keep running unnecessarily.

**Fix:** Use a sentinel flag or restructure with an early `return`:
```python
def extract_ctnr_location(ocr_output) -> CNRAI:
    ...
    for region in ocr_output.regions:
        for line in region.lines:
            for word in line.words:
                if len(tmp_cnrai.container_number) >= 11 and tmp_cnrai.container_type != "":
                    tmp_cnrai.bounding_box = [int(bb) for bb in bound_block]
                    return tmp_cnrai  # exit all loops immediately
                ...
```

---

### 5. Bounding Box Start Coordinate Fails When First Word Is at x=0 (Line 263)

`if bb[0] == 0` uses `0` as a sentinel for "not yet set", but a valid container word can legitimately start at x-coordinate 0 (left edge of image). Use `None` as the sentinel instead.

**Fix:**
```python
bb: list = [None, None, None, None]
...
if bb[0] is None:
    bb[0], bb[1] = x1, y1
    ...
elif bb[0] is not None:
    bb[2], bb[3] = x2, y2
    ...
# Before return, default unset values to 0
tmp_cnrai.bounding_box = [int(v) if v is not None else 0 for v in bb]
```

---

### 6. `downscale()` Divides by Zero for Images 10,000 px Wide (Line 39)

`(10000 // 1000) % 10 = 0`, making `down_factor = 0` and crashing with `ZeroDivisionError`.

**Fix:** Replace the formula with a straightforward target width:
```python
TARGET_WIDTH = 1000
def downscale(ori_img: np.ndarray) -> np.ndarray:
    h, w, _ = ori_img.shape
    if w >= TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        return cv2.resize(ori_img, (TARGET_WIDTH, int(h * scale)))
    return ori_img
```

---

## 🟢 Code Quality Improvements

### 7. Cache `get_carrier_prefixes()` at Module Load

The function opens and reads a file on every call and is invoked once per image. Cache the result once at startup.

```python
# At module level, after load_dotenv()
CARRIER_PREFIXES: Tuple = get_carrier_prefixes()
CONTAINER_T_PREFIXES: Tuple = ("G1", "R1", "U1", "P1", "T1")
```

---

### 8. Replace Magic Numbers with Named Constants

```python
CONTAINER_NUMBER_LENGTH = 11   # ISO standard container number length
CROP_PADDING_PX        = 100   # pixels added around bounding box for color sampling
DEFAULT_BUFFER_PX      = 50    # spatial proximity buffer for serial detection
ANGLE_BUFFER_FACTOR    = 3     # multiplier applied to label angle for buffer
```

---

### 9. Use a Dataclass for `CNRAI`

The class has no methods. A `@dataclass` removes the boilerplate `__init__` and adds `__repr__` for free.

```python
from dataclasses import dataclass, field

@dataclass
class CNRAI:
    container_number: str = ""
    container_type: str = ""
    bounding_box: list = field(default_factory=lambda: [0, 0, 0, 0])
    container_color: list = field(default_factory=lambda: [0, 0, 0])
    error: Optional[str] = None
```

---

### 10. Fix Variable Naming in `extract_ctnr_regex`

`intbb` implies the value is already an int, which is misleading.

```python
# Current
tmp_cnrai.bounding_box = [int(intbb) for intbb in bb]

# Fix
tmp_cnrai.bounding_box = [int(v) for v in bb]
```

---

## Priority Order

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 1 | Fix shape index swap in `get_ctnr_color_from_byte` | Low | 🔴 High |
| 2 | Guard against missing file extension | Low | 🔴 High |
| 3 | Move `flask` import to top level | Low | 🟡 Medium |
| 4 | Fix early exit to escape all loops | Low | 🟡 Medium |
| 5 | Use `None` sentinel in bounding box tracking | Low | 🟡 Medium |
| 6 | Fix `downscale()` divide-by-zero | Low | 🟡 Medium |
| 7 | Cache `get_carrier_prefixes()` | Low | 🟢 Low |
| 8 | Named constants for magic numbers | Low | 🟢 Low |
| 9 | Migrate `CNRAI` to `@dataclass` | Medium | 🟢 Low |
| 10 | Fix `intbb` variable name | Low | 🟢 Low |

