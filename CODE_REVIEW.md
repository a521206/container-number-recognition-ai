# Project Review: Container Number Recognition AI

## Project Overview
A Python-based OCR solution using Azure AI Vision to automatically detect and extract container numbers from images for logistics management.

---

## ✅ Strengths

1. **Clean Architecture**: Well-structured with separation between detection methods ([`extract_ctnr_regex()`](main.py:195), [`extract_ctnr_location()`](main.py:119)) and core processing
2. **Dual Detection Strategies**: Fallback approach using both regex matching and spatial bounding box analysis
3. **Comprehensive Container Prefixes**: [`container_prefix.txt`](container_prefix.txt) contains ~600 carrier prefixes for robust detection
4. **Good Documentation**: Clear README with setup instructions and references
5. **Production-Ready Deployment**: Google Cloud Functions configuration via [`cloudbuild.yaml`](cloudbuild.yaml)

---

## 🔴 Critical Issues

### 1. **Index Swap Bug in Crop Zone** ([Lines 109-113](main.py:109))
```python
cropped_img: np.ndarray = img_np[
    max(0, crop_zone[1] - 100): min(crop_zone[3] + 100, img_np.shape[1]),
    max(0, crop_zone[0] - 100): min(crop_zone[2] + 100, img_np.shape[0])]
```
**Problem**: `shape[1]` is width (x-axis) and `shape[0]` is height (y-axis). The bounds are swapped - `crop_zone[3]` should compare with `shape[0]` and `crop_zone[2]` with `shape[1]`.

### 2. **Early Exit Incomplete** ([Lines 144-146](main.py:144))
```python
if len(tmp_cnrai.container_number) >= 11 and tmp_cnrai.container_type != "":
    break  # Only breaks inner for loop, not outer
```
The `break` only exits the word loop, not the lines loop, leading to unnecessary iterations.

### 3. **Missing Extension Check** ([Line 325](main.py:325))
```python
if file and file.filename.rsplit('.', 1)[1].lower() not in ["jpg", "jpeg", "bmp", "png"]:
```
**Problem**: Will raise `IndexError` if filename has no extension (e.g., "image").

---

## 🟡 Logic Issues

### 4. **Bounding Box Tracking** ([Lines 218-230](main.py:218))
The logic only captures the first and last word's coordinates. For multi-word container numbers spanning multiple lines, this may not correctly represent the full bounding region.

### 5. **Regex Pattern Construction** ([Line 210](main.py:210))
```python
regex_ctnr_pattern: str = "".join(["(", "|".join(carrier_prefix), ")", "(\d{7})"])
```
Inefficient string building. Use f-string or `str.format()` instead.

### 6. **`downscale()` Scaling Factor** ([Line 37](main.py:37))
```python
down_factor: int = (w // 1000 % 10) * 2
```
This non-linear scaling (2, 4, 6, 8...) is confusing and may produce unexpected results for various image widths.

---

## 🟢 Code Quality Suggestions

### 7. **Replace Class with Dataclass**
The [`CNRAI`](main.py:19) class has no methods. Consider using `@dataclass`:
```python
from dataclasses import dataclass

@dataclass
class CNRAI:
    container_number: str = ""
    container_type: str = ""
    bounding_box: list = None
    container_color: list = None
    error: str | None = None
```

### 8. **Magic Numbers**
Extract constants like `50` (buffer), `100` (crop expansion), `11` (container length) as named constants.

### 9. **Runtime Prefix Loading**
[`get_carrier_prefixes()`](main.py:77) reads from file on every call. Consider caching or loading once.

### 10. **Redundant Type Casting**
```python
bb = [int(intbb) for intbb in bb]  # Line 236
```
Should be `bb = [int(b) for b in bb]` - `intbb` naming implies double conversion.

---

## 📋 Summary

| Category | Count |
|----------|-------|
| Critical Issues | 3 |
| Logic Issues | 3 |
| Quality Improvements | 4 |

**Overall Assessment**: The project demonstrates solid understanding of OCR integration and logistics domain. The two-method fallback approach is smart. However, the index swap bug in image cropping is a significant issue that could cause runtime errors or incorrect image processing. Recommended to fix the three critical issues before production deployment.

---

*Review generated: 2026-04-01*
