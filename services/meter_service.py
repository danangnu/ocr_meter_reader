# services/meter_service.py
import os, io, re, time, base64
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import pytesseract
from PIL import Image

# --- Optional: tesseract path (Windows) or env TESSERACT_CMD ---
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
else:
    win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.isfile(win):
        pytesseract.pytesseract.tesseract_cmd = win

# ------------------ helpers ------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def bgr_from_bytes(content: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return bgr

def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

# ------------------ LCD / band finders ------------------
def find_green_lcd(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([30, 15, 40], np.uint8)   # broad green
    upper = np.array([100, 255, 255], np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8), 1)
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return bgr
    h, w = bgr.shape[:2]
    best, best_area = None, -1
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        ar = ww/max(1,hh)
        if area > 0.01*w*h and 1.0 <= ar <= 14.0 and area > best_area:
            best_area, best = area, (x,y,ww,hh)
    if best is None:
        x,y,ww,hh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    else:
        x,y,ww,hh = best
    pad = max(2, int(0.02*max(w,h)))
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(w, x+ww+pad); y1 = min(h, y+hh+pad)
    return bgr[y0:y1, x0:x1].copy()

def find_band_failsafe(bgr: np.ndarray) -> np.ndarray:
    """Pick densest horizontal ‘ink’ band (works even if no green)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((15,15), np.uint8))
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
    proj = th.sum(axis=1).astype(np.float32)
    if proj.max() < 1:
        return bgr
    rows = np.where(proj > 0.25*proj.max())[0]
    y0, y1 = int(rows.min()), int(rows.max())
    pad = max(4, int(0.03*th.shape[0]))
    y0 = max(0, y0-pad); y1 = min(th.shape[0]-1, y1+pad)
    band = bgr[y0:y1+1, :]
    h,w = band.shape[:2]
    return band[:, int(0.30*w):]  # trim left icons, keep right

# ------------------ preprocessing & splitters ------------------
def preprocess(gray: np.ndarray, scale=3.0) -> np.ndarray:
    # slightly gentler defaults (help under glare / huge digits)
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    g = cv2.normalize(cv2.subtract(g, bg), None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.bilateralFilter(g, d=5, sigmaColor=35, sigmaSpace=35)
    g = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g,(0,0),1.0), -0.8, 0)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(g)
    return g

def gentle_roi(roi: np.ndarray) -> np.ndarray:
    """Reduce glare artifacts for OCR without destroying strokes."""
    r = cv2.medianBlur(roi, 3)
    hi = np.percentile(r, 99.5)
    r = np.clip(r, 0, hi).astype(np.uint8)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return r

def split_equal(g: np.ndarray, slots: int) -> List[np.ndarray]:
    W = g.shape[1]; cw = max(1, W//slots)
    return [g[:, i*cw:(W if i==slots-1 else (i+1)*cw)] for i in range(slots)]

def split_ccomp(g: np.ndarray, slots: int) -> List[np.ndarray]:
    inv = 255 - g
    bn = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bn = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    num, _, stats, _ = cv2.connectedComponentsWithStats(bn, 8)
    h, w = g.shape[:2]
    boxes = []
    for i in range(1, num):
        x,y,ww,hh,area = stats[i]
        if area < 0.002*w*h or hh < 0.25*h: continue
        ar = ww/float(hh)
        if 0.1 <= ar <= 1.6: boxes.append((x,y,ww,hh))
    if not boxes:
        return split_equal(g, slots)
    xs = np.array([x+ww/2 for x,y,ww,hh in boxes], np.float32)
    order = np.argsort(xs)
    edges = [int(round((i/slots)*len(xs))) for i in range(slots+1)]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = min(len(xs), edges[i-1]+1)
    cols = []
    for k in range(slots):
        lo, hi = edges[k], min(edges[k+1], len(order))
        if lo >= hi:
            cw = max(1, w//slots); cols.append((k*cw, (k+1)*cw if k<slots-1 else w)); continue
        members = [boxes[order[i]] for i in range(lo,hi)]
        xmin = min(b[0] for b in members); xmax = max(b[0]+b[2] for b in members)
        pad = max(1, int(0.01*w))
        cols.append((max(0,xmin-pad), min(w,xmax+pad)))
    cols = sorted(cols)
    cells = [g[:, l:r] for (l,r) in cols]
    return cells if len(cells)==slots else split_equal(g, slots)

# ------------------ Seven-seg fallback ------------------
SEG_MAP = {
    (1,1,1,1,1,1,0): "0",
    (0,1,1,0,0,0,0): "1",
    (1,1,0,1,1,0,1): "2",
    (1,1,1,1,0,0,1): "3",
    (0,1,1,0,0,1,1): "4",
    (1,0,1,1,0,1,1): "5",
    (1,0,1,1,1,1,1): "6",
    (1,1,1,0,0,0,0): "7",
    (1,1,1,1,1,1,1): "8",
    (1,1,1,1,0,1,1): "9",
}

def seg_sample_boxes(h: int, w: int):
    m = 0.08; t = 0.18
    boxes_rel = [
        (m,        m+t,     0.25,   0.75),  # a
        (m,        0.50-m,  0.75-m, 1.0-m),# b
        (0.50+m,   1.0-m,   0.75-m, 1.0-m),# c
        (1.0-m-t,  1.0-m,   0.25,   0.75),  # d
        (0.50+m,   1.0-m,   m,      0.25+m),# e
        (m,        0.50-m,  m,      0.25+m),# f
        (0.50-m-t, 0.50+m+t,0.25,   0.75),  # g
    ]
    boxes = []
    for (ry0, ry1, rx0, rx1) in boxes_rel:
        y0 = int(ry0*h); y1 = int(ry1*h)
        x0 = int(rx0*w); x1 = int(rx1*w)
        y1 = max(y1, y0+1); x1 = max(x1, x0+1)
        boxes.append((y0,y1,x0,x1))
    return boxes  # a,b,c,d,e,f,g

def sevenseg_digit(cell: np.ndarray) -> Optional[str]:
    """
    Adaptive 7-segment decode:
    - No aggressive close (reduces false 'all-on').
    - Threshold per segment decided from the distribution of segment densities.
    """
    inv = 255 - cell
    # light denoise only
    bn = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bn = cv2.medianBlur(bn, 3)

    ys, xs = np.where(bn > 0)
    if len(xs) and len(ys):
        x0, x1 = max(0, xs.min()-2), min(bn.shape[1]-1, xs.max()+2)
        y0, y1 = max(0, ys.min()-2), min(bn.shape[0]-1, ys.max()+2)
        bn = bn[y0:y1+1, x0:x1+1]

    h, w = bn.shape[:2]
    if h < 10 or w < 8:
        return None

    boxes = seg_sample_boxes(h, w)
    vals = []
    for (y0, y1, x0, x1) in boxes:
        roi = bn[y0:y1, x0:x1]
        vals.append((roi > 0).mean() if roi.size else 0.0)

    # adaptive ON threshold from segment density distribution
    vals_sorted = sorted(vals)
    lo = np.median(vals_sorted[:3]) if len(vals_sorted) >= 3 else np.median(vals_sorted)
    hi = np.median(vals_sorted[-3:]) if len(vals_sorted) >= 3 else np.median(vals_sorted)
    thr = max(0.58, min(0.85, 0.5*(hi + max(lo, 0.15))))  # stricter than before

    on = [1 if v >= thr else 0 for v in vals]  # a,b,c,d,e,f,g
    return SEG_MAP.get(tuple(on), None)

# ------------------ OCR ------------------
def ocr_digit(cell: np.ndarray, prefer_sevenseg: bool) -> Tuple[str, float]:
    # Try seven-seg first when glare/size suggests LCD seven-seg
    if prefer_sevenseg:
        s = sevenseg_digit(cell)
        if s is not None:
            return s, 60.0  # pseudo-confidence on Tesseract scale (0..100)

    h, _ = cell.shape[:2]

    def _one(img, invert: bool, do_dilate: bool):
        x = 255 - img if invert else img
        bn = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if do_dilate:
            bn = cv2.morphologyEx(bn, cv2.MORPH_DILATE, np.ones((2,2),np.uint8), 1)
        ys, xs = np.where(bn>0)
        if len(xs) and len(ys):
            x0,x1 = max(0, xs.min()-2), min(bn.shape[1]-1, xs.max()+2)
            y0,y1 = max(0, ys.min()-2), min(bn.shape[0]-1, ys.max()+2)
            bn = bn[y0:y1+1, x0:x1+1]
        cfg = "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
        d = pytesseract.image_to_data(bn, config=cfg, output_type=pytesseract.Output.DICT)
        best, conf = "", -1.0
        for t, cf in zip(d.get("text", []), d.get("conf", [])):
            t = (t or "").strip()
            if re.fullmatch(r"[0-9]", t):
                try: f = float(cf)
                except: f = -1.0
                if f > conf: best, conf = t, f
        return best, conf

    # Very large digits → avoid dilation (prevents 8→7)
    use_dilate = h < 150
    a, ca = _one(cell, True,  use_dilate)
    b, cb = _one(cell, False, use_dilate)
    return (a, ca) if ca >= cb else (b, cb)

def ocr_whole(gray: np.ndarray) -> Tuple[str, float]:
    """OCR whole value; tries several PSMs/inversions; returns (text, conf0..1)."""
    candidates = []
    for img in (255-gray, gray):
        for psm in (7, 6, 11):
            bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789."
            d = pytesseract.image_to_data(bn, config=cfg, output_type=pytesseract.Output.DICT)
            toks, confs = [], []
            for t, cf in zip(d.get("text", []), d.get("conf", [])):
                t = (t or "").strip()
                if re.fullmatch(r"[0-9.]+", t):
                    toks.append(t)
                    try: confs.append(float(cf))
                    except: pass
            if toks:
                s = max(toks, key=len)
                c = (sum(confs)/len(confs)/100.0) if confs else 0.0
                candidates.append((s,c))
    if not candidates: return "", 0.0
    # prefer any with a decimal, then length, then conf
    return max(candidates, key=lambda z: (bool(re.search(r"\d+\.\d+", z[0])), len(z[0]), z[1]))

# ------------------ Quality ------------------
@dataclass
class Quality:
    ok: bool
    issues: List[str]
    metrics: Dict[str, float]

def assess_quality(full_bgr: np.ndarray, lcd_bgr: np.ndarray, roi_gray: np.ndarray, cells: List[np.ndarray]) -> Quality:
    H,W = full_bgr.shape[:2]
    hL,wL = lcd_bgr.shape[:2]
    area_ratio = (hL*wL) / float(max(1, H*W))
    sharp = cv2.Laplacian(cv2.cvtColor(lcd_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    gray = cv2.cvtColor(lcd_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    v = cv2.cvtColor(lcd_bgr, cv2.COLOR_BGR2HSV)[:,:,2]
    glare = float((v > 250).mean())
    dh = float(np.median([c.shape[0] for c in cells]) if cells else 0)

    issues = []
    if area_ratio < 0.02: issues.append("LCD too small in frame (move closer / crop)")
    if dh < 60:          issues.append("Digits too small (target ≥ 80 px height)")
    if sharp < 60:       issues.append("Image blurry (hold steady / faster shutter)")
    if contrast < 25:    issues.append("Low contrast on LCD (avoid haze/dirty cover)")
    if glare > 0.03:     issues.append("Glare on LCD (tilt camera to avoid reflections)")

    return Quality(
        ok = len(issues) == 0,
        issues = issues,
        metrics = {
            "lcd_area_ratio": round(area_ratio,4),
            "digit_height_px": round(dh,1),
            "sharpness_varLaplacian": round(sharp,1),
            "contrast_stddev": round(contrast,1),
            "glare_ratio": round(glare,4),
        }
    )

# ------------------ Public service API ------------------
@dataclass
class Params:
    slots: int = 5
    keep_right: float = 0.90
    scale: float = 3.0
    splitter: str = "auto"  # "auto" | "ccomp" | "equal"
    failsafe: bool = False
    no_lcd: bool = False
    decimals: Optional[int] = None   # e.g., 1 -> 198.8

def detect_and_crop(bgr: np.ndarray, params: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lcd_bgr, band_bgr, roi_gray) without OCR.

    Fixes:
      - Even when failsafe=True, attempt green LCD detection first; only fall back if it fails.
      - Auto reduce scale when band is tall (huge digits) to avoid over-sharpening -> "8" illusions.
    """
    # Try color LCD first (robust if the screen is greenish). If it fails, fall back to whole image.
    try:
        lcd_candidate = bgr if params.no_lcd else find_green_lcd(bgr)
    except Exception:
        lcd_candidate = bgr

    # If color-detected LCD is basically the whole frame, treat it as "not detected" and use failsafe band.
    H, W = bgr.shape[:2]
    hL, wL = lcd_candidate.shape[:2]
    area_ratio = (hL * wL) / float(max(1, H * W))
    color_ok = (area_ratio < 0.95)  # < 95% of the full image → good crop

    if params.failsafe and not color_ok:
        lcd = bgr
    else:
        lcd = lcd_candidate

    # Band finder (works both with real LCD crop or whole image)
    band = find_band_failsafe(lcd)

    # Preprocess to gray ROI anchored on the right
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

    # Auto-tune scale for very tall bands (huge digits → less upscaling)
    base_scale = params.scale
    if gray.shape[0] > 500:
        base_scale = min(base_scale, 2.0)

    g = preprocess(gray, scale=base_scale)
    h, w = g.shape[:2]

    keep = clamp(params.keep_right, 0.5, 0.95)
    x0 = int((1.0 - keep) * w)
    roi = g[:, x0:]
    return lcd, band, roi

def _format_value(digits: str, slots: int, decimals: Optional[int], text_full: Optional[str]) -> str:
    d = re.sub(r"\D","", digits or "")
    if decimals is not None:
        need = max(1, decimals + 1)
        d = d.zfill(need)
        return (d[:-decimals] + "." + d[-decimals:]) if decimals > 0 else d
    if text_full:
        m = re.search(r"\d+\.\d+", text_full)
        if m: return m.group(0)
    return f"{d[:2]}.{d[2:]}" if slots == 5 else d

def suppress_glare_gray(roi: np.ndarray) -> np.ndarray:
    # remove bright islands relative to local background
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k)
    r = cv2.subtract(roi, (0.85 * tophat).astype(np.uint8))
    # clip extreme highs, renormalize
    hi = np.percentile(r, 99.0)
    r = np.clip(r, 0, hi).astype(np.uint8)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    return r

# --- NEW: tighten ROI to the densest right-run of ink (helps drop stray text) ---
def tighten_roi(roi: np.ndarray) -> np.ndarray:
    inv = 255 - roi
    bn = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    col = bn.sum(axis=0).astype(np.float32)
    if col.max() <= 0:
        return roi
    thr = 0.35 * col.max()
    mask = col > thr
    best_len, best = 0, (0, roi.shape[1]-1)
    s = None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None:
            e = i - 1
            if e - s > best_len: best_len, best = e - s, (s, e)
            s = None
    if s is not None:
        e = len(mask) - 1
        if e - s > best_len: best_len, best = e - s, (s, e)
    if best_len == 0:
        return roi
    pad = max(2, int(0.03 * roi.shape[1]))
    x0 = max(0, best[0] - pad)
    x1 = min(roi.shape[1]-1, best[1] + pad)
    return roi[:, x0:x1+1]

# --- NEW: whole-string OCR with token metadata (size + conf) ---
def ocr_whole_tokens(gray: np.ndarray):
    from pytesseract import Output
    H, W = gray.shape[:2]
    cands = []
    for img in (255 - gray, gray):
        for psm in (7, 6, 11):
            bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789."
            d = pytesseract.image_to_data(bn, config=cfg, output_type=Output.DICT)
            n = len(d.get("text", []))
            for i in range(n):
                t = (d["text"][i] or "").strip()
                if not re.fullmatch(r"[0-9.]+", t):
                    continue
                try: conf = float(d["conf"][i]) / 100.0
                except: conf = 0.0
                w = int(d["width"][i] or 0); h = int(d["height"][i] or 0)
                if w <= 0 or h <= 0: continue
                digits = re.sub(r"\D", "", t)
                cands.append({
                    "text": t, "digits": digits, "conf": conf,
                    "has_decimal": ("." in t),
                    "h_ratio": h / float(H), "w_ratio": w / float(W),
                })
    return cands

# --- NEW: penalize suspicious repetitive slot reads (e.g., 8888 under glare) ---
def _repetition_penalty(digits: str) -> int:
    if not digits:
        return 0
    from collections import Counter
    c = Counter(digits)
    ratio = max(c.values()) / len(digits)
    if ratio >= 0.75 and c.most_common(1)[0][0] in "80":
        return 4
    if ratio >= 0.75:
        return 3
    return 0

# --- Keep only the darkest strokes (ignore bright glare) ---
def binarize_dark_only(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), 5.0)
    ink = cv2.subtract(bg, gray)              # darker than local bg
    T = float(np.percentile(ink, 60))         # was 75 → keep more strokes
    T = max(8.0, T)
    bn = cv2.threshold(ink, T, 255, cv2.THRESH_BINARY)[1]
    bn = cv2.morphologyEx(bn, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    return bn  # white=ink

def ocr_digit_from_binary(bn: np.ndarray) -> tuple[str, float]:
    def _one(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            x0,x1 = max(0, xs.min()-2), min(mask.shape[1]-1, xs.max()+2)
            y0,y1 = max(0, ys.min()-2), min(mask.shape[0]-1, ys.max()+2)
            mask = mask[y0:y1+1, x0:x1+1]
        cfg = "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
        d = pytesseract.image_to_data(mask, config=cfg, output_type=pytesseract.Output.DICT)
        best, conf = "", -1.0
        for t, cf in zip(d.get("text", []), d.get("conf", [])):
            t = (t or "").strip()
            if re.fullmatch(r"[0-9]", t):
                try: f = float(cf)
                except: f = -1.0
                if f > conf: best, conf = t, f
        return best, (conf if conf >= 0 else 0.0)

    a, ca = _one(bn)
    # try light erosion to break over-connected segments that look like “8”
    er = cv2.morphologyEx(bn, cv2.MORPH_ERODE, np.ones((2,2), np.uint8), 1)
    b, cb = _one(er)
    return (a, ca) if ca >= cb else (b, cb)

def _count_holes(bin_img: np.ndarray) -> int:
    """
    Estimate the number of 'holes' in a binary glyph.
    bin_img: white=ink, black=background
    """
    # holes are black islands inside white regions -> check on inverted
    inv = 255 - bin_img
    cnts, hier = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return 0
    holes = 0
    area_min = max(8, 0.003 * bin_img.shape[0] * bin_img.shape[1])
    for i, h in enumerate(hier[0]):
        parent = h[3]
        if parent != -1:  # child contour => hole
            if cv2.contourArea(cnts[i]) >= area_min:
                holes += 1
    return holes

def _leading_one_rescue(bin_cell: np.ndarray, guess: str, conf: float) -> tuple[str, float]:
    """
    If the leftmost digit is a tall, skinny, single vertical stroke with no holes,
    treat it as '1' (boost confidence). Prevents 1→0 mistakes like '084.8' instead of '184.8'.
    bin_cell must be binary with white=ink.
    """
    h, w = bin_cell.shape[:2]
    if h < 20 or w < 5:
        return guess, conf

    ar = w / float(h)
    if ar > 0.42:   # too wide to be a typical '1'
        return guess, conf

    holes = _count_holes(bin_cell)
    if holes >= 1:  # '0' or '8' typically shows internal holes
        return guess, conf

    # central stroke should dominate
    col = bin_cell.sum(axis=0).astype(np.float32) / 255.0
    c0, c1 = int(0.40 * w), int(0.60 * w)
    center = col[c0:c1].mean() if c1 > c0 else col.mean()
    edges  = np.r_[col[:int(0.25 * w)], col[int(0.75 * w):]].mean() if w >= 4 else 0.0

    if center > max(3.0, 1.6 * edges):
        return "1", max(conf, 0.70)
    return guess, conf

def _rescue_first_by_width(cells_dark: list[np.ndarray], slot_digits: list[str], slot_confs: list[float]) -> None:
    """
    If the first digit's bounding box is much narrower than peers but similarly tall,
    treat it as '1'. Works well on seven-seg where '1' is a skinny bar.
    - cells_dark: list of binary (0/255) per-slot images, white=ink.
    - slot_digits / slot_confs: lists in-place modified.
    """
    wh = []
    for bc in cells_dark:
        ys, xs = np.where(bc > 0)
        if len(xs) == 0:
            wh.append((0, 0))
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        wh.append((w, h))

    if len(wh) < 2 or wh[0][0] <= 0:
        return

    widths  = [w for (w, _) in wh[1:] if w > 0]  # reference: slots 1..N-1
    heights = [h for (_, h) in wh[1:] if h > 0]
    if not widths or not heights:
        return

    ref_w = float(np.median(widths))
    ref_h = float(np.median(heights))
    if ref_w <= 0 or ref_h <= 0:
        return

    w0, h0 = wh[0]
    # Heuristics: clearly narrower but similarly tall → it's a '1'
    if (w0 / ref_w) <= 0.65 and (h0 / ref_h) >= 0.90:
        if slot_digits and slot_digits[0] in ("", "0", "8"):
            slot_digits[0] = "1"
            # boost confidence a bit; your code expects 0..1 after averaging/100 later
            slot_confs[0] = max(slot_confs[0], 0.80)

def read_from_bgr(bgr: np.ndarray, params: Params, want_debug: bool = False) -> dict:
    import time, re
    t0 = time.time()

    lcd, band, _ = detect_and_crop(bgr, params)

    # ---------- ROI builder ----------
    def build_roi(keep_right: float):
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        base_scale = params.scale if gray.shape[0] <= 500 else min(params.scale, 2.0)
        g = preprocess(gray, scale=base_scale)
        h, w = g.shape[:2]
        kr = clamp(keep_right, 0.50, 0.98)
        x0 = int((1.0 - kr) * w)
        roi_raw = g[:, x0:]

        glare = float((roi_raw > 245).mean())
        roi_proc = roi_raw.copy()
        if glare > 0.25 or h > 800:
            roi_proc = suppress_glare_gray(roi_proc)
        roi_proc = tighten_roi(roi_proc)
        roi_dark = binarize_dark_only(roi_proc)  # white=ink
        return roi_proc, roi_dark, roi_raw

    # ---------- helpers ----------
    def pad_cell(img: np.ndarray, frac: float = 0.08) -> np.ndarray:
        h, w = img.shape[:2]
        pad = max(2, int(frac * w))
        return cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_REPLICATE)

    def split_valleys_dark(bn: np.ndarray, slots: int) -> list[np.ndarray]:
        col = bn.sum(axis=0).astype(np.float32)
        if col.max() <= 0: return split_equal(bn, slots)
        sm = cv2.blur(col.reshape(1,-1), (1,25)).ravel()
        W = len(sm); approx = max(6, W // slots)
        cuts = []
        for k in range(1, slots):
            tgt = int(round(k * approx))
            lo = max(0, tgt - int(0.25 * approx))
            hi = min(W-1, tgt + int(0.25 * approx))
            if hi <= lo: continue
            idx = lo + int(np.argmin(sm[lo:hi+1]))
            cuts.append(idx)
        cuts = sorted(set([c for c in cuts if 3 <= c <= W-4]))
        xs = [0] + cuts + [W]
        cells = []
        for i in range(slots):
            x0, x1 = xs[i], xs[i+1]
            if x1 - x0 < 6: x0 = max(0, x0-3); x1 = min(W, x1+3)
            cells.append(bn[:, x0:x1])
        return cells if len(cells) == slots else split_equal(bn, slots)

    # ---- segment strengths & template voting ----
    def seg_strengths(gray_cell: np.ndarray) -> np.ndarray:
        best = np.zeros(7, dtype=np.float32)  # a..g
        for inv in (True, False):
            img = 255 - gray_cell if inv else gray_cell
            bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            bn = cv2.morphologyEx(bn, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
            ys, xs = np.where(bn>0)
            if len(xs) == 0: 
                continue
            x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
            bn = bn[y0:y1+1, x0:x1+1]
            h, w = bn.shape[:2]
            if h < 10 or w < 6: 
                continue
            vals = []
            for (yy0,yy1,xx0,xx1) in seg_sample_boxes(h,w):  # a,b,c,d,e,f,g
                roi = bn[yy0:yy1, xx0:xx1]
                vals.append((roi>0).mean() if roi.size else 0.0)
            best = np.maximum(best, np.array(vals, dtype=np.float32))
        return best

    TEMPLATES = {
        "0": np.array([1,1,1,1,1,1,0], dtype=np.float32),
        "1": np.array([0,1,1,0,0,0,0], dtype=np.float32),
        "2": np.array([1,1,0,1,1,0,1], dtype=np.float32),
        "3": np.array([1,1,1,1,0,0,1], dtype=np.float32),
        "4": np.array([0,1,1,0,0,1,1], dtype=np.float32),
        "5": np.array([1,0,1,1,0,1,1], dtype=np.float32),
        "6": np.array([1,0,1,1,1,1,1], dtype=np.float32),
        "7": np.array([1,1,1,0,0,0,0], dtype=np.float32),
        "8": np.array([1,1,1,1,1,1,1], dtype=np.float32),
        "9": np.array([1,1,1,1,0,1,1], dtype=np.float32),
    }
    WEIGHTS = np.array([1.0, 1.1, 1.1, 1.0, 1.0, 1.25, 1.20], dtype=np.float32)

    def seg_best_vs_second(gray_cell: np.ndarray) -> tuple[str, float]:
        s = seg_strengths(gray_cell)
        scores = []
        for d, t in TEMPLATES.items():
            err = WEIGHTS * ((t - s) ** 2)
            scores.append((d, float(err.sum())))
        scores.sort(key=lambda z: z[1])
        if not scores: return "", 0.0
        if len(scores) == 1: return scores[0][0], 1.0
        best_d, best_v = scores[0]
        second_v = scores[1][1]
        margin = max(0.0, (second_v - best_v)) / (second_v + 1e-6)  # 0..1
        return best_d, margin

    # repetition penalty already defined elsewhere:
    # _repetition_penalty(digits: str) -> int

    def best_decimal_token(tokens, target_len):
        good = []
        for t in tokens:
            if not t.get("has_decimal"): continue
            digs = t.get("digits", "")
            if len(digs) != target_len: continue
            if len(digs) >= 3 and len(set(digs)) == 1:  # 0000 / 8888 etc.
                continue
            if t.get("h_ratio",0) < 0.18 or t.get("w_ratio",0) < 0.16:
                continue
            good.append(t)
        if not good: return None
        return max(good, key=lambda z: (z["h_ratio"]*z["w_ratio"], z["conf"]))

    # Thin “1” score (slot-0 preference when picking ROI)
    def score_slot0_one(bin_cell: np.ndarray, peers_w: float, peers_h: float) -> float:
        ys, xs = np.where(bin_cell > 0)
        if len(xs) == 0 or peers_w <= 0 or peers_h <= 0: return 0.0
        x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
        bw, bh = (x1-x0+1), (y1-y0+1)
        wr = bw / peers_w
        hr = bh / peers_h
        col = (bin_cell[:, x0:x1+1].sum(axis=0) / 255.0)
        center = col[int(0.4*bw):int(0.6*bw)].mean() if bw>6 else col.mean()
        edges  = np.r_[col[:int(0.2*bw)], col[int(0.8*bw):]].mean() if bw>10 else col.mean()
        c_width  = np.clip((0.85 - wr) / 0.35, 0.0, 1.0)
        c_height = np.clip((hr - 0.88) / 0.12, 0.0, 1.0)
        c_center = np.clip((center / max(1e-3, edges) - 1.15) / 1.0, 0.0, 1.0)
        return 0.45*c_width + 0.30*c_height + 0.25*c_center

    # ---------- Per-slot fused OCR ----------
    def fuse_digit(cell_gray: np.ndarray, cell_bin: np.ndarray) -> tuple[str, float]:
        d_seg = sevenseg_digit(cell_gray); c_seg = 60.0 if d_seg is not None else 0.0
        d_bin, c_bin = ocr_digit_from_binary(cell_bin)
        d_gra, c_gra = ocr_digit(cell_gray, prefer_sevenseg=False)
        cand = [(d_seg or "", c_seg), (d_bin or "", c_bin), (d_gra or "", c_gra)]
        d_best, c_best = max(cand, key=lambda z: z[1])
        return d_best, c_best  # ~0..100

    def score_slot_like(ds: str, conf: float, pen: int, target_len: int) -> float:
        if not ds: return -999.0
        base = 2 if len(ds) == target_len else (1 if abs(len(ds)-target_len) <= 1 else 0)
        return base + conf - pen

    # ---------- Run one KR ----------
    def run_for_kr(kr: float):
        roi_proc, roi_dark, _ = build_roi(kr)
        cells_dark = split_valleys_dark(roi_dark, params.slots)

        # map grayscale cells & gather stats; pad both for OCR
        gray_cells, x_cur, total_w = [], 0, roi_dark.shape[1]
        padded_bin, padded_gray = [], []
        widths, heights = [], []
        for bc in cells_dark:
            w = bc.shape[1]
            gcell = roi_proc[:, x_cur:min(total_w, x_cur + w)]
            x_cur += w

            ys, xs = np.where(bc > 0)
            if len(xs):
                widths.append(int(xs.max()-xs.min()+1))
                heights.append(int(ys.max()-ys.min()+1))
            else:
                widths.append(0); heights.append(0)

            padded_bin.append(pad_cell(bc))
            padded_gray.append(pad_cell(gcell))
            gray_cells.append(gcell)

        # peers (exclude slot 0)
        peers_w = float(np.median([w for i,w in enumerate(widths) if i>0 and w>0])) if len(widths)>1 else 0.0
        peers_h = float(np.median([h for i,h in enumerate(heights) if i>0 and h>0])) if len(heights)>1 else 0.0

        # OCR per slot
        slot_digits, slot_confs = [], []
        for bcell, gcell in zip(padded_bin, padded_gray):
            d, cf = fuse_digit(gcell, bcell)
            slot_digits.append(d if d else ""); slot_confs.append(max(cf, 0.0))

        # ===== FIRST-DIGIT REPAIRS (more aggressive) =====
        if len(slot_digits) >= 1:
            override_one = False

            # (1) Narrow-ish + tall bar heuristic (loosened)
            if (peers_w > 0 and peers_h > 0 and widths[0] > 0 and heights[0] > 0):
                wr = widths[0] / peers_w
                hr = heights[0] / peers_h
                if wr <= 0.95 and hr >= 0.88 and slot_digits[0] in ("", "0", "8"):
                    override_one = True

            # (2) Segment pattern: strong b,c; others weak
            if not override_one:
                s0 = seg_strengths(gray_cells[0])  # a,b,c,d,e,f,g
                if s0[1] > 0.52 and s0[2] > 0.52 and max(s0[0], s0[3], s0[4], s0[5], s0[6]) < 0.38:
                    override_one = True

            # (3) Margin vote between 1 and second best
            if not override_one:
                d0, m0 = seg_best_vs_second(gray_cells[0])
                if d0 == "1" and m0 >= 0.10 and slot_digits[0] in ("", "0", "8", "7"):
                    override_one = True

            # (4) Hole-free vertical stroke dominance
            if not override_one:
                bc0 = padded_bin[0]
                holes = _count_holes(bc0)
                ys, xs = np.where(bc0 > 0)
                if holes == 0 and len(xs):
                    x0, x1 = xs.min(), xs.max()
                    bw = x1 - x0 + 1
                    col = (bc0[:, x0:x1+1].sum(axis=0) / 255.0)
                    mid = col[int(0.30*bw):int(0.70*bw)].sum()
                    tot = col.sum() + 1e-6
                    if (mid / tot) >= 0.65 and slot_digits[0] in ("", "0", "8"):
                        override_one = True

            if override_one:
                slot_digits[0] = "1"
                slot_confs[0] = max(slot_confs[0], 90.0)

        # ===== SLOT-1 “9 vs 8/1/7” override (stronger) =====
        if len(gray_cells) >= 2:
            d1, m1 = seg_best_vs_second(gray_cells[1])
            s = seg_strengths(gray_cells[1])  # a,b,c,d,e,f,g
            looks_like_9 = (
                (d1 == "9" and m1 >= 0.18) or
                (s[1] > 0.55 and s[2] > 0.55 and s[6] > 0.42 and s[4] < 0.30)  # b,c,g on; e off
            )
            if looks_like_9 and slot_digits[1] in ("", "1", "7", "8", "0"):
                slot_digits[1] = "9"
                slot_confs[1] = max(slot_confs[1], 85.0)

        # pack
        ds = re.sub(r"\D","", "".join(slot_digits))
        if ds:
            if len(ds) < params.slots: ds = ds.zfill(params.slots)
            elif len(ds) > params.slots: ds = ds[-params.slots:]
        conf_slot = (sum(slot_confs)/len(slot_confs)/100.0) if slot_confs else 0.0
        pen = _repetition_penalty(ds) if ds else 0

        # left_score (prefer ROIs whose first slot looks like a “1”)
        left_score = 0.0
        if len(padded_bin) >= 1 and peers_w > 0 and peers_h > 0:
            left_score = score_slot0_one(padded_bin[0], peers_w, peers_h)

        score = (2 if len(ds) == params.slots else 1 if abs(len(ds)-params.slots)<=1 else 0) + conf_slot - pen
        return {
            "kr": kr,
            "roi_proc": roi_proc, "roi_dark": roi_dark,
            "cells_dark": [c for c in padded_bin],
            "gray_cells": [c for c in padded_gray],
            "slot_digits": slot_digits, "slot_confs": slot_confs,
            "ds": ds, "conf_slot": conf_slot, "pen": pen,
            "score": score, "left_score": left_score
        }

    # ---------- Sweep KRs (favor good slot-0) ----------
    kr0 = clamp(params.keep_right, 0.50, 0.98)
    sweep = sorted(set([kr0] + [clamp(kr0-d,0.50,0.98) for d in (0.04,0.08,0.12,0.16,0.20,0.24)]))
    results = [run_for_kr(kr) for kr in sweep]
    best = max(results, key=lambda r: (r["left_score"] >= 0.40, round(r["left_score"],3), r["score"]))

    # ---------- Whole-line tokens (prefer clean decimal) ----------
    tokens_all = ocr_whole_tokens(best["roi_proc"]) + ocr_whole_tokens(best["roi_dark"])
    target_len = params.slots
    tok_strong = best_decimal_token(tokens_all, target_len)

    # Decide slot vs whole-line
    ds, conf_slot = best["ds"], best["conf_slot"]
    pen = _repetition_penalty(ds) if ds else 0
    score_slot = (2 if len(ds) == target_len else 1 if abs(len(ds)-target_len)<=1 else 0) + conf_slot - pen

    digits_full = tok_strong["digits"] if tok_strong else ""
    c_full = float(tok_strong["conf"]) if tok_strong else 0.0
    score_full = (6 if tok_strong else 0) + c_full
    choose_full = (score_full >= score_slot) and bool(digits_full)

    # Nudge: if slot starts "01" but token shows "19x.x", prefer token
    if not choose_full and ds and params.decimals is not None and ds.startswith("01") and tok_strong:
        if re.match(r"19\d\.\d", tok_strong["text"] or ""):
            choose_full = True

    chosen_digits_raw = digits_full if choose_full else ds
    value = _format_value(
        chosen_digits_raw, params.slots, params.decimals,
        (tok_strong["text"] if (choose_full and tok_strong) else None)
    )
    conf = c_full if choose_full else conf_slot
    source = "full" if choose_full else "slot"

    # ---------- Quality & debug ----------
    q = assess_quality(bgr, lcd, best["roi_proc"], best["cells_dark"])
    out = {
        "value": value,
        "digits": re.sub(r"\D","", chosen_digits_raw or ""),
        "confidence": round(float(conf), 3),
        "confidence_source": source,
        "timing_ms": int((time.time()-t0)*1000),
        "quality": {"ok": q.ok, "issues": q.issues, "metrics": q.metrics},
        "params": {
            "slots": params.slots, "keep_right": params.keep_right,
            "scale": params.scale, "splitter": params.splitter,
            "failsafe": params.failsafe, "no_lcd": params.no_lcd,
            "decimals": params.decimals,
        }
    }
    if want_debug:
        out["debug"] = {
            "lcd_b64": _png_b64(lcd),
            "band_b64": _png_b64(band),
            "roi_b64": _png_b64(best["roi_proc"]),
            "cells_b64": [ _png_b64(c) for c in best["cells_dark"][:params.slots] ]
        }
    return out

