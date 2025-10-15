# routers/meter_router.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests, re
from urllib.parse import urlparse, urljoin

from services.meter_service import (
    bgr_from_bytes, read_from_bgr, detect_and_crop, pytesseract, Params
)

router = APIRouter(prefix="", tags=["meter"])

class ReadResponse(BaseModel):
    value: str
    digits: str
    confidence: float
    confidence_source: str
    timing_ms: int
    quality: dict
    params: dict
    debug: dict | None = None

class DetectResponse(BaseModel):
    lcd_b64: str
    band_b64: str
    roi_b64: str
    params: dict

@router.get("/healthz")
def healthz():
    try:
        _ = pytesseract.get_tesseract_version()
        return {"ok": True}
    except Exception:
        return {"ok": False}

# -------- robust fetcher: handles HTML+og:image & hotlink headers --------
def _fetch_image_bytes(url: str) -> bytes:
    p = urlparse(url)
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{p.scheme}://{p.netloc}/",
        "Connection": "close",
    }
    r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()

    if "image" in ctype:
        return r.content

    if "html" in ctype or r.text.lstrip().startswith("<!"):
        html = r.text
        m = re.search(r'property=["\']og:image["\']\s*content=["\']([^"\']+)["\']', html, flags=re.I)
        if not m:
            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html, flags=re.I)
        if m:
            img_url = urljoin(url, m.group(1))
            p2 = urlparse(img_url)
            headers["Referer"] = f"{p2.scheme}://{p2.netloc}/"
            r2 = requests.get(img_url, headers=headers, timeout=15, allow_redirects=True)
            r2.raise_for_status()
            if "image" in r2.headers.get("content-type", "").lower():
                return r2.content

    raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL; content-type was {ctype}")

# ----------------------------- endpoints -----------------------------
@router.post("/read", response_model=ReadResponse)
async def read_endpoint(
    file: UploadFile = File(...),
    slots: int = Form(5),
    keep_right: float = Form(0.90),
    scale: float = Form(3.4),
    splitter: str = Form("auto"),   # auto | ccomp | equal
    failsafe: bool = Form(False),
    no_lcd: bool = Form(False),
    decimals: int | None = Form(None),
    debug: bool = Form(False),
):
    content = await file.read()
    bgr = bgr_from_bytes(content)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")
    params = Params(
        slots=slots, keep_right=keep_right, scale=scale,
        splitter=splitter, failsafe=failsafe, no_lcd=no_lcd,
        decimals=decimals,
    )
    return JSONResponse(read_from_bgr(bgr, params, want_debug=debug))

@router.post("/read-url", response_model=ReadResponse)
def read_url_endpoint(
    url: str = Query(..., description="Direct image URL or a web page with an og:image"),
    slots: int = Query(5),
    keep_right: float = Query(0.90),
    scale: float = Query(3.4),
    splitter: str = Query("auto"),  # auto | ccomp | equal
    failsafe: bool = Query(False),
    no_lcd: bool = Query(False),
    decimals: int | None = Query(None),
    debug: bool = Query(False),
):
    try:
        content = _fetch_image_bytes(url)
    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    bgr = bgr_from_bytes(content)
    if bgr is None:
        raise HTTPException(400, "Could not decode image from fetched bytes")

    params = Params(
        slots=slots, keep_right=keep_right, scale=scale,
        splitter=splitter, failsafe=failsafe, no_lcd=no_lcd,
        decimals=decimals,
    )
    return JSONResponse(read_from_bgr(bgr, params, want_debug=debug))

@router.post("/detect", response_model=DetectResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    keep_right: float = Form(0.90),
    scale: float = Form(3.4),
    failsafe: bool = Form(False),
    no_lcd: bool = Form(False),
):
    """Detect/crop only (no OCR). Returns base64 crops: lcd, band, roi."""
    content = await file.read()
    bgr = bgr_from_bytes(content)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")
    params = Params(keep_right=keep_right, scale=scale, failsafe=failsafe, no_lcd=no_lcd)
    lcd, band, roi = detect_and_crop(bgr, params)
    from services.meter_service import _png_b64
    return JSONResponse({
        "lcd_b64": _png_b64(lcd),
        "band_b64": _png_b64(band),
        "roi_b64": _png_b64(roi),
        "params": {
            "keep_right": keep_right,
            "scale": scale,
            "failsafe": failsafe,
            "no_lcd": no_lcd
        }
    })