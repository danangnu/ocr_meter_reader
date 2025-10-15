# main.py
from fastapi import FastAPI
from routers.meter_router import router as meter_router

app = FastAPI(title="Meter OCR API", version="1.2.0")
app.include_router(meter_router)