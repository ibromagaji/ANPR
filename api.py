from datetime import datetime
from multiprocessing import connection
from uuid import UUID

import psycopg
from fastapi import FastAPI
from pydantic import BaseModel, Field
from psycopg.rows import dict_row
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

DB_URL = "postgresql://ju5ti5:magaji%40123@localhost:5432/anpr_db"

app = FastAPI()

class DetectionIn(BaseModel):
    event_id : UUID
    camera_id: str
    plate_number : str
    confidence : float = Field(..., ge=0, le=1)
    captured_at : datetime


@app.get("/")
def index():
    return FileResponse("static/index.html")

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: str):
        for ws in list(self.active):
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
           
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.post('/detections')
async def create_detection(d: DetectionIn):
    async with await psycopg.AsyncConnection.connect(DB_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO detections(
                    event_id, camera_id, plate_number, confidence, captured_at
                )
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (d.event_id, d.camera_id, d.plate_number, d.confidence, d.captured_at)
            )
            rowcount = cur.rowcount

    print("rowcount:", rowcount, "| open sockets:", len(manager.active))
    if rowcount == 1:
        await manager.broadcast(d.model_dump_json())
        
    return {'status': 'ok'}
@app.get('/detections')
async def list_detections():
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT * FROM detections
            ORDER BY captured_at DESC
            LIMIT 50
            """
        ).fetchall()
        return rows

