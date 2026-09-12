from collections import defaultdict, deque
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# Load YOLO model and OCR reader
model = YOLO("license_plate_best.pt")
reader = easyocr.Reader(["en"], gpu=True)

# Tracks plate detections over frames
plate_history = defaultdict(lambda: deque(maxlen=12))
plate_final = {}
plate_confidence = defaultdict(int)

CONF_THRES = 0.3


def plate_preprocess(plate_image):
    """Preprocesses a cropped license plate image for OCR extraction."""
    if plate_image.size == 0:
        return ""

    grey_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_image, (3, 3), 0)
    thres = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    plate_resized = cv2.resize(
        thres, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            paragraph=True,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        )
        if ocr_result:
            return ocr_result[0].strip().upper().replace(" ", "")
    except Exception:
        pass

    return ""


def get_stable_pattern(track_id, new_text):
    """Stores OCR readings across frames and returns the most frequently seen text."""
    if new_text and len(new_text.replace(" ", "")) >= 3:
        cleaned_text = new_text.strip().upper()

        if cleaned_text:
            plate_history[track_id].append(cleaned_text)

            text_counts = {}
            for text in plate_history[track_id]:
                text_counts[text] = text_counts.get(text, 0) + 1

            most_common = max(text_counts, key=text_counts.get)
            max_count = text_counts[most_common]

            if max_count >= 2:
                if (
                    track_id not in plate_final
                    or max_count > plate_confidence[track_id]
                ):
                    plate_final[track_id] = most_common
                    plate_confidence[track_id] = max_count

    return plate_final.get(track_id, "")


def cleanup_stale_tracks(active_track_ids):
    """Removes tracked IDs from memory when they are no longer in frame."""
    active_set = set(active_track_ids)

    tracked_keys = list(plate_history.keys())
    for track_id in tracked_keys:
        if track_id not in active_set:
            del plate_history[track_id]
            plate_final.pop(track_id, None)
            plate_confidence.pop(track_id, None)


def zoom_overlay(frame, x1, y1, x2, y2, text, track_id):
    """Draws a zoomed view of the license plate and its text above the bounding box."""
    stable_text = get_stable_pattern(track_id, text)

    if stable_text and plate_confidence[track_id] >= 2:
        overlay_h, overlay_w = 150, 400

        oy1 = max(0, y1 - overlay_h - 40)
        ox1 = x1
        oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

        if ox2 > frame.shape[1]:
            ox1 = frame.shape[1] - overlay_w
            ox2 = frame.shape[1]

        if y2 > y1 and x2 > x1:
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop is not None and plate_crop.size > 0:
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))
                if (
                    oy1 >= 0
                    and oy2 <= frame.shape[0]
                    and ox1 >= 0
                    and ox2 <= frame.shape[1]
                ):
                    frame[oy1:oy2, ox1:ox2] = plate_resized

        display_text = f"{stable_text}"
        cv2.putText(
            frame,
            display_text,
            (ox1, oy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            5,
        )
        cv2.putText(
            frame,
            display_text,
            (ox1, oy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4,
        )


def main():
    # Use 0 for default webcam, or replace with RTSP/HTTP stream URL
    # e.g., camera_source = "rtsp://admin:password@192.168.1.100:554/stream1"
    camera_source = 0

    cap = cv2.VideoCapture(camera_source)

    # Disable hardware video buffer to process real-time frames only
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open live camera stream.")
        return

    print("Live stream active. Press 'q' to quit.")

    while True:
        # Flush outdated frame buffer for low latency
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            print("Failed to grab frame from stream. Reconnecting...")
            break

        results = model.track(frame, persist=True, verbose=False)
        current_frame_track_ids = []

        for r in results:
            boxes = r.boxes

            if boxes.id is not None:
                track_ids = boxes.id.int().tolist()
                high_conf_indices = [
                    i
                    for i, conf in enumerate(boxes.conf)
                    if float(conf) > CONF_THRES
                ]

                for i in high_conf_indices:
                    box = boxes[i]
                    track_id = track_ids[i]
                    current_frame_track_ids.append(track_id)

                    x1, y1, x2, y2 = box.xyxy.int().tolist()[0]

                    plate_crop = frame[y1:y2, x1:x2]
                    text = plate_preprocess(plate_crop)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    zoom_overlay(frame, x1, y1, x2, y2, text, track_id)

        # Remove memory state for targets that have left the scene
        cleanup_stale_tracks(current_frame_track_ids)

        cv2.imshow("Live License Plate Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()