"""
Finger Frame Camera
--------------------
Make a rectangle with both hands (thumbs + index fingers), hold it steady,
and the app counts down and captures a photo cropped to that rectangle.

Usage:
    python main.py [--camera 0] [--output captured_photos]
"""

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import cv2
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("finger_frame_camera")

Rect = Tuple[int, int, int, int]


@dataclass
class Config:
    camera_index: int = 0
    output_dir: str = "captured_photos"
    smoothing_frames: int = 5
    stability_threshold_px: int = 8
    stable_time_required_s: float = 1.0
    countdown_seconds: float = 3.0
    post_capture_hold_s: float = 1.0
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    window_name: str = "Finger Frame Camera"


class FingerFrameCamera:
    def __init__(self, config: Config):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        self.frame_history: deque[Rect] = deque(maxlen=config.smoothing_frames)
        self.stable_start_time: Optional[float] = None

        self.countdown_started = False
        self.countdown_start_time: Optional[float] = None
        self.locked_rectangle: Optional[Rect] = None
        self.photo_captured = False

        os.makedirs(config.output_dir, exist_ok=True)

    # -- lifecycle -----------------------------------------------------

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.cfg.camera_index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()

    def run(self):
        assert self.cap is not None, "Camera not initialized; use 'with FingerFrameCamera(...) as cam'"
        while True:
            success, frame = self.cap.read()
            if not success:
                log.warning("Failed to read frame from camera; stopping.")
                break

            frame = cv2.flip(frame, 1)
            frame = self._process_frame(frame)

            cv2.imshow(self.cfg.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # -- per-frame logic -------------------------------------------------

    def _process_frame(self, frame):
        hand_points = self._detect_hand_points(frame)

        if not self.countdown_started:
            self._update_tracking(frame, hand_points)
        else:
            self._update_countdown(frame)

        return frame

    def _detect_hand_points(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_points = []
        if results.multi_hand_landmarks:
            height, width, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                hand_points.append({
                    "thumb": (int(thumb.x * width), int(thumb.y * height)),
                    "index": (int(index.x * width), int(index.y * height)),
                })
        return hand_points

    def _update_tracking(self, frame, hand_points):
        if len(hand_points) != 2:
            self.frame_history.clear()
            self.stable_start_time = None
            return

        current_rect = self._smoothed_rectangle(hand_points)
        self._check_stability(current_rect)

        x1, y1, x2, y2 = current_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)

        if self.stable_start_time is not None and not self.countdown_started:
            stable_duration = time.time() - self.stable_start_time
            if stable_duration < self.cfg.stable_time_required_s:
                cv2.putText(frame, "Hold still...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    def _smoothed_rectangle(self, hand_points) -> Rect:
        all_points = [
            hand_points[0]["thumb"], hand_points[0]["index"],
            hand_points[1]["thumb"], hand_points[1]["index"],
        ]
        x_values = [p[0] for p in all_points]
        y_values = [p[1] for p in all_points]
        raw_rect = (min(x_values), min(y_values), max(x_values), max(y_values))

        self.frame_history.append(raw_rect)
        n = len(self.frame_history)
        smooth = tuple(
            int(sum(p[i] for p in self.frame_history) / n) for i in range(4)
        )
        return smooth  # type: ignore[return-value]

    def _check_stability(self, current_rect: Rect):
        if len(self.frame_history) < 2:
            return

        prev = self.frame_history[-2]
        movement = max(abs(a - b) for a, b in zip(current_rect, prev))

        if movement < self.cfg.stability_threshold_px:
            if self.stable_start_time is None:
                self.stable_start_time = time.time()

            stable_duration = time.time() - self.stable_start_time
            if stable_duration >= self.cfg.stable_time_required_s:
                self._lock_and_start_countdown(current_rect)
        else:
            self.stable_start_time = None

    def _lock_and_start_countdown(self, rect: Rect):
        self.locked_rectangle = rect
        self.countdown_started = True
        self.countdown_start_time = time.time()
        log.info("Rectangle locked at %s; starting countdown.", rect)

    def _update_countdown(self, frame):
        assert self.locked_rectangle is not None and self.countdown_start_time is not None
        x1, y1, x2, y2 = self.locked_rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)

        elapsed = time.time() - self.countdown_start_time
        countdown_text = self._countdown_text(elapsed)
        self._draw_centered_text(frame, countdown_text)

        if elapsed >= self.cfg.countdown_seconds and not self.photo_captured:
            self._capture_photo(frame)
            self.photo_captured = True

        if elapsed >= self.cfg.countdown_seconds + self.cfg.post_capture_hold_s:
            self._reset_state()

    def _countdown_text(self, elapsed: float) -> str:
        remaining = self.cfg.countdown_seconds - elapsed
        if remaining > 0:
            return str(int(remaining) + 1)
        return "CAPTURE!"

    def _draw_centered_text(self, frame, text: str):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)

    def _capture_photo(self, frame):
        assert self.locked_rectangle is not None
        x1, y1, x2, y2 = self.locked_rectangle
        h, w, _ = frame.shape

        cx1, cy1 = max(0, min(x1, x2)), max(0, min(y1, y2))
        cx2, cy2 = min(w, max(x1, x2)), min(h, max(y1, y2))

        if cx2 <= cx1 or cy2 <= cy1:
            log.warning("Invalid crop region %s; skipping capture.", (cx1, cy1, cx2, cy2))
            return

        captured_photo = frame[cy1:cy2, cx1:cx2].copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.cfg.output_dir, f"photo_{timestamp}.jpg")
        cv2.imwrite(filename, captured_photo)
        log.info("Photo saved: %s", filename)

    def _reset_state(self):
        self.countdown_started = False
        self.countdown_start_time = None
        self.locked_rectangle = None
        self.stable_start_time = None
        self.photo_captured = False
        self.frame_history.clear()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Finger frame gesture camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default="captured_photos", help="Output directory for photos")
    args = parser.parse_args()
    return Config(camera_index=args.camera, output_dir=args.output)


def main():
    config = parse_args()
    try:
        with FingerFrameCamera(config) as camera:
            camera.run()
    except RuntimeError as e:
        log.error(str(e))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")


if __name__ == "__main__":
    main()