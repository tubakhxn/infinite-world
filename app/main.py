from __future__ import annotations

import argparse
import os
from typing import Tuple, Optional

import cv2
import numpy as np

from .hand_tracking import HandTracker, HandResult, is_pinching, fingertip, palm_center, thumb_index_tips, count_extended_fingers
from .hud import HUD
from .utils import HoldTimer, RateLimiter
from .worldgen import generate_world, quick_preview_world


def estimate_open_palm(hand: HandResult) -> bool:
    # Heuristic: distance between wrist (0) and middle tip (12) relative to bbox height
    wrist = hand.landmarks[0][:2]
    mid_tip = hand.landmarks[12][:2]
    dist = float(np.linalg.norm(wrist - mid_tip))
    return dist > hand.bbox[3] * 0.6  # fairly open


def _open_camera(index: int, backend: str) -> cv2.VideoCapture:
    # Windows: try DirectShow first if requested
    if backend == "dshow":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(index, cv2.CAP_MSMF)
    return cv2.VideoCapture(index)


def _backend_name(backend: str) -> str:
    return {"dshow": "DirectShow", "msmf": "Media Foundation", "any": "Auto"}.get(backend, backend)


def list_cameras(max_index: int = 8) -> list[tuple[int, str]]:
    found = []
    for i in range(max_index + 1):
        for bname, bflag in [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", 0)]:
            cap = cv2.VideoCapture(i, bflag) if bflag else cv2.VideoCapture(i)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                found.append((i, bname))
                break
    return found


def run(camera: int, width: int, height: int, flip: bool, backend: str, auto_camera: bool):
    cap = _open_camera(camera, backend)
    if not cap.isOpened():
        # Try alternative backends and indices if auto
        tried = [(camera, backend)]
        if auto_camera:
            for idx in range(0, 8):
                for b in ([backend] if backend != "any" else ["dshow", "msmf", "any"]):
                    if (idx, b) in tried:
                        continue
                    cap = _open_camera(idx, b)
                    if cap.isOpened():
                        camera, backend = idx, b
                        tried.append((idx, b))
                        break
                if cap.isOpened():
                    break
    if not cap.isOpened():
        print("ERROR: Could not open camera. Try Settings > Privacy & Security > Camera: allow for Desktop apps, and close other apps using the webcam.")
        print("Tip: Run with --list-cameras to discover a working index/backend.")
        return
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    tracker = HandTracker(max_hands=1)
    hud = HUD((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720))
    hold = HoldTimer(hold_seconds=1.0)
    preview_rate = RateLimiter(fps=6.0)
    # snapshots folder
    import os, time
    snap_dir = os.path.join(os.getcwd(), "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    print(f"Camera index: {camera} | Backend: {_backend_name(backend)} | Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)

        hands = tracker.process(frame)
        grabbing = False
        cursor = (frame.shape[1] // 2, frame.shape[0] // 2)
        open_palm = False
        three_fingers = False
        hand_landmarks = None
        if hands:
            hand = hands[0]
            hand_landmarks = hand.landmarks
            cursor = fingertip(hand, 8)
            grabbing = is_pinching(hand)
            finger_count = count_extended_fingers(hand)
            # Rotation: angle between wrist and index tip
            wrist = hand.landmarks[0][:2]
            index_tip = hand.landmarks[8][:2]
            dx, dy = index_tip[0] - wrist[0], index_tip[1] - wrist[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angle = (angle + 360) % 360
            # Gesture mapping
            thumb_up = finger_count == 1
            two_fingers = finger_count == 2
            three_fingers = finger_count == 3
            open_palm = estimate_open_palm(hand) and not grabbing and not thumb_up and not two_fingers and not three_fingers
            # Draw indicators
            color = (0, 255, 0) if grabbing else ((0, 255, 0) if thumb_up else ((255, 165, 0) if two_fingers else ((255, 0, 255) if three_fingers else ((0, 165, 255) if open_palm else (200, 200, 200)))))
            cv2.circle(frame, cursor, 10, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, palm_center(hand), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            tpt, ipt = thumb_index_tips(hand)
            cv2.line(frame, tpt, ipt, (0, 255, 0) if grabbing else (180, 180, 180), 2, lineType=cv2.LINE_AA)
            # Control dials and preview
            if thumb_up and preview_rate.ready():
                hud.tech.value = min(1.0, max(0.0, angle / 360.0))
                b, t, c = hud.values()
                world = quick_preview_world(b, t, c)
                hud.set_world(world.title, world.description, preview=True, animate=True)
            elif two_fingers and preview_rate.ready():
                hud.bio.value = min(1.0, max(0.0, angle / 360.0))
                b, t, c = hud.values()
                world = quick_preview_world(b, t, c)
                hud.set_world(world.title, world.description, preview=True, animate=True)
            elif three_fingers and preview_rate.ready():
                hud.culture.value = min(1.0, max(0.0, angle / 360.0))
                b, t, c = hud.values()
                world = quick_preview_world(b, t, c)
                hud.set_world(world.title, world.description, preview=True, animate=True)

        # The following code should be inside the main loop, not inside the gesture block
        hud.update(frame, cursor, grabbing, hand_detected=len(hands) > 0, hand_landmarks=hand_landmarks)
        # Draw open-palm progress ring when charging
        if open_palm and not grabbing and hold.progress > 0.0:
            pct = int(hold.progress * 360)
            cv2.ellipse(frame, cursor, (28, 28), 0, 0, pct, (255, 255, 255), 4, lineType=cv2.LINE_AA)

        if hold.update(open_palm):
            b, t, c = hud.values()
            world = generate_world(b, t, c)
            hud.set_world(world.title, world.description, preview=False, animate=True)
            # Draw updated status and save snapshot
            hud._draw_status(frame)
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe_title = (world.title or "world").replace(" ", "_")[:30]
            img_path = os.path.join(snap_dir, f"{ts}_{safe_title}.png")
            txt_path = os.path.join(snap_dir, f"{ts}_{safe_title}.txt")
            cv2.imwrite(img_path, frame)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(world.title + "\n\n" + world.description)

        cv2.imshow("Gesture World Builder", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            # manual snapshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(snap_dir, f"snapshot_{ts}.png")
            cv2.imwrite(img_path, frame)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gesture-driven world builder with OpenCV + MediaPipe")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror the camera image")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="any", help="Windows camera backend")
    parser.add_argument("--auto-camera", action="store_true", help="Auto-try indices/backends if the chosen one fails")
    parser.add_argument("--list-cameras", action="store_true", help="List detected cameras and exit")
    args = parser.parse_args()
    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("No cameras detected. Check Privacy settings and other apps.")
        else:
            print("Detected cameras (index, backend):")
            for idx, b in cams:
                print(f"  {idx}  {b}")
        return
    run(args.camera, args.width, args.height, not args.no_flip, args.backend, args.auto_camera)


if __name__ == "__main__":
    main()
