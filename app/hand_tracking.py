from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe indices for landmarks
TIP_IDS = [4, 8, 12, 16, 20]


@dataclass
class HandResult:
    handedness: str  # "Left" or "Right"
    landmarks: np.ndarray  # shape (21, 3) in pixel coords (x,y,z-ish)
    bbox: Tuple[int, int, int, int]  # x, y, w, h


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7, tracking_confidence: float = 0.6):
        self.max_hands = max_hands
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self._drawer = mp.solutions.drawing_utils
        self._drawer_style = mp.solutions.drawing_styles

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(frame_rgb)
        out: List[HandResult] = []
        if not res.multi_hand_landmarks:
            return out
        for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([(int(p.x * w), int(p.y * h), p.z) for p in lm.landmark], dtype=np.float32)
            x, y, w_box, h_box = self._calc_bbox(pts, (w, h))
            out.append(HandResult(handedness.classification[0].label, pts, (x, y, w_box, h_box)))
        return out

    @staticmethod
    def _calc_bbox(pts: np.ndarray, wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1 = max(int(xs.min()) - 10, 0), max(int(ys.min()) - 10, 0)
        x2, y2 = min(int(xs.max()) + 10, wh[0] - 1), min(int(ys.max()) + 10, wh[1] - 1)
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def draw(frame: np.ndarray, hand: HandResult) -> None:
        # Use mp drawing utils for quick skeleton
        mp_hands = mp.solutions.hands
        # Need normalized landmarks; convert back roughly
        h, w = frame.shape[:2]
        norm = []
        for x, y, z in hand.landmarks:
            norm.append(mp.framework.formats.landmark_pb2.NormalizedLandmark(x=float(x) / w, y=float(y) / h, z=float(z)))
        norm_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark=norm)
        self_draw = mp.solutions.drawing_utils
        self_draw.draw_landmarks(
            frame,
            norm_list,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )


def pinch_distance(hand: HandResult) -> float:
    thumb_tip = hand.landmarks[TIP_IDS[0]][:2]
    index_tip = hand.landmarks[TIP_IDS[1]][:2]
    return float(np.linalg.norm(thumb_tip - index_tip))


def is_pinching(hand: HandResult, threshold: float = 40.0) -> bool:
    # Use dynamic threshold based on hand bbox size for robustness across distances
    dist = pinch_distance(hand)
    dyn = max(hand.bbox[2], hand.bbox[3]) * 0.18  # ~18% of hand size
    return dist < min(threshold, dyn)


def count_extended_fingers(hand: HandResult) -> int:
    """Count how many fingers are extended (up)."""
    # Thumb: tip further from wrist than knuckle
    thumb_extended = hand.landmarks[4][0] > hand.landmarks[3][0] if hand.handedness == "Right" else hand.landmarks[4][0] < hand.landmarks[3][0]
    
    # Other 4 fingers: tip higher (lower y) than PIP joint
    fingers_extended = 0
    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if hand.landmarks[tip_id][1] < hand.landmarks[pip_id][1]:
            fingers_extended += 1
    
    return fingers_extended + (1 if thumb_extended else 0)


def palm_center(hand: HandResult) -> Tuple[int, int]:
    # Approximate as mean of some landmarks (0, 1, 5, 9, 13, 17)
    idxs = [0, 1, 5, 9, 13, 17]
    pts = hand.landmarks[idxs, :2]
    c = np.mean(pts, axis=0)
    return int(c[0]), int(c[1])


def vector_angle(pivot: Tuple[int, int], p: Tuple[int, int]) -> float:
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return math.degrees(math.atan2(dy, dx)) % 360


def fingertip(hand: HandResult, tip_id: int = 8) -> Tuple[int, int]:
    x, y, _ = hand.landmarks[tip_id]
    return int(x), int(y)


def thumb_index_tips(hand: HandResult) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    tx, ty, _ = hand.landmarks[TIP_IDS[0]]
    ix, iy, _ = hand.landmarks[TIP_IDS[1]]
    return (int(tx), int(ty)), (int(ix), int(iy))
