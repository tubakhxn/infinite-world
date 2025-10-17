from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple
from collections import deque

import cv2
import numpy as np


@dataclass
class Dial:
    name: str
    color: Tuple[int, int, int]  # BGR
    center: Tuple[int, int]
    radius: int
    value: float = 0.5  # 0..1
    _target: float = 0.5
    grabbed: bool = False

    def contains(self, pt: Tuple[int, int]) -> bool:
        return np.linalg.norm(np.array(pt) - np.array(self.center)) <= self.radius + 10

    def handle(self, frame: np.ndarray, cursor: Tuple[int, int], grabbing: bool, hand_landmarks=None):
        # If grabbing and cursor near dial, mark grabbed; update value by angle
        if grabbing and (self.grabbed or self.contains(cursor)):
            self.grabbed = True
            ang = math.degrees(math.atan2(cursor[1] - self.center[1], cursor[0] - self.center[0]))
            ang = (ang + 360) % 360
            self._target = (ang % 360) / 360.0
        elif not grabbing:
            self.grabbed = False
        # smooth toward target for premium feel
        self.value += (self._target - self.value) * 0.25
        self.draw(frame, hand_landmarks)

    def draw(self, frame: np.ndarray, hand_landmarks=None):
        # Neon style: draw to overlay then blur and add
        if hand_landmarks is None:
            return
        overlay = np.zeros_like(frame)
        # Define colors and labels for each finger
        colors = [(0,165,255), (0,255,0), (255,0,255), (255,255,255)] # orange, green, magenta, white
        labels = ["biology", "tech", "culture", ""]
        finger_ids = [4, 8, 12, 16] # thumb, index, middle, ring
        palm_base = tuple(int(x) for x in hand_landmarks[0][:2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        # Draw neon lines and labels
        for i, fid in enumerate(finger_ids):
            tip = tuple(int(x) for x in hand_landmarks[fid][:2])
            cv2.line(overlay, palm_base, tip, colors[i], 6, lineType=cv2.LINE_AA)
            cv2.circle(overlay, tip, 10, colors[i], -1, lineType=cv2.LINE_AA)
            if labels[i]:
                # Position label slightly past the fingertip
                label_pos = (int(tip[0] + 18), int(tip[1] - 8))
                cv2.putText(overlay, labels[i], label_pos, font, font_scale, colors[i], font_thickness, lineType=cv2.LINE_AA)
        # Draw activation circle on palm if dial is grabbed
        if self.grabbed:
            # Use dial color for palm circle
            cv2.circle(overlay, palm_base, 22, self.color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, palm_base, 22, (255,255,255), 2, lineType=cv2.LINE_AA)
        # Glow effect
        blur = cv2.GaussianBlur(overlay, (0, 0), 8)
        cv2.addWeighted(blur, 0.7, frame, 1.0, 0, frame)
        cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)


class HUD:
    def __init__(self, size: Tuple[int, int]):
        w, h = size
        r = 80
        # Start dials near hand (will be repositioned dynamically)
        self.bio = Dial("biology", (0, 165, 255), (int(0.22 * w), int(0.82 * h)), r)
        self.tech = Dial("tech", (0, 255, 0), (int(0.50 * w), int(0.82 * h)), r)
        self.culture = Dial("culture", (255, 0, 255), (int(0.78 * w), int(0.82 * h)), r)
        self.status_text = "Show 3 fingers to generate. Move hand to steer values."
        self.last_world: Tuple[str, str] | None = None
        self.trail = deque(maxlen=20)
        self.active: Dial | None = None
        self._last_cursor: Tuple[int, int] = (0, 0)
        # introductory banner control
        import time
        self._intro_start = time.time()
        self._intro_done = False
        # typewriter reveal state for world text
        self._reveal_progress = 1.0  # 1.0 = fully shown
        self._is_preview = False
        # glitch mode state (shown while grabbing)
        self._glitch_on = False
        self._glitch_tick = 0
        # track hand for dial positioning
        self._hand_detected = False

    def update(self, frame: np.ndarray, cursor: Tuple[int, int], grabbing: bool, hand_detected: bool = False, hand_landmarks=None):
        # Dim background a bit for a cinematic look
        dim = np.zeros_like(frame)
        cv2.rectangle(dim, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(dim, 0.25, frame, 1.0, 0, frame)

        self._hand_detected = hand_detected
        
        # Position dials around hand when detected
        if hand_detected:
            # Arrange dials in a circle around the hand cursor
            offset = 180
            angle_bio = -120  # upper left
            angle_tech = 0     # right
            angle_culture = -240  # lower left
            
            self.bio.center = (
                int(cursor[0] + offset * np.cos(np.deg2rad(angle_bio))),
                int(cursor[1] + offset * np.sin(np.deg2rad(angle_bio)))
            )
            self.tech.center = (
                int(cursor[0] + offset * np.cos(np.deg2rad(angle_tech))),
                int(cursor[1] + offset * np.sin(np.deg2rad(angle_tech)))
            )
            self.culture.center = (
                int(cursor[0] + offset * np.cos(np.deg2rad(angle_culture))),
                int(cursor[1] + offset * np.sin(np.deg2rad(angle_culture)))
            )

        # Update dials and track which is active
        self.bio.handle(frame, cursor, grabbing, hand_landmarks)
        self.tech.handle(frame, cursor, grabbing, hand_landmarks)
        self.culture.handle(frame, cursor, grabbing, hand_landmarks)
        self.active = next((d for d in (self.bio, self.tech, self.culture) if d.grabbed), None)
        self._last_cursor = cursor

        # Gesture trail while grabbing
        if grabbing:
            self.trail.append(cursor)
        else:
            self.trail.clear()
        for i in range(1, len(self.trail)):
            c1, c2 = self.trail[i - 1], self.trail[i]
            color = self.active.color if self.active else (200, 200, 200)
            cv2.line(frame, c1, c2, color, 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, c2, 5, color, -1, lineType=cv2.LINE_AA)

        # Active dial focus elements (extra arcs) near upper-left
        if self.active:
            self._draw_focus(frame, self.active, anchor=cursor)
            self._intro_done = True

        # Panel: glitch when grabbing, otherwise show world text
        self._glitch_on = grabbing
        self._draw_status(frame)
        self._draw_intro(frame)

    def _draw_status(self, frame: np.ndarray):
        (tw, th), _ = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
        cv2.putText(frame, self.status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        # Right side text panel like the reference
        if self._glitch_on:
            self._draw_glitch_panel(frame)
        elif self.last_world:
            title, body = self._typewriter_text()
            h, w = frame.shape[:2]
            panel_x = int(w * 0.56)
            pad = 14
            # Dynamically wrap based on panel width to avoid cutoff
            max_chars = max(28, int((w - panel_x - 30) / 10))
            lines = [title, ""] + wrap_text(body, max_chars)
            # semi-transparent dark backing rectangle
            y0 = 80
            y1 = y0 + 28 * len(lines) + pad
            cv2.rectangle(frame, (panel_x - pad, y0 - pad), (w - 15, y1), (0, 0, 0), -1)
            y = y0
            for i, line in enumerate(lines):
                scale = 0.8 if i == 0 else 0.65
                weight = 2 if i == 0 else 1
                cv2.putText(frame, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), weight, lineType=cv2.LINE_AA)
                y += 28
            # preview indicator
            if self._is_preview:
                cv2.putText(frame, "preview", (panel_x, y0 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    def values(self) -> Tuple[float, float, float]:
        return self.bio.value, self.tech.value, self.culture.value

    # --- world text handling ---
    def set_world(self, title: str, body: str, preview: bool = False, animate: bool = True):
        self.last_world = (title, body)
        self._is_preview = preview
        self._reveal_progress = 0.0 if animate else 1.0

    def _typewriter_text(self) -> Tuple[str, str]:
        if not self.last_world:
            return ("", "")
        title, body = self.last_world
        # advance reveal
        import time
        if self._reveal_progress < 1.0:
            # Reveal roughly 120 chars per second for body
            dt = 1/30.0  # approx frame step
            self._reveal_progress = min(1.0, self._reveal_progress + dt * (2.0 if self._is_preview else 1.0))
        n = int(len(body) * self._reveal_progress)
        return (title, body[:max(0, n)])

    # --- glitch rendering ---
    def _draw_glitch_panel(self, frame: np.ndarray):
        # Produce a block of glyphy nonsense, refreshed often while grabbing
        h, w = frame.shape[:2]
        panel_x = int(w * 0.58)
        pad = 14
        y0 = 80
        rows = 14
        cols = 36
        # semi-transparent dark backing rectangle
        y1 = y0 + 22 * rows + pad
        cv2.rectangle(frame, (panel_x - pad, y0 - pad), (w - 20, y1), (0, 0, 0), -1)
        # generate glyphs deterministically but wiggly
        rng = np.random.default_rng(int(self._glitch_tick // 2) + int(self.bio.value*1000) + int(self.tech.value*100) + int(self.culture.value*10))
        self._glitch_tick += 1
        glyphs = list(";:+=[]{}()^@*&$#%/\\|<>?!._-~")
        for r in range(rows):
            line = "".join(rng.choice(glyphs) for _ in range(cols))
            y = y0 + r * 22
            cv2.putText(frame, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, lineType=cv2.LINE_AA)


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) - 1 + len(w) + 1 > width and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


    
def _ring(overlay: np.ndarray, center: Tuple[int, int], r: int, color: Tuple[int, int, int]):
    cv2.circle(overlay, center, r, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, int(r*0.75), color, 2, lineType=cv2.LINE_AA)
    for a0 in range(-60, 241, 20):
        ang = np.deg2rad(a0)
        p1 = (int(center[0] + r * np.cos(ang)), int(center[1] + r * np.sin(ang)))
        p2 = (int(center[0] + (r+10) * np.cos(ang)), int(center[1] + (r+10) * np.sin(ang)))
        cv2.line(overlay, p1, p2, color, 2, lineType=cv2.LINE_AA)


def _neon_add(frame: np.ndarray, overlay: np.ndarray, sigma: float = 6.0, glow: float = 0.6, bright: float = 1.0):
    blur = cv2.GaussianBlur(overlay, (0, 0), sigma)
    cv2.addWeighted(blur, glow, frame, 1.0, 0, frame)
    cv2.addWeighted(overlay, bright, frame, 1.0, 0, frame)


def draw_arc(overlay: np.ndarray, center: Tuple[int, int], r: int, start: int, end: int, color: Tuple[int, int, int], thickness: int = 6):
    cv2.ellipse(overlay, center, (r, r), 0, start, end, color, thickness, lineType=cv2.LINE_AA)


def draw_focus_text(frame: np.ndarray, dial: Dial, topleft: Tuple[int, int]):
    x, y = topleft
    overlay = np.zeros_like(frame)
    # build arcs stack like the biology reference
    draw_arc(overlay, (x+120, y+120), 90, -30, 210, dial.color, 6)
    draw_arc(overlay, (x+120, y+120), 70, -20, 200, dial.color, 3)
    draw_arc(overlay, (x+120, y+120), 110, 160, 230, (255, 255, 255), 3)
    _neon_add(frame, overlay)
    # value digits
    vtxt = f"{dial.value:0.2f}"
    cv2.putText(frame, vtxt, (x+95, y+135), cv2.FONT_HERSHEY_SIMPLEX, 2.0, dial.color, 4, lineType=cv2.LINE_AA)
    # label
    cv2.putText(frame, dial.name, (x+160, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dial.color, 2, lineType=cv2.LINE_AA)


def HUD__draw_focus(self: 'HUD', frame: np.ndarray, dial: Dial, anchor: Tuple[int, int]):
    # place the focus graphics slightly up-left of the hand cursor
    ax = max(anchor[0] - 200, 10)
    ay = max(anchor[1] - 200, 50)
    draw_focus_text(frame, dial, (ax, ay))


# Bind as a method
HUD._draw_focus = HUD__draw_focus


def HUD__draw_intro(self: 'HUD', frame: np.ndarray):
    import time
    if self._intro_done:
        return
    elapsed = time.time() - self._intro_start
    if elapsed > 6.0:
        self._intro_done = True
        return
    title = "an infinite world builder"
    subtitle = "with gestures"
    tagline = "steer bio · tech · culture as parameters for llm"
    h, w = frame.shape[:2]
    y = 40
    cv2.putText(frame, title, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, subtitle, (20, y+36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, tagline, (20, y+72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1, lineType=cv2.LINE_AA)


HUD._draw_intro = HUD__draw_intro
