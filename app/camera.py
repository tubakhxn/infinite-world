from __future__ import annotations

import cv2

# Map friendly names to OpenCV API backends
BACKENDS = {
    "auto": 0,  # let OpenCV decide
    "dshow": cv2.CAP_DSHOW,  # often best on Windows
    "msmf": cv2.CAP_MSMF,    # Windows Media Foundation
    "vfw": cv2.CAP_VFW,
}


def open_camera(index: int = 0, backend: str = "auto", width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    api = BACKENDS.get(backend, 0)
    cap = cv2.VideoCapture(index, api)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def probe_cameras(max_index: int = 4, backend: str = "auto") -> list[int]:
    found = []
    for i in range(max_index + 1):
        cap = open_camera(i, backend)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found
