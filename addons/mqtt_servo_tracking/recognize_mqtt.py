# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
Run:
python addons/mqtt_servo_tracking/recognize_mqtt.py
Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
"FaceMesh points not consistent with Haar box" problem and enables multi-face.
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
"""
from __future__ import annotations
import argparse
import json
import time
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Reuse your known-good alignment method
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.camera_utils import (
    ROTATE_MODES,
    apply_camera_orientation,
    configure_low_latency_capture,
    read_latest_frame,
)
from src.haar_5pt import align_face_5pt
from src.onnx_providers import select_provider_interactive, get_provider_display_name
from src.speaker_protocol import (
    CENTERED,
    MOVED_LEFT,
    MOVED_RIGHT,
    OUT_OF_FRAME,
    SCAN,
    STOPPED,
    EvidenceLogger,
    command_from_error_with_hysteresis,
    command_when_speaker_lost,
    motor_command_for_servo,
    normalize_motor_command,
    tracking_phase_label,
)
from src.speaker_recognition import FaceDisposition, SpeakerMatchView, classify_face, smooth_speaker_label

# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray # (5,2) float32 in FULL-frame coords

class ActionType(Enum):
    FACE_LOCKED = auto()
    FACE_LOST = auto()
    HEAD_LEFT = auto()
    HEAD_RIGHT = auto()
    EYE_BLINK = auto()
    SMILE = auto()

@dataclass
class Action:
    type: ActionType
    timestamp: float
    details: str = ""

@dataclass
class FaceLock:
    target_name: str
    target_emb: np.ndarray
    last_seen: float = field(default_factory=time.time)
    last_position: Optional[Tuple[float, float]] = None
    last_kps: Optional[np.ndarray] = None
    last_eye_dist: Optional[float] = None
    last_mouth_size: Optional[float] = None
    history: List[Action] = field(default_factory=list)
    consecutive_frames: int = 0
    
    def update_position(self, kps: np.ndarray) -> List[Action]:
        actions = []
        current_time = time.time()
        
        # Calculate face center
        center_x = kps[:, 0].mean()
        center_y = kps[:, 1].mean()
        
        # Detect head movement
        if self.last_position is not None:
            dx = center_x - self.last_position[0]
            if dx > 10:  # Threshold for right movement
                actions.append(Action(ActionType.HEAD_RIGHT, current_time, f"Moved right by {dx:.1f}px"))
            elif dx < -10:  # Threshold for left movement
                actions.append(Action(ActionType.HEAD_LEFT, current_time, f"Moved left by {abs(dx):.1f}px"))
        
        # Detect eye blink (using vertical distance between eyes and nose)
        eye_level = (kps[0, 1] + kps[1, 1]) / 2  # Average y of both eyes
        nose_y = kps[2, 1]
        eye_dist = abs(eye_level - nose_y)
        
        if self.last_eye_dist is not None:
            if eye_dist < self.last_eye_dist * 0.7:  # Threshold for blink
                actions.append(Action(ActionType.EYE_BLINK, current_time, "Blink detected"))
        
        # Detect smile (using mouth width/height ratio)
        mouth_width = abs(kps[3, 0] - kps[4, 0])
        mouth_height = abs(kps[3, 1] - kps[4, 1])
        mouth_ratio = mouth_width / (mouth_height + 1e-5)
        
        if self.last_mouth_size is not None and mouth_ratio > 1.5 * self.last_mouth_size:
            actions.append(Action(ActionType.SMILE, current_time, f"Smile detected (ratio: {mouth_ratio:.2f})"))
        
        # Update state
        self.last_position = (center_x, center_y)
        self.last_kps = kps.copy()
        self.last_eye_dist = eye_dist
        self.last_mouth_size = mouth_ratio
        self.last_seen = current_time
        self.consecutive_frames += 1
        
        return actions


@dataclass
class AnalyzedFace:
    index: int
    det: FaceDet
    emb: np.ndarray
    aligned: np.ndarray
    match: MatchResult
    view: SpeakerMatchView


def pick_primary_speaker_index(
    analyses: List[AnalyzedFace],
    speaker_name: str,
    face_lock: Optional[FaceLock],
) -> Optional[int]:
    """One authorized speaker per frame — best match, biased toward the locked track."""
    candidates = [a for a in analyses if a.match.name == speaker_name and a.match.accepted]
    if not candidates:
        return None
    if face_lock and face_lock.last_position is not None:
        ref_x, ref_y = face_lock.last_position

        def track_score(item: AnalyzedFace) -> float:
            cx = float(item.det.kps[:, 0].mean())
            cy = float(item.det.kps[:, 1].mean())
            pos_dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            return float(item.match.distance) + (pos_dist / 400.0)

        return min(candidates, key=track_score).index
    return min(candidates, key=lambda item: item.match.distance).index


def reacquire_locked_speaker(
    analyses: List[AnalyzedFace],
    speaker_name: str,
    face_lock: FaceLock,
    max_dist_px: float = 300.0,
) -> Optional[AnalyzedFace]:
    """Re-lock onto the authorized speaker near the last known position."""
    if face_lock.last_position is None:
        return None
    ref_x, ref_y = face_lock.last_position
    best: Optional[AnalyzedFace] = None
    best_score = float(max_dist_px)
    for item in analyses:
        if item.match.name != speaker_name or not item.match.accepted:
            continue
        cx = float(item.det.kps[:, 0].mean())
        cy = float(item.det.kps[:, 1].mean())
        pos_dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
        score = pos_dist + (item.match.distance * 120.0)
        if score < best_score:
            best_score = score
            best = item
    return best


def nearest_face_to_position(
    faces: List[FaceDet],
    ref_xy: Tuple[float, float],
    max_dist_px: float = 280.0,
) -> Optional[int]:
    """Pick the face landmark center closest to the last tracked position."""
    if not faces:
        return None
    best_idx: Optional[int] = None
    best_dist = float(max_dist_px)
    ref_x, ref_y = ref_xy
    for i, face in enumerate(faces):
        cx = float(face.kps[:, 0].mean())
        cy = float(face.kps[:, 1].mean())
        dist = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
        if dist <= best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

# -------------------------
# Math helpers
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.
    kps: (5,2) in full-frame coords
    """
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    """
    Minimal geometry sanity:
    - eyes not collapsed
    - mouth generally below nose
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True

# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    try:
        data = np.load(str(db_path), allow_pickle=True)
        out: Dict[str, np.ndarray] = {}
        for k in data.files:
            out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
        return out
    except Exception as e:
        print(f"Warning: Failed to load database {db_path}: {e}. Starting with empty DB.")
        return {}

# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.
    Input: 112x112 BGR -> internally RGB + (x-127.5)/128, NHWC float32.
    Output: (1,D) or (D,)
    """
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        if self.debug:
            print("[embed] model:", model_path)
            print("[embed] providers:", self.sess.get_providers())
            print("[embed] input:", self.sess.get_inputs()[0].name, self.sess.get_inputs()[0].shape, self.sess.get_inputs()[0].type)
            print("[embed] output:", self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape, self.sess.get_outputs()[0].type)

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = rgb[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)

# -------------------------
# Multi-face Haar + FaceMesh(ROI) 5pt
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        model_path: str = "models/face_landmarker.task",
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
        
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe"
            )
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for ROI processing
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 5pt indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32) # (x,y,w,h)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        
        if not res.face_landmarks:
            return None
        
        lm = res.face_landmarks[0]
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)
        # enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []
        
        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        
        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            # expand ROI a bit for FaceMesh stability
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue
            
            # map ROI kps back to full-frame coords
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            
            # sanity: eye distance relative to Haar width
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue
            
            # build bbox from kps (centered)
            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            
            out.append(
                FaceDet(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=1.0,
                    kps=kps.astype(np.float32),
                )
            )
        return out

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()

# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.40):
        """
        Face database matcher with cosine similarity.
        
        Args:
            db: Dictionary mapping names to embedding vectors
            dist_thresh: Cosine distance threshold (default 0.40 for better recall).
                        Lower = stricter (fewer false positives, more false negatives).
                        Higher = more lenient (more false positives, fewer false negatives).
        """
        self.db = db
        self.dist_thresh = float(dist_thresh)
        # pre-stack for speed
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        e = emb.reshape(1, -1).astype(np.float32) # (1,D)
        # cosine similarity since both sides are normalized: sim = dot
        sims = (self._mat @ e.T).reshape(-1) # (K,)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )

# -------------------------
# UI Helpers
# -------------------------
def draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    shadow_offset: int = 1,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    font: int = cv2.FONT_HERSHEY_DUPLEX,
) -> None:
    """
    Draw text with a shadow/outline for better readability.
    Uses FONT_HERSHEY_DUPLEX for a cleaner, more modern look.
    """
    x, y = pos
    # Draw shadow (lighter shadow for less bold appearance)
    for dx, dy in [(shadow_offset, shadow_offset), (-shadow_offset, shadow_offset), 
                   (shadow_offset, -shadow_offset), (-shadow_offset, -shadow_offset)]:
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_text_box(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
    padding: int = 8,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
) -> Tuple[int, int]:
    """
    Draw text with a semi-transparent background box for better readability.
    Returns (width, height) of the text box.
    """
    x, y = pos
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
    
    # Create overlay for transparency
    overlay = img.copy()
    box_x1 = x - padding
    box_y1 = y - text_height - padding
    box_x2 = x + text_width + padding
    box_y2 = y + baseline + padding
    
    # Ensure coordinates are within image bounds
    h, w = img.shape[:2]
    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(w, box_x2)
    box_y2 = min(h, box_y2)
    
    if box_x2 > box_x1 and box_y2 > box_y1:
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    draw_text_with_shadow(img, text, (x, y), font_scale, text_color, 1, font=font)
    
    return (text_width + padding * 2, text_height + padding * 2 + baseline)

# -------------------------
# MQTT movement control (BENAX protocol)
# -------------------------
DEFAULT_MQTT_BROKER = "157.173.101.159"
DEFAULT_MOVEMENT_TOPIC = "vision/faithlock/movement"
DEFAULT_STATUS_TOPIC = "vision/faithlock/status"


def compute_face_error_x(kps: np.ndarray, frame_width: int) -> float:
    face_center_x = float(np.mean(kps[:, 0]))
    return face_center_x - (float(frame_width) / 2.0)


def build_dashboard_status(
    movement_command: str,
    movement_error_x: float,
    face_lock: Optional[FaceLock],
    faces_count: int,
    locked_face_found: bool,
    fps: Optional[float],
    threshold: float,
    provider_name: str,
    speaker_confidence: float = 0.0,
    speaker_similarity: float = 0.0,
    speaker_name: Optional[str] = None,
) -> Dict[str, object]:
    cmd = normalize_motor_command(movement_command)
    spk = face_lock.target_name if face_lock else speaker_name
    phase = tracking_phase_label(
        speaker_name=spk,
        locked=face_lock is not None,
        speaker_visible=locked_face_found,
        motor_command=cmd,
    )
    return {
        "timestamp": time.time(),
        "movement": cmd,
        "motor_command": cmd,
        "phase": phase,
        "error_x": round(float(movement_error_x), 2),
        "locked": face_lock is not None,
        "target": face_lock.target_name if face_lock else speaker_name,
        "speaker_id": face_lock.target_name if face_lock else speaker_name,
        "locked_face_found": bool(locked_face_found),
        "speaker_visible": bool(locked_face_found),
        "faces": int(faces_count),
        "fps": round(float(fps), 2) if fps is not None else None,
        "threshold": round(float(threshold), 3),
        "confidence": round(float(speaker_confidence), 4),
        "similarity": round(float(speaker_similarity), 4),
        "provider": provider_name,
    }


def resolve_speaker_name(db: Dict[str, np.ndarray], requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in db:
            print(f"Warning: speaker '{requested}' not in database. Available: {sorted(db.keys())}")
            return None
        return requested
    if len(db) == 1:
        return next(iter(db.keys()))
    if len(db) > 1:
        print(f"Multiple identities in DB {sorted(db.keys())}. Use --speaker-name to select the main speaker.")
    return None


class MqttMovementPublisher:
    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        topic: str,
        status_topic: str,
        client_id: str,
        min_publish_interval: float = 0.15,
        status_min_publish_interval: float = 0.25,
    ):
        self.topic = topic
        self.status_topic = status_topic
        self.min_publish_interval = float(max(0.0, min_publish_interval))
        self.status_min_publish_interval = float(max(0.0, status_min_publish_interval))
        self.last_command: Optional[str] = None
        self.last_publish_at = 0.0
        self.last_status_publish_at = 0.0
        self.connected = False
        self._broker_host = broker_host
        self._broker_port = int(broker_port)
        self._connect_failures = 0

        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed. Add it to requirements and run pip install -r requirements.txt")

        self.client = mqtt.Client(client_id=client_id, clean_session=True, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=2, max_delay=30)
        self.client.loop_start()
        self._connect_blocking(timeout_sec=12.0)

    def _connect_blocking(self, timeout_sec: float = 12.0) -> bool:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            if self.connected:
                return True
            try:
                self.client.reconnect()
            except Exception:
                try:
                    self.client.connect(self._broker_host, self._broker_port, keepalive=45)
                except Exception as exc:
                    self._connect_failures += 1
                    if self._connect_failures <= 3 or self._connect_failures % 5 == 0:
                        print(f"[MQTT] Connect attempt failed: {exc}")
            time.sleep(0.5)
        print(
            f"[MQTT] Could not reach broker {self._broker_host}:{self._broker_port} "
            f"after {timeout_sec:.0f}s — pan commands paused until reconnect."
        )
        return False

    def _on_connect(self, _client, _userdata, _flags, rc, _properties=None):
        self.connected = (rc == 0)
        if self.connected:
            self._connect_failures = 0
            print(f"[MQTT] Connected to {self._broker_host}:{self._broker_port}")
            print(f"[MQTT] Movement topic: {self.topic}")
            print(f"[MQTT] Status topic: {self.status_topic}")
        else:
            print(f"[MQTT] Broker rejected connection (rc={rc})")

    def _on_disconnect(self, _client, _userdata, *args):
        # paho-mqtt V1 passes: (rc)
        # paho-mqtt V2 passes: (disconnect_flags, reason_code, properties)
        rc = 0
        if len(args) == 1:
            rc = int(args[0])
        elif len(args) >= 2:
            rc = int(args[1])

        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}), retrying...")

    def publish(self, command: str, force: bool = False):
        servo_cmd = motor_command_for_servo(command)
        if servo_cmd is None:
            return

        now = time.time()
        is_nudge = servo_cmd in (MOVED_LEFT, MOVED_RIGHT)
        is_center = servo_cmd == CENTERED
        if is_nudge:
            interval = self.min_publish_interval
        elif is_center:
            interval = max(self.min_publish_interval, 0.25)
        else:
            interval = max(self.min_publish_interval, 0.1)

        if (
            not force
            and servo_cmd == self.last_command
            and (now - self.last_publish_at) < interval
        ):
            return
        if not self.connected:
            self._connect_blocking(timeout_sec=2.0)
            if not self.connected:
                return

        info = self.client.publish(self.topic, payload=servo_cmd, qos=0, retain=False)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.last_command = servo_cmd
            self.last_publish_at = now

    def publish_status(self, status: Dict[str, object], force: bool = False):
        now = time.time()
        if not force and (now - self.last_status_publish_at) < self.status_min_publish_interval:
            return
        if not self.connected:
            return

        payload = json.dumps(status, separators=(",", ":"))
        info = self.client.publish(self.status_topic, payload=payload, qos=0, retain=False)
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.last_status_publish_at = now

    def close(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face lock tracking with MQTT direction publishing for ESP8266 servo control.",
    )
    parser.add_argument("--mqtt-broker", default=DEFAULT_MQTT_BROKER, help="MQTT broker host/IP.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument(
        "--mqtt-topic",
        default=DEFAULT_MOVEMENT_TOPIC,
        help="MQTT topic to publish movement commands.",
    )
    parser.add_argument(
        "--mqtt-status-topic",
        default=DEFAULT_STATUS_TOPIC,
        help="MQTT topic to publish dashboard status JSON.",
    )
    parser.add_argument(
        "--mqtt-client-id",
        default=f"face-lock-{int(time.time())}",
        help="MQTT client id.",
    )
    parser.add_argument(
        "--deadzone-px",
        type=float,
        default=85.0,
        help="Horizontal pixel deadzone around frame center for CENTER command.",
    )
    parser.add_argument(
        "--center-exit-hysteresis-px",
        type=float,
        default=60.0,
        help="Extra pixels required to leave CENTER and start LEFT/RIGHT movement.",
    )
    parser.add_argument(
        "--error-smooth-alpha",
        type=float,
        default=0.28,
        help="EMA smoothing factor for horizontal error (0..1). Lower = smoother pan.",
    )
    parser.add_argument(
        "--command-confirm-frames",
        type=int,
        default=4,
        help="Consecutive frames before changing LEFT/RIGHT/CENTER.",
    )
    parser.add_argument(
        "--face-lost-confirm-frames",
        type=int,
        default=20,
        help="Frames without a visible locked face before hold/scan behavior.",
    )
    parser.add_argument(
        "--search-delay-sec",
        type=float,
        default=1.0,
        help="Seconds after lock is lost before the camera starts SCAN sweep.",
    )
    parser.add_argument(
        "--allow-scan",
        action="store_true",
        default=True,
        help="Sweep camera (SCAN) when locked person disappears until they reappear (default: on).",
    )
    parser.add_argument(
        "--no-scan",
        action="store_false",
        dest="allow_scan",
        help="Hold pan position instead of searching when the locked person is lost.",
    )
    parser.add_argument(
        "--mqtt-min-interval",
        type=float,
        default=0.08,
        help="Minimum seconds between repeated pan nudges (MOVED_LEFT/RIGHT).",
    )
    parser.add_argument(
        "--mqtt-status-min-interval",
        type=float,
        default=0.25,
        help="Minimum seconds between dashboard status MQTT messages.",
    )
    parser.add_argument(
        "--disable-mqtt",
        action="store_true",
        help="Run face lock tracking without MQTT publishing.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index for the pan-mounted external USB camera (run scripts/probe_cameras.py to confirm).",
    )
    parser.add_argument(
        "--camera-rotate",
        choices=ROTATE_MODES,
        default="180",
        help="Fix camera mounting orientation (default 180 = upside-down USB cams).",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Capture width (640 recommended for low-latency pan tracking).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Capture height (480 recommended for low-latency pan tracking).",
    )
    parser.add_argument(
        "--lock-reverify-frames",
        type=int,
        default=8,
        help="When locked, re-run face embedding every N frames (landmarks every frame).",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Use CPU ONNX provider without interactive prompt.",
    )
    parser.add_argument(
        "--speaker-name",
        default="Faith",
        help="Enrolled speaker to track (default: Faith).",
    )
    parser.add_argument(
        "--auto-lock-speaker",
        action="store_true",
        default=False,
        help="Automatically lock when the enrolled speaker appears (default: off — press L).",
    )
    parser.add_argument(
        "--no-auto-lock-speaker",
        action="store_false",
        dest="auto_lock_speaker",
        help="Same as default: manual lock with L key.",
    )
    parser.add_argument(
        "--evidence-log",
        action="store_true",
        default=True,
        help="Write CSV/JSONL evidence logs under logs/evidence/ (default: on).",
    )
    parser.add_argument(
        "--no-evidence-log",
        action="store_false",
        dest="evidence_log",
        help="Disable assessment evidence logging.",
    )
    return parser.parse_args()

# -------------------------
# Demo
# -------------------------
def save_action_history(face_name: str, actions: List[Action]):
    if not actions:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{face_name}_history_{timestamp}.txt"
    os.makedirs("logs", exist_ok=True)
    
    with open(f"logs/{filename}", "w") as f:
        for action in actions:
            time_str = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"{time_str} - {action.type.name}: {action.details}\n")

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()
    args.error_smooth_alpha = float(max(0.01, min(1.0, args.error_smooth_alpha)))
    args.command_confirm_frames = int(max(1, args.command_confirm_frames))
    args.face_lost_confirm_frames = int(max(1, args.face_lost_confirm_frames))
    args.lock_reverify_frames = int(max(1, args.lock_reverify_frames))
    args.search_delay_sec = float(max(0.0, args.search_delay_sec))
    args.mqtt_min_interval = float(max(0.0, args.mqtt_min_interval))
    args.mqtt_status_min_interval = float(max(0.0, args.mqtt_status_min_interval))
    args.center_exit_hysteresis_px = float(max(0.0, args.center_exit_hysteresis_px))
    db_path = Path("data/db/face_db.npz")
    os.makedirs("logs", exist_ok=True)
    
    # Select execution provider (CPU/GPU)
    if args.cpu_only:
        providers = ["CPUExecutionProvider"]
    else:
        providers = select_provider_interactive()
    provider_name = get_provider_display_name(providers)
    print(f"\nUsing: {provider_name}")
    
    # Note about CUDA warnings
    if "CUDAExecutionProvider" in providers and providers[0] == "CUDAExecutionProvider":
        print("\nNote: CUDA is selected for maximum performance.")
        print("      If you see CUDA errors about missing DLLs, try DirectML instead.\n")
    
    print("=" * 60 + "\n")
    
    det = HaarFaceMesh5pt(
        min_size=(70, 70),
        debug=False,
    )
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
        providers=providers,
    )
    db = load_db_npz(db_path)
    if not db:
        print("Warning: Database is empty. Please enroll identities first.")
        det.close()
        return
    
    # Stricter threshold when only one speaker is enrolled — reduces false locks on other people.
    # 0.24 distance = 0.76 cosine similarity: strangers rarely exceed ~0.5 sim with a good template,
    # so if the authorized speaker is missed at this level, re-enroll with more samples (15+) in the
    # same lighting/camera used for tracking. Adjustable live with +/-.
    single_speaker_thresh = 0.24 if len(db) == 1 else 0.32
    matcher = FaceDBMatcher(db=db, dist_thresh=single_speaker_thresh)
    speaker_name = resolve_speaker_name(db, args.speaker_name)
    if speaker_name is None:
        print("Error: No authorized speaker configured. Enroll with src.enroll and use --speaker-name.")
        det.close()
        return

    evidence: Optional[EvidenceLogger] = None
    if args.evidence_log:
        evidence = EvidenceLogger()
    last_evidence_at = 0.0
    last_logged_command = ""
    speaker_confidence = 0.0
    speaker_similarity = 0.0
    speaker_distance = 1.0
    
    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        print(f"Camera not available at index {args.camera_index}")
        det.close()
        return
    
    actual_width, actual_height = configure_low_latency_capture(
        cap,
        width=int(args.camera_width),
        height=int(args.camera_height),
    )
    camera_width = int(args.camera_width)
    camera_height = int(args.camera_height)
    print(f"Camera index: {args.camera_index}")
    print(f"Camera orientation: {args.camera_rotate}")
    print(f"Camera resolution: {actual_width}x{actual_height}")
    if actual_width != camera_width or actual_height != camera_height:
        print(f"  (Requested {camera_width}x{camera_height}, camera using {actual_width}x{actual_height})")
    
    print(f"\nBENAX Single-Speaker Tracking - Using {provider_name}")
    print(f"Authorized speaker: {speaker_name}")
    print(f"Auto-lock speaker: {'ON' if args.auto_lock_speaker else 'OFF - press L to lock'}")
    print(f"Search when lost: {'ON (SCAN until reacquire)' if args.allow_scan else 'OFF (hold pan)'}")
    print("Controls: L=lock/unlock | A/F or arrow keys=select face | q=quit | r=reload DB | +/- threshold")
    print("Motor commands: MOVED_LEFT | MOVED_RIGHT | CENTERED | STOPPED | SCAN | OUT_OF_FRAME")
    print(
        f"MQTT movement topic: {args.mqtt_topic} @ {args.mqtt_broker}:{args.mqtt_port}"
        if not args.disable_mqtt
        else "MQTT movement publishing disabled"
    )
    if not args.disable_mqtt:
        print(f"MQTT dashboard status topic: {args.mqtt_status_topic}")
    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    movement_command = STOPPED
    movement_error_x = 0.0
    filtered_error_x: Optional[float] = None
    stable_track_command = CENTERED
    pending_track_command: Optional[str] = None
    pending_track_count = 0
    face_missing_since: Optional[float] = None
    mqtt_publisher: Optional[MqttMovementPublisher] = None

    if args.disable_mqtt:
        print("[MQTT] Disabled by flag (--disable-mqtt)")
    elif mqtt is None:
        print("[MQTT] paho-mqtt is not installed. Movement commands will not be published.")
    else:
        try:
            mqtt_publisher = MqttMovementPublisher(
                broker_host=args.mqtt_broker,
                broker_port=args.mqtt_port,
                topic=args.mqtt_topic,
                status_topic=args.mqtt_status_topic,
                client_id=args.mqtt_client_id,
                min_publish_interval=args.mqtt_min_interval,
                status_min_publish_interval=args.mqtt_status_min_interval,
            )
            mqtt_publisher.publish(STOPPED, force=True)
        except Exception as e:
            print(f"[MQTT] Failed to initialize publisher: {e}")
            mqtt_publisher = None
    
    # Face locking state
    face_lock: Optional[FaceLock] = None
    max_timeout = 40.0  # seconds before unlocking if face is lost
    
    # Face selection for locking (when multiple faces present)
    selected_face_index: Optional[int] = None  # Index of currently selected face (None = auto-select first)
    potential_face_to_lock: Optional[Tuple[str, np.ndarray, np.ndarray]] = None  # (name, emb, kps) of selected face
    
    # Temporal smoothing for better accuracy (average recent embeddings per face)
    # Maps face_id -> list of recent embeddings (max smoothing_window)
    face_embedding_history: Dict[int, List[np.ndarray]] = {}
    smoothing_window = 3  # temporal smoothing for stable identity

    # Additional label smoothing to reduce flicker in displayed names.
    # Maps face_id -> recent predicted labels (e.g., ["Alice", "Alice", "Unknown", ...])
    face_label_history: Dict[int, List[str]] = {}
    label_smoothing_window = 3
    face_lost_streak = 0
    frame_index = 0
    last_scan_publish_at = 0.0
    
    try:
        while True:
            ok, frame = read_latest_frame(cap)
            if not ok or frame is None:
                break

            frame_index += 1
            frame = apply_camera_orientation(frame, args.camera_rotate)
            
            max_faces = 2 if face_lock else 3
            faces = det.detect(frame, max_faces=max_faces)
            vis = frame.copy()
            
            # compute fps
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            # draw + recognize each face
            h, w = vis.shape[:2]
            thumb = 112
            pad = 8
            x0 = w - thumb - pad
            y0 = 80
            shown = 0
            
            # Check if we should unlock due to timeout
            current_time = time.time()
            if (
                face_lock
                and not args.allow_scan
                and (current_time - face_lock.last_seen) > max_timeout
            ):
                save_action_history(face_lock.target_name, face_lock.history)
                print(f"[FaceLock] Timeout - Unlocked {face_lock.target_name}")
                face_lock = None
                selected_face_index = None
                potential_face_to_lock = None
                filtered_error_x = None
                stable_track_command = CENTERED
                pending_track_command = None
                pending_track_count = 0
                face_missing_since = None
                face_lost_streak = 0
                if mqtt_publisher is not None:
                    mqtt_publisher.publish(STOPPED, force=True)

            # Clean up embedding / label history for faces that disappeared
            active_face_ids = set(range(len(faces)))
            face_embedding_history = {k: v for k, v in face_embedding_history.items() if k in active_face_ids}
            face_label_history = {k: v for k, v in face_label_history.items() if k in active_face_ids}
            
            # Reset selection if selected face disappeared
            if selected_face_index is not None and selected_face_index >= len(faces):
                selected_face_index = None
                potential_face_to_lock = None

            locked_face_found = False
            locked_face_kps: Optional[np.ndarray] = None
            ignored_faces_count = 0
            use_fast_track = False

            if (
                face_lock
                and face_lock.last_position is not None
                and faces
                and (current_time - face_lock.last_seen) < 0.5
                and (frame_index % args.lock_reverify_frames != 0)
            ):
                track_idx = nearest_face_to_position(faces, face_lock.last_position)
                if track_idx is not None:
                    use_fast_track = True
                    f = faces[track_idx]
                    locked_face_found = True
                    locked_face_kps = f.kps.copy()
                    # Landmarks only for pan — do not update lock state without identity verify.
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (255, 165, 0), 4)
                    for (x, y) in f.kps.astype(int):
                        cv2.circle(vis, (int(x), int(y)), 2, (255, 165, 0), -1)
                    draw_text_with_shadow(
                        vis,
                        f"{face_lock.target_name} (LOCKED)",
                        (f.x1, max(0, f.y1 - 28)),
                        0.85,
                        (255, 165, 0),
                        1,
                        font=cv2.FONT_HERSHEY_DUPLEX,
                    )

            if not use_fast_track:
                analyses: List[AnalyzedFace] = []
                for i, f in enumerate(faces):
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb_raw = embedder.embed(aligned)

                    if i not in face_embedding_history:
                        face_embedding_history[i] = []
                    face_embedding_history[i].append(emb_raw)
                    if len(face_embedding_history[i]) > smoothing_window:
                        face_embedding_history[i].pop(0)

                    if len(face_embedding_history[i]) > 0:
                        emb_stack = np.stack(face_embedding_history[i], axis=0)
                        emb_smooth = emb_stack.mean(axis=0)
                        emb_smooth = emb_smooth / (np.linalg.norm(emb_smooth) + 1e-12)
                        emb = emb_smooth.astype(np.float32)
                    else:
                        emb = emb_raw

                    mr = matcher.match(emb)
                    view = classify_face(
                        speaker_name=speaker_name,
                        matched_name=mr.name,
                        accepted=mr.accepted,
                        similarity=mr.similarity,
                        distance=mr.distance,
                    )
                    analyses.append(
                        AnalyzedFace(
                            index=i,
                            det=f,
                            emb=emb,
                            aligned=aligned,
                            match=mr,
                            view=view,
                        )
                    )

                primary_speaker_idx = pick_primary_speaker_index(analyses, speaker_name, face_lock)
                if face_lock and primary_speaker_idx is None:
                    reacquired = reacquire_locked_speaker(analyses, speaker_name, face_lock)
                    if reacquired is not None:
                        primary_speaker_idx = reacquired.index

                for item in analyses:
                    i = item.index
                    f = item.det
                    emb = item.emb
                    aligned = item.aligned
                    mr = item.match
                    view = item.view

                    is_primary_speaker = primary_speaker_idx is not None and i == primary_speaker_idx
                    if not is_primary_speaker and (view.is_authorized or mr.name == speaker_name):
                        view = SpeakerMatchView(
                            disposition=FaceDisposition.UNKNOWN,
                            label="Unknown",
                            is_authorized=False,
                            confidence=0.0,
                            similarity=mr.similarity,
                            distance=mr.distance,
                        )

                    is_authorized_speaker = is_primary_speaker
                    if is_authorized_speaker:
                        speaker_confidence = view.confidence
                        speaker_similarity = view.similarity
                        speaker_distance = view.distance
                    elif view.label == "Ignored":
                        ignored_faces_count += 1
                    elif not is_primary_speaker and len(faces) > 1:
                        ignored_faces_count += 1

                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                    for (x, y) in f.kps.astype(int):
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

                    if i not in face_label_history:
                        face_label_history[i] = []
                    stable_label = smooth_speaker_label(
                        face_label_history[i],
                        speaker_name=speaker_name,
                        raw_label=view.label,
                        window=label_smoothing_window,
                    )

                    is_locked_face = False
                    if face_lock and is_primary_speaker:
                        actions = face_lock.update_position(f.kps)
                        face_lock.history.extend(actions)
                        if show_debug:
                            for action in actions:
                                print(f"[Action] {action.type.name}: {action.details}")
                        is_locked_face = True
                        locked_face_found = True
                        locked_face_kps = f.kps.copy()

                    label = stable_label if is_primary_speaker else view.label
                    status = " (LOCKED)" if is_locked_face else ""
                    conf_pct = int(max(0.0, min(100.0, mr.similarity * 100.0)))
                    line1 = f"{label}{status}"
                    line2 = f"conf={conf_pct}% dist={mr.distance:.3f}"

                    is_selected = (selected_face_index == i) if selected_face_index is not None else False

                    if selected_face_index is None and not face_lock and is_primary_speaker:
                        selected_face_index = i
                        is_selected = True

                    if (
                        not face_lock
                        and selected_face_index is not None
                        and i == selected_face_index
                        and mr.name == speaker_name
                        and mr.accepted
                    ):
                        potential_face_to_lock = (speaker_name, emb, f.kps)

                    if (
                        args.auto_lock_speaker
                        and not face_lock
                        and is_primary_speaker
                    ):
                        face_lock = FaceLock(
                            target_name=speaker_name,
                            target_emb=emb,
                            last_seen=current_time,
                        )
                        face_lock.update_position(f.kps)
                        face_lock.history.append(Action(
                            ActionType.FACE_LOCKED,
                            current_time,
                            f"Speaker auto-locked: {speaker_name}",
                        ))
                        filtered_error_x = None
                        stable_track_command = CENTERED
                        pending_track_command = None
                        pending_track_count = 0
                        face_missing_since = None
                        is_locked_face = True
                        locked_face_found = True
                        locked_face_kps = f.kps.copy()
                        print(f"[SpeakerLock] Auto-locked {speaker_name}")

                    if is_locked_face:
                        color = (255, 165, 0)
                        border_thickness = 4
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                    elif is_selected:
                        color = (255, 255, 0)
                        border_thickness = 4
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                        cv2.circle(vis, (f.x1 + 15, f.y1 + 15), 8, color, -1)
                        draw_text_with_shadow(vis, "SPEAKER", (f.x1, f.y2 + 25), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                    elif label == "Ignored":
                        color = (128, 128, 128)
                        border_thickness = 2
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)
                        draw_text_with_shadow(vis, "IGNORED", (f.x1, f.y2 + 25), 0.6, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                    else:
                        color = (0, 255, 0) if is_primary_speaker else (0, 0, 255)
                        border_thickness = 2
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, border_thickness)

                    draw_text_with_shadow(vis, line1, (f.x1, max(0, f.y1 - 28)), 0.85, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)
                    draw_text_with_shadow(vis, line2, (f.x1, max(0, f.y1 - 6)), 0.65, color, 1, font=cv2.FONT_HERSHEY_DUPLEX)

                    if is_selected and not face_lock and potential_face_to_lock:
                        hint_text = f"Press 'l' to lock {potential_face_to_lock[0]}"
                        draw_text_with_shadow(vis, hint_text, (f.x1, f.y2 + 45), 0.65, (255, 255, 0), 1, font=cv2.FONT_HERSHEY_DUPLEX)

                    if y0 + thumb <= h and shown < 4:
                        vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                        draw_text_with_shadow(
                            vis,
                            f"{i+1}:{label}",
                            (x0, y0 - 6),
                            0.6,
                            color,
                            1,
                            font=cv2.FONT_HERSHEY_DUPLEX,
                        )
                        y0 += thumb + pad
                        shown += 1

                    if show_debug:
                        dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                        draw_text_with_shadow(vis, dbg, (10, h - 20), 0.65, (255, 255, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)

            # Movement command for ESP servo
            if face_lock and locked_face_found and locked_face_kps is not None:
                face_missing_since = None
                face_lost_streak = 0
                raw_error_x = compute_face_error_x(locked_face_kps, frame_width=w)
                if filtered_error_x is None:
                    filtered_error_x = raw_error_x
                else:
                    filtered_error_x = (
                        args.error_smooth_alpha * raw_error_x
                        + (1.0 - args.error_smooth_alpha) * filtered_error_x
                    )
                movement_error_x = float(filtered_error_x)

                desired_track_command = command_from_error_with_hysteresis(
                    error_x=movement_error_x,
                    deadzone_px=args.deadzone_px,
                    center_exit_hysteresis_px=args.center_exit_hysteresis_px,
                    previous_command=stable_track_command,
                )

                if desired_track_command == stable_track_command:
                    pending_track_command = None
                    pending_track_count = 0
                else:
                    if pending_track_command == desired_track_command:
                        pending_track_count += 1
                    else:
                        pending_track_command = desired_track_command
                        pending_track_count = 1
                    if pending_track_count >= args.command_confirm_frames:
                        stable_track_command = desired_track_command
                        pending_track_command = None
                        pending_track_count = 0

                movement_command = stable_track_command
            elif face_lock:
                face_lost_streak += 1
                if face_lost_streak < args.face_lost_confirm_frames:
                    movement_command = stable_track_command
                    movement_error_x = float(filtered_error_x or 0.0)
                else:
                    if face_missing_since is None:
                        face_missing_since = current_time
                    lost_for = current_time - face_missing_since
                    movement_command = command_when_speaker_lost(
                        lost_for,
                        args.search_delay_sec,
                        hold_command=stable_track_command,
                        allow_scan=args.allow_scan,
                    )
                    movement_error_x = float(filtered_error_x or 0.0)
            else:
                filtered_error_x = None
                stable_track_command = CENTERED
                pending_track_command = None
                pending_track_count = 0
                face_missing_since = None
                face_lost_streak = 0
                movement_command = STOPPED
                movement_error_x = 0.0

            movement_command = normalize_motor_command(movement_command)

            if mqtt_publisher is not None:
                cmd_changed = movement_command != mqtt_publisher.last_command
                force_scan = False
                if movement_command == SCAN and face_lock and not locked_face_found:
                    if (current_time - last_scan_publish_at) >= 0.45:
                        force_scan = True
                        last_scan_publish_at = current_time
                mqtt_publisher.publish(movement_command, force=cmd_changed or force_scan)
                mqtt_publisher.publish_status(
                    build_dashboard_status(
                        movement_command=movement_command,
                        movement_error_x=movement_error_x,
                        face_lock=face_lock,
                        faces_count=len(faces),
                        locked_face_found=locked_face_found,
                        fps=fps,
                        threshold=matcher.dist_thresh,
                        provider_name=provider_name,
                        speaker_confidence=speaker_confidence,
                        speaker_similarity=speaker_similarity,
                        speaker_name=speaker_name,
                    )
                )

            if evidence is not None:
                should_log = (
                    movement_command != last_logged_command
                    or (current_time - last_evidence_at) >= 1.0
                )
                if should_log:
                    note = ""
                    if ignored_faces_count > 0:
                        note = f"ignored_other_faces={ignored_faces_count}"
                    if movement_command == OUT_OF_FRAME:
                        note = (note + "; occlusion").strip("; ")
                    if movement_command == SCAN:
                        note = (note + "; reacquire_scan").strip("; ")
                    evidence.log(
                        speaker_id=speaker_name if face_lock else None,
                        confidence=speaker_confidence,
                        similarity=speaker_similarity,
                        distance=speaker_distance,
                        motor_command=movement_command,
                        error_x_px=movement_error_x,
                        locked=face_lock is not None,
                        speaker_visible=locked_face_found,
                        faces_in_frame=len(faces),
                        ignored_faces=ignored_faces_count,
                        fps=fps,
                        event_note=note,
                    )
                    last_evidence_at = current_time
                    last_logged_command = movement_command
            
            # Draw UI elements with proper spacing and modern styling
            y_offset = 35
            
            # Main header with background box
            header = f"IDs: {len(matcher._names)} | Threshold: {matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" | FPS: {fps:.1f}"
            draw_text_box(vis, header, (12, y_offset), 0.75, (200, 255, 200), (20, 20, 20), 0.75, 6, cv2.FONT_HERSHEY_DUPLEX)
            y_offset += 40

            movement_text = f"Motor: {movement_command}"
            if movement_command in (MOVED_LEFT, MOVED_RIGHT, CENTERED):
                movement_text += f" (err_x={movement_error_x:+.1f}px)"
            draw_text_with_shadow(vis, movement_text, (12, y_offset), 0.62, (180, 220, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)
            y_offset += 28
            
            # Lock status with enhanced styling
            if face_lock:
                if movement_command == SCAN and not locked_face_found:
                    status_text = f"[SEARCH] Sweeping for {face_lock.target_name}..."
                    draw_text_box(vis, status_text, (12, y_offset), 0.85, (100, 180, 255), (0, 20, 40), 0.8, 8, cv2.FONT_HERSHEY_DUPLEX)
                else:
                    status_text = f"[LOCKED] {face_lock.target_name} conf={int(speaker_confidence*100)}%"
                    draw_text_box(vis, status_text, (12, y_offset), 0.85, (255, 200, 100), (40, 20, 0), 0.8, 8, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 40
                
                # Last action
                if face_lock.history and len(face_lock.history) > 0:
                    last_action = face_lock.history[-1]
                    action_text = f"Last Action: {last_action.type.name} - {last_action.details}"
                    draw_text_with_shadow(vis, action_text, (12, h - 25), 0.65, (200, 255, 255), 1, font=cv2.FONT_HERSHEY_DUPLEX)
            else:
                if len(faces) > 1:
                    if selected_face_index is not None and selected_face_index < len(faces):
                        status_text = f"Selected face {selected_face_index + 1}/{len(faces)} | A/F to change | 'L' to lock"
                    else:
                        status_text = f"{len(faces)} faces detected | A/F to select | 'L' to lock"
                elif len(faces) == 1:
                    status_text = "Press 'L' to lock the recognized face"
                else:
                    status_text = "No faces detected"
                draw_text_box(vis, status_text, (12, y_offset), 0.75, (200, 255, 200), (0, 40, 0), 0.7, 6, cv2.FONT_HERSHEY_DUPLEX)
                y_offset += 40
            
            # Handle key presses
            key_raw = cv2.waitKey(1)
            key = key_raw & 0xFF
            
            # Handle arrow keys and navigation
            # Arrow keys in OpenCV: when key==0 or 224, the next byte contains the arrow key code
            # Left=75, Right=77, Up=72, Down=80
            # Also support 'a' for left, 'f' for right (to avoid conflict with 'd' for debug)
            arrow_key = None
            if key == 0 or key == 224:  # Extended key indicator
                # Get the actual arrow key code from the next byte
                arrow_key = (key_raw >> 8) & 0xFF
            
            if arrow_key == 75 or key == ord('a'):  # Left arrow or 'a' - previous face
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = len(faces) - 1
                    else:
                        selected_face_index = (selected_face_index - 1) % len(faces)
                    potential_face_to_lock = None  # Will be updated in next frame
                    print(f"[FaceSelect] Selected face {selected_face_index + 1}/{len(faces)}")
            elif arrow_key == 77 or key == ord('f'):  # Right arrow or 'f' - next face
                if not face_lock and len(faces) > 0:
                    if selected_face_index is None:
                        selected_face_index = 0
                    else:
                        selected_face_index = (selected_face_index + 1) % len(faces)
                    potential_face_to_lock = None  # Will be updated in next frame
                    print(f"[FaceSelect] Selected face {selected_face_index + 1}/{len(faces)}")
            
            if key == ord("q"):  # Quit
                break
            elif key == ord("r"):  # Reload DB
                matcher.reload_from(db_path)
                print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):  # Increase threshold
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
            elif key == ord("-"):  # Decrease threshold
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
            elif key == ord("d"):  # Toggle debug
                show_debug = not show_debug
                print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
            elif key == ord('l'):  # Lock/unlock face
                if face_lock:  # Unlock if already locked
                    save_action_history(face_lock.target_name, face_lock.history)
                    print(f"[FaceLock] Unlocked {face_lock.target_name}")
                    face_lock = None
                    selected_face_index = None
                    potential_face_to_lock = None
                    filtered_error_x = None
                    stable_track_command = CENTERED
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    face_lost_streak = 0
                    if mqtt_publisher is not None:
                        mqtt_publisher.publish(STOPPED, force=True)
                # Only allow locking if we have a selected recognized face
                elif potential_face_to_lock and potential_face_to_lock[0] is not None:
                    name, emb, kps = potential_face_to_lock
                    face_lock = FaceLock(
                        target_name=name,
                        target_emb=emb,
                        last_seen=current_time
                    )
                    face_lock.update_position(kps)  # Initialize position tracking
                    face_lock.history.append(Action(
                        ActionType.FACE_LOCKED,
                        current_time,
                        f"Face locked: {name}"
                    ))
                    filtered_error_x = None
                    stable_track_command = CENTERED
                    pending_track_command = None
                    pending_track_count = 0
                    face_missing_since = None
                    face_lost_streak = 0
                    last_scan_publish_at = 0.0
                    print(f"[SpeakerLock] Locked onto {name} (face {selected_face_index + 1 if selected_face_index is not None else '?'})")
                    print("[SpeakerLock] Tracking active. Leave frame to trigger SEARCH sweep.")
            
            cv2.imshow("BENAX Speaker Tracking - Press 'q' to quit", vis)
    finally:
        if evidence is not None:
            evidence.close()
        if mqtt_publisher is not None:
            mqtt_publisher.publish(STOPPED, force=True)
            mqtt_publisher.publish_status(
                {
                    "timestamp": time.time(),
                    "movement": STOPPED,
                    "error_x": 0.0,
                    "locked": False,
                    "target": None,
                    "locked_face_found": False,
                    "faces": 0,
                    "fps": None,
                    "threshold": round(float(matcher.dist_thresh), 3),
                    "provider": provider_name,
                    "shutdown": True,
                },
                force=True,
            )
            mqtt_publisher.close()
        det.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
