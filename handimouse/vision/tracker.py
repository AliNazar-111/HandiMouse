"""
Vision Layer: Modern MediaPipe Hand Tracking Module.
Utilizes the high-performance MediaPipe Tasks HandLandmarker API,
implements dynamic model caching, coordinate filtering, and fingers-up checks.
"""

import cv2
import logging
import math
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple, Union

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from handimouse.vision.filter import OneEuroFilter

logger = logging.getLogger("handimouse.vision.tracker")


class HandTracker:
    """
    A high-performance hand tracking wrapper using the modern MediaPipe Tasks API.

    Optimizations & Constraints:
    1. Tasks HandLandmarker API: Designed for high efficiency on modern systems.
    2. Video Running Mode: Specifically tuned for frame streams with frame timestamps.
    3. Auto Model Downloader: Downloads the pre-trained float16 model automatically if missing.
    4. 3D Coordinates Smoothing: Uses 21 independent OneEuroFilters to filter landmarks.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        dominant_hand_preference: str = "largest",
        filter_min_cutoff: float = 1.0,
        filter_beta: float = 0.007
    ):
        """
        Initialize the Hand Tracker.

        Args:
            max_hands: Max hands to detect (default 2 to allow selection filter).
            min_detection_confidence: Detection confidence (0.0 to 1.0).
            min_tracking_confidence: Tracking confidence (0.0 to 1.0).
            dominant_hand_preference: Rule to select target hand ("right", "left", "largest").
            filter_min_cutoff: Minimum cutoff frequency for OneEuroFilter.
            filter_beta: Speed coefficient for OneEuroFilter.
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.dominant_hand_preference = dominant_hand_preference.lower()

        # Dynamic model download
        self.model_path = self._ensure_model_exists()

        # Initialize the modern HandLandmarker
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # 21 independent 3D OneEuroFilters
        self.filter_min_cutoff = filter_min_cutoff
        self.filter_beta = filter_beta
        self._landmark_filters = [
            OneEuroFilter(min_cutoff=self.filter_min_cutoff, beta=self.filter_beta) 
            for _ in range(21)
        ]
        self._filters_active = False

        logger.info(
            f"Modern HandTracker initialized successfully | "
            f"Dominant preference: {self.dominant_hand_preference}"
        )

    def _ensure_model_exists(self) -> str:
        """
        Ensures the pre-trained MediaPipe Hand Landmarker task model file is present.
        If missing, it downloads it from the official Google Cloud storage bucket.
        """
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            logger.info(f"Model not found locally. Downloading from official storage: {url}...")
            start_time = time.time()
            try:
                # Add standard User-Agent headers to prevent potential blocking
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                    out_file.write(response.read())
                logger.info(f"Successfully downloaded model to {model_path} in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.critical(f"Failed to download hand landmarker model: {e}", exc_info=True)
                if os.path.exists(model_path):
                    os.remove(model_path)
                raise RuntimeError("Could not retrieve pre-trained tracking model.") from e

        return model_path

    def _reset_filters(self) -> None:
        """Reset historical states of OneEuroFilters to prevent rubber-banding on re-entry."""
        if self._filters_active:
            for filt in self._landmark_filters:
                filt.last_time = None
                filt.last_value = None
            self._filters_active = False
            logger.debug("OneEuroFilters reset successfully.")

    def _calculate_hand_area(self, landmarks) -> float:
        """Calculates the normalized bounding box area of a hand."""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width * height

    def _select_dominant_hand(self, result) -> Tuple[Optional[int], str, float]:
        """
        Applies dominant hand selection heuristics to choose a single target hand.

        Returns:
            Tuple[Optional[int], str, float]: (Selected index, handedness string "Left"/"Right", confidence score).
        """
        if not result.hand_landmarks:
            return None, "Unknown", 0.0

        num_hands = len(result.hand_landmarks)
        hand_data = []

        for i in range(num_hands):
            landmarks = result.hand_landmarks[i]
            # In the modern Tasks API, handedness contains a list of Category objects
            classification = result.handedness[i][0]
            
            hand_label = classification.category_name  # "Left" or "Right"
            score = classification.score
            area = self._calculate_hand_area(landmarks)

            hand_data.append({
                "index": i,
                "label": hand_label,
                "score": score,
                "area": area,
                "landmarks": landmarks
            })

        selected_hand = None

        if self.dominant_hand_preference == "right":
            right_hands = [h for h in hand_data if h["label"] == "Right"]
            if right_hands:
                selected_hand = max(right_hands, key=lambda x: x["area"])
            else:
                selected_hand = max(hand_data, key=lambda x: x["area"])

        elif self.dominant_hand_preference == "left":
            left_hands = [h for h in hand_data if h["label"] == "Left"]
            if left_hands:
                selected_hand = max(left_hands, key=lambda x: x["area"])
            else:
                selected_hand = max(hand_data, key=lambda x: x["area"])

        else:  # "largest"
            selected_hand = max(hand_data, key=lambda x: x["area"])

        if selected_hand:
            return selected_hand["index"], selected_hand["label"], selected_hand["score"]

        return None, "Unknown", 0.0

    def _calculate_fingers_up(self, landmarks_list: List[Tuple[float, float, float]], handedness: str) -> List[int]:
        """
        Heuristically calculates which fingers are extended (1) or folded (0).

        Hand landmark indices:
        Thumb: 1-4 (4 tip)
        Index: 5-8 (8 tip)
        Middle: 9-12 (12 tip)
        Ring: 13-16 (16 tip)
        Pinky: 17-20 (20 tip)
        """
        fingers = [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]
        
        wrist = landmarks_list[0]
        middle_knuckle = landmarks_list[9]
        hand_scale = math.sqrt(
            (middle_knuckle[0] - wrist[0]) ** 2 + 
            (middle_knuckle[1] - wrist[1]) ** 2
        )
        if hand_scale == 0:
            hand_scale = 0.1

        # 1. THUMB CHECK: Uses rotation-invariant distance between thumb tip and index knuckle
        thumb_tip = landmarks_list[4]
        index_knuckle = landmarks_list[5]
        thumb_distance = math.sqrt(
            (thumb_tip[0] - index_knuckle[0]) ** 2 + 
            (thumb_tip[1] - index_knuckle[1]) ** 2
        )
        if thumb_distance > 0.6 * hand_scale:
            thumb_ip = landmarks_list[3]
            # Category name accounts for mirrored frames. Standard: Left thumb points right, right points left.
            if handedness == "Right":
                if thumb_tip[0] < thumb_ip[0]:
                    fingers[0] = 1
            else:
                if thumb_tip[0] > thumb_ip[0]:
                    fingers[0] = 1
            if thumb_distance > 0.85 * hand_scale:
                fingers[0] = 1

        # 2. FOUR FINGERS: Height-based knuckle vs tip comparison
        # Index: Tip (8) vs Joint (6)
        if landmarks_list[8][1] < landmarks_list[6][1]:
            fingers[1] = 1
            
        # Middle: Tip (12) vs Joint (10)
        if landmarks_list[12][1] < landmarks_list[10][1]:
            fingers[2] = 1
            
        # Ring: Tip (16) vs Joint (14)
        if landmarks_list[16][1] < landmarks_list[14][1]:
            fingers[3] = 1
            
        # Pinky: Tip (20) vs Joint (18)
        if landmarks_list[20][1] < landmarks_list[18][1]:
            fingers[4] = 1

        return fingers

    def process_frame(self, frame: cv2.Mat, timestamp: Optional[float] = None) -> Dict[str, Union[bool, float, List]]:
        """
        Process a single video frame.

        Args:
            frame: Raw BGR camera image frame.
            timestamp: Optional epoch timestamp in seconds.

        Returns:
            Dict: vision state output matching requirements.
        """
        current_time = timestamp if timestamp is not None else time.time()
        timestamp_ms = int(current_time * 1000)

        empty_response = {
            "hand_detected": False,
            "landmarks": [],
            "fingers_up": [0, 0, 0, 0, 0],
            "confidence": 0.0,
            "handedness": "Unknown"
        }

        if frame is None:
            self._reset_filters()
            return empty_response

        # Convert image to RGB format as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap the frame in an mp.Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Execute landmarker for video tracking mode
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._reset_filters()
            return empty_response

        # Dominant Hand Filter
        selected_index, hand_label, confidence = self._select_dominant_hand(result)
        if selected_index is None:
            self._reset_filters()
            return empty_response

        raw_landmarks = result.hand_landmarks[selected_index]
        self._filters_active = True

        # Process and Filter Coordinates via OneEuroFilters
        filtered_landmarks = []
        for i, lm in enumerate(raw_landmarks):
            raw_coord = (lm.x, lm.y, lm.z)
            filtered_coord = self._landmark_filters[i](raw_coord, timestamp=current_time)
            filtered_landmarks.append((filtered_coord[0], filtered_coord[1], filtered_coord[2]))

        # Calculate extended fingers
        fingers_up = self._calculate_fingers_up(filtered_landmarks, hand_label)

        return {
            "hand_detected": True,
            "landmarks": filtered_landmarks,
            "fingers_up": fingers_up,
            "confidence": float(confidence),
            "handedness": hand_label
        }

    def close(self) -> None:
        """Close the landmarker instance and clean up resources."""
        self.landmarker.close()
        logger.info("MediaPipe tracking module closed successfully.")
