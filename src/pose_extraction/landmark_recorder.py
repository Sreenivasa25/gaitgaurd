import json
import os
import time
from datetime import datetime, timezone

import cv2

from src.pose_extraction.pose_detector import PoseDetector
from src.utils.config import config


class LandmarkRecorder:
    """Record timestamped MediaPipe pose landmarks."""

    def __init__(self, output_path=None):
        self.detector = PoseDetector()

        if output_path is None:
            keypoints_dir = os.path.join(
                config.DATA_DIR,
                "keypoints",
            )
            os.makedirs(keypoints_dir, exist_ok=True)

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = os.path.join(
                keypoints_dir,
                f"gait_session_{timestamp}.json",
            )

        self.output_path = output_path
        self.frames = []

    def record_camera(self, duration_seconds=30):
        """Record pose landmarks from the webcam."""

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera."
            )

        start_time = time.time()
        frame_number = 0

        print("GaitGuard Landmark Recorder")
        print(
            f"Recording for {duration_seconds} seconds..."
        )
        print("Press 'q' to stop early.")

        try:
            while True:
                success, frame = camera.read()

                if not success:
                    print(
                        "WARNING: Could not read camera frame."
                    )
                    continue

                elapsed = time.time() - start_time

                if elapsed >= duration_seconds:
                    break

                results = self.detector.process_frame(frame)
                landmarks = self.detector.get_landmarks(
                    results
                )

                if landmarks:
                    self.frames.append(
                        {
                            "frame_number": frame_number,
                            "timestamp": elapsed,
                            "datetime_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "landmarks": landmarks,
                        }
                    )

                display_frame = (
                    self.detector.draw_landmarks(
                        frame,
                        results,
                    )
                )

                cv2.putText(
                    display_frame,
                    f"Time: {elapsed:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Frames: {frame_number}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "GaitGuard - Landmark Recorder",
                    display_frame,
                )

                frame_number += 1

                if (
                    cv2.waitKey(1) & 0xFF
                ) == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.detector.close()

        self.save()

    def save(self):
        """Save recorded landmarks as JSON."""

        output_dir = os.path.dirname(
            self.output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        data = {
            "project": "GaitGuard",
            "subject_id": os.getenv(
                "SUBJECT_ID",
                "worker_001",
            ),
            "experiment_id": os.getenv(
                "EXPERIMENT_ID",
                "baseline_trial_001",
            ),
            "device_type": config.DEVICE_TYPE,
            "pose_backend": config.POSE_BACKEND,
            "frame_count": len(self.frames),
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "frames": self.frames,
        }

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

        print(
            f"\n✓ Saved {len(self.frames)} pose frames"
        )
        print(
            f"✓ Output: {self.output_path}"
        )


def main():
    recorder = LandmarkRecorder()

    try:
        recorder.record_camera(
            duration_seconds=30
        )
    except RuntimeError as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
