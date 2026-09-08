import cv2
import mediapipe as mp


class PoseDetector:
    """Extract human pose landmarks using MediaPipe."""

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame):
        """Process an OpenCV BGR frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_landmarks(self, frame, results):
        """Draw detected pose landmarks on the frame."""

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        return frame

    def get_landmarks(self, results):
        """Convert MediaPipe landmarks into dictionaries."""

        if not results.pose_landmarks:
            return []

        landmarks = []

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append(
                {
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )

        return landmarks

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def main():
    """Run GaitGuard pose detection using the webcam."""

    detector = PoseDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        detector.close()
        return

    print("GaitGuard Pose Detection")
    print("Press 'q' to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        results = detector.process_frame(frame)

        frame = detector.draw_landmarks(frame, results)

        cv2.imshow("GaitGuard - Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()