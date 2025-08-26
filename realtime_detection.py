
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import GTSRB_CLASS_NAMES
import argparse
import time
import threading
from collections import deque

class RealTimeTrafficSignDetector:
    def __init__(self, model_path, target_size=(32, 32), confidence_threshold=0.7):
        self.model = load_model(model_path)
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.class_names = GTSRB_CLASS_NAMES

        # For smoothing predictions
        self.prediction_history = deque(maxlen=10)
        self.stable_prediction = None
        self.stable_confidence = 0.0

        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)

    def preprocess_frame(self, frame, roi=None):
        """Preprocess frame for model inference"""
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]

        # Resize to model input size
        processed = cv2.resize(frame, self.target_size)
        processed = processed.astype('float32') / 255.0
        processed = np.expand_dims(processed, axis=0)

        return processed

    def predict_frame(self, frame, roi=None):
        """Predict traffic sign in frame"""
        start_time = time.time()

        # Preprocess
        processed_frame = self.preprocess_frame(frame, roi)

        # Predict
        prediction = self.model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return {
            'class_id': int(predicted_class),
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'processing_time': processing_time
        }

    def smooth_predictions(self, prediction):
        """Smooth predictions over time to reduce flickering"""
        self.prediction_history.append(prediction)

        if len(self.prediction_history) < 5:
            return prediction

        # Get recent predictions
        recent_predictions = list(self.prediction_history)[-5:]

        # Find most common prediction with high confidence
        class_counts = {}
        confidence_sums = {}

        for pred in recent_predictions:
            if pred['confidence'] > self.confidence_threshold:
                class_id = pred['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                confidence_sums[class_id] = confidence_sums.get(class_id, 0) + pred['confidence']

        if class_counts:
            # Find most frequent class
            most_common_class = max(class_counts.keys(), key=lambda x: class_counts[x])
            avg_confidence = confidence_sums[most_common_class] / class_counts[most_common_class]

            # Update stable prediction if it appears frequently enough
            if class_counts[most_common_class] >= 3:
                self.stable_prediction = {
                    'class_id': most_common_class,
                    'class_name': self.class_names[most_common_class],
                    'confidence': avg_confidence
                }

        return self.stable_prediction if self.stable_prediction else prediction

    def detect_traffic_sign_regions(self, frame):
        """Simple traffic sign region detection using color and shape"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for traffic signs (red, blue, yellow)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])

        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])

        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])

        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2

        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Combine masks
        combined_mask = red_mask + blue_mask + yellow_mask

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio
        potential_signs = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Traffic signs are usually square-ish or circular
                if 0.7 <= aspect_ratio <= 1.4:
                    potential_signs.append((x, y, w, h))

        return potential_signs

    def run_webcam(self, camera_index=0, detect_regions=True):
        """Run real-time detection on webcam"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        print("Starting webcam detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            # Detect potential traffic sign regions
            if detect_regions:
                regions = self.detect_traffic_sign_regions(frame)
            else:
                # Use entire frame
                h, w = frame.shape[:2]
                regions = [(w//4, h//4, w//2, h//2)]  # Center region

            # Process each detected region
            for i, (x, y, w, h) in enumerate(regions):
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Predict for this region
                roi = (x, y, w, h)
                prediction = self.predict_frame(frame, roi)

                # Smooth prediction
                smooth_pred = self.smooth_predictions(prediction)

                if smooth_pred and smooth_pred['confidence'] > self.confidence_threshold:
                    # Draw prediction
                    label = f"{smooth_pred['class_name'][:20]}"
                    confidence = f"{smooth_pred['confidence']*100:.1f}%"

                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y), (0, 255, 0), -1)

                    # Draw text
                    cv2.putText(frame, label, (x + 5, y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, confidence, (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Calculate FPS
            frame_time = time.time() - frame_start
            self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            avg_fps = np.mean(self.fps_counter)

            # Display FPS and processing info
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.processing_times:
                avg_processing_time = np.mean(self.processing_times) * 1000
                cv2.putText(frame, f"Processing: {avg_processing_time:.1f}ms", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Traffic Sign Detection', frame)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset stable prediction
                self.stable_prediction = None
                self.prediction_history.clear()

        cap.release()
        cv2.destroyAllWindows()

    def process_video(self, video_path, output_path=None):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Set up video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame (similar to webcam processing)
            regions = self.detect_traffic_sign_regions(frame)

            for x, y, w, h in regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi = (x, y, w, h)
                prediction = self.predict_frame(frame, roi)

                if prediction['confidence'] > self.confidence_threshold:
                    label = f"{prediction['class_name'][:20]}"
                    confidence = f"{prediction['confidence']*100:.1f}%"

                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y), (0, 255, 0), -1)

                    cv2.putText(frame, label, (x + 5, y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, confidence, (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Write frame if output is specified
            if writer:
                writer.write(frame)

            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")

        cap.release()
        if writer:
            writer.release()

        print(f"Video processing completed. Processed {frame_count} frames.")

def main():
    parser = argparse.ArgumentParser(description='Real-time Traffic Sign Detection')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video'], default='webcam',
                       help='Detection mode')
    parser.add_argument('--video_path', type=str,
                       help='Path to input video (for video mode)')
    parser.add_argument('--output_path', type=str,
                       help='Path to output video (for video mode)')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera index for webcam mode')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Confidence threshold for detections')
    parser.add_argument('--target_size', type=int, nargs=2, default=[32, 32],
                       help='Target image size (width height)')
    parser.add_argument('--detect_regions', action='store_true', default=True,
                       help='Enable automatic traffic sign region detection')

    args = parser.parse_args()

    # Initialize detector
    print(f"Loading model from {args.model_path}")
    detector = RealTimeTrafficSignDetector(
        model_path=args.model_path,
        target_size=tuple(args.target_size),
        confidence_threshold=args.confidence_threshold
    )

    if args.mode == 'webcam':
        detector.run_webcam(
            camera_index=args.camera_index,
            detect_regions=args.detect_regions
        )
    elif args.mode == 'video':
        if not args.video_path:
            print("Error: video_path is required for video mode")
            return

        detector.process_video(
            video_path=args.video_path,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()
