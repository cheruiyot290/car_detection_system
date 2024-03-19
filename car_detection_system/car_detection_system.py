import cv2

# Load pre-trained car detection model
car_cascade_path = r'C:\Users\cheru\source\repos\car_detection_system\car_detection_system\resources\haarcascade_car.xml'
model = cv2.CascadeClassifier(car_cascade_path)

# Define parking slot rectangles
# Format: (x, y, width, height)
parking_slots = [
    (550, 320, 45, 80),
    (490, 325, 45, 80),
    (440, 340, 45, 80),
    (380, 350, 45, 80),
    (320, 360, 45, 80),
    (260, 365, 45, 80),
    (200, 370, 45, 80),
    (140, 370, 45, 80),
    (80, 370, 45, 80),
    (25, 370, 45, 80),
]

# Resize frame function
def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Check if a point (x, y) is inside a rectangle defined by (x, y, w, h)
def is_point_inside_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

# Process video frames
def process_video(video_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize parking slot status
    slot_status = [True] * len(parking_slots)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame (adjust scale percent as needed)
        frame_resized = resize_frame(frame, 50)  # Adjust scale percent (e.g., 50 for half size)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame
        cars = model.detectMultiScale(gray, 1.1, 3)

        # Update parking slot status based on car detection
        for i, slot in enumerate(parking_slots):
            slot_status[i] = True  # Reset slot status
            for (x, y, w, h) in cars:
                if is_point_inside_rect(x + w // 2, y + h // 2, slot):
                    slot_status[i] = False  # Slot occupied if car detected inside

        # Draw parking slots on frame
        for i, (x, y, w, h) in enumerate(parking_slots):
            if slot_status[i]:
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for available
                cv2.putText(frame_resized, "Available", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for occupied
                cv2.putText(frame_resized, "Occupied", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Parking Occupancy Detection", frame_resized)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Adjust delay for slower video playback
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    video_path = r'C:\Users\cheru\source\repos\car_detection_system\car_detection_system\resources\parking1.mp4'
    process_video(video_path)

if __name__ == "__main__":
    main()
