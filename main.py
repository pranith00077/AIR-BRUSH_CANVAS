import cv2
import numpy as np

def detect_hand_sign(use_static_image=False, image_path='test_hand.jpg'):
    if use_static_image:
        # Load static image for testing
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            print("Creating a test image...")
            # Create a simple test image with a hand-like shape
            frame = create_test_image()
        process_frame(frame)
    else:
        # Try different camera indices
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera opened successfully with index {i}")
                break
        else:
            print("Error: Could not open any camera. Please check camera permissions or if another application is using the camera.")
            print("You can test with a static image by running: python main.py --static test_hand.jpg")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            process_frame(frame)

            cv2.imshow('Hand Sign Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    # Apply skin color detection to filter out non-skin objects
    skin_mask = detect_skin_color(frame)

    # Apply Gaussian blur to the skin mask
    blur = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # Threshold to get binary image
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # Minimum area to consider as hand
            # Draw joints (approximate landmarks)
            draw_joints(frame, max_contour)

            # Enhanced gesture recognition for alphabets
            detected_sign = recognize_alphabet_gesture(max_contour)

            cv2.putText(frame, f"Detected Sign: {detected_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No hand detected
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



def detect_skin_color(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range (more accurate)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def draw_joints(frame, contour):
    # Approximate hand joints using contour analysis
    # Find convex hull
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is not None and len(defects) > 0:
        # Draw defects as joints
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 10000:  # Threshold for finger detection
                far = tuple(contour[f][0])
                cv2.circle(frame, far, 5, (0, 0, 255), -1)  # Draw joint as red circle

def recognize_alphabet_gesture(contour):
    """
    Enhanced gesture recognition for ASL alphabet signs using contour analysis
    """
    try:
        # Calculate convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None and len(defects) > 0:
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 10000:  # Threshold for finger detection
                    finger_count += 1

            # Enhanced ASL Alphabet Recognition Logic
            if finger_count == 0:
                return "A"  # Fist - all fingers closed
            elif finger_count == 1:
                return "B"  # One finger (index)
            elif finger_count == 2:
                return "V"  # Victory sign (index and middle)
            elif finger_count == 3:
                return "W"  # Three fingers (index, middle, ring)
            elif finger_count == 4:
                return "4"  # Four fingers
            elif finger_count == 5:
                return "5"  # All five fingers
            else:
                return f"{finger_count} fingers"
        else:
            return "A"  # No defects = fist
    except cv2.error:
        return "Processing error"



def create_test_image():
    # Create a simple test image with a hand-like shape
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple hand shape (fist for testing)
    # Palm
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
    
    # Fingers (simplified)
    cv2.rectangle(img, (280, 140), (300, 200), (255, 255, 255), -1)  # Thumb
    cv2.rectangle(img, (320, 140), (340, 200), (255, 255, 255), -1)  # Index
    cv2.rectangle(img, (340, 140), (360, 200), (255, 255, 255), -1)  # Middle
    cv2.rectangle(img, (360, 140), (380, 200), (255, 255, 255), -1)  # Ring
    cv2.rectangle(img, (380, 140), (400, 200), (255, 255, 255), -1)  # Pinky
    
    return img

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--static':
        image_path = sys.argv[2] if len(sys.argv) > 2 else 'test_hand.jpg'
        detect_hand_sign(use_static_image=True, image_path=image_path)
    else:
        detect_hand_sign()
