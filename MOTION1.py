import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import threading
import pyaudio
import wave
import time
import os
import tempfile
import subprocess

# Set path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Custom output directory for recordings
output_dir = r"C:\MOTION1\output"  # or another folder with permission

print("Creating output directory at:", output_dir)
os.makedirs(output_dir, exist_ok=True)


# Set this if ffmpeg is not in your system PATH, otherwise leave as 'ffmpeg'
ffmpeg_path = 'ffmpeg'  
# Example full path if needed:
# ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
temp_canvas = canvas.copy()
prev_x, prev_y = None, None
eraser_size = 50

palette = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'black': (0, 0, 0),
    'purple': (255, 0, 255),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'pink': (203, 192, 255)
}
palette_keys = list(palette.keys())
selected_color_idx = 0
draw_paths = []  # Each path is [(x1,y1),(x2,y2),..., color]
current_path = []
shapes = []
undo_stack = []  # Each item: ('add_shape'/'remove_shape'/'draw'/'erase_draw', shape/path)
redo_stack = []

pen_thickness = 6  # Default line thickness

class Shape:
    def __init__(self, shape_type, center, size, color, filled=False, fill_color=None):
        self.shape_type = shape_type
        self.center = center
        self.size = size
        self.color = color
        self.filled = filled
        self.fill_color = fill_color

    def draw(self, img):
        x, y = self.center
        s = self.size
        if self.shape_type == 'circle':
            if self.filled and self.fill_color:
                cv2.circle(img, (x, y), s, self.fill_color, -1)
            cv2.circle(img, (x, y), s, self.color, 2)
        elif self.shape_type == 'square':
            if self.filled and self.fill_color:
                cv2.rectangle(img, (x - s, y - s), (x + s, y + s), self.fill_color, -1)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), self.color, 2)
        elif self.shape_type == 'triangle':
            pts = np.array([[x, y - s], [x - s, y + s], [x + s, y + s]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            if self.filled and self.fill_color:
                cv2.fillPoly(img, [pts], self.fill_color)
            cv2.polylines(img, [pts], True, self.color, 2)

    def inside(self, point):
        x, y = point
        cx, cy = self.center
        s = self.size
        if self.shape_type == 'circle':
            return (x - cx)**2 + (y - cy)**2 <= s**2
        elif self.shape_type == 'square':
            return cx - s <= x <= cx + s and cy - s <= y <= cy + s
        elif self.shape_type == 'triangle':
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], np.int32)
            return cv2.pointPolygonTest(pts, (x, y), False) >= 0
        return False

def count_fingers(hand_landmarks, handedness):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    if handedness == "Right":
        fingers.append(hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x)
    else:
        fingers.append(hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x)

    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y)
    return fingers

def get_hand_depth(hand_landmarks):
    return hand_landmarks.landmark[0].z

palette_positions = [(50, i * 40 + 50) for i in range(len(palette))]
palette_size = 25
shapes_top = ['circle', 'square', 'triangle']
shape_positions_top = [(i * 60 + 120, 30) for i in range(len(shapes_top))]

button_labels = ["Undo", "Redo", "Clear", "Exit"]
button_positions = {label: (i * 80 + 320, 30) for i, label in enumerate(button_labels)}
button_size = (70, 30)

selected_shape_type = None
selected_shape_idx = None
current_shape = None
click_triggered = {key: False for key in button_positions.keys()}

text_box_position = (50, 400)
text_box_size = (540, 60)
recognized_text = ""

# Ensure output directory exists
output_dir = r"C:\Users\prane\OneDrive\Desktop\MOTION1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def draw_ui(img):
    for i, (x, y) in enumerate(palette_positions):
        color = palette[palette_keys[i]]
        cv2.rectangle(img, (x - palette_size, y - palette_size),
                      (x + palette_size, y + palette_size), color, -1)
        cv2.putText(img, palette_keys[i], (x - 20, y + palette_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        if i == selected_color_idx:
            cv2.rectangle(img, (x - palette_size - 2, y - palette_size - 2),
                          (x + palette_size + 2, y + palette_size + 2), (0, 0, 0), 2)

    for i, (x, y) in enumerate(shape_positions_top):
        cv2.circle(img, (x, y), 20, (0, 255, 255), -1)
        cv2.putText(img, shapes_top[i][0].upper(), (x - 10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if i == selected_shape_idx:
            cv2.circle(img, (x, y), 23, (255, 0, 0), 2)

    draw_buttons(img)

    cv2.rectangle(img, text_box_position,
                  (text_box_position[0] + text_box_size[0], text_box_position[1] + text_box_size[1]),
                  (255, 255, 255), -1)
    cv2.rectangle(img, text_box_position,
                  (text_box_position[0] + text_box_size[0], text_box_position[1] + text_box_size[1]),
                  (0, 0, 0), 2)
    cv2.putText(img, "Recognized: " + recognized_text,
                (text_box_position[0] + 10, text_box_position[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_buttons(img):
    for label, (x, y) in button_positions.items():
        cv2.rectangle(img, (x, y), (x + button_size[0], y + button_size[1]), (200, 200, 200), -1)
        cv2.rectangle(img, (x, y), (x + button_size[0], y + button_size[1]), (0, 0, 0), 2)
        cv2.putText(img, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def two_finger_click_in_rect(fingers, landmarks, rect):
    if fingers == [0, 1, 1, 0, 0]:
        x1 = int(landmarks.landmark[8].x * 640)
        y1 = int(landmarks.landmark[8].y * 480)
        x2 = int(landmarks.landmark[12].x * 640)
        y2 = int(landmarks.landmark[12].y * 480)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x, y, w, h = rect
        return x <= cx <= x + w and y <= cy <= y + h
    return False


def redraw_canvas():
    global canvas
    canvas[:] = 255
    for path in draw_paths:
        color = path[-1]
        points = path[:-1]
        for i in range(1, len(points)):
            cv2.line(canvas, points[i - 1], points[i], color, pen_thickness)
    for shape in shapes:
        shape.draw(canvas)


def recognize_text_from_canvas():
    global recognized_text
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    recognized_text = text.strip() if text else "No text recognized"

# Create a named window and set to fullscreen
cv2.namedWindow("Gesture Drawing", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gesture Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    temp_canvas[:] = canvas.copy()

    if results.multi_hand_landmarks:
        hand_info = []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            depth = get_hand_depth(hand_landmarks)
            hand_info.append((hand_landmarks, hand_label, depth))
        hand_info.sort(key=lambda x: x[2])
        hand_landmarks, hand_label, _ = hand_info[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = count_fingers(hand_landmarks, hand_label)
        h, w, _ = frame.shape
        x = int(hand_landmarks.landmark[8].x * w)
        y = int(hand_landmarks.landmark[8].y * h)

        for label, (bx, by) in button_positions.items():
            triggered = two_finger_click_in_rect(fingers, hand_landmarks,
                                                 (bx, by, button_size[0], button_size[1]))
            if triggered and not click_triggered[label]:
                click_triggered[label] = True
                if label == "Undo" and undo_stack:
                    action = undo_stack.pop()
                    redo_stack.append(action)
                    act_type = action[0]
                    if act_type == 'add_shape':
                        shape = action[1]
                        if shape in shapes:
                            shapes.remove(shape)
                    elif act_type == 'remove_shape':
                        shape = action[1]
                        shapes.append(shape)
                    elif act_type == 'draw':
                        path = action[1]
                        if path in draw_paths:
                            draw_paths.remove(path)
                    elif act_type == 'erase_draw':
                        path = action[1]
                        draw_paths.append(path)
                    redraw_canvas()
                elif label == "Redo" and redo_stack:
                    action = redo_stack.pop()
                    undo_stack.append(action)
                    act_type = action[0]
                    if act_type == 'add_shape':
                        shape = action[1]
                        shapes.append(shape)
                    elif act_type == 'remove_shape':
                        shape = action[1]
                        if shape in shapes:
                            shapes.remove(shape)
                    elif act_type == 'draw':
                        path = action[1]
                        draw_paths.append(path)
                    elif act_type == 'erase_draw':
                        path = action[1]
                        if path in draw_paths:
                            draw_paths.remove(path)
                    redraw_canvas()
                elif label == "Clear":
                    undo_stack.extend([('remove_shape', s) for s in shapes])
                    undo_stack.extend([('erase_draw', p) for p in draw_paths])
                    shapes.clear()
                    draw_paths.clear()
                    redo_stack.clear()
                    canvas[:] = 255
                    recognized_text = ""
                elif label == "Exit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
            elif not triggered:
                click_triggered[label] = False

        if fingers == [1, 1, 0, 0, 0]:
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)
            x2 = int(hand_landmarks.landmark[8].x * w)
            y2 = int(hand_landmarks.landmark[8].y * h)
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            pen_thickness = max(1, min(20, int(dist // 10)))
            cv2.line(temp_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(temp_canvas, f"Pen Thickness: {pen_thickness}", (x2 + 10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if fingers == [0, 1, 0, 0, 0]:
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y
                current_path = [(x, y)]
            else:
                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         palette[palette_keys[selected_color_idx]], pen_thickness)
                current_path.append((x, y))
                prev_x, prev_y = x, y
        else:
            if current_path:
                current_path.append(palette[palette_keys[selected_color_idx]])
                draw_paths.append(current_path)
                undo_stack.append(('draw', current_path))
                redo_stack.clear()
                current_path = []
                recognize_text_from_canvas()
            prev_x, prev_y = None, None

        if fingers == [0, 1, 1, 1, 0]:
            removed_shapes = []
            for s in shapes[:]:
                if s.inside((x, y)):
                    shapes.remove(s)
                    removed_shapes.append(s)
            if removed_shapes:
                for s in removed_shapes:
                    undo_stack.append(('remove_shape', s))
                redo_stack.clear()
            redraw_canvas()

        if fingers == [0, 1, 1, 1, 1]:
            cv2.circle(temp_canvas, (x, y), eraser_size, (255, 0, 0), 2)
            cv2.circle(temp_canvas, (x, y), 3, (255, 0, 0), -1)

            erased_paths = []
            modified_paths = []
            for path in draw_paths:
                points = path[:-1]
                color = path[-1]
                new_segments = []
                current_segment = []

                for i, (px, py) in enumerate(points):
                    if (px - x)**2 + (py - y)**2 <= eraser_size**2:
                        if len(current_segment) > 1:
                            current_segment.append(color)
                            new_segments.append(current_segment)
                        current_segment = []
                    else:
                        current_segment.append((px, py))

                if len(current_segment) > 1:
                    current_segment.append(color)
                    new_segments.append(current_segment)

                if new_segments:
                    modified_paths.extend(new_segments)
                elif len(points) > 0:
                    erased_paths.append(path)

            if erased_paths or modified_paths != draw_paths:
                for p in erased_paths:
                    undo_stack.append(('erase_draw', p))
                redo_stack.clear()
                draw_paths = modified_paths
                redraw_canvas()

        if fingers == [0, 1, 1, 0, 0]:
            for i, (px, py) in enumerate(palette_positions):
                if (px - palette_size <= x <= px + palette_size and py - palette_size <= y <= py + palette_size):
                    selected_color_idx = i
                    print("[DEBUG] Selected color:", palette_keys[selected_color_idx])

            for i, (sx, sy) in enumerate(shape_positions_top):
                if abs(x - sx) < 20 and abs(y - sy) < 20:
                    selected_shape_idx = i
                    selected_shape_type = shapes_top[i]

            for shape in shapes:
                if shape.inside((x, y)):
                    shape.filled = True
                    shape.fill_color = palette[palette_keys[selected_color_idx]]
                    redraw_canvas()
                    print(f"[DEBUG] Filled {shape.shape_type} with {palette_keys[selected_color_idx]}")

        if selected_shape_type:
            color = palette[palette_keys[selected_color_idx]]
            if current_shape is None:
                current_shape = Shape(selected_shape_type, (x, y), 30, color, False)
            else:
                current_shape.center = (x, y)
                if fingers == [1, 1, 1, 1, 1]:
                    current_shape.size += 1
                    if current_shape.size > 200:
                        current_shape.size = 200
            current_shape.draw(temp_canvas)
            if fingers == [0, 0, 0, 0, 0]:
                shapes.append(current_shape)
                undo_stack.append(('add_shape', current_shape))
                redo_stack.clear()
                current_shape = None
                selected_shape_type = None
                selected_shape_idx = None
                redraw_canvas()

    draw_ui(temp_canvas)

    gray = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(temp_canvas, temp_canvas, mask=mask)
    final = cv2.add(frame_bg, fg)

    cv2.imshow("Gesture Drawing", final)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()