import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load model
model = load_model('lenet5_model.h5')

drawing = False
last_point = None
last_draw_time = 0
predict_delay = 1.5  # thời gian chờ 1.5 giây sau lần vẽ cuối mới nhận diện
window_name = "Draw digits (press 'c' to clear, ESC to exit)"

canvas = np.zeros((680, 1200), dtype=np.uint8)

# Vẽ nét mượt bằng đoạn thẳng
def draw(event, x, y, flags, param):
    global drawing, last_point, last_draw_time
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_point:
            cv2.line(canvas, last_point, (x, y), 255, 12, lineType=cv2.LINE_AA)
        last_point = (x, y)
        last_draw_time = time.time()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None
        last_draw_time = time.time()

# Xử lý và tách các chữ số
def preprocess_digits(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    positions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 400:  # loại bỏ các nét quá nhỏ
            continue

        digit = thresh[y:y+h, x:x+w]
        resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        padded = padded.astype('float32') / 255.0
        padded = padded.reshape(1, 28, 28, 1)

        digit_images.append(padded)
        positions.append((x, y, w, h))

    if digit_images:
        sorted_digits = sorted(zip(digit_images, positions), key=lambda x: x[1][0])
        digit_images, positions = zip(*sorted_digits)
    else:
        digit_images, positions = [], []

    return digit_images, positions

# Main loop
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

while True:
    canvas_rgb = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)

    now = time.time()
    if now - last_draw_time > predict_delay and np.count_nonzero(canvas) > 0:
        digit_imgs, positions = preprocess_digits(canvas)
        for i, digit_img in enumerate(digit_imgs):
            pred = model.predict(digit_img, verbose=0)
            digit = np.argmax(pred)
            x, y, w, h = positions[i]
            cv2.rectangle(canvas_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(canvas_rgb, str(digit), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(window_name, canvas_rgb)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        canvas[:] = 0
        last_draw_time = 0

cv2.destroyAllWindows()
