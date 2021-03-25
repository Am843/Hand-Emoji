import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os

model1 = load_model('emoji.h5')
print(model1)


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("Processed : " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def emojis():
    emojis_folder = 'hand emoji/'
    emoji = []
    for emoj in range(len(os.listdir(emojis_folder))):
        emoji.append(cv2.imread(emojis_folder + str(emoj) + '.jpg'))
    return emoji


def overlay(image, emoji, x, y):
    emoji = cv2.resize(emoji, (150, 150))
    try:
        image[y:y + emoji.shape[0], x:x + emoji.shape[1]] = emoji
    except:
        pass
    return image


# def blend_transparent(face_img, overlay_t_img):
#     overlay_img = overlay_t_img[:, :, :3]
#     overlay_mask = overlay_t_img[:, :, :3]
#     background_mask = 255 - overlay_mask
#     overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
#     background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
#     face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
#     overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
#     return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


emojis = emojis()
print(len(emojis))
cap = cv2.VideoCapture(0)
x, y, w, h = 300, 50, 350, 350
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([100, 170, 255]))
    res = cv2.bitwise_and(frame, frame, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    median = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel_square = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(median, kernel_square, iterations=2)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
    ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
    thresh = thresh[y:y + h, x:x + w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 10000:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            newImage = thresh[y:y + h1, x:x + w1]
            newImage = cv2.resize(newImage, (50, 50))
            pred_prob, pred_class = keras_predict(model1, newImage)
            print(pred_class, pred_prob)
            img = overlay(frame, emojis[pred_class], 350, 300)
    x, y, w, h = 300, 50, 350, 350
    cv2.imshow("frame", frame)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(1)
    if k == 27:
        break
