import cv2
import numpy as np
from flask import Flask,  request, jsonify, json
import base64
import mediapipe as mp
import joblib
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
model = joblib.load('hand_gesture_model.pkl')

categories=[
["fine",'بخير'],
["hello","مرحبا, كيف حالك؟"],
["stop",'قف'],
["yes",'نعم']]
words = []
sequence = ''
fontFile = "Sahel.ttf"
font = ImageFont.truetype(fontFile, 40)
app = Flask(__name__)

camera = cv2.VideoCapture(0)
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
        return lmlist, bbox
"""
def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            detector = handDetector()
            img = detector.findHands(frame)
            lmlist, bbox = detector.findPosition(img)
            if bbox:
                # Your custom model and prediction code here
                # Ensure all processing is done correctly
                x, y, x2, y2 = bbox
                hand_img = img[y:y2, x:x2]
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (64, 64))  # Ensure the image is resized as per training
                hand_img_flat = hand_img.flatten().reshape(1, -1)
                prediction = model.predict(hand_img_flat)
                for category in categories:
                    if category[0] == prediction[0]:
                        sequence = category[1]
                probability = model.predict_proba(hand_img_flat).max()

                reshaped_text = arabic_reshaper.reshape(sequence)   
                bidi_text = get_display(reshaped_text) 
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 300), bidi_text, (0,0,0), font=font)
                img = np.array(img_pil)
                words.append(reshaped_text)   

                cv2.putText(img, f'{prediction[0]} {probability:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # After processing the image and adding any overlays or text:
                prediction_data = json.dumps({'prediction': prediction[0]})
                yield f"event: prediction\ndata: {prediction_data}\n\n"
                ret, buffer = cv2.imencode('.jpg', img)
                if not ret:
                    continue  # Skip the frame if it fails to encode
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Correct format for HTTP streaming
"""

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = handDetector()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)
        if bbox:
            x, y, x2, y2 = bbox
            hand_img = img[y:y2, x:x2]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img_flat = hand_img.flatten().reshape(1, -1)
            results = model.predict(hand_img_flat)
            return jsonify({'result': results[0]})
        else:
            return jsonify({'result': 'No hand detected'})
    except Exception as e:
        return jsonify({'error': str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)
