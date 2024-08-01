from flask import Flask, render_template, request, jsonify , send_file
import cv2
import base64
import numpy as np
import detect_boxes
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result_img = process_image(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'result': result_img_str})

@app.route('/capture', methods=['POST'])
def capture():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    result_img = process_image(img)
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'result': result_img_str})

@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')

@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

def process_image(img):
    try:
        model_path = "./pothole_model.pt"
        model = detect_boxes.load_model(model_path)


        image = cv2.imread('./input.jpg')
        results = model(image)

        img,count = detect_boxes.draw_boxes(results,image,model.names)
        return img
    
    except Exception:
        print('Error Occurred')
        traceback.print_exc()
        return img

if __name__ == '__main__':
    app.run(debug=True)
