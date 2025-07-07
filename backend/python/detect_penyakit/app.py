from flask import Flask, render_template, Response
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
import cv2
import uuid
import pymysql.cursors
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import threading
import time

app = Flask(__name__)

# ========== KONFIGURASI MYSQL ==========
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'pi'
app.config['MYSQL_PASSWORD'] = 'Petaniasik123#$'
app.config['MYSQL_DB'] = 'padi_sawah'

# ========== INISIALISASI FOLDER ==========
SAVE_DIR = '/var/www/html/img/hama_padi'
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== LOAD MODEL DAN CLASSIFIER ==========
leafCascade = cv2.CascadeClassifier("static/src/cascade3.xml")

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

model = load_model(
    'static/src/efficientnetb0_model.h5',
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
)

label = [
    'Hawar daun bakteri',
    'Garis daun bakteri',
    'Hawar malai bakteri',
    'Penyakit blas (jamur)',
    'Bercak coklat',
    'Pucuk mati',
    'Embun bulu (jamur bulu halus)',
    'Hama penggerek daun (Hispa)',
    'Normal / Sehat',
    'Penyakit tungro (virus)'
]

# ========== VARIABEL GLOBAL ==========
latest_frame = None
frame_lock = threading.Lock()

# ========== FUNGSI DETEKSI UTAMA ==========
def process_frame(frame):
    current_hour = datetime.now().hour
    if not (6 <= current_hour < 17):
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    leaf = leafCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=7,
        minSize=(150, 150),
        maxSize=(800, 800)
    )

    for (x, y, w, h) in leaf:
        try:
            load = frame[y:y+h, x:x+w]
            resized_load = cv2.resize(load, (224, 224))

            z = tf.keras.utils.img_to_array(resized_load)
            z = np.expand_dims(z, axis=0)
            z = preprocess_input(z)
            classes = model.predict(z, verbose=0)

            index = np.argmax(classes)
            prob = np.max(classes) * 100
            disease = label[index]

            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, resized_load)

            connection = pymysql.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DB'],
                cursorclass=pymysql.cursors.DictCursor
            )

            with connection.cursor() as cursor:
                sql = """INSERT INTO sensor_kameras 
                        (iot_id, penyakit, probabilitas, image) 
                        VALUES (%s, %s, %s, %s)"""
                cursor.execute(sql, ('jTZids5M', disease, float(prob), filename))
                connection.commit()

                cursor.execute("SELECT COUNT(*) AS total FROM sensor_kameras")
                result = cursor.fetchone()
                data_count = result['total']

                if data_count >= 1000:
                    cursor.execute("DELETE FROM sensor_kameras")
                    cursor.execute("ALTER TABLE sensor_kameras AUTO_INCREMENT = 1")
                    connection.commit()

                    for fname in os.listdir(SAVE_DIR):
                        fpath = os.path.join(SAVE_DIR, fname)
                        try:
                            if os.path.isfile(fpath):
                                os.unlink(fpath)
                        except Exception as e:
                            print(f"Gagal menghapus {fpath}: {e}")

            cv2.putText(frame, f"{disease} ({prob:.2f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")
            if 'connection' in locals():
                connection.rollback()
        finally:
            if 'connection' in locals():
                connection.close()

    return frame

# ========== THREAD KAMERA ==========
def camera_thread():
    global latest_frame
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if success:
            processed_frame = process_frame(frame)
            with frame_lock:
                latest_frame = processed_frame.copy()
        time.sleep(0.1)

# ========== GENERATE VIDEO STREAM ==========
def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ========== ROUTES ==========
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('detect.html')

if __name__ == '__main__':
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=7000, debug=False, use_reloader=False)
