from flask import Flask, render_template, Response, url_for
import cv2
import torch
from ultralytics import YOLO
import datetime
import numpy as np
from flask_socketio import SocketIO, emit

SPEED_THRESHOLD = 0.01
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)
YELLOW = (255, 255, 0)  # 정지하지 않은 차량의 색상
CONFIDENCE_THRESHOLD = 0.3
STOP_THRESHOLD_SECONDS = 5
OVERLOAD_THRESHOLD_SECONDS = 10
TRIPOD_THRESHOLD_SECONDS = 3
ALERT_COOLDOWN_SECONDS = 30

socketio = SocketIO()

def create_app():
    app = Flask(__name__, template_folder='../templates', static_url_path='/static', static_folder='../static')

    # YOLO 모델 로드
    model_path = "static/models/car_yolov8n_best.pt"
    model = YOLO(model_path)

    # 과적재 화물차 인식 모델
    overload_model_path = "static/models/overload_truck_yolov8n.pt"
    overload_model = YOLO(overload_model_path)

    # 삼각대 인식 모델
    tripod_model_path = "static/models/tripod_yolov8n.pt"
    tripod_model = YOLO(tripod_model_path)

    # DeepSort 트래커 초기화
    from deep_sort_realtime.deepsort_tracker import DeepSort
    tracker = DeepSort(max_age=30)

    # 비디오 파일 경로
    video_path = "static/videos/test_video_crop.avi"
    cap = cv2.VideoCapture(video_path)

    FRAME_WIDTH = 420
    FRAME_HEIGHT = 280 

    # 이전 프레임 객체 위치 저장
    global previous_positions, overload_timestamps, tripod_timestamps, stop_timestamps, warned_tracks, last_alert_time
    previous_positions = {}
    overload_timestamps = {}
    tripod_timestamps = {}
    stop_timestamps = {}
    warned_tracks = set()
    last_alert_time = {}

    @app.route('/')
    def index():
        image_file = url_for('static', filename='images/warning.jpeg')
        return render_template('index.html', image_file=image_file )

    @app.route('/cctv')
    def video_show():
        image_file = url_for('static', filename='images/warning.jpeg')
        return render_template('video_show.html', image_file=image_file)

    @app.route('/drone')
    def drone_control():
        return render_template('drone_control.html')

    def draw_boxes(frame, boxes, color):
        for box in boxes:
            xmin, ymin, width, height = box
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)

    def gen_frames():
        global previous_positions, overload_timestamps, tripod_timestamps, stop_timestamps, warned_tracks, last_alert_time

        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                start = datetime.datetime.now()

                detection = model.predict(source=[frame], save=False)[0]
                results = []

                for data in detection.boxes.data.tolist():
                    confidence = float(data[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, int(data[5])])

                tracks = tracker.update_tracks(results, frame=frame)

                current_time = datetime.datetime.now()
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()

                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                    if track_id in previous_positions:
                        prev_x, prev_y = previous_positions[track_id]
                        speed = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                    else:
                        speed = 0

                    previous_positions[track_id] = (center_x, center_y)

                    if speed < SPEED_THRESHOLD:
                        if track_id in stop_timestamps:
                            first_detected = stop_timestamps[track_id]
                            if (current_time - first_detected).total_seconds() > STOP_THRESHOLD_SECONDS:
                                if track_id not in warned_tracks:
                                    if (track_id not in last_alert_time or 
                                            (current_time - last_alert_time[track_id]).total_seconds() > ALERT_COOLDOWN_SECONDS):
                                        socketio.emit('warning', {'message': '정지 차량 감지!'})
                                        last_alert_time[track_id] = current_time
                                        warned_tracks.add(track_id)
                                draw_boxes(frame, [[xmin, ymin, xmax - xmin, ymax - ymin]], RED)
                            else:
                                draw_boxes(frame, [[xmin, ymin, xmax - xmin, ymax - ymin]], BLUE)
                        else:
                            stop_timestamps[track_id] = current_time
                            draw_boxes(frame, [[xmin, ymin, xmax - xmin, ymax - ymin]], BLUE)
                    else:
                        draw_boxes(frame, [[xmin, ymin, xmax - xmin, ymax - ymin]], GREEN)
                        if track_id in stop_timestamps:
                            del stop_timestamps[track_id]
                            warned_tracks.discard(track_id)

                end = datetime.datetime.now()
                total = (end - start).total_seconds()
                fps_text = f'FPS: {1 / total:.2f}'
                cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video')
    def video():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    socketio.init_app(app)
    return app