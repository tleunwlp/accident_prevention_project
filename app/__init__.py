from flask import Flask, render_template, Response, url_for
import cv2
import torch
from ultralytics import YOLO
import datetime
import time
import numpy as np
from flask_socketio import SocketIO, emit

SPEED_THRESHOLD = 0.01
RED = (255, 0, 0)  # 기존 빨강색을 현대적이고 직관적인 파랑색으로 변경
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)  # 파랑색을 조금 더 진한 색으로 변경
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

    def draw_boxes(frame, boxes, colors):
        for box in boxes:
            class_id = int(box[0])
            color = colors[class_id % len(colors)]
            xmin, ymin, width, height = box[1:]
            xmax = xmin + width
            ymax = ymin + height

            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)

    COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]

    def gen_frames():
        global previous_positions, overload_timestamps, tripod_timestamps, stop_timestamps, warned_tracks, last_alert_time

        while True:
            success, frame = cap.read()  # 비디오에서 프레임 읽기
            if not success:
                break  # 프레임을 읽지 못하면 종료
            else:
                # 프레임 크기 조정
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # 프레임 처리 시작 시간 기록
                start = datetime.datetime.now()

                # YOLO 모델을 이용한 객체 검출
                detection = model.predict(source=[frame], save=False)[0]
                results = []

                # 검출된 객체의 정보를 리스트에 저장
                for data in detection.boxes.data.tolist():  # data: [xmin, ymin, xmax, ymax, confidence_score, class_id]
                    confidence = float(data[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    label = int(data[5])
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])

                # DeepSort를 이용한 객체 추적
                tracks = tracker.update_tracks(results, frame=frame)
                
                # 과적재 화물차 검출
                overload_detection = overload_model.predict(source=[frame], save=False)[0]
                
                # 과적재 화물차 인식된 객체 정보를 저장
                overload_results = []
                for data in overload_detection.boxes.data.tolist():
                    confidence = float(data[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    label = int(data[5])
                    overload_results.append([xmin, ymin, xmax - xmin, ymax - ymin, confidence, label])
                    
                # 삼각대 검출
                tripod_detection = tripod_model.predict(source=[frame], save=False)[0]
                
                # 삼각대 인식된 객체 정보를 저장
                tripod_results = []
                for data in tripod_detection.boxes.data.tolist():
                    confidence = float(data[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    label = int(data[5])
                    tripod_results.append([xmin, ymin, xmax - xmin, ymax - ymin, confidence, label])

                # 추적된 객체 정보를 프레임에 그리기
                current_time = datetime.datetime.now()
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()

                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                    # 이전 위치와 현재 위치를 이용하여 속도 계산
                    if track_id in previous_positions:
                        prev_x, prev_y = previous_positions[track_id]
                        speed = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                    else:
                        speed = 0

                    previous_positions[track_id] = (center_x, center_y)

                    # 속도 임계값을 이용하여 정지 차량 판별
                    if speed < SPEED_THRESHOLD:
                        color = RED
                        label = f"Stopped: {track_id}"
                        if track_id in stop_timestamps:
                            first_detected = stop_timestamps[track_id]
                            if (current_time - first_detected).total_seconds() > STOP_THRESHOLD_SECONDS:
                                if track_id not in warned_tracks:
                                    # 정지 차량이 5초 이상 감지된 경우 클라이언트로 알림 전송
                                    if (track_id not in last_alert_time or 
                                            (current_time - last_alert_time[track_id]).total_seconds() > ALERT_COOLDOWN_SECONDS):
                                        socketio.emit('warning', {'message': '정지 차량 감지!'})
                                        last_alert_time[track_id] = current_time  # 마지막 알림 시간을 기록
                                        warned_tracks.add(track_id)  # 경고가 발생한 객체를 추적
                        else:
                            stop_timestamps[track_id] = current_time
                    else:
                        color = GREEN
                        label = f"Moving: {track_id}"
                        if track_id in stop_timestamps:
                            del stop_timestamps[track_id]

                    draw_boxes(frame, [[track_id, xmin, ymin, xmax - xmin, ymax - ymin]], COLORS)

                    # 레이블 추가
                    #cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 120, ymin), color, -1)
                    cv2.putText(frame, label, (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, WHITE, 1)
                    
                # 과적재 화물차 인식된 객체에 대한 처리
                # for data in overload_results:
                #     xmin, ymin, xmax, ymax, confidence, label = data
                #     center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                #     if (center_x, center_y) in overload_timestamps:
                #         first_detected, count = overload_timestamps[(center_x, center_y)]
                #         if (current_time - first_detected).total_seconds() > OVERLOAD_THRESHOLD_SECONDS:
                #             cv2.putText(frame, f"Overload: {label}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, WHITE, 1)
                #             if (center_x, center_y) not in last_alert_time or (current_time - last_alert_time[(center_x, center_y)]).total_seconds() > ALERT_COOLDOWN_SECONDS:
                #                 socketio.emit('warning', {'message': '과적재 화물차 감지!'})
                #                 last_alert_time[(center_x, center_y)] = current_time
                #     else:
                #         overload_timestamps[(center_x, center_y)] = (current_time, 1)

                # 삼각대 인식된 객체에 대한 처리
                for data in tripod_results:
                    xmin, ymin, xmax, ymax, confidence, label = data
                    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                    if (center_x, center_y) in tripod_timestamps:
                        first_detected, count = tripod_timestamps[(center_x, center_y)]
                        if (current_time - first_detected).total_seconds() > TRIPOD_THRESHOLD_SECONDS:
                            cv2.putText(frame, f"Tripod: {label}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.2, WHITE, 1)
                            if (center_x, center_y) not in last_alert_time or (current_time - last_alert_time[(center_x, center_y)]).total_seconds() > ALERT_COOLDOWN_SECONDS:
                                socketio.emit('warning', {'message': '삼각대 감지!'})
                                last_alert_time[(center_x, center_y)] = current_time
                    else:
                        tripod_timestamps[(center_x, center_y)] = (current_time, 1)
                
                # 프레임 처리 종료 시간 기록
                end = datetime.datetime.now()

                # 프레임 처리 시간 계산 및 FPS 표시
                total = (end - start).total_seconds()
                fps_text = f'FPS: {1 / total:.2f}'
                cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

                # 프레임을 JPEG 형식으로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # 생성된 프레임을 반환하여 스트리밍
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video')
    def video():
        # 비디오 스트리밍 엔드포인트
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    socketio.init_app(app)
    return app
