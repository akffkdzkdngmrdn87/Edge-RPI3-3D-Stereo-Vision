import cv2           # 비전 알고리즘 처리를 위한 OpenCV
import numpy as np   # 행렬 연산 및 필터링을 위한 Numpy
from flask import Flask, Response # 브라우저에 영상 스트리밍을 제공하기 위한 웹 프레임워크
import sys           # 시스템 종료 예외 처리를 위한 패키지

app = Flask(__name__) # Flask 웹 서버 인스턴스를 초기화합니다.

# 사전에 연산해 둔 캘리브레이션 맵핑 데이터를 불러옵니다.
try:
    calib_data = np.load('stereo_calib.npz')
    map1_l, map2_l = calib_data['map1_l'], calib_data['map2_l']
    map1_r, map2_r = calib_data['map1_r'], calib_data['map2_r']
    print("캘리브레이션 매트릭스 로드 완료.")
except Exception as e:
    # 파일이 존재하지 않거나 손상되었을 경우 서버 구동을 중단합니다.
    sys.exit("오류: 'stereo_calib.npz' 파일을 찾을 수 없습니다. 캘리브레이션을 먼저 수행하십시오.")

def find_cameras():
    # 가용 카메라 디바이스 인덱스를 탐색하여 리스트로 반환합니다.
    indices = []
    for i in range(10):
        cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None: indices.append(i)
        cam.release()
    return indices

cam_slots = find_cameras()
if len(cam_slots) < 2: 
    sys.exit("오류: 시스템에서 2대의 카메라 장치를 인식하지 못했습니다.")

# 좌우 카메라 인덱스 할당 및 초기화
left_idx, right_idx = cam_slots[1], cam_slots[0]
cam_left = cv2.VideoCapture(left_idx, cv2.CAP_V4L2)
cam_right = cv2.VideoCapture(right_idx, cv2.CAP_V4L2)

width, height = 320, 240
for cam in (cam_left, cam_right):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 실시간성을 확보하기 위해 내부 버퍼 크기를 1로 제한하여 과거 프레임 지연(Lag)을 방지합니다.
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# SGBM (Semi-Global Block Matching) 인스턴스를 생성합니다.
# 다방향 동적 프로그래밍을 통해 로컬 블록 매칭의 노이즈를 줄이고 일관된 깊이 맵을 산출하는 알고리즘입니다.
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=11,
    P1=8 * 1 * 11 ** 2, P2=32 * 1 * 11 ** 2,
    disp12MaxDiff=10, uniquenessRatio=5,
    speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # 연산량 감소를 위한 3-Way 모드 적용
)

def generate_frames():
    # 프레임의 정중앙 좌표를 계산합니다.
    center_x, center_y = width // 2, height // 2
    box_size = 10 # 중앙 관심 영역(ROI)의 크기(픽셀 단위의 반지름 개념)
    
    # EMA(Exponential Moving Average) 필터 처리를 위해 과거 시차 데이터를 보존할 변수입니다.
    smoothed_d = 0.0 

    while True:
        ret_l, frame_l = cam_left.read()
        ret_r, frame_r = cam_right.read()

        if not ret_l or not ret_r: 
            continue # 프레임 읽기에 실패하면 건너뜁니다.
        
        # cv2.remap을 사용하여 카메라 렌즈의 왜곡을 보정하고 에피폴라 선을 수평으로 정렬합니다.
        rectified_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # 블록 매칭 알고리즘은 픽셀의 강도(Intensity) 차이를 비교하므로 흑백 이미지로 변환합니다.
        gray_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)
        
        # SGBM 연산을 수행하여 16비트 정수형(Fixed-point) 시차 맵을 반환받습니다.
        disparity_16 = stereo.compute(gray_l, gray_r)
        # SGBM의 출력 구조에 따라 16.0으로 나누어 실수 형태의 실제 픽셀 시차 값으로 변환합니다.
        disparity = disparity_16.astype(np.float32) / 16.0
        
        # 프레임 중앙부(크기 20x20)의 시차 데이터를 슬라이싱하여 관심 영역(ROI)을 추출합니다.
        roi = disparity[center_y - box_size : center_y + box_size, 
                        center_x - box_size : center_x + box_size]
        
        # 음수나 0은 거리 측정이 실패한 부분(Hole)이므로, 유효한 양수 시차 데이터만 필터링합니다.
        valid_d = roi[roi > 0]
        distance_cm = 0.0
        
        # 유효한 픽셀이 일정 개수(30개) 이상일 경우에만 통계적으로 신뢰할 수 있다고 판단합니다.
        if len(valid_d) > 30:
            current_d = np.median(valid_d) # 아웃라이어(이상치)의 영향을 줄이기 위해 평균(Mean) 대신 중앙값(Median)을 산출합니다.
            
            # EMA(지수 이동 평균) 필터 적용: 단기적 노이즈(Jittering)를 억제하여 데이터를 안정화합니다.
            if smoothed_d == 0.0:
                smoothed_d = current_d # 최초 측정 시 데이터 초기화
            else:
                # 최신 측정치를 10%, 이전 평활화 데이터를 90% 반영하여 값의 급격한 변동을 억제합니다.
                smoothed_d = 0.1 * current_d + 0.9 * smoothed_d 

            # 경험적 보정 상수(Calibration Factor) 정의. 
            # 3D 기하학의 기본 거리 공식 (Z = f * B / d) 구조를 단일 상수로 간소화한 형태입니다.
            # 실측 데이터를 기반으로 Z축의 척도를 보정하기 위해 도출된 상숫값입니다.
            MAGIC_FACTOR = 1800.0 
            distance_cm = MAGIC_FACTOR / smoothed_d # 시차(d)에 반비례하여 거리(cm)를 도출합니다.
        else:
            # 유효 데이터가 부족하면 필터링 변수를 초기화합니다.
            smoothed_d = 0.0 
        
        # 시각화를 위해 시차 데이터를 0~255 범위의 8비트 이미지로 정규화(Normalization)합니다.
        norm_disp = cv2.normalize(disparity_16, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 인간의 시각적 인지력을 높이기 위해 컬러 맵(JET: 가까우면 붉은색, 멀면 파란색)을 덧입힙니다.
        color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)

        # 사용자에게 거리를 측정 중인 ROI 영역을 시각적으로 나타내기 위해 사각형과 십자선을 묘화합니다.
        cv2.rectangle(color_disp, (center_x - box_size, center_y - box_size), 
                      (center_x + box_size, center_y + box_size), (0, 255, 0), 2)
        cv2.drawMarker(color_disp, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        
        # 산출된 거리가 유효 범위(0 ~ 5미터) 내에 있을 경우에만 화면에 텍스트를 출력합니다.
        if distance_cm > 0 and distance_cm < 500: 
            text = f"Dist: {distance_cm:.1f} cm"
            cv2.putText(color_disp, text, (center_x - 70, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(color_disp, "No Target", (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 화면에 송출하기 위해 프레임을 JPEG 포맷으로 인코딩하여 바이너리 버퍼로 변환합니다.
        ret, buffer = cv2.imencode('.jpg', color_disp)
        frame = buffer.tobytes()

        # HTTP MJPEG 스트림 형식으로 프레임을 연속적으로 반환(Yield)합니다.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    # Flask 서버의 루트('/') 경로 접속 시 스트리밍 함수를 Response 객체로 반환합니다.
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("실시간 3D 심도 및 거리 측정 서버 구동을 시작합니다.")
    # 라즈베리파이의 내부망 IP(0.0.0.0)를 개방하여 5000번 포트로 서비스를 실행합니다.
    app.run(host='0.0.0.0', port=5000, threaded=True)
