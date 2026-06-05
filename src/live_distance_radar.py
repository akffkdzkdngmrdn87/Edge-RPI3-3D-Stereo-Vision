import cv2
import numpy as np
from flask import Flask, Response
import sys

# Flask 웹 서버 환경을 설정합니다.
app = Flask(__name__)

# 심도(Depth) 산출을 위한 실측 기반 보정 상수입니다.
MAGIC_FACTOR = 1800.0 
# 카메라 이미지 센서와 렌즈 사이의 픽셀 단위 초점 거리(Focal Length in pixels)입니다.
# 320x240 해상도 기반으로 캘리브레이션 단계에서 산출 또는 근사된 값입니다.
FOCAL_PIXELS = 277.0 

# 사전 연산된 왜곡 보정 데이터를 시스템에 로드합니다.
try:
    calib_data = np.load('stereo_calib.npz')
    map1_l, map2_l = calib_data['map1_l'], calib_data['map2_l']
    map1_r, map2_r = calib_data['map1_r'], calib_data['map2_r']
except Exception:
    sys.exit("오류: 캘리브레이션 매트릭스를 찾을 수 없습니다.")

def find_cameras():
    # 0~9 번 장치 중 사용 가능한 카메라 인덱스 리스트를 반환하는 리스트 컴프리헨션 구문입니다.
    indices = [i for i in range(10) if cv2.VideoCapture(i, cv2.CAP_V4L2).isOpened()]
    return indices

cam_slots = find_cameras()
if len(cam_slots) < 2: 
    sys.exit("오류: 시스템에서 2대의 카메라 장치를 인식하지 못했습니다.")

left_idx, right_idx = cam_slots[1], cam_slots[0]
cam_left = cv2.VideoCapture(left_idx, cv2.CAP_V4L2)
cam_right = cv2.VideoCapture(right_idx, cv2.CAP_V4L2)

width, height = 320, 240
for cam in (cam_left, cam_right):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 로컬 블록 매칭 알고리즘 객체를 생성합니다. (사양 최적화 설정 포함)
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=11,
    P1=8 * 1 * 11 ** 2, P2=32 * 1 * 11 ** 2,
    disp12MaxDiff=10, uniquenessRatio=5,
    speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

def generate_frames():
    center_x, center_y = width // 2, height // 2
    smoothed_d = 0.0 # EMA 필터용 변수

    while True:
        ret_l, frame_l = cam_left.read()
        ret_r, frame_r = cam_right.read()
        if not ret_l or not ret_r: continue
        
        # 렌즈 왜곡 보정 및 이미지 평탄화 (에피폴라 1D 제약 만족용)
        rectified_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rectified_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)
        
        # 스테레오 시차 연산 수행 및 실수 픽셀 스케일 변환
        disparity_16 = stereo.compute(gray_l, gray_r)
        disparity = disparity_16.astype(np.float32) / 16.0
        
        # 화면 정중앙 영역 20x20 크기 샘플 추출
        roi = disparity[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10]
        valid_d = roi[roi > 0]
        distance_cm = 0.0
        
        # 뎁스 맵 렌더링용 컬러 정규화 처리
        norm_disp = cv2.normalize(disparity_16, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)

        if len(valid_d) > 30:
            current_d = np.median(valid_d)
            # 측정 노이즈 완화를 위한 EMA 필터 연산 적용
            smoothed_d = current_d if smoothed_d == 0.0 else 0.1 * current_d + 0.9 * smoothed_d 
            distance_cm = MAGIC_FACTOR / smoothed_d
            
            # 중앙 타겟과 동일한 깊이(거리)를 가진 인접 객체를 찾기 위해 임계값 경계를 설정합니다.
            # OpenCV의 inRange 함수 구동 시 Numpy float32 포맷의 구조적 타입 에러가 발생하는 것을 
            # 원천 방지하기 위해, 파이썬 내장 float() 함수로 명시적 형 변환(Type Casting)을 수행합니다.
            lower_bound = float(smoothed_d - 5.0)
            upper_bound = float(smoothed_d + 5.0)
            
            # 현재 중앙 물체와 비슷한 시차 범위에 속하는 픽셀들만 하얗게(255) 남기는 마스크를 생성합니다.
            mask = cv2.inRange(disparity, lower_bound, upper_bound)
            
            # 윤곽선 검출 알고리즘(findContours)을 실행하여 연속된 픽셀 덩어리의 경계를 추출합니다.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # 추출된 윤곽선을 둘러싸는 최소 크기의 직사각형 테두리(Bounding Box)를 계산합니다.
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 계산된 Bounding Box 영역 내부에 화면의 정중앙 지점(center)이 포함되는지 검사합니다.
                if x < center_x < x + w and y < center_y < y + h:
                    # 삼각 함수 비례식 기반의 기하학적 투영 역산식 (W_real = W_pixel * Z / F_pixel)
                    # 이미지 픽셀 단위의 폭과 높이를 실제 물리적 규격(cm)으로 변환합니다.
                    real_width_cm = (w * distance_cm) / FOCAL_PIXELS
                    real_height_cm = (h * distance_cm) / FOCAL_PIXELS
                    
                    # 추정된 객체의 외곽선을 영상 스트림에 시각적으로 랜더링합니다.
                    cv2.rectangle(color_disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    dim_text = f"{real_width_cm:.1f}cm x {real_height_cm:.1f}cm"
                    # 산출된 가로/세로 길이 수치(cm)를 화면에 묘화합니다.
                    cv2.putText(color_disp, dim_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    break # 조건을 만족하는 대상 객체를 찾았으므로 윤곽선 순회를 중단합니다.
                    
        else:
            smoothed_d = 0.0 # 측정 범위를 벗어나면 필터를 초기화합니다.

        # 화면 중심 십자선 마커 및 추정된 Z축 거리(cm) 텍스트를 출력합니다.
        cv2.drawMarker(color_disp, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        if distance_cm > 0:
            cv2.putText(color_disp, f"Dist: {distance_cm:.1f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 결과 렌더링 프레임 인코딩 후 브라우저 송출
        ret, buffer = cv2.imencode('.jpg', color_disp)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    # 서버 엔드포인트 라우팅 함수
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("객체 면적 측정 및 3D 스캐너 서버 구동을 시작합니다.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
