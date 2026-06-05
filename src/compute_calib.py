import cv2      # 컴퓨터 비전 알고리즘 활용을 위한 라이브러리입니다.
import numpy as np # 행렬 및 다차원 배열 연산을 위한 수학 라이브러리입니다.
import glob     # 지정된 패턴(예: *.jpg)과 일치하는 파일 경로를 탐색하는 모듈입니다.

# 캘리브레이션에 사용된 체스보드의 내부 코너 개수(가로 9개, 세로 6개)를 정의합니다.
CHESSBOARD_SIZE = (9, 6)
# 체스보드 격자 1칸의 실제 물리적 크기(단위: mm)를 정의합니다.
SQUARE_SIZE_MM = 25.0 

def main():
    print("카메라 캘리브레이션 및 스테레오 정렬 연산을 시작합니다.")

    # 3D 공간 상의 체스보드 코너 좌표(객체 포인트)를 담을 영행렬(Zero Matrix)을 생성합니다.
    # 체스보드는 평면이므로 Z축 좌표는 0으로 고정되며, X와 Y 좌표만 기록됩니다.
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    # np.mgrid를 사용하여 격자망 인덱스를 생성하고 물리적 크기(25.0mm)를 곱하여 실제 3D 좌표계를 완성합니다.
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    # objpoints: 모든 이미지에서 공통적으로 사용되는 3D 객체 좌표 리스트
    # imgpoints_l, imgpoints_r: 영상에서 검출된 2D 픽셀 좌표 리스트
    objpoints, imgpoints_l, imgpoints_r = [], [], []

    # calib_images 폴더 내의 좌/우 이미지를 이름 오름차순으로 정렬하여 불러옵니다.
    left_images = sorted(glob.glob('calib_images/left_*.jpg'))
    right_images = sorted(glob.glob('calib_images/right_*.jpg'))

    # 이미지가 존재하지 않으면 알고리즘을 수행할 수 없으므로 종료합니다.
    if not left_images:
        print("오류: 캘리브레이션 이미지를 찾을 수 없습니다. 이미지 수집을 먼저 진행하십시오.")
        return

    img_shape = None # 이미지의 해상도 크기를 저장할 변수입니다.
    
    # 좌우 이미지를 쌍(Pair) 단위로 묶어서 반복 처리합니다.
    for i, (img_l_path, img_r_path) in enumerate(zip(left_images, right_images)):
        # 디스크에서 이미지를 메모리로 적재합니다.
        img_l, img_r = cv2.imread(img_l_path), cv2.imread(img_r_path)
        # 코너 검출 알고리즘은 컬러 데이터가 필요 없으므로 연산량 감소를 위해 그레이스케일(흑백)로 변환합니다.
        gray_l, gray_r = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        # 첫 번째 반복에서 이미지의 해상도(width, height)를 저장합니다.
        if img_shape is None: 
            img_shape = gray_l.shape[::-1]

        # 흑백 이미지에서 체스보드의 내부 코너 위치(픽셀 좌표)를 1차적으로 탐색합니다.
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

        # 양쪽 이미지 모두에서 코너 검출에 성공한 경우에만 캘리브레이션 데이터로 사용합니다.
        if ret_l and ret_r:
            objpoints.append(objp) # 3D 기준 좌표를 리스트에 추가합니다.
            
            # SubPix(서브픽셀) 연산의 종료 조건(최대 반복 횟수 30회, 최소 오차 0.001)을 정의합니다.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # 정수 단위 픽셀로 검출된 코너 좌표를 소수점 이하 단위(Sub-pixel)의 고정밀도로 재조정합니다.
            imgpoints_l.append(cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria))
            imgpoints_r.append(cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria))
            print(f"  [{i+1}/{len(left_images)}] 코너 검출 완료")
        else:
            print(f"  [{i+1}/{len(left_images)}] 코너 검출 실패 (해당 프레임 제외)")

    # 유효한 캘리브레이션 쌍이 10장 미만일 경우 오차가 커지므로 연산을 거부합니다.
    if len(objpoints) < 10:
        print("\n오류: 유효한 체스보드 패턴 데이터가 부족합니다. 촬영을 다시 진행하십시오.")
        return

    print(f"\n총 {len(objpoints)}개의 데이터 세트로 파라미터 산출을 시작합니다.")
    
    # 단일 카메라에 대한 내부 파라미터(K: Camera Matrix)와 왜곡 계수(D: Distortion Coefficients)를 계산합니다.
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # 두 카메라 간의 상관관계를 구하기 위해, 산출된 내부 파라미터를 고정(CALIB_FIX_INTRINSIC)하고 스테레오 캘리브레이션을 수행합니다.
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l, K_r, D_r, img_shape,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=flags
    )
    # R: 회전 행렬(Rotation), T: 병진 이동 벡터(Translation), E: 본질 행렬(Essential), F: 기본 행렬(Fundamental)

    print(f"연산 완료. 재투영 오차율(RMS Error): {ret:.4f} 픽셀")
    
    # 스테레오 카메라의 에피폴라 선을 수평으로 정렬(Rectification)하기 위한 변환 행렬(R1, R2, P1, P2)을 계산합니다.
    # Q 매트릭스는 시차(Disparity)를 3D 깊이(Depth)로 변환할 때 사용되는 투영 행렬입니다.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, img_shape, R, T)
    
    # 실시간 영상 처리 속도를 높이기 위해, 왜곡 보정 및 수평 정렬을 위한 픽셀 매핑(Map) 데이터를 사전 계산하여 저장합니다.
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, img_shape, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, img_shape, cv2.CV_16SC2)

    # 산출된 맵 데이터와 행렬들을 .npz (Numpy 압축 배열 포맷) 파일로 직렬화하여 디스크에 영구 저장합니다.
    np.savez('stereo_calib.npz', map1_l=map1_l, map2_l=map2_l, map1_r=map1_r, map2_r=map2_r, Q=Q)
    print("데이터 저장 완료: 'stereo_calib.npz' 파일이 성공적으로 생성되었습니다.")

if __name__ == '__main__':
    main()
