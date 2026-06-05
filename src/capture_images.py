import cv2  # 영상 처리 및 카메라 제어를 위한 OpenCV 라이브러리입니다.
import os   # 디렉토리 생성 및 파일 경로 제어를 위한 표준 라이브러리입니다.
import sys  # 프로그램 강제 종료 등 시스템 제어를 위한 라이브러리입니다.

def find_cameras():
    # 사용 가능한 카메라 장치(Index)를 탐색하여 리스트로 반환하는 함수입니다.
    print("시스템에 연결된 카메라 장치를 탐색합니다...")
    available_indices = []
    
    # 0번부터 9번까지의 장치 번호를 순회하며 카메라 연결 상태를 점검합니다.
    for i in range(10):
        # Linux 환경의 V4L2(Video4Linux2) 드라이버를 사용하여 카메라를 호출합니다.
        cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cam.isOpened(): # 카메라 객체가 정상적으로 초기화되었는지 확인합니다.
            ret, frame = cam.read() # 프레임을 1장 읽어옵니다.
            if ret and frame is not None: 
                available_indices.append(i) # 정상적인 프레임이 반환되면 유효한 장치로 리스트에 추가합니다.
        cam.release() # 테스트가 끝난 카메라 객체는 메모리 반환을 위해 해제합니다.
        
    return available_indices

def main():
    cam_slots = find_cameras() # 함수를 호출하여 유효한 카메라 인덱스를 가져옵니다.
    
    # 스테레오 비전은 반드시 2대 이상의 카메라가 필요하므로 예외 처리를 수행합니다.
    if len(cam_slots) < 2:
        print("오류: 2대의 카메라를 인식할 수 없습니다. 연결 상태를 확인하십시오.")
        sys.exit(1) # 프로그램 강제 종료
        
    # 물리적 배치에 맞게 좌측과 우측 카메라의 인덱스를 할당합니다. 
    # (하드웨어 연결 순서에 따라 인덱스는 달라질 수 있습니다.)
    left_idx, right_idx = cam_slots[1], cam_slots[0]
    print(f"카메라 할당 완료 - 좌측: video{left_idx} | 우측: video{right_idx}")

    # 수집한 이미지를 저장할 디렉토리 경로를 설정합니다.
    save_dir = "calib_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) # 디렉토리가 존재하지 않으면 새로 생성합니다.

    # 캡처를 진행할 좌/우 카메라 객체를 생성합니다.
    cam_left = cv2.VideoCapture(left_idx, cv2.CAP_V4L2)
    cam_right = cv2.VideoCapture(right_idx, cv2.CAP_V4L2)

    # 연산 부하 감소 및 메모리 절약을 위해 카메라 해상도를 320x240으로 고정합니다.
    width, height = 320, 240
    for cam in (cam_left, cam_right):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)   # 프레임의 너비를 설정합니다.
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 프레임의 높이를 설정합니다.

    count = 0 # 캡처된 이미지 쌍의 개수를 기록할 변수입니다.
    print("\n[안내] 25.0mm 규격의 체스보드를 카메라 전방에 위치시킨 후 Enter를 누르십시오.")
    
    try:
        while True:
            # 사용자로부터 터미널 입력을 대기합니다. (q 입력 시 반복문 종료)
            cmd = input(f">>> [누적 {count}쌍] 캡처를 진행하려면 Enter, 종료하려면 'q'를 입력하세요: ")
            if cmd.lower() == 'q': 
                break

            # 버퍼에 남아있는 과거 프레임을 비우기 위해 5장의 프레임을 공회전시킵니다.
            # 이는 좌우 카메라의 캡처 시점을 최대한 일치시키기 위한 동기화(Synchronization) 기법입니다.
            for _ in range(5): 
                cam_left.read()
                cam_right.read()

            # 좌/우 카메라에서 최신 프레임을 각각 1장씩 읽어옵니다.
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            # 두 카메라 모두 프레임을 정상적으로 읽어왔을 경우에만 저장 프로세스를 수행합니다.
            if ret_l and ret_r:
                # 저장될 파일의 이름을 00, 01, 02 형식으로 포맷팅하여 생성합니다.
                img_name_l = os.path.join(save_dir, f"left_{count:02d}.jpg")
                img_name_r = os.path.join(save_dir, f"right_{count:02d}.jpg")
                
                # 프레임을 디스크에 이미지 파일(.jpg)로 저장합니다.
                cv2.imwrite(img_name_l, frame_l)
                cv2.imwrite(img_name_r, frame_r)
                
                print(f"저장 완료: {img_name_l}, {img_name_r}")
                count += 1 # 캡처 카운트를 1 증가시킵니다.

    finally:
        # 정상 종료 또는 오류 발생 여부와 상관없이 카메라 객체를 안전하게 해제합니다.
        cam_left.release()
        cam_right.release()
        print(f"\n작업 종료: 총 {count}쌍의 이미지 수집이 완료되었습니다.")

if __name__ == '__main__':
    main()
