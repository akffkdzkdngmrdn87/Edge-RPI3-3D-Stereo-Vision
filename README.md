# Edge AI 3D Stereo Vision System
**Raspberry Pi 3 기반 실시간 3D 심도 및 객체 면적 추정 시스템**

[![Platform](https://img.shields.io/badge/platform-Raspberry_Pi_3-red.svg)](https://www.raspberrypi.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)

## 1. 프로젝트 개요 (Project Overview)

* **연구 목적:** 본 프로젝트는 상용 3D 센서(LiDAR, Depth Camera)를 대체하여, 범용 웹캠 2대와 단일 보드 컴퓨터(SBC)를 활용한 엣지 컴퓨팅 기반의 3D 스테레오 비전 시스템 구현 가능성을 검증(PoC)하기 위해 진행되었습니다.
* **연구 범위:** 1GB RAM 수준의 제한된 컴퓨팅 자원 환경(Raspberry Pi 3)에서 발생하는 메모리 병목 현상을 최적화하고, 수학적 필터링을 결합하여 실시간 사물 거리 및 객체 면적(Bounding Box) 추정 시스템을 구축하였습니다.
* **개발 방법론:** 본 시스템의 알고리즘 설계 및 문제 해결 과정은 대형 언어 모델(LLM)을 활용한 AI-Assisted 프로그래밍 방식을 적용하여 진행되었으며, 하드웨어 최적화부터 웹 스트리밍 기반 관제 시스템까지 통합 파이프라인을 구축하였습니다.

## 2. 시스템 개발 환경 (System Environment)

서버 환경에서의 비전 연산 및 원격 관제를 위해 크로스 컴파일 및 패키지 의존성을 최적화하였습니다.
* **엣지 디바이스:** Raspberry Pi 3 Model B (1GB RAM)
* **운영체제 (OS):** Ubuntu 22.04 LTS (Headless Server)
* **카메라 제원:** ABKO HD720p Webcams x 2 (Stereo Configuration)
* **미들웨어 및 프레임워크:** Python 3.8+, OpenCV (`opencv-python-headless`), NumPy, Flask

## 3. 핵심 아키텍처 및 알고리즘 (Core Architecture)

1. **메모리 최적화 및 SGBM 적용:** 자원 한계를 극복하기 위해 ZRAM 스왑을 적용하고 연산 해상도를 320x240으로 제한하였습니다. 깊이 맵의 텍스처 소실 현상을 개선하기 위해 SGBM(Semi-Global Block Matching) 3-Way 모드를 적용하였습니다.
2. **투영 오차 보정 (Geometric Calibration):** 3D 투영 매트릭스 변환 시 발생하는 Z축 역전 오류를 방지하기 위해, 기하학적 비례 수식($Z = f \cdot B / d$)을 직접 구현하고 실측 기반의 상수를 적용하여 측정 정밀도를 개선하였습니다.
3. **데이터 안정화 필터링:** 저해상도 연산 시 발생하는 거리 수치의 요동(Jittering) 현상을 제어하기 위해 관심 영역(ROI)의 중앙값(Median)을 추출하고, 지수 이동 평균(EMA) 필터를 적용하여 데이터를 평활화(Smoothing)하였습니다.
4. **객체 면적 산출 (Dimension Estimation):** 산출된 거리 데이터를 바탕으로 카메라의 시야각(FOV) 비례식을 융합하여, 대상 객체의 물리적 가로 및 세로 길이(cm)를 실시간으로 추정합니다.

## 4. 퀵 스타트 (Quick Start)

시스템을 구동하기 전, 운영체제의 ZRAM 스왑 영역이 정상적으로 활성화되어 있는지 확인해야 합니다.

### ⚙️ [사전 준비] 시스템 패키지 및 환경 구성
```bash
# 1. 패키지 업데이트 및 영상 처리 유틸리티 설치
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy v4l-utils

# 2. Headless 전용 라이브러리 및 종속성 패키지 설치
pip3 install opencv-python-headless flask numpy
```

### 🚀 [실행 가이드] 캘리브레이션 및 스캐너 가동
디스플레이 출력 장치 없이 SSH 터미널을 통해 실행 및 제어가 가능합니다.

```bash
# 1. 하드웨어 I/O 권한 부여 및 카메라 디바이스 확인
sudo chmod 777 /dev/video*
ls -l /dev/video*

# 2. 스테레오 캘리브레이션 (초기 1회 수행)
# 25.0mm 규격 체스보드 패턴을 활용하여 영점 매트릭스(stereo_calib.npz)를 추출합니다.
python3 capture_images.py
python3 compute_calib.py

# 3. 객체 면적 추정 스캐너 구동 (웹 스트리밍 서버)
python3 live_dimension_scanner.py
```

* **원격 관제망 접속:** 시스템 구동 후, 동일 네트워크에 접속된 PC의 웹 브라우저를 통해 `http://[라즈베리파이IP]:5000` 에 접속하면 실시간 심도 맵 및 면적 추정 결과를 모니터링할 수 있습니다.
* **라즈베리파이3의 최적화는 반드시 다음 링크를 확인하고 작업을 해주시길 바랍니다.** https://github.com/akffkdzkdngmrdn87/-3-Rev1.3/blob/main/%ED%95%84%EB%8F%85.md
