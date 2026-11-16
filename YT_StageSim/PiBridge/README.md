# PiBridge — 라즈베리파이 중계 (임시 틀)

Unity 시뮬레이터가 파이의 카메라·STM32 를 쓰도록 중계한다. **아직 실제 파이에서 실행해 본 적 없음 (월요일 현장 첫 시험).**

## 실행 (파이, 저장소 루트)

```bash
cd /home/pi/S15P31A507
demo_web/.venv/bin/python /path/to/pi_bridge.py --host 0.0.0.0
```

- 카메라만: `--no-stage` / STM32 만: `--no-camera` / 카메라 없이 화면 시험: `--camera off`
- 카메라 설정은 verify_web 기본과 같다: 노출 8333 µs, 게인 3.0, 10 fps, 캘리브레이션 `hardware/camera_calib/camera_calib.yml`
- verify_web·demo_web 와 동시에 켜지 않는다 (카메라·USB 포트는 한 프로그램만).
- STM32 를 커널 드라이버(`/dev/robot_stage`)로 쓰는 중이면 시리얼 중계를 하지 않는다.

## Unity 에서

연결·설정 탭
- 스테이지: `라즈베리파이 경유` → `<파이 IP>:5555`
- 카메라: `라즈베리파이 영상` → `http://<파이 IP>:8091`

## 제공 주소

| 주소 | 내용 |
| --- | --- |
| tcp 5555 | STM32 시리얼 바이트 그대로 (robot_stage 프레임) |
| GET /health | 카메라 상태, 플레이트 표시 위치, 알고리즘 좌표 → 화면 좌표 변환 값 (view_map, QR·나사 윤곽용, 11-16 추가) |
| GET /video | 1280×720 흑백 MJPEG (verify_web 화면과 같은 변환) |
| POST /view | `{"zoom": 1~8, "center": "plate"|"image"}` |
| GET /settle?px=0.25&timeout=6 | 흔들림 판정 (verify_web 과 같음) |
| GET /frame?average=1 | 알고리즘 입력 3840×2160 8bit 흑백 원시 바이트 |
