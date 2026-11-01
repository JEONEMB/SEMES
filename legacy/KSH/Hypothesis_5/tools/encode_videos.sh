#!/bin/bash
# 보고서 영상 인코딩 : report_media 가 만든 PNG 프레임 → H.264 mp4 (브라우저 재생용)
# 사용 : tools/encode_videos.sh <프레임 폴더 상위> <출력 폴더>
set -e
IN="$1"; OUT="$2"; mkdir -p "$OUT"
for d in v1_parallax v2_phasecorr v3_subpixel v4_gaussnewton; do
  ffmpeg -y -loglevel error -framerate 12 -i "$IN/$d/%04d.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
         -c:v libx264 -pix_fmt yuv420p -crf 22 -movflags +faststart "$OUT/$d.mp4"
done
ls -la "$OUT"
