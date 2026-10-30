"""Compare every decoded animation frame with the prior C++ output."""
import json
import math
import numpy as np
import cv2
from PIL import Image
from recheck_20261027 import ROOT,OUT

def main():
    checks=[]
    old=ROOT/'results/historical_robot_motion_002/robot_teaching_motion.gif'
    new=OUT/'robot/robot_teaching_motion.gif'
    with Image.open(old) as a,Image.open(new) as b:
        different=0
        for i in range(min(a.n_frames,b.n_frames)):
            a.seek(i);b.seek(i)
            different+=a.convert('RGB').tobytes()!=b.convert('RGB').tobytes() or a.info.get('duration')!=b.info.get('duration')
        checks.append(dict(kind='gif',frames=b.n_frames,different_frames=different,passed=a.n_frames==b.n_frames and different==0))
    old=ROOT/'results/historical_video_preview_001/SSAFY_TestImageSet_2x_annotated.mp4'
    new=OUT/'video/SSAFY_TestImageSet_2x_annotated.mp4'
    a,b=cv2.VideoCapture(str(old)),cv2.VideoCapture(str(new))
    count=different=maximum=0;squares=0;values=0;length_match=True
    while True:
        oka,x=a.read();okb,y=b.read()
        if oka!=okb:length_match=False;break
        if not oka:break
        if x.shape!=y.shape:raise RuntimeError('Video frame shape mismatch')
        delta=x.astype(np.int16)-y.astype(np.int16)
        peak=int(np.abs(delta).max());maximum=max(maximum,peak)
        different+=peak!=0;count+=1
        squares+=float(np.square(delta.astype(np.float64)).sum());values+=delta.size
    a.release();b.release()
    checks.append(dict(kind='mp4',frames=count,different_frames=different,max_channel_difference=maximum,rmse=math.sqrt(squares/values),length_match=length_match,passed=length_match and count==300 and different==0))
    (OUT/'decoded_media_comparison.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(checks)

if __name__=='__main__':main()
