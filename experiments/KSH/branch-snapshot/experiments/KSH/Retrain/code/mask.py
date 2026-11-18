# 마스크 : 정렬에 쓰는 영역만 남기고 나머지는 0 (PNG 크기 축소용). 결과가 원본과 같은지 검증 후 사용
import cv2,numpy as np,sys
R=[(1300,860,2520,1280),(2520,1190,3220,1530),(660,1500,1360,1850)]
def masked(g):
    m=np.zeros_like(g)
    for x0,y0,x1,y1 in R: m[y0:y1,x0:x1]=g[y0:y1,x0:x1]
    return m
if __name__=='__main__':
    for src,dst in zip(sys.argv[1::2],sys.argv[2::2]):
        g=cv2.imread(src,cv2.IMREAD_GRAYSCALE); cv2.imwrite(dst,masked(g),[cv2.IMWRITE_PNG_COMPRESSION,6])
