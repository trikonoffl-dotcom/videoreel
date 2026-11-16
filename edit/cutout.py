import cv2, numpy as np, sys
src, dst, tol = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
if img.shape[2]==4: img=img[:,:,:3]
h,w=img.shape[:2]
print(src, img.shape, img[2,2], img[h//2, 3])
mask=np.zeros((h+2,w+2),np.uint8)
flags=4|(255<<8)|cv2.FLOODFILL_MASK_ONLY|cv2.FLOODFILL_FIXED_RANGE
for p in [(0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]:
    cv2.floodFill(img,mask,p,0,(tol,)*3,(tol,)*3,flags)
bg=mask[1:-1,1:-1]>0
a=np.where(bg,0,255).astype(np.uint8)
a=cv2.erode(a,np.ones((3,3),np.uint8)); a=cv2.GaussianBlur(a,(3,3),0)
out=np.dstack([img,a])
ys,xs=np.where(a>0); out=out[ys.min():ys.max()+1, xs.min():xs.max()+1]
# square pad
hh,ww=out.shape[:2]; m=max(hh,ww); sq=np.zeros((m,m,4),np.uint8); sq[(m-hh)//2:(m-hh)//2+hh,(m-ww)//2:(m-ww)//2+ww]=out
cv2.imwrite(dst,sq); print(dst, sq.shape)
