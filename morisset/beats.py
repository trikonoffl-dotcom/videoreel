import numpy as np,json,sys
S=sys.argv[1]
x=np.fromfile(f"{S}/a.raw",dtype=np.float32); sr=22050
n=1024;h=512
w=np.hanning(n); fr=[np.abs(np.fft.rfft(x[i:i+n]*w)) for i in range(0,len(x)-n,h)]
fr=np.log1p(10*np.array(fr)); of=np.maximum(0,np.diff(fr,axis=0)).sum(1); of=(of-of.mean())/of.std()
fps=sr/h
best=None
for bpm in np.arange(70,170,0.5):
    p=60/bpm*fps; idx=(np.arange(0,len(of)-1,p)).astype(int)
    for ph in range(0,int(p),2):
        sc=of[np.clip(idx+ph,0,len(of)-1)].mean()
        if best is None or sc>best[0]: best=(sc,bpm,ph)
sc,bpm,ph=best; p=60/bpm*fps
beats=[(ph+k*p)/fps for k in range(int((len(of)-ph)/p))]
print(bpm,ph/fps,len(beats)); print([round(b,2) for b in beats[:12]])
json.dump({"bpm":bpm,"beats":beats},open("beats.json","w"))
