import numpy as np, wave, sys
S=sys.argv[1]; sr=44100; dur=float(sys.argv[2]) if len(sys.argv)>2 else 30.2; bpm=100; beat=60/bpm; n=int(sr*dur)
rng=np.random.default_rng(7); out=np.zeros((n,2),np.float32)
def lp(x,fc):
    a=np.exp(-2*np.pi*fc/sr); y=np.zeros_like(x); s=0.0
    for i in range(len(x)): s=(1-a)*x[i]+a*s; y[i]=s
    return y
def add(t,x,g=1.0,pan=0.0):
    i=int(t*sr)
    if i>=n: return
    x=x[:n-i]; out[i:i+len(x),0]+=x*g*(1-pan)/1; out[i:i+len(x),1]+=x*g*(1+pan)/1
def mid(m): return 440*2**((m-69)/12)
def tone(f,d,kind="saw",att=.01,rel=.2,det=0.0):
    t=np.arange(int(d*sr))/sr
    w=lambda ff: (2*((t*ff)%1)-1) if kind=="saw" else np.sin(2*np.pi*ff*t)
    x=w(f)+ (w(f*(1+det))+w(f/(1+det)) if det else 0)
    env=np.minimum(1,t/att)*np.minimum(1,np.maximum(0,(d-t)/rel))
    return x*env
# drums
tk=np.arange(int(.4*sr))/sr
kick=np.sin(2*np.pi*(46*tk+ 70*0.05*(1-np.exp(-tk/0.05))))*np.exp(-tk/0.16)
th=np.arange(int(.05*sr))/sr; hat=np.diff(rng.standard_normal(len(th)+1))*np.exp(-th/0.011)
tc=np.arange(int(.25*sr))/sr; snare=lp(rng.standard_normal(len(tc)),6000)*np.exp(-tc/0.07)+np.sin(2*np.pi*190*tc)*np.exp(-tc/0.05)*.5
prog=[(45,[57,60,64]),(41,[53,57,60]),(48,[55,60,64]),(43,[55,59,62])]   # Am F C G
bars=int(dur/(beat*4))+1
for b in range(bars):
    t0=b*beat*4; root,ch=prog[b%4]
    # pad (soft, wide)
    for k,m in enumerate(ch):
        pad=lp(tone(mid(m),beat*4.05,"saw",att=.5,rel=.9,det=.006),1400)
        add(t0,pad,.07,pan=(-.4,0,.4)[k])
    # bass
    for e in range(8):
        tb=t0+e*beat/2
        if e in (0,3,4,7): add(tb,lp(tone(mid(root),beat*.45,"sin",att=.005,rel=.12),300)*1.0,.34)
    # plucks (16th arp)
    arp=[ch[0],ch[1],ch[2],ch[1]+12,ch[2],ch[1],ch[0]+12,ch[1]]
    for e in range(8):
        tp=t0+e*beat/2
        p=lp(tone(mid(arp[e]+12),.35,"saw",att=.002,rel=.3),2600)*np.exp(-np.arange(int(.35*sr))/sr/0.12)
        add(tp,p,.075,pan=-.3 if e%2 else .3); add(tp+beat*.75/1,p*.5,.04,pan=.4 if e%2 else -.4)   # echo
    for q in range(4):
        tq=t0+q*beat
        add(tq,kick,.55)
        if q in (1,3): add(tq,snare,.22)
        add(tq+beat/2,hat,.07,pan=.2); 
        if q%2==0: add(tq+beat*.75,hat,.03,pan=-.2)
out=np.tanh(out*1.2)
fi=np.minimum(1,np.arange(n)/(0.6*sr)); fo=np.minimum(1,np.maximum(0,(n-np.arange(n))/(2.2*sr)))
out*= (fi*fo)[:,None]; out*=0.9/np.abs(out).max()
w=wave.open(f"{S}/music2.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((out*32767).astype('<i2').tobytes()); w.close()
