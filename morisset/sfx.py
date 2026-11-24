import numpy as np, wave, sys
S=sys.argv[1]; sr=44100; n=int(sr*38.4); out=np.zeros(n,np.float32); rng=np.random.default_rng(3)
def whoosh(tc, dur=0.30, g=0.55):
    m=int(dur*sr); t=np.arange(m)/sr; x=rng.standard_normal(m).astype(np.float32)
    fc=np.linspace(500,7000,m); a=np.exp(-2*np.pi*fc/sr); y=np.zeros(m,np.float32); s=0.0
    for i in range(m): s=(1-a[i])*x[i]+a[i]*s; y[i]=s            # sweeping low-pass -> rising whoosh
    y=y-np.concatenate([[0],y[:-1]])*0.6                          # tilt toward highs
    env=np.minimum(1,(t/dur)**2.2)*np.minimum(1,(dur-t)/0.04+0.0)  # swell into the cut
    y*=env; y/=np.abs(y).max()+1e-9
    i=int((tc-dur)*sr+0.05*sr)
    if i>=0: out[i:i+m]+=y*g
def hit(tc, g=0.7):
    m=int(0.28*sr); t=np.arange(m)/sr
    k=np.sin(2*np.pi*(52*t+40*0.04*(1-np.exp(-t/0.04))))*np.exp(-t/0.11)
    c=rng.standard_normal(m)*np.exp(-t/0.02)*0.25
    i=int(tc*sr); out[i:i+m]+=((k+c)*g).astype(np.float32)
cuts=[25.34+k*0.3733 for k in range(6)]
for c in cuts: whoosh(c); hit(c, 0.55)
whoosh(27.58, 0.35, 0.5); hit(27.58, 0.85)                        # exit hit back into the story
out=np.tanh(out*1.1)*0.9
st=np.stack([out,out],1); w=wave.open(f"{S}/sfx.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((st*32767).astype('<i2').tobytes()); w.close()
