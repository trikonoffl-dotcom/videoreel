import numpy as np, json, sys, wave
S=sys.argv[1]; sr=44100; dur=30.1
b0=json.load(open("beats.json"))["beats"][0]; beat=60/76.0
n=int(sr*dur); out=np.zeros(n,np.float32); rng=np.random.default_rng(1)
def add(t,s,g=1):
    i=int(t*sr)
    if i<0 or i>=n: return
    s=s[:n-i]; out[i:i+len(s)]+=s*g
tk=np.arange(int(.35*sr))/sr
kick=np.sin(2*np.pi*(48*tk+ (110-48)*0.045*(1-np.exp(-tk/0.045))))*np.exp(-tk/0.14)*0.95
th=np.arange(int(.06*sr))/sr
hat=np.diff(rng.standard_normal(len(th)+1)).astype(np.float32)*np.exp(-th/0.012)*0.25
tc=np.arange(int(.22*sr))/sr
clap=rng.standard_normal(len(tc)).astype(np.float32)*np.exp(-tc/0.05)*0.35
bass_notes=[41.2,41.2,49.0,36.7]
tb=np.arange(int(beat*1.9*sr))/sr
k=0; t=b0
while t<dur:
    pos=k%4
    add(t,kick,1.0)
    if pos in (1,3): add(t,clap,.8)
    add(t+beat/2,hat,.9)
    if pos in (0,2): add(t+beat*0.75,kick,.55)
    f=bass_notes[(k//4)%4]; env=np.exp(-tb/0.9)*(1-np.exp(-tb/0.01))
    add(t,(np.sin(2*np.pi*f*tb)*env*0.35).astype(np.float32),1.0)
    k+=1; t+=beat
out=np.tanh(out*1.1); out*=0.9/max(1e-6,np.abs(out).max())
st=np.stack([out,out],1)
w=wave.open(f"{S}/beat.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((st*32767).astype('<i2').tobytes()); w.close()
