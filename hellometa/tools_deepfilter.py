"""Neural speech enhancement with DeepFilterNet3 (48 kHz).
Run with a Python env that has `deepfilternet` installed:
  python tools_deepfilter.py <in.wav> <out.wav> <atten_limit_db|none> <DeepFilterNet3 model dir>"""
import sys, subprocess, numpy as np, torch
from df.enhance import enhance, init_df
src, out, lim = sys.argv[1], sys.argv[2], sys.argv[3]
model, st, _ = init_df(model_base_dir=sys.argv[4])
raw = subprocess.run(['ffmpeg','-v','error','-i',src,'-ac','1','-ar','48000','-f','f32le','-'],capture_output=True,check=True).stdout
x = torch.from_numpy(np.frombuffer(raw,np.float32).copy())[None]
y = enhance(model, st, x, atten_lim_db=None if lim=='none' else float(lim))
y = y[0].numpy().astype(np.float32)
subprocess.run(['ffmpeg','-v','error','-y','-f','f32le','-ar','48000','-ac','1','-i','-','-c:a','pcm_s24le',out],input=y.tobytes(),check=True)
print('done',out,len(y)/48000)
