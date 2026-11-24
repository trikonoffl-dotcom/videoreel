import sys,numpy as np,os
from PIL import Image
from rembg import new_session, remove
S=sys.argv[1]; os.makedirs(f"{S}/mk",exist_ok=True)
s=new_session("u2net")
for i in range(1,722):
    im=Image.open(f"{S}/fr/{i:04d}.jpg").resize((540,960),Image.LANCZOS)
    remove(im,session=s,only_mask=True).save(f"{S}/mk/{i:04d}.png")
print("done")
