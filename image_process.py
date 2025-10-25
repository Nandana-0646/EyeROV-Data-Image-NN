# image_process.py
import cv2
import numpy as np
from skimage import util
from skimage.filters import median, unsharp_mask
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle
from pathlib import Path

out_dir = Path("oculus_output")
out_dir.mkdir(exist_ok=True)

img_path = Path("oculus.jpg")
if not img_path.exists():
    raise FileNotFoundError("oculus.jpg not found in this folder.")

# load
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# original
cv2.imwrite(str(out_dir / "original.png"), img_bgr)

# histogram equalization (grayscale -> color)
eq = cv2.equalizeHist(img_gray)
cv2.imwrite(str(out_dir / "hist_equal_gray.png"), cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(img_gray)
cv2.imwrite(str(out_dir / "clahe_gray.png"), cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))

# bilateral (edge-preserving blur)
bilat = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
cv2.imwrite(str(out_dir / "bilateral.png"), bilat)

# nl means color denoise
nlm = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
cv2.imwrite(str(out_dir / "nlmeans_color.png"), nlm)

# TV denoise (skimage)
img_float = util.img_as_float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
try:
    tv = denoise_tv_chambolle(img_float, weight=0.1, channel_axis=2)
except TypeError:
    tv = denoise_tv_chambolle(img_float, weight=0.1, multichannel=True)
tv_uint8 = (np.clip(tv,0,1)*255).astype(np.uint8)
# convert back to BGR for saving
tv_bgr = cv2.cvtColor(tv_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(out_dir / "tv_denoise.png"), tv_bgr)

# median filter per channel
med = np.zeros_like(img_bgr)
for c in range(3):
    med[:,:,c] = median(img_bgr[:,:,c], disk(3))
cv2.imwrite(str(out_dir / "median.png"), med)

# unsharp (sharpen)
sharp = unsharp_mask(img_float, radius=1.0, amount=1.0)
sharp_uint8 = (np.clip(sharp,0,1)*255).astype(np.uint8)
cv2.imwrite(str(out_dir / "unsharp.png"), cv2.cvtColor(sharp_uint8, cv2.COLOR_RGB2BGR))

print("Saved processed images to folder 'oculus_output'")
