from PIL import Image
import sys

IN = r"results/analysis_50epoch_train_val_loss.png"
OUT_CROPPED_PNG = r"results/analysis_50epoch_train_val_loss_no_title.png"
OUT_CROPPED_EPS = r"results/analysis_50epoch_train_val_loss_no_title.eps"

try:
    img = Image.open(IN)
    width, height = img.size
    
    # Crop top ~60 pixels to remove title
    crop_top = 60
    cropped = img.crop((0, crop_top, width, height))
    cropped.save(OUT_CROPPED_PNG)
    print(f"Saved cropped PNG: {OUT_CROPPED_PNG}")
    
    # Convert cropped PNG to EPS
    if cropped.mode not in ("RGB", "L"):
        cropped = cropped.convert("RGB")
    cropped.save(OUT_CROPPED_EPS, format="EPS")
    print(f"Saved cropped EPS: {OUT_CROPPED_EPS}")
    
except Exception as e:
    print("Error:", e)
    sys.exit(2)
