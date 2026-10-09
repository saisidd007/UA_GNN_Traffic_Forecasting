from PIL import Image
import sys

IN = r"results/analysis_50epoch_train_val_loss.png"
OUT = r"results/analysis_50epoch_train_val_loss.eps"

try:
    img = Image.open(IN)
    # Ensure image is in RGB (EPS via Pillow expects RGB or L mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(OUT, format="EPS")
    print(f"Saved: {OUT}")
except Exception as e:
    print("Error:", e)
    sys.exit(2)
