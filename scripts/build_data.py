"""Build the explorer's images and metrics from the Major-Project results (no model run here).

- Nine cases with the cloudy input and ground truth (sample_images/), the model's output (results/<set>/predictions)
  and its uncertainty map. The uncertainty PNGs were saved with matplotlib's magma colormap scaled to each image's
  99.5th percentile; they are mapped back to an 8-bit relative scale here so the browser can threshold them.
- Every test image's output and uncertainty at 384 px for the scatter, plus per-image metrics from the CSVs.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import colormaps

SRC = Path(sys.argv[1]); OUT = Path(__file__).resolve().parents[1] / "docs"
LUT = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)

def unmagma(path):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32).reshape(-1, 3)
    idx = np.empty(len(a), np.uint8)
    for s in range(0, len(a), 65536):
        d = ((a[s:s + 65536, None, :] - LUT[None]) ** 2).sum(-1)
        idx[s:s + 65536] = d.argmin(1)
    side = int(np.sqrt(len(a)))
    return Image.fromarray(idx.reshape(side, side), "L")

metrics = {}
for ds in ("RICE1", "RICE2"):
    with open(SRC / "results" / ds / "per_image_metrics.csv") as f:
        for r in csv.DictReader(f):
            m = {"set": ds, "name": r["name"], "psnr": round(float(r["psnr"]), 2), "ssim": round(float(r["ssim"]), 4),
                 "psnrIn": round(float(r["psnr_input"]), 2), "ssimIn": round(float(r["ssim_input"]), 4),
                 "corr": round(float(r["unc_err_corr"]), 3)}
            if r.get("cloud_fraction"):
                num = lambda v: None if not np.isfinite(float(v)) else round(float(v), 2)
                m.update(cloud=round(float(r["cloud_fraction"]), 3), psnrCloud=num(r["psnr_cloud_region"]), psnrClear=num(r["psnr_clear_region"]))
            metrics[f"{ds}-{r['name']}"] = m

cases = []
for p in sorted((SRC / "sample_images").glob("*.png")):
    ds, rest = p.stem.split("_", 1)
    name = re.search(r"_(\d+)$", p.stem).group(1)
    label = re.sub(r"_\d+$", "", rest).replace("_", " ")
    key = f"{ds}-{name}"; slug = key.lower()
    Image.open(p).convert("RGB").save(OUT / "img/cases" / f"{slug}-cloudy.webp", quality=92)
    Image.open(SRC / "sample_images/ground_truth" / f"{p.stem}_ground_truth.png").convert("RGB").save(OUT / "img/cases" / f"{slug}-truth.webp", quality=92)
    Image.open(SRC / "results" / ds / "predictions" / f"{name}.png").convert("RGB").save(OUT / "img/cases" / f"{slug}-output.webp", quality=92)
    unmagma(SRC / "results" / ds / "uncertainty" / f"{name}.png").save(OUT / "img/cases" / f"{slug}-unc.png", optimize=True)
    cases.append({"key": key, "slug": slug, "label": label[0].upper() + label[1:], **metrics[key]})

test = []
for key, m in metrics.items():
    ds, name = key.split("-")
    pred = SRC / "results" / ds / "predictions" / f"{name}.png"; unc = SRC / "results" / ds / "uncertainty" / f"{name}.png"
    if not pred.exists(): continue
    slug = key.lower()
    Image.open(pred).convert("RGB").resize((384, 384), Image.LANCZOS).save(OUT / "img/test" / f"{slug}.webp", quality=80)
    m = dict(m, slug=slug, hasUnc=unc.exists())
    if unc.exists():
        Image.open(unc).convert("RGB").resize((384, 384), Image.LANCZOS).save(OUT / "img/test" / f"{slug}-unc.webp", quality=80)
    test.append(m)

summary = {}
for ds in ("RICE1", "RICE2"):
    txt = (SRC / "results" / ds / "results_summary.txt").read_text()
    g = lambda pat: float(re.search(pat, txt).group(1))
    summary[ds] = {"n": sum(1 for t in test if t["set"] == ds), "psnr": g(r"Average PSNR : ([\d.]+)"), "ssim": g(r"Average SSIM : ([\d.]+)"),
                   "psnrIn": g(r"Cloudy input : PSNR ([\d.]+)"), "ssimIn": g(r"SSIM ([\d.]+)\s+\(gain"), "ms": g(r"Median inference time: ([\d.]+)"),
                   "corr": g(r"Uncertainty-error correlation: ([-\d.]+)")}
    if "inside cloud mask" in txt:
        summary[ds].update(psnrCloud=g(r"PSNR inside cloud mask : ([\d.]+)"), psnrClear=g(r"PSNR outside cloud mask: ([\d.]+)"))
(OUT / "data/explorer.json").write_text(json.dumps({"cases": cases, "test": test, "summary": summary}, separators=(",", ":"), allow_nan=False))
print(len(cases), "cases", len(test), "test images", summary)
