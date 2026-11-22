# Cloud removal explorer

**Live: [mehtaaarush.github.io/cloud-removal-explorer](https://mehtaaarush.github.io/cloud-removal-explorer/)**

An interactive explorer for my B.Tech major project: an uncertainty-aware generative model that removes cloud, haze and shadow from satellite images. It shows the model's recorded outputs on the held-out RICE test set.

## What it does

- Drag a divider across nine held-out scenes to compare the cloudy input with the model's output, the ground truth, its per-pixel uncertainty or its error.
- Mark the least certain pixels and see, live, how well the model's uncertainty predicts its own error in that scene.
- Browse all 247 test images on a scatter of input against output PSNR, with each image's output and uncertainty map.
- Summary PSNR, SSIM and GPU time for RICE1 (thin cloud and haze) and RICE2 (thick cloud and shadow).

| Test set | PSNR in | PSNR out | SSIM out |
|---|---:|---:|---:|
| RICE1 | 19.24 dB | 35.55 dB | 0.951 |
| RICE2 | 22.23 dB | 34.29 dB | 0.878 |

## The model

A SwinIR transformer with Fourier Units as the global branch, a TransGAN generator as the local branch, wavelet fusion of the two, and an uncertainty head trained with a Laplace likelihood that also weights the adversarial loss. Training ran on a GPU cluster; the weights are not served here, so the page shows the model's saved test outputs rather than running the network.

## Build the data

`scripts/build_data.py` reads the project's results folder (sample images, per-image predictions, uncertainty maps and metric CSVs) and writes `docs/data/explorer.json` and the images under `docs/img/`.

Imagery: the [RICE dataset](https://github.com/BUPTLdy/RICE_DATASET).
