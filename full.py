import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import time
import math
from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


def load_model(weights_path, scale, device):
    model = FSRCNN(scale_factor=scale).to(device)
    state_dict = model.state_dict()

    for n, p in torch.load(weights_path, map_location=device).items():
        if n in state_dict:
            state_dict[n].copy_(p)
        else:
            raise KeyError(f"Unexpected key in weights: {n}")

    model.eval()
    return model


def process_image_tiled(image_path, ref_path, model, device, scale, tiles):

    # --------------------------------------------
    # Load LR image
    # --------------------------------------------
    lr_img = pil_image.open(image_path).convert("RGB")
    lr_w, lr_h = lr_img.width, lr_img.height

    # Convert LR to YCbCr
    lr_y, lr_ycbcr = preprocess(lr_img, device)

    # ---- FIX: accept numpy or torch ----
    if isinstance(lr_ycbcr, torch.Tensor):
        lr_ycbcr_np = lr_ycbcr.cpu().numpy()
    else:
        lr_ycbcr_np = lr_ycbcr

    # HR output resolution
    HR_H = lr_h * scale
    HR_W = lr_w * scale

    preds_full = torch.zeros((1, 1, HR_H, HR_W), device=device)

    # --------------------------------------------
    # Load reference HR image
    # --------------------------------------------
    reference = pil_image.open(ref_path).convert("RGB")
    reference = reference.resize((HR_W, HR_H), pil_image.BICUBIC)
    reference_y, _ = preprocess(reference, device)

    # --------------------------------------------
    # Tiled inference
    # --------------------------------------------
    _, _, H, W = lr_y.shape
    tile_h = math.ceil(H / tiles)
    tile_w = math.ceil(W / tiles)

    total_infer_time = 0.0

    for i in range(tiles):
        for j in range(tiles):
            y0, y1 = i * tile_h, min((i + 1) * tile_h, H)
            x0, x1 = j * tile_w, min((j + 1) * tile_w, W)

            tile = lr_y[:, :, y0:y1, x0:x1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                pred = model(tile).clamp(0.0, 1.0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_infer_time += (t1 - t0) * 1000

            preds_full[
                :, :,
                y0 * scale : y1 * scale,
                x0 * scale : x1 * scale
            ] = pred

    print(f"[INFO] Avg inference/tile: {total_infer_time/(tiles*tiles):.2f} ms")
    print(f"[INFO] Total inference: {total_infer_time:.2f} ms")

    # --------------------------------------------
    # PSNR (Y channel)
    # --------------------------------------------
    psnr = calc_psnr(reference_y, preds_full)
    print(f"[INFO] PSNR vs reference: {psnr:.2f} dB")

    # --------------------------------------------
    # COLOR RESTORATION
    # --------------------------------------------

    # Predicted Y
    pred_y = preds_full.mul(255.0).cpu().numpy().squeeze()

    # Upscale Cb & Cr by bicubic
    cb = pil_image.fromarray(lr_ycbcr_np[..., 1]).resize((HR_W, HR_H), pil_image.BICUBIC)
    cr = pil_image.fromarray(lr_ycbcr_np[..., 2]).resize((HR_W, HR_H), pil_image.BICUBIC)

    cb = np.array(cb)
    cr = np.array(cr)

    # Merge Y Cb Cr
    ycbcr = np.stack([pred_y, cb, cr], axis=2)

    # Convert to RGB
    rgb = convert_ycbcr_to_rgb(ycbcr)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Save final output
    output_img = pil_image.fromarray(rgb)
    out_path = image_path.replace('.', f'_fsrcnn_x{scale}.')
    output_img.save(out_path)

    print(f"[INFO] Saved COLOR output → {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--reference-file", type=str, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])

    args = parser.parse_args()
    cudnn.benchmark = True
    device = torch.device(args.device)

    model = load_model(args.weights_file, args.scale, device)

    start = time.perf_counter()
    process_image_tiled(args.image_file, args.reference_file,
                        model, device, args.scale, args.tiles)
    end = time.perf_counter()

    print(f"[INFO] Total time: {end - start:.2f} sec")


if __name__ == "__main__":
    main()
