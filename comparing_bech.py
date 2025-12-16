# ================================
# STEP 1: EXPORT FSRCNN TO ONNX
# ================================
import torch
from models import FSRCNN


def export_fsrcnn_to_onnx(weights_path, scale, onnx_path):
    device = torch.device("cpu")
    model = FSRCNN(scale_factor=scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_path, map_location=device).items():
        if n in state_dict:
            state_dict[n].copy_(p)
        else:
            raise KeyError(f"Unexpected key in weights: {n}")

    model.eval()

    # Dummy input (dynamic spatial size)
    dummy = torch.randn(1, 1, 64, 64, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {2: "h", 3: "w"},
            "output": {2: "oh", 3: "ow"}
        }
    )

    print(f"[OK] Exported ONNX → {onnx_path}")


# ================================
# STEP 2: ONNX RUNTIME INFERENCE
# ================================
import onnxruntime as ort
import numpy as np
import PIL.Image as pil_image
import time
import math
from utils import preprocess, convert_ycbcr_to_rgb
from utils import calc_psnr as calc_psnr_torch


def process_image_tiled_onnx(
    image_path,
    ref_path,
    onnx_path,
    scale,
    tiles,
    compute_psnr=True,
    warmup_iters=3,
    batch_tiles=False
):
    # -------- ONNX Runtime session --------
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.intra_op_num_threads = 8
    sess_opt.inter_op_num_threads = 1

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opt,
        providers=["CPUExecutionProvider"]
    )

    # -------- Load LR image --------
    lr_img = pil_image.open(image_path).convert("RGB")
    lr_w, lr_h = lr_img.width, lr_img.height

    lr_y, lr_ycbcr = preprocess(lr_img, device="cpu")
    lr_y_np = lr_y.numpy()

    HR_H, HR_W = lr_h * scale, lr_w * scale
    preds_full = np.zeros((1, 1, HR_H, HR_W), dtype=np.float32)

    # -------- Reference image (PSNR) --------
    if compute_psnr:
        reference = pil_image.open(ref_path).convert("RGB")
        reference = reference.resize((HR_W, HR_H), pil_image.BICUBIC)
        reference_y, _ = preprocess(reference, device="cpu")
        reference_y_np = reference_y.numpy()

    _, _, H, W = lr_y_np.shape
    tile_h = math.ceil(H / tiles)
    tile_w = math.ceil(W / tiles)

    # -------- Warm-up --------
    warm_tile = lr_y_np[:, :, :max(tile_h, 1), :max(tile_w, 1)]
    for _ in range(warmup_iters):
        session.run(None, {"input": warm_tile})

    # -------- Inference timing --------
    total_time = 0.0

    if not batch_tiles:
        for i in range(tiles):
            for j in range(tiles):
                y0, y1 = i * tile_h, min((i + 1) * tile_h, H)
                x0, x1 = j * tile_w, min((j + 1) * tile_w, W)

                tile = lr_y_np[:, :, y0:y1, x0:x1]
                if tile.shape[2] == 0 or tile.shape[3] == 0:
                    continue

                t0 = time.perf_counter()
                pred = session.run(None, {"input": tile})[0]
                t1 = time.perf_counter()

                total_time += (t1 - t0) * 1000
                preds_full[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = np.clip(pred, 0.0, 1.0)
    else:
        tiles_np = []
        coords = []
        for i in range(tiles):
            for j in range(tiles):
                y0, y1 = i * tile_h, min((i + 1) * tile_h, H)
                x0, x1 = j * tile_w, min((j + 1) * tile_w, W)

                tile = lr_y_np[:, :, y0:y1, x0:x1]
                if tile.shape[2] == 0 or tile.shape[3] == 0:
                    continue

                tiles_np.append(tile)
                coords.append((y0, y1, x0, x1))

        batch = np.concatenate(tiles_np, axis=0)
        t0 = time.perf_counter()
        preds = session.run(None, {"input": batch})[0]
        t1 = time.perf_counter()
        total_time = (t1 - t0) * 1000

        for idx, (y0, y1, x0, x1) in enumerate(coords):
            preds_full[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = np.clip(
                preds[idx:idx + 1], 0.0, 1.0
            )

    print(f"[INFO] Avg inference/tile: {total_time / (tiles * tiles):.2f} ms")
    print(f"[INFO] Total inference (model only): {total_time:.2f} ms")

    # -------- PSNR (after timing) --------
    if compute_psnr:
        import torch
        reference_y_t = torch.from_numpy(reference_y_np)
        preds_full_t = torch.from_numpy(preds_full)
        psnr = calc_psnr_torch(reference_y_t, preds_full_t)
        print(f"[INFO] PSNR: {psnr.item():.2f} dB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["export", "infer"], required=True)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--image", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-psnr", action="store_true")

    args = parser.parse_args()

    if args.mode == "export":
        export_fsrcnn_to_onnx(args.weights, args.scale, args.onnx)
    else:
        process_image_tiled_onnx(
            image_path=args.image,
            ref_path=args.reference,
            onnx_path=args.onnx,
            scale=args.scale,
            tiles=args.tiles,
            compute_psnr=not args.no_psnr,
            warmup_iters=args.warmup
        )