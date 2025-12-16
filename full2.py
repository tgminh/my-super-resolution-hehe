import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import time
import math
from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
import torch.nn.functional as F
import os

def load_model(weights_path, scale, device, use_jit=False, use_fp16=True):
    model = FSRCNN(scale_factor=scale).to(device)
    state = torch.load(weights_path, map_location=device)
    # allow either raw state_dict or wrapped dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    if device.type == 'cuda' and use_fp16:
        model.half()            # use FP16 for faster inference on GPU

    if use_jit:
        # trace with a small example shape
        example = torch.randn(1, 1, 64, 64).to(device)
        if device.type == 'cuda' and use_fp16:
            example = example.half()
        with torch.no_grad():
            model = torch.jit.trace(model, example)
            model = torch.jit.optimize_for_inference(model)
        print("[INFO] model traced with TorchScript")

    return model

def process_image_tiled_batched(image_path, ref_path, model, device, scale, tiles):
    # 1. Load Input Image (Treat as Low Resolution)
    lr = pil_image.open(image_path).convert('RGB')
    
    # Calculate target HR resolution based on scale
    target_w = lr.width * scale
    target_h = lr.height * scale
    
    # 2. Upscale Chroma (CbCr) using Bicubic (Standard SR approach)
    # We resize LR -> HR dimension directly for the color channels
    bicubic = lr.resize((target_w, target_h), resample=pil_image.BICUBIC)
    # Optional: Save bicubic for visual comparison
    # bicubic.save(image_path.replace('.', f'_bicubic_x{scale}.'))

    # 3. Preprocess Input (LR) -> Tensor
    lr_y, _ = preprocess(lr, device)   # shape (1,1,H_lr,W_lr)
    
    # Preprocess Bicubic -> used later to merge CbCr channels
    _, ycbcr = preprocess(bicubic, device)

    # tile geometry (LR domain)
    _, _, H, W = lr_y.shape
    tile_h = math.ceil(H / tiles)
    tile_w = math.ceil(W / tiles)

    # collect tiles and coords; compute max tile size
    tiles_list = []
    coords = []  # (y0,y1,x0,x1)
    max_h = 0
    max_w = 0
    for i in range(tiles):
        for j in range(tiles):
            y0 = i * tile_h
            y1 = min((i + 1) * tile_h, H)
            x0 = j * tile_w
            x1 = min((j + 1) * tile_w, W)
            t = lr_y[:, :, y0:y1, x0:x1]
            tiles_list.append(t)
            coords.append((y0, y1, x0, x1))
            h_c = t.shape[2]; w_c = t.shape[3]
            if h_c > max_h: max_h = h_c
            if w_c > max_w: max_w = w_c

    # pad each tile to (max_h, max_w) so we can batch
    padded_tiles = []
    for t in tiles_list:
        _,_,h_c,w_c = t.shape
        pad_h = max_h - h_c
        pad_w = max_w - w_c
        # pad right and bottom using reflection to avoid border artifacts
        padded = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
        padded_tiles.append(padded)

    # stack to batch on device
    batch = torch.cat(padded_tiles, dim=0).to(device)   # shape (N,1,max_h,max_w)

    # if model is half, convert batch to half
    first_param = None
    try:
        first_param = next(model.parameters())
    except Exception:
        first_param = None

    if first_param is not None and first_param.dtype == torch.float16:
        batch = batch.half()

    # single model forward (batched)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(batch).clamp(0.0, 1.0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_model_ms = (t1 - t0) * 1000.0

    # Assemble outputs into full HR tensor
    # Note: Output size is Input size * scale
    preds_full = torch.zeros((1,1,H*scale,W*scale), device=device, dtype=preds.dtype)
    idx = 0
    for (y0,y1,x0,x1) in coords:
        ph = (y1 - y0) * scale
        pw = (x1 - x0) * scale
        pred_tile = preds[idx:idx+1, :, :ph, :pw]   # crop top-left valid area
        oy0, oy1 = y0*scale, y1*scale
        ox0, ox1 = x0*scale, x1*scale
        preds_full[:, :, oy0:oy1, ox0:ox1] = pred_tile
        idx += 1

    # 4. PSNR Calculation (Only if Reference is provided)
    if ref_path is not None:
        if os.path.exists(ref_path):
            ref_img = pil_image.open(ref_path).convert('RGB')
            # Ensure reference matches the output dimensions for strict PSNR calculation
            if ref_img.size != (target_w, target_h):
                ref_img = ref_img.resize((target_w, target_h), resample=pil_image.BICUBIC)
            
            ref_y, _ = preprocess(ref_img, device)
            
            # Compute PSNR (convert both to float32 for calculation)
            psnr_val = calc_psnr(ref_y.float(), preds_full.float())
            print(f'PSNR: {psnr_val:.2f} dB')
        else:
            print(f"[WARNING] Reference file not found at {ref_path}")

    print(f'Model (batched) time: {total_model_ms:.2f} ms for {len(coords)} tiles')
    print(f'Avg per-tile (batched): {total_model_ms / len(coords):.2f} ms')

    # save RGB output (merge Y with bicubic CbCr)
    preds_np = (preds_full.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)).astype(np.uint8)
    out = np.array([preds_np, ycbcr[...,1], ycbcr[...,2]]).transpose([1,2,0])
    out = np.clip(convert_ycbcr_to_rgb(out), 0.0, 255.0).astype(np.uint8)
    
    save_path = image_path.replace('.', f'_fsrcnn_f2_x{scale}.')
    pil_image.fromarray(out).save(save_path)
    print(f"Saved inference result to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True, help="Path to input (LR) image")
    parser.add_argument('--reference-file', type=str, default=None, help="Path to reference (HR) image for PSNR calculation")
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--tiles', type=int, default=1, help='Number of tiles per side (2->2x2)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu','cuda'])
    parser.add_argument('--jit', action='store_true', help='Optional TorchScript trace (off by default)')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device(args.device)
    
    # Scale is now used to initialize the model correctly
    model = load_model(args.weights_file, args.scale, device, use_jit=args.jit, use_fp16=(device.type=='cuda'))

    t0 = time.perf_counter()
    process_image_tiled_batched(args.image_file, args.reference_file, model, device, args.scale, args.tiles)
    t1 = time.perf_counter()
    print(f'Total elapsed (incl I/O): {(t1-t0):.2f} s')

if __name__ == '__main__':
    main()