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


def process_image_tiled(image_path, model, device, scale, tiles, overlap_percent):
    # Load HR image
    image = pil_image.open(image_path).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    # Create LR and bicubic baseline
    lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
    bicubic.save(image_path.replace('.', f'_bicubic_x{scale}.'))

    lr_y, _ = preprocess(lr, device)
    hr_y, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    # Divide into tiles
    _, _, H, W = lr_y.shape
    tile_h = math.ceil(H / tiles)
    tile_w = math.ceil(W / tiles)
    overlap_h = int(tile_h * overlap_percent / 100)
    overlap_w = int(tile_w * overlap_percent / 100)

    preds_full = torch.zeros((1, 1, H * scale, W * scale), device=device)
    total_infer_time = 0.0

    for i in range(tiles):
        for j in range(tiles):
            y0 = max(i * tile_h - overlap_h, 0)
            y1 = min((i + 1) * tile_h + overlap_h, H)
            x0 = max(j * tile_w - overlap_w, 0)
            x1 = min((j + 1) * tile_w + overlap_w, W)

            tile = lr_y[:, :, y0:y1, x0:x1]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                pred = model(tile).clamp(0.0, 1.0)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            infer_time = (end - start) * 1000
            total_infer_time += infer_time

            y0_out = y0 * scale
            y1_out = y1 * scale
            x0_out = x0 * scale
            x1_out = x1 * scale
            preds_full[:, :, y0_out:y1_out, x0_out:x1_out] = pred

    avg_infer_time = total_infer_time / (tiles * tiles)
    print(f'Average inference time per tile: {avg_infer_time:.2f} ms')
    print(f'Total inference time (all tiles): {total_infer_time:.2f} ms')

    psnr = calc_psnr(hr_y, preds_full)
    print(f'PSNR: {psnr:.2f} dB')

    preds = preds_full.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_path.replace('.', f'_fsrcnn_tile{tiles}_ov{overlap_percent}_x{scale}.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--tiles', type=int, default=1, help='Number of tiles per dimension (e.g., 2 = 2x2)')
    parser.add_argument('--overlap', type=float, default=0, help='Overlap percentage between tiles')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device(args.device)
    model = load_model(args.weights_file, args.scale, device)

    total_start = time.perf_counter()
    process_image_tiled(args.image_file, model, device, args.scale, args.tiles, args.overlap)
    total_end = time.perf_counter()

    print(f'Total elapsed time (including I/O & preprocessing): {(total_end - total_start):.2f} s')


if __name__ == '__main__':
    main()
