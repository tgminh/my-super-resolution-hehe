import argparse
import time
import math
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class FSRCNN(nn.Module):
    def __init__(self, scale, num_channels=3):
        super(FSRCNN, self).__init__()
        # Feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 56, kernel_size=5, padding=2), nn.PReLU()
        )
        # Shrinking, mapping, and expanding layers
        self.layers = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1), nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(12, 56, kernel_size=1), nn.PReLU()
        )
        # Final deconvolution (upsampling) layer
        self.last_part = nn.ConvTranspose2d(
            56, num_channels, kernel_size=9, stride=scale, padding=3, output_padding=1
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.layers(x)
        x = self.last_part(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="FSRCNN Tiled Super-Resolution")
    parser.add_argument("--weights-file", type=str, required=True,
                        help="Path to FSRCNN weights file (state dict).")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to input high-resolution image.")
    parser.add_argument("--scale", type=int, default=2,
                        help="Upscaling factor (e.g., 2, 3, 4).")
    parser.add_argument("--tiles", type=int, default=4,
                        help="Number of tiles (4, 9, or 16).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device: 'cuda' or 'cpu'.")
    parser.add_argument("--overlap", type=float, default=0,
                        help="Overlap percentage between tiles (0-100).")
    return parser.parse_args()

def compute_psnr(img1, img2):
    """Compute PSNR (dB) between two uint8 images."""
    mse = ((img1.astype(np.float64) - img2.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    args = parse_args()
    # Select device
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

    # Load high-resolution image and downscale to low-res
    hr_image = Image.open(args.image_file).convert("RGB")
    w_hr, h_hr = hr_image.size
    w_lr = w_hr // args.scale
    h_lr = h_hr // args.scale
    lr_image = hr_image.resize((w_lr, h_lr), Image.BICUBIC)

    # Prepare HR array for PSNR comparison (crop to exact multiple of scale)
    hr_np = np.array(hr_image)
    hr_cropped = hr_np[:h_lr*args.scale, :w_lr*args.scale, :]

    # Determine tiling layout
    tiles = args.tiles
    root = int(math.sqrt(tiles))
    if root * root != tiles:
        raise ValueError("tiles must be a perfect square (4, 9, 16)")
    nx = ny = root
    tile_w = math.ceil(w_lr / nx)
    tile_h = math.ceil(h_lr / ny)
    overlap_frac = args.overlap / 100.0
    overlap_x = int(tile_w * overlap_frac)
    overlap_y = int(tile_h * overlap_frac)

    # Collect tile and extended-region coordinates
    tile_regions = []
    for i in range(ny):
        for j in range(nx):
            x0 = j * tile_w
            y0 = i * tile_h
            x1 = min(x0 + tile_w, w_lr)
            y1 = min(y0 + tile_h, h_lr)
            ex0 = max(0, x0 - overlap_x)
            ey0 = max(0, y0 - overlap_y)
            ex1 = min(w_lr, x1 + overlap_x)
            ey1 = min(h_lr, y1 + overlap_y)
            tile_regions.append((x0, y0, x1, y1, ex0, ey0, ex1, ey1))

    # Prepare output canvas for SR result
    final_sr = torch.zeros((3, h_lr*args.scale, w_lr*args.scale), dtype=torch.float32)

    # Load FSRCNN model and weights
    model = FSRCNN(args.scale, num_channels=3).to(device)
    state = torch.load(args.weights_file, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    # Inference with CUDA streams
    start_time = time.time()
    # Create CUDA streams for overlapping transfers (or dummy for CPU)
    streams = [torch.cuda.Stream(device=device) for _ in tile_regions] if device.type=="cuda" else [None]*len(tile_regions)
    results = []
    with torch.inference_mode():  # disable grad, speed up
        for idx, (x0, y0, x1, y1, ex0, ey0, ex1, ey1) in enumerate(tile_regions):
            # Crop LR tile (with overlap)
            tile_crop = lr_image.crop((ex0, ey0, ex1, ey1))
            tile_np = np.array(tile_crop).astype(np.float32) / 255.0
            tile_tensor = torch.from_numpy(tile_np.transpose(2,0,1)).unsqueeze(0)  # shape 1xCxHxW

            if device.type == "cuda":
                # Pin to CPU memory and launch async copy+inference on stream
                tile_pin = tile_tensor.pin_memory()
                stream = streams[idx]
                with torch.cuda.stream(stream):
                    tile_gpu = tile_pin.to(device, non_blocking=True)
                    sr_gpu = model(tile_gpu)  # run FSRCNN
                    results.append((sr_gpu.squeeze(0).cpu(), x0, y0, x1, y1, ex0, ey0, ex1, ey1))
            else:
                # CPU fallback
                sr = model(tile_tensor.to(device))
                results.append((sr.squeeze(0).cpu(), x0, y0, x1, y1, ex0, ey0, ex1, ey1))

        if device.type == "cuda":
            torch.cuda.synchronize()
    end_time = time.time()

    # Assemble tiles into final SR image (crop overlapping borders)
    for (sr_tile_cpu, x0, y0, x1, y1, ex0, ey0, ex1, ey1) in results:
        _, Hs, Ws = sr_tile_cpu.shape
        # Amount to remove (in high-res pixels) from each side
        left_rm   = (x0 - ex0) * args.scale
        top_rm    = (y0 - ey0) * args.scale
        right_rm  = (ex1 - x1) * args.scale
        bottom_rm = (ey1 - y1) * args.scale
        # Crop the SR tile to remove overlap margins
        sr_inner = sr_tile_cpu[:,
                                top_rm:Hs-bottom_rm,
                                left_rm:Ws-right_rm]
        # Place into final image canvas
        final_sr[:, y0*args.scale:y1*args.scale, x0*args.scale:x1*args.scale] = sr_inner

    total_time = end_time - start_time
    avg_tile_time = total_time / len(tile_regions)
    fps = 1.0 / total_time if total_time > 0 else float('inf')

    # Save the output image
    final_sr_np = (final_sr.clamp(0.0,1.0) * 255.0).byte().permute(1,2,0).numpy()
    sr_image = Image.fromarray(final_sr_np)
    sr_image.save("sr_output.png")

    # Compute PSNR against original HR
    sr_np = final_sr_np.astype(np.uint8)
    hr_eval = hr_cropped.astype(np.uint8)
    psnr_val = compute_psnr(sr_np, hr_eval)

    print(f"Output saved as 'sr_output.png'")
    print(f"Total time: {total_time:.4f} sec, FPS: {fps:.3f}")
    print(f"Average tile time: {avg_tile_time:.4f} sec, PSNR: {psnr_val:.2f} dB")
