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

# NOTE: ORT_ENABLE_ALL graph optimizations will be enabled when creating the session
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
    compute_psnr=True
):
    pass



if __name__ == "__main__":
    pass













# # ================================
# # STEP 1: EXPORT FSRCNN TO ONNX
# # ================================
# import torch
# from models import FSRCNN

# def export_fsrcnn_to_onnx(weights_path, scale, onnx_path):
#     device = torch.device("cpu")
#     model = FSRCNN(scale_factor=scale).to(device)

#     state_dict = model.state_dict()
#     for n, p in torch.load(weights_path, map_location=device).items():
#         if n in state_dict:
#             state_dict[n].copy_(p)
#         else:
#             raise KeyError(f"Unexpected key in weights: {n}")

#     model.eval()

#     # Dummy input (dynamic spatial size)
#     dummy = torch.randn(1, 1, 64, 64, device=device)

#     torch.onnx.export(
#         model,
#         dummy,
#         onnx_path,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=17,
#         dynamic_axes={
#             "input": {2: "h", 3: "w"},
#             "output": {2: "oh", 3: "ow"}
#         }
#     )

#     print(f"[OK] Exported ONNX → {onnx_path}")


# # ================================
# # STEP 2: ONNX RUNTIME INFERENCE
# # ================================
# import onnxruntime as ort
# import numpy as np
# import PIL.Image as pil_image
# import time
# import math
# from utils import preprocess, convert_ycbcr_to_rgb
# from utils import calc_psnr as calc_psnr_torch


# def process_image_tiled_onnx(image_path, ref_path, onnx_path, scale, tiles):

#     # -------- Load ONNX Runtime session --------
#     sess_opt = ort.SessionOptions()
#     sess_opt.intra_op_num_threads = 8   # i7-10700K = 8 cores
#     sess_opt.inter_op_num_threads = 1

#     session = ort.InferenceSession(
#         onnx_path,
#         sess_options=sess_opt,
#         providers=["CPUExecutionProvider"]
#     )

#     # -------- Load LR image --------
#     lr_img = pil_image.open(image_path).convert("RGB")
#     lr_w, lr_h = lr_img.width, lr_img.height

#     lr_y, lr_ycbcr = preprocess(lr_img, device="cpu")
#     lr_y_np = lr_y.numpy()

#     HR_H, HR_W = lr_h * scale, lr_w * scale
#     preds_full = np.zeros((1, 1, HR_H, HR_W), dtype=np.float32)

#     # -------- Reference image --------
#     reference = pil_image.open(ref_path).convert("RGB")
#     reference = reference.resize((HR_W, HR_H), pil_image.BICUBIC)
#     reference_y, _ = preprocess(reference, device="cpu")

#     _, _, H, W = lr_y_np.shape
#     tile_h = math.ceil(H / tiles)
#     tile_w = math.ceil(W / tiles)

#     total_time = 0.0

#     for i in range(tiles):
#         for j in range(tiles):
#             y0, y1 = i * tile_h, min((i + 1) * tile_h, H)
#             x0, x1 = j * tile_w, min((j + 1) * tile_w, W)

#             tile = lr_y_np[:, :, y0:y1, x0:x1]

#             t0 = time.perf_counter()
#             pred = session.run(None, {"input": tile})[0]
#             t1 = time.perf_counter()

#             total_time += (t1 - t0) * 1000

#             preds_full[:, :, y0*scale:y1*scale, x0*scale:x1*scale] = np.clip(pred, 0.0, 1.0)

#     print(f"[INFO] Avg inference/tile: {total_time/(tiles*tiles):.2f} ms")
#     print(f"[INFO] Total inference: {total_time:.2f} ms")

#     # -------- PSNR --------
#     # calc_psnr in utils.py is Torch-based → convert NumPy to Tensor here
#     import torch
#     reference_y_t = torch.from_numpy(reference_y_np)
#     preds_full_t = torch.from_numpy(preds_full)
#     psnr = calc_psnr_torch(reference_y_t, preds_full_t)
#     print(f"[INFO] PSNR: {psnr.item():.2f} dB")

#     # -------- Color restoration --------
#     pred_y = preds_full.squeeze() * 255.0
#     cb = pil_image.fromarray(lr_ycbcr[..., 1]).resize((HR_W, HR_H), pil_image.BICUBIC)
#     cr = pil_image.fromarray(lr_ycbcr[..., 2]).resize((HR_W, HR_H), pil_image.BICUBIC)

#     ycbcr = np.stack([pred_y, np.array(cb), np.array(cr)], axis=2)
#     rgb = convert_ycbcr_to_rgb(ycbcr)
#     rgb = np.clip(rgb, 0, 255).astype(np.uint8)

#     out = pil_image.fromarray(rgb)
#     out_path = image_path.replace('.', f'_fsrcnn_onnx_x{scale}.')
#     out.save(out_path)

#     print(f"[OK] Saved → {out_path}")


# # ================================
# # ARGUMENT PARSER + MAIN
# # ================================
# import argparse


# def main():
#     parser = argparse.ArgumentParser(description="FSRCNN ONNX CPU-only inference")

#     parser.add_argument("--mode", type=str, required=True,
#                         choices=["export", "infer"],
#                         help="export: export PyTorch model to ONNX | infer: run ONNX inference")

#     parser.add_argument("--weights", type=str, help="Path to FSRCNN .pth weights (for export)")
#     parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
#     parser.add_argument("--scale", type=int, default=3, help="Upscale factor (default: 3)")

#     parser.add_argument("--image", type=str, help="Low-resolution input image (for inference)")
#     parser.add_argument("--reference", type=str, help="High-resolution reference image (for PSNR)")
#     parser.add_argument("--tiles", type=int, default=1, help="Number of tiles per dimension")

#     args = parser.parse_args()

#     if args.mode == "export":
#         if args.weights is None:
#             raise ValueError("--weights is required for export mode")
#         export_fsrcnn_to_onnx(args.weights, args.scale, args.onnx)

#     elif args.mode == "infer":
#         if args.image is None or args.reference is None:
#             raise ValueError("--image and --reference are required for infer mode")

#         process_image_tiled_onnx(
#             image_path=args.image,
#             ref_path=args.reference,
#             onnx_path=args.onnx,
#             scale=args.scale,
#             tiles=args.tiles
#         )


# if __name__ == "__main__":
#     main()
