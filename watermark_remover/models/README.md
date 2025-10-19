# Models

Place third-party weights under `~/.wmr/models/` by default.

- **LaMa (ONNX)**: export or download a LaMa inpainting ONNX that accepts `(image, mask)` tensors and produces an inpainted image in `[0,1]`. Rename the file to `lama.onnx`.
- **RAFT (optional)**: download RAFT weights (TorchScript or standard checkpoint) and store them as `raft.pth` to enable flow-guided blending.

You can override the location in future releases via CLI flags or environment variables.
