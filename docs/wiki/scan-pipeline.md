# Scan Pipeline

> The scan pipeline is a five-stage sequential process that converts raw session uploads (video, stills, depth, audio) into structured product signals for confidence scoring.

## Overview

The pipeline runs inside `DraftAssembler.run()` (in `draft.py`) as a single `async` method called from the background task created by `/capture/finalize`. It has six stages: frame selection, background removal, barcode extraction, ASR transcription, VLM extraction, and dimension estimation. Every stage is written to fail silently: exceptions are caught internally and the stage contributes a zero or None to `assemble_draft()` rather than aborting the pipeline. The stages produce collectively: a UPC string, a `VLMResult` dataclass, physical dimension tuple, a narration transcript, and a list of usable frame paths.

## How It Works

### Stage 1: Frame Extraction (`pipeline/frames.py`)

`frames_from_video()` calls `ffmpeg` via subprocess at 3 fps to extract JPEG frames from the rotation video into the session's `frames/` subdirectory. Frames are then mapped to 360-degree sectors (12 sectors at 30° each) based on their index position in the full extracted set. Within each sector, `laplacian_variance()` scores each frame using a pure-numpy Laplacian convolution (no OpenCV); frames below a variance threshold of `50.0` are discarded as too blurry. Only the sharpest surviving frame per sector is kept, yielding at most 12 frames. Still images uploaded via `/capture/still` are appended after deduplication. The combined frame list feeds all subsequent stages.

### Stage 2: Barcode Extraction (`pipeline/barcode.py`)

`extract_upc()` opens an image with Pillow and passes it to `pyzbar.decode()`. It recognises six barcode types: EAN-13, UPC-A, UPC-E, EAN-8, CODE-128, and CODE-39. The assembler checks the dedicated `upc` still angle first (if the operator shot a close-up of the barcode), then falls back to scanning the first five pipeline frames. Returns the decoded string or None. This is the only stage that is purely synchronous.

### Stage 3: Narration Transcription (`draft.py: DraftAssembler._transcribe`)

If the session has a `narration_path`, the assembler POSTs the raw audio bytes to `http://localhost:8004/transcribe` (the faster-whisper `fw-server`) with a 10-second timeout. The response JSON's `text` field becomes the narration string passed to the VLM. On any failure — network error, non-200 status, empty text — `None` is returned silently and the VLM receives no narration context.

### Stage 4: VLM Extraction (`pipeline/vlm.py`)

`VLMClient.extract_product_info()` assembles an OpenAI-compatible multimodal chat request: the system prompt (`EXTRACT_PROMPT`) instructs the model to return a strict JSON object with brand, flavor, category, puff_count, nicotine_mg, ocr_text, warnings, and a confidence float. Up to four images are base64-encoded and included as `image_url` content parts. If narration is present it is appended as a final text part. The request goes to `http://localhost:8001/v1/chat/completions` (the Lemonade NPU inference server running a local VLM) with `temperature=0.1` and a 15-second timeout. The response text is stripped of markdown fences before JSON parsing. On any failure, `VLMResult(vlm_status="unavailable")` is returned and the VLM signal score becomes 0.0. The confidence field in `VLMResult` is the model's self-reported confidence (0–1), used directly as the `vlm` signal score.

### Stage 5: Dimension Estimation (`pipeline/dimensions.py`)

`depth_to_dimensions()` takes a 2D numpy array (the ARKit LiDAR depth grid) and a scan distance (default 350 mm). Using iPhone 15 Pro Max horizontal FOV of 69°, it computes the physical frame size at that distance, identifies foreground pixels as those with depth less than 95% of the scan distance, and measures the bounding-box span of foreground rows and columns. Width and height are derived from the fraction of frame pixels occupied. Depth estimate uses the 5th percentile of the raw depth grid (the closest surface, approximately the front face of the product). Returns a `(width_mm, height_mm, depth_mm)` tuple or None if the grid is empty or non-2D. The assembler reads depth from `.npy` (numpy binary) or falls back to `json.loads`.

### Stage 1.5: Background Removal (`pipeline/background.py`)

`remove_background()` wraps `rembg` to strip backgrounds from extracted frames, writing results to a `bg/` subdirectory under `frame_out_dir`. The background-removed frame paths (`bg_frame_paths`) are then passed to VLM extraction (Stage 4) in place of the original frames. If `rembg` is unavailable or raises for an individual image, the function falls back to `shutil.copy2` (passthrough) so the pipeline continues with the original frame. If the entire background removal stage fails (e.g. out-of-disk), the assembler falls back to the original `frame_paths`.

## Key Decisions

- **Sector-based frame selection rather than uniform sampling**: The rotation video covers a full product 360°. Selecting the sharpest frame per 30° sector guarantees coverage of all faces while discarding motion blur. A naïve top-N-by-sharpness approach would cluster frames around whichever angle the operator held longest; the sector approach distributes coverage evenly.

- **Barcode checked against UPC still first**: The dedicated `upc` still angle is a close-up specifically for barcode scanning. Checking it before falling back to video frames avoids false negatives caused by motion blur or distance in the rotation video. The 5-frame fallback limit prevents burning VLM-image quota on barcode attempts.

- **VLM limited to four images**: The `content.append` loop in `VLMClient.extract_product_info` slices to `image_paths[:4]`. VLMs have context window limits and degrade on excess images; four covers front, back, and two sides, which is sufficient for brand/flavor/OCR extraction without inflating token cost.

- **Pure-numpy Laplacian, no OpenCV**: The blur scoring in `frames.py` implements the Laplacian operator using `numpy.lib.stride_tricks.sliding_window_view` and a matrix multiply rather than `cv2.Laplacian`. This avoids an OpenCV dependency that would add significant binary weight on the ARM/NPU target.

## Gotchas

- **Background removal is applied to VLM frames only**: The background removal stage (1.5) processes frames before they reach VLM extraction (Stage 4). Barcode extraction (Stage 2) continues to use the original frames since barcode detection works best on unmodified images. Background-stripped images improve VLM extraction accuracy by eliminating shelf/store clutter from the product image.

- **Depth JSON vs numpy mismatch**: The `_run_pipeline` coroutine in `capture.py` saves the depth upload always as `depth_<frame_id>.json`, but `DraftAssembler.run()` checks the file extension: `.npy` → `np.load`, otherwise → `json.loads`. If the client sends a numpy binary (which is more efficient for large depth grids), it must name the file with a `.npy` extension for the correct branch to execute; there is no content-sniffing.

- **ffmpeg is a hard runtime dependency**: `extract_frames_from_video` calls `subprocess.run(["ffmpeg", ...], check=True)`. If `ffmpeg` is not on `PATH`, the stage raises `FileNotFoundError`, which is caught by the assembler's `except Exception` block, leaving `frame_paths` empty. The pipeline then proceeds with only still images (if any), and both VLM quality and embedding coverage degrade silently.

## Related

- [[architecture]] — how the pipeline fits into the overall request flow
- [[confidence-scoring]] — how stage outputs are translated into the final score
