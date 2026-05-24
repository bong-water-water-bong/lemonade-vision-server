# Vision Mobile Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Vision" feature to lemonade-mobile (Flutter) that lets a cashier onboard new products by scanning them with an iPhone and look up existing products by text or voice — all routed through the Lemonade Nexus mesh instead of ngrok.

**Architecture:** One new `VisionApiClient` (Dart) talking to the vision server at port 8787, URL-derived from the user's active Lemonade server config. Two new screens: `CaptureScreen` (product onboarding wizard) and `DeduceScreen` (text/voice lookup). One server-side fix: ffmpeg audio conversion in `deduce.py` and `capture.py` to accept M4A from iOS alongside WAV.

**Tech Stack:** Flutter/Dart, Riverpod, existing `record`/`image_picker`/`http` packages, Python/FastAPI (vision server patch), Lemonade Nexus (connectivity — already deployed, no changes needed here).

---

## Repos

- **lemonade-mobile** — `https://github.com/lemonade-sdk/lemonade-mobile` — fork to `bong-water-water-bong/lemonade-mobile`, clone locally to `/home/bcloud/lemonade-mobile`
- **lemonade-vision-server** — `/home/bcloud/lemonade-vision-server` — one server-side audio fix

---

## Vision Server API Surface (reference for implementers)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/session/start` | none | Returns `{session_id, qr_png_b64}` |
| DELETE | `/session/{session_id}` | none | Tears down session |
| POST | `/capture/video` | X-Session-Token | Multipart `file` upload |
| POST | `/capture/still` | X-Session-Token | Multipart `file` + form `angle` |
| POST | `/capture/audio` | X-Session-Token | Multipart `file` (narration) |
| POST | `/capture/finalize` | X-Session-Token | Returns `{job_id}` |
| GET | `/product/draft/{job_id}` | none | Poll until `status` is `ready` or `failed` |
| POST | `/product/commit` | none | Commits draft → product DB |
| POST | `/deduce/text` | none | `{query, top_k}` → candidates |
| POST | `/deduce/audio` | none | Multipart WAV/M4A → candidates |

Valid `angle` values for `/capture/still`: `upc`, `label`, `front`, `rear`, `left`, `right`, `top`, `bottom`.

---

## Dart Types (`lib/api/types/vision_types.dart`)

```dart
class DeduceRequest {
  final String query;
  final int topK;
}

class DeduceCandidate {
  final String sku;
  final double confidence;
  final String matchReason;
  final String? brand;
  final String? flavor;
}

class DeduceResponse {
  final List<DeduceCandidate> candidates;
  final String queryUsed;
}

class SessionStartResponse {
  final String sessionId;
}

class FinalizeResponse {
  final String jobId;
}

class SignalScores {
  final double upc;
  final double vlm;
  final double embedding;
  final double dimension;
}

class DraftProduct {
  final String jobId;
  final String status;   // "processing" | "ready" | "failed"
  final String? upc;
  final String? brand;
  final String? flavor;
  final String? category;
  final int? puffCount;
  final int? nicotineMg;
  final String? ocrText;
  final SignalScores? signalScores;
}

class CommitRequest {
  final String jobId;
  final String sku;
  final String brand;
  final String flavor;
  final String category;
  final int? puffCount;
  final int? nicotineMg;
  final bool requiresAttendant;
  final double confidenceThreshold;
  final List<String> aliases;
}

class CommitResponse {
  final String sku;
  final String message;
}
```

JSON field mapping (snake_case server ↔ camelCase Dart):
- `session_id` → `sessionId`
- `job_id` → `jobId`
- `top_k` → `topK`
- `match_reason` → `matchReason`
- `query_used` → `queryUsed`
- `puff_count` → `puffCount`
- `nicotine_mg` → `nicotineMg`
- `requires_attendant` → `requiresAttendant`
- `confidence_threshold` → `confidenceThreshold`

Use `json_serializable` if already configured; otherwise write `fromJson`/`toJson` manually (the existing codebase does so manually — follow that pattern).

---

## VisionApiClient (`lib/api/vision_client.dart`)

```dart
class VisionApiClient {
  final String baseUrl;   // "http://10.64.0.X:8787"
  final http.Client _http;

  // --- Session ---
  Future<String> startSession();          // returns sessionId
  Future<void> deleteSession(String sid);

  // --- Capture (all require X-Session-Token header) ---
  Future<void> uploadVideo(String sid, File video);
  Future<void> uploadStill(String sid, String angle, File image);
  Future<void> uploadNarration(String sid, File audio);
  Future<String> finalize(String sid);    // returns jobId

  // --- Pipeline ---
  // Polls GET /product/draft/{jobId} every 2s, up to 60 attempts.
  // Throws VisionException if status == "failed" or attempts exhausted.
  Future<DraftProduct> pollJob(String jobId);

  // --- Product ---
  Future<CommitResponse> commitProduct(CommitRequest req);

  // --- Deduce ---
  Future<DeduceResponse> deduceText(String query, {int topK = 3});
  Future<DeduceResponse> deduceAudio(Uint8List bytes, String mimeType, {int topK = 3});
}

class VisionException implements Exception {
  final String message;
  final int? statusCode;
}
```

`uploadVideo`, `uploadStill`, `uploadNarration` all use `http.MultipartRequest` following the same pattern as `LemonadeApiClient.postMultipart`. Pass `X-Session-Token: $sid` as a request header (not a form field).

`deduceAudio` sends the audio bytes as `multipart/form-data` with field name `file`, filename `query.m4a` (or `query.wav`), content-type equal to the passed `mimeType`.

---

## Riverpod Provider (`lib/providers/vision_provider.dart`)

```dart
final visionClientProvider = Provider<VisionApiClient?>((ref) {
  final server = ref.watch(selectedServerProvider);
  if (server == null) return null;
  final uri = Uri.parse(server.baseUrl);
  final baseUrl = '${uri.scheme}://${uri.host}:8787';
  return VisionApiClient(baseUrl);
});
```

Returns `null` when no server is selected. Screens show a "Select a server first" placeholder in that case.

---

## CaptureScreen (`lib/screens/capture_screen.dart`)

Five-step wizard, each step is a full-screen stage with a header showing current step number.

**Step 1 — Mode select:** Two large buttons — "Video" (record a slow 360° rotation) or "Stills" (take individual angle shots). Selecting Video proceeds to Step 2a. Selecting Stills proceeds to Step 2b.

**Step 2a — Video capture:** Calls `ImagePicker().pickVideo(source: ImageSource.camera)`. On return, calls `startSession()` then `uploadVideo()` with a progress indicator. Proceeds to Step 3.

**Step 2b — Stills capture:** Displays angle buttons in a grid: `front`, `rear`, `upc`, `label`, `top`, `bottom`. Each tap calls `ImagePicker().pickImage(source: ImageSource.camera)`, uploads via `uploadStill()`, and marks the angle as captured (green check). At least one angle required. "Done" button proceeds to Step 3.  
Session is started before the first upload (lazy: call `startSession()` on first image tap).

**Step 3 — Narration (optional):** Mic button using the `record` package. Hold to record, release to stop. Uploads via `uploadNarration()`. "Skip" button advances without uploading audio.

**Step 4 — Processing:** Calls `finalize()` then enters a polling loop via `pollJob()`. Shows a spinner with "Analysing product…". On `status == "ready"`, advances to Step 5. On `status == "failed"`, shows error with a retry-from-Step-1 button.

**Step 5 — Review & Commit:** Shows `DraftReviewCard` — a form pre-filled from `DraftProduct`. Editable fields: SKU (auto-generated default: `brand-flavor` slugified), Brand, Flavor, Category (dropdown: `disposable_vape`, `e-liquid`, `accessory`, `other`), Puff Count, Nicotine mg, Requires Attendant (toggle), Aliases (chip input). Signal score badges (UPC/VLM/embedding/dimension) shown read-only. "Commit" button calls `commitProduct()` and shows a success snackbar with the assigned SKU, then pops back to the Vision home.

Session is deleted (`deleteSession`) after a successful commit or on explicit user cancel.

---

## DeduceScreen (`lib/screens/deduce_screen.dart`)

Single screen with two input modes, toggled by a segmented control at the top: **Text** / **Voice**.

**Text mode:** Text field + search button. On submit, calls `deduceText(query)`. Shows `DeduceResultTile` list below.

**Voice mode:** Large mic button. Tap-to-start, tap-to-stop recording (using `record` package, M4A). On stop, immediately calls `deduceAudio(bytes, 'audio/m4a')`. Shows results below.

**DeduceResultTile:** One row per `DeduceCandidate`. Left: SKU + brand/flavor in secondary text. Right: confidence as a percentage + color-coded dot (≥0.85 green, ≥0.60 amber, <0.60 red). Below the SKU: `matchReason` in caption style.

Empty state: "No candidates found" with a ghost icon. Error state: red banner with error message.

---

## Navigation

Add "Vision" to the app drawer alongside existing Chat/Transcription/Settings entries. The Vision destination shows a tab bar with two tabs: **Lookup** (DeduceScreen) and **Onboard** (CaptureScreen entry point). Both tabs share `visionClientProvider`.

---

## Server-Side Fix: Audio Format (`lemonade-vision-server`)

The vision server currently assumes audio is WAV. iOS `record` package produces M4A. Fix: use ffmpeg to convert to 16kHz mono PCM WAV before passing to fw-server.

**`src/lemonade_vision/api/deduce.py` — `deduce_audio`:**

After saving the uploaded file to `audio_path`, add:
```python
import subprocess
wav_path = Path(d) / "query.wav"
subprocess.run(
    ["ffmpeg", "-y", "-i", str(audio_path),
     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
    check=True, capture_output=True,
)
```
Then use `wav_path` instead of `audio_path` when reading bytes for the transcription POST.

**`src/lemonade_vision/api/capture.py` — `upload_audio`:**

Same ffmpeg conversion after saving the narration file, before storing `narration_path` in the DB. Convert to `narration_converted.wav` and store that path.

Tests: add `test_deduce_audio_m4a_accepted` (mock ffmpeg subprocess) and `test_capture_audio_m4a_accepted`.

---

## What's Out of Scope

- LiDAR depth upload (Flutter has no reliable LiDAR API; `capture/depth` endpoint remains available for future native extension)
- Product edit after commit (`PATCH /product/{sku}` exists but no UI planned here)
- ngrok removal (ngrok.sh stays as a fallback; Nexus makes it redundant but removing it is a separate housekeeping task)
- Nexus iOS client setup (already deployed per confirmed precondition)
