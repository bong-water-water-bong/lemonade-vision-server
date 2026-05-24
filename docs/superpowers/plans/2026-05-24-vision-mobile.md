# Vision Mobile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Vision" feature to lemonade-mobile (Flutter) giving cashiers a product onboarding wizard (capture → pipeline → review → commit) and a deduce lookup screen (text/voice → ranked candidates), connected to lemonade-vision-server over the Lemonade Nexus mesh.

**Architecture:** One `VisionApiClient` (Dart) derived from the active Lemonade server's host at port 8787. Two new screens (`CaptureScreen`, `DeduceScreen`) accessed via a new "Vision" drawer entry. One server-side ffmpeg fix allows iOS M4A audio alongside WAV in both `/deduce/audio` and `/capture/audio`.

**Tech Stack:** Flutter 3 / Dart, Riverpod, `record`/`image_picker`/`http`/`http_parser` (all already in pubspec), Python 3.12 / FastAPI (vision server patch), ffmpeg (already installed via lemonade-vision-server).

---

## Repos

| Repo | Local path | Branch |
|------|-----------|--------|
| `lemonade-vision-server` | `/home/bcloud/lemonade-vision-server` | `main` |
| `lemonade-mobile` (fork) | `/home/bcloud/lemonade-mobile` | `feat/vision-cashier` |

---

## File Map

**lemonade-vision-server** (modify):
- `src/lemonade_vision/api/deduce.py` — add ffmpeg WAV conversion before fw-server call
- `src/lemonade_vision/api/capture.py` — add ffmpeg WAV conversion for narration upload
- `tests/test_deduce.py` — add M4A audio test
- `tests/test_capture_audio.py` — new file, M4A narration test

**lemonade-mobile** (fork → `/home/bcloud/lemonade-mobile`):
- `lib/api/types/vision_types.dart` — all Dart models: DeduceCandidate, DeduceResponse, DraftProduct, CommitRequest, CommitResponse, SignalScores, VisionException
- `lib/api/vision_client.dart` — VisionApiClient with all 9 methods
- `lib/providers/vision_provider.dart` — `visionClientProvider` deriving URL from selected server
- `lib/widgets/vision/deduce_result_tile.dart` — single candidate row widget
- `lib/widgets/vision/draft_review_card.dart` — editable draft form widget
- `lib/screens/deduce_screen.dart` — text + voice lookup screen
- `lib/screens/capture_screen.dart` — 5-step capture wizard
- `lib/screens/vision_home_screen.dart` — tab bar: Lookup / Onboard
- `lib/widgets/chat_drawer.dart` — add Vision ListTile (modify lines ~109-155)
- `test/api/vision_types_test.dart` — JSON round-trip tests
- `test/api/vision_client_test.dart` — unit tests with MockClient

---

## Task 1: Fork and clone lemonade-mobile

**Files:** none yet — setup only

- [ ] **Step 1: Fork the repo**

```bash
gh repo fork lemonade-sdk/lemonade-mobile --org bong-water-water-bong --clone=false
```

Expected: `✓ Created fork bong-water-water-bong/lemonade-mobile`

- [ ] **Step 2: Clone the fork**

```bash
git clone https://github.com/bong-water-water-bong/lemonade-mobile.git /home/bcloud/lemonade-mobile
cd /home/bcloud/lemonade-mobile
```

- [ ] **Step 3: Set git identity**

```bash
git config user.email "277547417+bong-water-water-bong@users.noreply.github.com"
git config user.name "bcloud"
```

- [ ] **Step 4: Create feature branch**

```bash
git checkout -b feat/vision-cashier
```

- [ ] **Step 5: Verify Flutter setup**

```bash
flutter pub get
flutter analyze --no-fatal-infos 2>&1 | tail -5
```

Expected: no errors, warnings are ok.

---

## Task 2: Server-side audio fix (lemonade-vision-server)

**Files:**
- Modify: `src/lemonade_vision/api/deduce.py`
- Modify: `src/lemonade_vision/api/capture.py`
- Modify: `tests/test_deduce.py`
- Create: `tests/test_capture_audio.py`

- [ ] **Step 1: Write the failing test for deduce M4A**

Add to `tests/test_deduce.py`:

```python
import subprocess

def test_deduce_audio_calls_ffmpeg_for_m4a(client_with_products):
    """Non-WAV uploads trigger ffmpeg conversion before transcription."""
    with patch("lemonade_vision.api.deduce.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        resp = client_with_products.post(
            "/deduce/audio",
            files={"file": ("query.m4a", b"\x00" * 16, "audio/m4a")},
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "pcm_s16le" in cmd
    # ffmpeg failed → 500 (not 422 which would mean no conversion attempted)
    assert resp.status_code == 500
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/bcloud/lemonade-vision-server
pytest tests/test_deduce.py::test_deduce_audio_calls_ffmpeg_for_m4a -v
```

Expected: `FAILED` — `AssertionError: assert 0 == 1` (mock not called yet)

- [ ] **Step 3: Add subprocess import and ffmpeg conversion to deduce.py**

In `src/lemonade_vision/api/deduce.py`, add `import subprocess` at the top (after existing imports), then replace the `deduce_audio` function body:

```python
import subprocess
```

Replace the full `deduce_audio` function (lines 79–104) with:

```python
@router.post("/deduce/audio", response_model=DeduceResponse)
async def deduce_audio(request: Request, file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        suffix = Path(file.filename or "query.bin").suffix or ".bin"
        audio_path = Path(d) / f"query{suffix}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        wav_path = Path(d) / "query.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            check=True, capture_output=True,
        )

        transcript: Optional[str] = None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                with open(wav_path, "rb") as af:
                    resp = await client.post(
                        "http://localhost:8004/transcribe",
                        content=af.read(),
                        headers={"Content-Type": "audio/wav"},
                    )
                    resp.raise_for_status()
                    transcript = resp.json().get("text")
        except Exception as exc:
            _logger.warning("fw-server transcription failed: %s", exc)
            raise HTTPException(status_code=503, detail="fw-server :8004 unreachable")

        if not transcript:
            raise HTTPException(status_code=503, detail="transcription returned empty")

        return await deduce_text(body=DeduceRequest(query=transcript), request=request)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_deduce.py::test_deduce_audio_calls_ffmpeg_for_m4a -v
```

Expected: `PASSED`

- [ ] **Step 5: Write failing test for capture M4A narration**

Create `tests/test_capture_audio.py`:

```python
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def authed_client(tmp_path):
    import os
    os.environ["VISION_DATA_DIR"] = str(tmp_path / "data")
    from lemonade_vision.server import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as client:
        resp = client.post("/session/start")
        assert resp.status_code == 200
        token = resp.json()["session_id"]
        yield client, token


def test_capture_audio_calls_ffmpeg_for_m4a(authed_client):
    """Narration M4A upload triggers ffmpeg conversion to WAV."""
    client, token = authed_client
    with patch("lemonade_vision.api.capture.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        resp = client.post(
            "/capture/audio",
            headers={"X-Session-Token": token},
            files={"file": ("narration.m4a", b"\x00" * 16, "audio/m4a")},
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "pcm_s16le" in cmd
    assert resp.status_code == 500
```

- [ ] **Step 6: Run test to verify it fails**

```bash
pytest tests/test_capture_audio.py::test_capture_audio_calls_ffmpeg_for_m4a -v
```

Expected: `FAILED` — mock not called yet

- [ ] **Step 7: Add subprocess import and ffmpeg conversion to capture.py**

In `src/lemonade_vision/api/capture.py`, add `import subprocess` at the top, then replace the `upload_audio` function:

```python
import subprocess
```

Replace full `upload_audio` function (lines 91–109) with:

```python
@router.post("/capture/audio")
async def upload_audio(
    request: Request,
    session: Annotated[dict, Depends(_require_session)],
    file: UploadFile = File(...),
):
    tmp_dir = Path(session["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "narration.bin").suffix or ".bin"
    raw_path = tmp_dir / f"narration_raw{suffix}"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    narration_path = tmp_dir / "narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(narration_path)],
        check=True, capture_output=True,
    )

    db = request.app.state.db
    db.execute(
        "UPDATE capture_sessions SET narration_path = ? WHERE session_id = ?",
        (str(narration_path), session["session_id"]),
    )
    db.commit()
    return {"message": "audio received"}
```

- [ ] **Step 8: Run both new tests**

```bash
pytest tests/test_deduce.py::test_deduce_audio_calls_ffmpeg_for_m4a \
       tests/test_capture_audio.py::test_capture_audio_calls_ffmpeg_for_m4a -v
```

Expected: both `PASSED`

- [ ] **Step 9: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_pipeline_integration.py
```

Expected: all tests pass

- [ ] **Step 10: Commit**

```bash
cd /home/bcloud/lemonade-vision-server
git add src/lemonade_vision/api/deduce.py \
        src/lemonade_vision/api/capture.py \
        tests/test_deduce.py \
        tests/test_capture_audio.py
git commit -m "fix: convert uploaded audio to WAV via ffmpeg before transcription

Accepts any format (M4A, WebM, etc.) from iOS/Android clients.
Applies to both /deduce/audio and /capture/audio endpoints."
git push
```

---

## Task 3: Dart types

**Files:**
- Create: `lib/api/types/vision_types.dart`
- Create: `test/api/vision_types_test.dart`

Work in `/home/bcloud/lemonade-mobile`.

- [ ] **Step 1: Write the failing test**

Create `test/api/vision_types_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:lemonade_mobile/api/types/vision_types.dart';

void main() {
  group('DeduceCandidate.fromJson', () {
    test('parses all fields', () {
      final c = DeduceCandidate.fromJson({
        'sku': 'ELFBAR001',
        'confidence': 0.92,
        'match_reason': 'brand match',
        'brand': 'Elf Bar',
        'flavor': 'Mango Ice',
      });
      expect(c.sku, 'ELFBAR001');
      expect(c.confidence, 0.92);
      expect(c.matchReason, 'brand match');
      expect(c.brand, 'Elf Bar');
      expect(c.flavor, 'Mango Ice');
    });

    test('nullable brand and flavor', () {
      final c = DeduceCandidate.fromJson({
        'sku': 'X1',
        'confidence': 0.5,
        'match_reason': 'embedding similarity',
      });
      expect(c.brand, isNull);
      expect(c.flavor, isNull);
    });
  });

  group('DeduceResponse.fromJson', () {
    test('parses candidates and queryUsed', () {
      final r = DeduceResponse.fromJson({
        'candidates': [
          {'sku': 'A', 'confidence': 0.9, 'match_reason': 'upc'},
        ],
        'query_used': 'elf bar mango',
      });
      expect(r.candidates, hasLength(1));
      expect(r.queryUsed, 'elf bar mango');
    });
  });

  group('DraftProduct.fromJson', () {
    test('parses ready draft with signal scores', () {
      final d = DraftProduct.fromJson({
        'job_id': 'j1',
        'status': 'ready',
        'brand': 'Elf Bar',
        'flavor': 'Mango Ice',
        'category': 'disposable_vape',
        'puff_count': 5000,
        'nicotine_mg': 50,
        'upc': '012345678901',
        'ocr_text': 'ELF BAR',
        'signal_scores': {
          'upc': 1.0, 'vlm': 0.8, 'embedding': 0.9, 'dimension': 0.5,
        },
      });
      expect(d.status, 'ready');
      expect(d.brand, 'Elf Bar');
      expect(d.puffCount, 5000);
      expect(d.signalScores!.upc, 1.0);
    });

    test('parses minimal processing draft', () {
      final d = DraftProduct.fromJson({'job_id': 'j2', 'status': 'processing'});
      expect(d.brand, isNull);
      expect(d.signalScores, isNull);
    });
  });

  group('CommitRequest.toJson', () {
    test('serialises required fields', () {
      final req = CommitRequest(
        jobId: 'j1', sku: 'ELF-MANGO-5K',
        brand: 'Elf Bar', flavor: 'Mango Ice',
        category: 'disposable_vape',
      );
      final j = req.toJson();
      expect(j['job_id'], 'j1');
      expect(j['sku'], 'ELF-MANGO-5K');
      expect(j['requires_attendant'], false);
      expect(j['confidence_threshold'], 0.85);
      expect(j['aliases'], isEmpty);
      expect(j.containsKey('puff_count'), isFalse);
    });

    test('includes optional fields when set', () {
      final req = CommitRequest(
        jobId: 'j1', sku: 'S', brand: 'B', flavor: 'F',
        category: 'other', puffCount: 1000, nicotineMg: 50,
        requiresAttendant: true, aliases: ['elfie'],
      );
      final j = req.toJson();
      expect(j['puff_count'], 1000);
      expect(j['nicotine_mg'], 50);
      expect(j['requires_attendant'], true);
      expect(j['aliases'], ['elfie']);
    });
  });

  group('CommitResponse.fromJson', () {
    test('parses sku and message', () {
      final r = CommitResponse.fromJson({'sku': 'ELF-001', 'message': 'committed'});
      expect(r.sku, 'ELF-001');
    });
  });

  group('VisionException', () {
    test('toString includes status code', () {
      final e = VisionException('not found', statusCode: 404);
      expect(e.toString(), contains('404'));
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/bcloud/lemonade-mobile
flutter test test/api/vision_types_test.dart 2>&1 | tail -10
```

Expected: compilation error — `vision_types.dart` not found

- [ ] **Step 3: Create the types file**

Create `lib/api/types/vision_types.dart`:

```dart
import 'dart:typed_data';

// ---------- Deduce ----------

class DeduceCandidate {
  final String sku;
  final double confidence;
  final String matchReason;
  final String? brand;
  final String? flavor;

  const DeduceCandidate({
    required this.sku,
    required this.confidence,
    required this.matchReason,
    this.brand,
    this.flavor,
  });

  factory DeduceCandidate.fromJson(Map<String, dynamic> j) => DeduceCandidate(
        sku: j['sku'] as String,
        confidence: (j['confidence'] as num).toDouble(),
        matchReason: j['match_reason'] as String,
        brand: j['brand'] as String?,
        flavor: j['flavor'] as String?,
      );
}

class DeduceResponse {
  final List<DeduceCandidate> candidates;
  final String queryUsed;

  const DeduceResponse({required this.candidates, required this.queryUsed});

  factory DeduceResponse.fromJson(Map<String, dynamic> j) => DeduceResponse(
        candidates: (j['candidates'] as List)
            .map((e) => DeduceCandidate.fromJson(e as Map<String, dynamic>))
            .toList(),
        queryUsed: (j['query_used'] as String?) ?? '',
      );
}

// ---------- Session ----------

class SessionStartResponse {
  final String sessionId;

  const SessionStartResponse({required this.sessionId});

  factory SessionStartResponse.fromJson(Map<String, dynamic> j) =>
      SessionStartResponse(sessionId: j['session_id'] as String);
}

// ---------- Pipeline ----------

class SignalScores {
  final double upc;
  final double vlm;
  final double embedding;
  final double dimension;

  const SignalScores({
    required this.upc,
    required this.vlm,
    required this.embedding,
    required this.dimension,
  });

  factory SignalScores.fromJson(Map<String, dynamic> j) => SignalScores(
        upc: (j['upc'] as num?)?.toDouble() ?? 0.0,
        vlm: (j['vlm'] as num?)?.toDouble() ?? 0.0,
        embedding: (j['embedding'] as num?)?.toDouble() ?? 0.0,
        dimension: (j['dimension'] as num?)?.toDouble() ?? 0.0,
      );
}

class DraftProduct {
  final String jobId;
  final String status;
  final String? upc;
  final String? brand;
  final String? flavor;
  final String? category;
  final int? puffCount;
  final int? nicotineMg;
  final String? ocrText;
  final SignalScores? signalScores;

  const DraftProduct({
    required this.jobId,
    required this.status,
    this.upc,
    this.brand,
    this.flavor,
    this.category,
    this.puffCount,
    this.nicotineMg,
    this.ocrText,
    this.signalScores,
  });

  factory DraftProduct.fromJson(Map<String, dynamic> j) {
    final scores = j['signal_scores'];
    return DraftProduct(
      jobId: j['job_id'] as String,
      status: j['status'] as String,
      upc: j['upc'] as String?,
      brand: j['brand'] as String?,
      flavor: j['flavor'] as String?,
      category: j['category'] as String?,
      puffCount: j['puff_count'] as int?,
      nicotineMg: j['nicotine_mg'] as int?,
      ocrText: j['ocr_text'] as String?,
      signalScores: scores != null
          ? SignalScores.fromJson(scores as Map<String, dynamic>)
          : null,
    );
  }
}

// ---------- Commit ----------

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

  const CommitRequest({
    required this.jobId,
    required this.sku,
    required this.brand,
    required this.flavor,
    required this.category,
    this.puffCount,
    this.nicotineMg,
    this.requiresAttendant = false,
    this.confidenceThreshold = 0.85,
    this.aliases = const [],
  });

  Map<String, dynamic> toJson() => {
        'job_id': jobId,
        'sku': sku,
        'brand': brand,
        'flavor': flavor,
        'category': category,
        if (puffCount != null) 'puff_count': puffCount,
        if (nicotineMg != null) 'nicotine_mg': nicotineMg,
        'requires_attendant': requiresAttendant,
        'confidence_threshold': confidenceThreshold,
        'aliases': aliases,
      };
}

class CommitResponse {
  final String sku;
  final String message;

  const CommitResponse({required this.sku, required this.message});

  factory CommitResponse.fromJson(Map<String, dynamic> j) => CommitResponse(
        sku: j['sku'] as String,
        message: (j['message'] as String?) ?? 'committed',
      );
}

// ---------- Error ----------

class VisionException implements Exception {
  final String message;
  final int? statusCode;

  const VisionException(this.message, {this.statusCode});

  @override
  String toString() => statusCode != null
      ? 'VisionException($statusCode): $message'
      : 'VisionException: $message';
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
flutter test test/api/vision_types_test.dart
```

Expected: `All tests passed!`

- [ ] **Step 5: Commit**

```bash
git add lib/api/types/vision_types.dart test/api/vision_types_test.dart
git commit -m "feat: add vision API Dart types with JSON serialisation"
```

---

## Task 4: VisionApiClient — session + capture

**Files:**
- Create: `lib/api/vision_client.dart`
- Create: `test/api/vision_client_test.dart` (session + capture tests)

- [ ] **Step 1: Write failing tests for session and capture**

Create `test/api/vision_client_test.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:lemonade_mobile/api/vision_client.dart';
import 'package:lemonade_mobile/api/types/vision_types.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';

void main() {
  group('VisionApiClient.startSession', () {
    test('returns sessionId from response', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async =>
            http.Response(jsonEncode({'session_id': 'abc123', 'qr_png_b64': 'x'}), 200)),
      );
      final sid = await client.startSession();
      expect(sid, 'abc123');
    });

    test('throws VisionException on 500', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async => http.Response('error', 500)),
      );
      expect(client.startSession(), throwsA(isA<VisionException>()));
    });
  });

  group('VisionApiClient.finalize', () {
    test('returns jobId from response', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async =>
            http.Response(jsonEncode({'job_id': 'job-xyz', 'message': 'processing'}), 200)),
      );
      final jobId = await client.finalize('sid-1');
      expect(jobId, 'job-xyz');
    });
  });

  group('VisionApiClient.deduceText', () {
    test('sends query and topK, returns DeduceResponse', () async {
      http.Request? captured;
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((req) async {
          captured = req;
          return http.Response(
            jsonEncode({
              'candidates': [
                {'sku': 'E1', 'confidence': 0.9, 'match_reason': 'upc'},
              ],
              'query_used': 'elf bar',
            }),
            200,
          );
        }),
      );
      final resp = await client.deduceText('elf bar', topK: 2);
      expect(resp.candidates, hasLength(1));
      expect(resp.candidates.first.sku, 'E1');
      final body = jsonDecode(captured!.body) as Map;
      expect(body['query'], 'elf bar');
      expect(body['top_k'], 2);
    });
  });

  group('VisionApiClient.deduceAudio', () {
    test('sends bytes as multipart and returns DeduceResponse', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async => http.Response(
              jsonEncode({
                'candidates': [],
                'query_used': '',
              }),
              200,
            )),
      );
      final resp = await client.deduceAudio(
          Uint8List.fromList([0, 1, 2]), 'audio/m4a');
      expect(resp.candidates, isEmpty);
    });

    test('throws VisionException on 503', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async => http.Response('unavailable', 503)),
      );
      expect(
        client.deduceAudio(Uint8List(4), 'audio/m4a'),
        throwsA(isA<VisionException>()),
      );
    });
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
flutter test test/api/vision_client_test.dart 2>&1 | tail -10
```

Expected: compilation error — `vision_client.dart` not found

- [ ] **Step 3: Create VisionApiClient**

Create `lib/api/vision_client.dart`:

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import 'types/vision_types.dart';

class VisionApiClient {
  final String baseUrl;
  final http.Client _http;

  VisionApiClient(this.baseUrl, {http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  void dispose() => _http.close();

  // ---- Session ----

  Future<String> startSession() async {
    final uri = Uri.parse('$baseUrl/session/start');
    final resp = await _http.post(uri);
    _check(resp);
    return SessionStartResponse.fromJson(
        jsonDecode(resp.body) as Map<String, dynamic>).sessionId;
  }

  Future<void> deleteSession(String sessionId) async {
    final uri = Uri.parse('$baseUrl/session/$sessionId');
    await _http.delete(uri);
  }

  // ---- Capture ----

  Future<void> uploadVideo(String sessionId, File video) async {
    final uri = Uri.parse('$baseUrl/capture/video');
    final req = http.MultipartRequest('POST', uri)
      ..headers['X-Session-Token'] = sessionId
      ..files.add(await http.MultipartFile.fromPath(
        'file', video.path,
        contentType: MediaType('video', 'mp4'),
      ));
    _checkStreamed(await _http.send(req));
  }

  Future<void> uploadStill(
      String sessionId, String angle, File image) async {
    final uri = Uri.parse('$baseUrl/capture/still');
    final req = http.MultipartRequest('POST', uri)
      ..headers['X-Session-Token'] = sessionId
      ..fields['angle'] = angle
      ..files.add(await http.MultipartFile.fromPath(
        'file', image.path,
        contentType: MediaType('image', 'jpeg'),
      ));
    _checkStreamed(await _http.send(req));
  }

  Future<void> uploadNarration(String sessionId, File audio) async {
    final uri = Uri.parse('$baseUrl/capture/audio');
    final ext = audio.path.split('.').last;
    final req = http.MultipartRequest('POST', uri)
      ..headers['X-Session-Token'] = sessionId
      ..files.add(await http.MultipartFile.fromPath(
        'file', audio.path,
        contentType: MediaType('audio', ext),
      ));
    _checkStreamed(await _http.send(req));
  }

  Future<String> finalize(String sessionId) async {
    final uri = Uri.parse('$baseUrl/capture/finalize');
    final req = http.Request('POST', uri)
      ..headers['X-Session-Token'] = sessionId
      ..headers['Content-Length'] = '0';
    final streamed = await _http.send(req);
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw VisionException(body, statusCode: streamed.statusCode);
    }
    return (jsonDecode(body) as Map<String, dynamic>)['job_id'] as String;
  }

  // ---- Pipeline ----

  /// Polls GET /product/draft/{jobId} every 2 s up to [maxAttempts] times.
  /// Returns when status is 'ready' or 'failed'. Throws after timeout.
  Future<DraftProduct> pollJob(String jobId, {int maxAttempts = 60}) async {
    for (var i = 0; i < maxAttempts; i++) {
      final uri = Uri.parse('$baseUrl/product/draft/$jobId');
      final resp = await _http.get(uri);
      _check(resp);
      final draft = DraftProduct.fromJson(
          jsonDecode(resp.body) as Map<String, dynamic>);
      if (draft.status == 'ready' || draft.status == 'failed') return draft;
      if (i < maxAttempts - 1) {
        await Future.delayed(const Duration(seconds: 2));
      }
    }
    throw VisionException('Pipeline timed out after ${maxAttempts * 2}s');
  }

  // ---- Product ----

  Future<CommitResponse> commitProduct(CommitRequest request) async {
    final uri = Uri.parse('$baseUrl/product/commit');
    final resp = await _http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(request.toJson()),
    );
    _check(resp);
    return CommitResponse.fromJson(
        jsonDecode(resp.body) as Map<String, dynamic>);
  }

  // ---- Deduce ----

  Future<DeduceResponse> deduceText(String query, {int topK = 3}) async {
    final uri = Uri.parse('$baseUrl/deduce/text');
    final resp = await _http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'query': query, 'top_k': topK}),
    );
    _check(resp);
    return DeduceResponse.fromJson(
        jsonDecode(resp.body) as Map<String, dynamic>);
  }

  Future<DeduceResponse> deduceAudio(
      Uint8List bytes, String mimeType, {int topK = 3}) async {
    final uri = Uri.parse('$baseUrl/deduce/audio');
    final parts = mimeType.split('/');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'file', bytes,
        filename: 'query.${parts.last}',
        contentType: MediaType(parts[0], parts[1]),
      ));
    final streamed = await _http.send(req);
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw VisionException(body, statusCode: streamed.statusCode);
    }
    return DeduceResponse.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  // ---- Helpers ----

  void _check(http.Response resp) {
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw VisionException(resp.body, statusCode: resp.statusCode);
    }
  }

  void _checkStreamed(http.StreamedResponse resp) {
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw VisionException(
          'HTTP ${resp.statusCode}', statusCode: resp.statusCode);
    }
  }
}
```

- [ ] **Step 4: Run tests**

```bash
flutter test test/api/vision_client_test.dart
```

Expected: `All tests passed!`

- [ ] **Step 5: Add pollJob test** (add to existing `test/api/vision_client_test.dart` inside `main()`):

```dart
  group('VisionApiClient.pollJob', () {
    test('returns draft when status is ready on first poll', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async => http.Response(
              jsonEncode({'job_id': 'j1', 'status': 'ready', 'brand': 'Elf Bar'}),
              200,
            )),
      );
      final draft = await client.pollJob('j1', maxAttempts: 1);
      expect(draft.status, 'ready');
      expect(draft.brand, 'Elf Bar');
    });

    test('throws after maxAttempts when still processing', () async {
      final client = VisionApiClient(
        'http://localhost:8787',
        httpClient: MockClient((_) async => http.Response(
              jsonEncode({'job_id': 'j1', 'status': 'processing'}),
              200,
            )),
      );
      expect(
        client.pollJob('j1', maxAttempts: 1),
        throwsA(isA<VisionException>()),
      );
    });
  });
```

- [ ] **Step 6: Run tests again**

```bash
flutter test test/api/vision_client_test.dart
```

Expected: `All tests passed!`

- [ ] **Step 7: Commit**

```bash
git add lib/api/vision_client.dart test/api/vision_client_test.dart
git commit -m "feat: add VisionApiClient with session, capture, pipeline, and deduce methods"
```

---

## Task 5: Riverpod provider

**Files:**
- Create: `lib/providers/vision_provider.dart`

- [ ] **Step 1: Write failing test**

Add `test/providers/vision_provider_test.dart`:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:lemonade_mobile/api/vision_client.dart';
import 'package:lemonade_mobile/models/server_config.dart';
import 'package:lemonade_mobile/providers/servers_provider.dart';
import 'package:lemonade_mobile/providers/vision_provider.dart';

void main() {
  test('visionClientProvider returns null when no server selected', () {
    final container = ProviderContainer(
      overrides: [
        selectedServerProvider.overrideWith((ref) => null),
      ],
    );
    addTearDown(container.dispose);
    expect(container.read(visionClientProvider), isNull);
  });

  test('visionClientProvider derives port 8787 from selected server', () {
    final container = ProviderContainer(
      overrides: [
        selectedServerProvider.overrideWith((ref) => ServerConfig(
              name: 'Test',
              baseUrl: 'http://10.64.0.5:13305',
            )),
      ],
    );
    addTearDown(container.dispose);
    final client = container.read(visionClientProvider);
    expect(client, isNotNull);
    expect(client!.baseUrl, 'http://10.64.0.5:8787');
  });

  test('visionClientProvider uses same scheme as selected server', () {
    final container = ProviderContainer(
      overrides: [
        selectedServerProvider.overrideWith((ref) => ServerConfig(
              name: 'Test',
              baseUrl: 'https://myserver.local:13305',
            )),
      ],
    );
    addTearDown(container.dispose);
    final client = container.read(visionClientProvider);
    expect(client!.baseUrl, 'https://myserver.local:8787');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
flutter test test/providers/vision_provider_test.dart 2>&1 | tail -5
```

Expected: compilation error — `vision_provider.dart` not found

- [ ] **Step 3: Create the provider**

Create `lib/providers/vision_provider.dart`:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/vision_client.dart';
import 'servers_provider.dart';

final visionClientProvider = Provider<VisionApiClient?>((ref) {
  final server = ref.watch(selectedServerProvider);
  if (server == null) return null;
  final uri = Uri.parse(server.baseUrl);
  final baseUrl = '${uri.scheme}://${uri.host}:8787';
  return VisionApiClient(baseUrl);
});
```

- [ ] **Step 4: Run test**

```bash
flutter test test/providers/vision_provider_test.dart
```

Expected: `All tests passed!`

- [ ] **Step 5: Commit**

```bash
git add lib/providers/vision_provider.dart test/providers/vision_provider_test.dart
git commit -m "feat: add visionClientProvider deriving port 8787 from active server"
```

---

## Task 6: Widgets — DeduceResultTile and DraftReviewCard

**Files:**
- Create: `lib/widgets/vision/deduce_result_tile.dart`
- Create: `lib/widgets/vision/draft_review_card.dart`

- [ ] **Step 1: Create DeduceResultTile**

Create `lib/widgets/vision/deduce_result_tile.dart`:

```dart
import 'package:flutter/material.dart';
import '../../api/types/vision_types.dart';

class DeduceResultTile extends StatelessWidget {
  final DeduceCandidate candidate;

  const DeduceResultTile({super.key, required this.candidate});

  Color _confidenceColor(double c) {
    if (c >= 0.85) return Colors.green;
    if (c >= 0.60) return Colors.amber;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    final pct = (candidate.confidence * 100).toStringAsFixed(0);
    final color = _confidenceColor(candidate.confidence);
    return ListTile(
      title: Text(candidate.sku,
          style: const TextStyle(fontWeight: FontWeight.bold)),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (candidate.brand != null || candidate.flavor != null)
            Text('${candidate.brand ?? ''} ${candidate.flavor ?? ''}'.trim()),
          Text(candidate.matchReason,
              style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$pct%',
              style: TextStyle(
                  color: color, fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(width: 6),
          Icon(Icons.circle, color: color, size: 10),
        ],
      ),
      isThreeLine: true,
    );
  }
}
```

- [ ] **Step 2: Create DraftReviewCard**

Create `lib/widgets/vision/draft_review_card.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../api/types/vision_types.dart';

class DraftReviewCard extends StatefulWidget {
  final DraftProduct draft;
  final void Function(CommitRequest) onCommit;
  final VoidCallback onCancel;

  const DraftReviewCard({
    super.key,
    required this.draft,
    required this.onCommit,
    required this.onCancel,
  });

  @override
  State<DraftReviewCard> createState() => _DraftReviewCardState();
}

class _DraftReviewCardState extends State<DraftReviewCard> {
  static const _categories = [
    'disposable_vape', 'e-liquid', 'accessory', 'other',
  ];

  late final TextEditingController _sku;
  late final TextEditingController _brand;
  late final TextEditingController _flavor;
  late final TextEditingController _puffCount;
  late final TextEditingController _nicotineMg;
  late final TextEditingController _aliases;
  late String _category;
  late bool _requiresAttendant;

  @override
  void initState() {
    super.initState();
    final d = widget.draft;
    final defaultSku = [d.brand, d.flavor]
        .where((s) => s != null)
        .join('-')
        .toLowerCase()
        .replaceAll(' ', '-');
    _sku = TextEditingController(text: defaultSku);
    _brand = TextEditingController(text: d.brand ?? '');
    _flavor = TextEditingController(text: d.flavor ?? '');
    _puffCount = TextEditingController(
        text: d.puffCount != null ? '${d.puffCount}' : '');
    _nicotineMg = TextEditingController(
        text: d.nicotineMg != null ? '${d.nicotineMg}' : '');
    _aliases = TextEditingController();
    _category = _categories.contains(d.category) ? d.category! : 'other';
    _requiresAttendant = false;
  }

  @override
  void dispose() {
    for (final c in [_sku, _brand, _flavor, _puffCount, _nicotineMg, _aliases]) {
      c.dispose();
    }
    super.dispose();
  }

  void _submit() {
    final aliasList = _aliases.text
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    widget.onCommit(CommitRequest(
      jobId: widget.draft.jobId,
      sku: _sku.text.trim(),
      brand: _brand.text.trim(),
      flavor: _flavor.text.trim(),
      category: _category,
      puffCount: int.tryParse(_puffCount.text),
      nicotineMg: int.tryParse(_nicotineMg.text),
      requiresAttendant: _requiresAttendant,
      aliases: aliasList,
    ));
  }

  Widget _signalBadge(String label, double score) {
    Color color;
    if (score >= 0.8) {
      color = Colors.green;
    } else if (score >= 0.5) {
      color = Colors.amber;
    } else {
      color = Colors.grey;
    }
    return Chip(
      label: Text('$label ${(score * 100).round()}%',
          style: TextStyle(color: color, fontSize: 11)),
      backgroundColor: color.withOpacity(0.1),
      visualDensity: VisualDensity.compact,
    );
  }

  @override
  Widget build(BuildContext context) {
    final scores = widget.draft.signalScores;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (scores != null) ...[
            Wrap(spacing: 6, children: [
              _signalBadge('UPC', scores.upc),
              _signalBadge('VLM', scores.vlm),
              _signalBadge('Embed', scores.embedding),
              _signalBadge('Dim', scores.dimension),
            ]),
            const SizedBox(height: 12),
          ],
          TextField(
            controller: _sku,
            decoration: const InputDecoration(labelText: 'SKU *'),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _brand,
            decoration: const InputDecoration(labelText: 'Brand'),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _flavor,
            decoration: const InputDecoration(labelText: 'Flavor / Variant'),
          ),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _category,
            decoration: const InputDecoration(labelText: 'Category'),
            items: _categories
                .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                .toList(),
            onChanged: (v) => setState(() => _category = v!),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _puffCount,
            decoration: const InputDecoration(labelText: 'Puff Count'),
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _nicotineMg,
            decoration:
                const InputDecoration(labelText: 'Nicotine (mg, optional)'),
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _aliases,
            decoration: const InputDecoration(
              labelText: 'Aliases (comma-separated)',
              hintText: 'elfie, mango elf',
            ),
          ),
          const SizedBox(height: 8),
          SwitchListTile(
            value: _requiresAttendant,
            onChanged: (v) => setState(() => _requiresAttendant = v),
            title: const Text('Requires attendant'),
            contentPadding: EdgeInsets.zero,
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: widget.onCancel,
                  child: const Text('Cancel'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton(
                  onPressed: _sku.text.trim().isEmpty ? null : _submit,
                  child: const Text('Commit to database'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Run flutter analyze**

```bash
flutter analyze lib/widgets/vision/ --no-fatal-infos 2>&1 | tail -5
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add lib/widgets/vision/
git commit -m "feat: add DeduceResultTile and DraftReviewCard widgets"
```

---

## Task 7: DeduceScreen

**Files:**
- Create: `lib/screens/deduce_screen.dart`

- [ ] **Step 1: Create DeduceScreen**

Create `lib/screens/deduce_screen.dart`:

```dart
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../api/types/vision_types.dart';
import '../providers/vision_provider.dart';
import '../widgets/vision/deduce_result_tile.dart';

class DeduceScreen extends ConsumerStatefulWidget {
  const DeduceScreen({super.key});

  @override
  ConsumerState<DeduceScreen> createState() => _DeduceScreenState();
}

class _DeduceScreenState extends ConsumerState<DeduceScreen> {
  bool _voiceMode = false;
  final _queryCtrl = TextEditingController();
  List<DeduceCandidate> _candidates = [];
  String? _error;
  bool _loading = false;
  bool _recording = false;
  final _recorder = AudioRecorder();

  @override
  void dispose() {
    _queryCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _searchText() async {
    final client = ref.read(visionClientProvider);
    if (client == null) return;
    final q = _queryCtrl.text.trim();
    if (q.isEmpty) return;
    setState(() { _loading = true; _error = null; });
    try {
      final resp = await client.deduceText(q);
      setState(() => _candidates = resp.candidates);
    } on VisionException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _startRecording() async {
    final permitted = await _recorder.hasPermission();
    if (!permitted) {
      setState(() => _error = 'Microphone permission denied');
      return;
    }
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/deduce_query.m4a';
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.aacLc),
      path: path,
    );
    setState(() { _recording = true; _error = null; });
  }

  Future<void> _stopRecordingAndSearch() async {
    final path = await _recorder.stop();
    setState(() => _recording = false);
    if (path == null) return;

    final client = ref.read(visionClientProvider);
    if (client == null) return;
    setState(() { _loading = true; _error = null; });
    try {
      final bytes = await File(path).readAsBytes();
      final resp = await client.deduceAudio(
          Uint8List.fromList(bytes), 'audio/m4a');
      setState(() => _candidates = resp.candidates);
    } on VisionException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final client = ref.watch(visionClientProvider);

    if (client == null) {
      return const Center(child: Text('Select a server first'));
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(12),
          child: SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: false, label: Text('Text')),
              ButtonSegment(value: true, label: Text('Voice')),
            ],
            selected: {_voiceMode},
            onSelectionChanged: (s) =>
                setState(() { _voiceMode = s.first; _candidates = []; }),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: _voiceMode ? _buildVoiceInput() : _buildTextInput(),
        ),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(_error!,
                style: const TextStyle(color: Colors.red)),
          ),
        if (_loading) const LinearProgressIndicator(),
        Expanded(
          child: _candidates.isEmpty && !_loading
              ? const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.search_off, size: 48, color: Colors.grey),
                      SizedBox(height: 8),
                      Text('No candidates yet',
                          style: TextStyle(color: Colors.grey)),
                    ],
                  ),
                )
              : ListView.separated(
                  itemCount: _candidates.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (_, i) =>
                      DeduceResultTile(candidate: _candidates[i]),
                ),
        ),
      ],
    );
  }

  Widget _buildTextInput() {
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _queryCtrl,
            decoration: const InputDecoration(
              hintText: 'e.g. elf bar mango ice 5000',
              border: OutlineInputBorder(),
            ),
            onSubmitted: (_) => _searchText(),
          ),
        ),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: _loading ? null : _searchText,
          child: const Text('Search'),
        ),
      ],
    );
  }

  Widget _buildVoiceInput() {
    return Center(
      child: GestureDetector(
        onTap: _loading
            ? null
            : (_recording ? _stopRecordingAndSearch : _startRecording),
        child: CircleAvatar(
          radius: 36,
          backgroundColor: _recording ? Colors.red : Colors.blue,
          child: Icon(
            _recording ? Icons.stop : Icons.mic,
            color: Colors.white,
            size: 32,
          ),
        ),
      ),
    );
  }
}
```

- [ ] **Step 2: Run analyze**

```bash
flutter analyze lib/screens/deduce_screen.dart --no-fatal-infos 2>&1 | tail -5
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add lib/screens/deduce_screen.dart
git commit -m "feat: add DeduceScreen with text and voice query modes"
```

---

## Task 8: CaptureScreen

**Files:**
- Create: `lib/screens/capture_screen.dart`

- [ ] **Step 1: Create CaptureScreen**

Create `lib/screens/capture_screen.dart`:

```dart
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../api/types/vision_types.dart';
import '../providers/vision_provider.dart';
import '../widgets/vision/draft_review_card.dart';

enum _Step { modeSelect, videoCapture, stillsCapture, narration, processing, review }

class CaptureScreen extends ConsumerStatefulWidget {
  const CaptureScreen({super.key});

  @override
  ConsumerState<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends ConsumerState<CaptureScreen> {
  _Step _step = _Step.modeSelect;
  bool _videoMode = false;
  String? _sessionId;
  DraftProduct? _draft;
  final Set<String> _capturedAngles = {};
  bool _recording = false;
  bool _busy = false;
  String? _error;
  final _recorder = AudioRecorder();
  final _picker = ImagePicker();

  static const _angles = [
    'front', 'rear', 'upc', 'label', 'top', 'bottom',
  ];

  @override
  void dispose() {
    _recorder.dispose();
    if (_sessionId != null) {
      ref.read(visionClientProvider)?.deleteSession(_sessionId!);
    }
    super.dispose();
  }

  Future<void> _startSession() async {
    final client = ref.read(visionClientProvider)!;
    _sessionId = await client.startSession();
  }

  Future<void> _pickAndUploadVideo() async {
    setState(() { _busy = true; _error = null; });
    try {
      final xfile = await _picker.pickVideo(source: ImageSource.camera);
      if (xfile == null) { setState(() => _busy = false); return; }
      await _startSession();
      final client = ref.read(visionClientProvider)!;
      await client.uploadVideo(_sessionId!, File(xfile.path));
      setState(() => _step = _Step.narration);
    } on Exception catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _pickAndUploadStill(String angle) async {
    setState(() { _busy = true; _error = null; });
    try {
      final xfile = await _picker.pickImage(
        source: ImageSource.camera, imageQuality: 90);
      if (xfile == null) { setState(() => _busy = false); return; }
      if (_sessionId == null) await _startSession();
      final client = ref.read(visionClientProvider)!;
      await client.uploadStill(_sessionId!, angle, File(xfile.path));
      setState(() => _capturedAngles.add(angle));
    } on Exception catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _startRecordingNarration() async {
    final permitted = await _recorder.hasPermission();
    if (!permitted) {
      setState(() => _error = 'Microphone permission denied');
      return;
    }
    final dir = await getTemporaryDirectory();
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.aacLc),
      path: '${dir.path}/narration.m4a',
    );
    setState(() => _recording = true);
  }

  Future<void> _stopNarrationAndProceed() async {
    final path = await _recorder.stop();
    setState(() => _recording = false);
    if (path != null && _sessionId != null) {
      final client = ref.read(visionClientProvider)!;
      await client.uploadNarration(_sessionId!, File(path));
    }
    await _finalize();
  }

  Future<void> _finalize() async {
    setState(() { _step = _Step.processing; _busy = true; _error = null; });
    try {
      final client = ref.read(visionClientProvider)!;
      final jobId = await client.finalize(_sessionId!);
      final draft = await client.pollJob(jobId);
      setState(() { _draft = draft; _step = _Step.review; });
    } on Exception catch (e) {
      setState(() { _error = e.toString(); _step = _Step.narration; });
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _commit(CommitRequest req) async {
    setState(() { _busy = true; _error = null; });
    try {
      final client = ref.read(visionClientProvider)!;
      final result = await client.commitProduct(req);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Committed: ${result.sku}')),
      );
      Navigator.of(context).pop();
    } on Exception catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final client = ref.watch(visionClientProvider);
    if (client == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Scan Product')),
        body: const Center(child: Text('Select a server first')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(_stepTitle()),
        leading: _step == _Step.review
            ? null
            : IconButton(
                icon: const Icon(Icons.close),
                onPressed: () => Navigator.of(context).pop(),
              ),
      ),
      body: _buildBody(),
    );
  }

  String _stepTitle() {
    switch (_step) {
      case _Step.modeSelect: return 'Scan Product';
      case _Step.videoCapture: return 'Record Video';
      case _Step.stillsCapture: return 'Take Stills';
      case _Step.narration: return 'Add Narration';
      case _Step.processing: return 'Processing…';
      case _Step.review: return 'Review Draft';
    }
  }

  Widget _buildBody() {
    if (_error != null && _step != _Step.processing) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text(_error!, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => setState(() => _error = null),
              child: const Text('Try again'),
            ),
          ]),
        ),
      );
    }
    switch (_step) {
      case _Step.modeSelect: return _buildModeSelect();
      case _Step.videoCapture: return _buildVideoCapture();
      case _Step.stillsCapture: return _buildStillsCapture();
      case _Step.narration: return _buildNarration();
      case _Step.processing: return _buildProcessing();
      case _Step.review: return _buildReview();
    }
  }

  Widget _buildModeSelect() => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      const Text('How would you like to capture this product?',
          style: TextStyle(fontSize: 16)),
      const SizedBox(height: 24),
      Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        _ModeButton(
          icon: Icons.videocam,
          label: 'Video',
          onTap: () => setState(() {
            _videoMode = true;
            _step = _Step.videoCapture;
          }),
        ),
        const SizedBox(width: 24),
        _ModeButton(
          icon: Icons.photo_camera,
          label: 'Stills',
          onTap: () => setState(() {
            _videoMode = false;
            _step = _Step.stillsCapture;
          }),
        ),
      ]),
    ]),
  );

  Widget _buildVideoCapture() => Center(
    child: FilledButton.icon(
      icon: const Icon(Icons.videocam),
      label: const Text('Record 360° rotation'),
      onPressed: _busy ? null : _pickAndUploadVideo,
    ),
  );

  Widget _buildStillsCapture() => Column(children: [
    const Padding(
      padding: EdgeInsets.all(16),
      child: Text('Tap each angle to capture. At least one required.'),
    ),
    Expanded(
      child: GridView.count(
        crossAxisCount: 2,
        padding: const EdgeInsets.all(16),
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        children: _angles.map((angle) {
          final done = _capturedAngles.contains(angle);
          return GestureDetector(
            onTap: _busy ? null : () => _pickAndUploadStill(angle),
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(
                  color: done ? Colors.green : Colors.grey,
                  width: 2,
                ),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(done ? Icons.check_circle : Icons.camera_alt,
                      color: done ? Colors.green : Colors.grey,
                      size: 36),
                  const SizedBox(height: 8),
                  Text(angle,
                      style: TextStyle(
                          color: done ? Colors.green : null,
                          fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    ),
    Padding(
      padding: const EdgeInsets.all(16),
      child: FilledButton(
        onPressed: (_capturedAngles.isEmpty || _busy)
            ? null
            : () => setState(() => _step = _Step.narration),
        child: const Text('Continue'),
      ),
    ),
  ]);

  Widget _buildNarration() => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      const Text('Optional: narrate the product aloud',
          style: TextStyle(fontSize: 16)),
      const SizedBox(height: 24),
      GestureDetector(
        onTap: _recording ? _stopNarrationAndProceed : _startRecordingNarration,
        child: CircleAvatar(
          radius: 40,
          backgroundColor: _recording ? Colors.red : Colors.blue,
          child: Icon(
            _recording ? Icons.stop : Icons.mic,
            color: Colors.white, size: 36,
          ),
        ),
      ),
      const SizedBox(height: 12),
      Text(_recording ? 'Tap to stop' : 'Tap to record'),
      const SizedBox(height: 24),
      TextButton(
        onPressed: _recording ? null : _finalize,
        child: const Text('Skip narration'),
      ),
    ]),
  );

  Widget _buildProcessing() => const Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      CircularProgressIndicator(),
      SizedBox(height: 16),
      Text('Analysing product…'),
    ]),
  );

  Widget _buildReview() => DraftReviewCard(
    draft: _draft!,
    onCommit: _commit,
    onCancel: () => Navigator.of(context).pop(),
  );
}

class _ModeButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ModeButton(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      CircleAvatar(radius: 32, child: Icon(icon, size: 32)),
      const SizedBox(height: 8),
      Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
    ]),
  );
}
```

- [ ] **Step 2: Run analyze**

```bash
flutter analyze lib/screens/capture_screen.dart --no-fatal-infos 2>&1 | tail -5
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add lib/screens/capture_screen.dart
git commit -m "feat: add CaptureScreen 5-step product onboarding wizard"
```

---

## Task 9: Navigation — VisionHomeScreen + drawer entry

**Files:**
- Create: `lib/screens/vision_home_screen.dart`
- Modify: `lib/widgets/chat_drawer.dart`

- [ ] **Step 1: Create VisionHomeScreen**

Create `lib/screens/vision_home_screen.dart`:

```dart
import 'package:flutter/material.dart';

import 'capture_screen.dart';
import 'deduce_screen.dart';

class VisionHomeScreen extends StatelessWidget {
  const VisionHomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Vision'),
          bottom: const TabBar(tabs: [
            Tab(icon: Icon(Icons.search), text: 'Lookup'),
            Tab(icon: Icon(Icons.camera_alt), text: 'Onboard'),
          ]),
        ),
        body: TabBarView(
          children: [
            const DeduceScreen(),
            _OnboardTab(),
          ],
        ),
      ),
    );
  }
}

class _OnboardTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      const Icon(Icons.add_a_photo, size: 64, color: Colors.grey),
      const SizedBox(height: 16),
      const Text('Scan a new product to add it\nto the cashier database',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.grey)),
      const SizedBox(height: 24),
      FilledButton.icon(
        icon: const Icon(Icons.camera_alt),
        label: const Text('Scan New Product'),
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const CaptureScreen()),
        ),
      ),
    ]),
  );
}
```

- [ ] **Step 2: Add Vision entry to ChatDrawer**

In `lib/widgets/chat_drawer.dart`:

1. Add import at the top (alongside other screen imports):
```dart
import '../screens/vision_home_screen.dart';
```

2. Add the Vision `ListTile` directly before the Transcription tile (around line 109). The existing Transcription tile looks like:

```dart
ListTile(
  leading: const Icon(Icons.mic),
  title: const Text('Transcription'),
  onTap: () {
    Navigator.pop(context);
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const TranscriptionScreen()),
    );
  },
),
```

Add this tile immediately before it:

```dart
ListTile(
  leading: const Icon(Icons.camera_enhance),
  title: const Text('Vision'),
  onTap: () {
    Navigator.pop(context);
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const VisionHomeScreen()),
    );
  },
),
```

- [ ] **Step 3: Run analyze on the full project**

```bash
flutter analyze --no-fatal-infos 2>&1 | tail -10
```

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add lib/screens/vision_home_screen.dart lib/widgets/chat_drawer.dart
git commit -m "feat: add VisionHomeScreen and Vision drawer entry"
```

---

## Task 10: Push PR

- [ ] **Step 1: Run all tests**

```bash
cd /home/bcloud/lemonade-mobile
flutter test
```

Expected: all tests pass

- [ ] **Step 2: Run full analyze**

```bash
flutter analyze --no-fatal-infos
```

Expected: no errors

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/vision-cashier
```

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --repo bong-water-water-bong/lemonade-mobile \
  --title "feat: Vision — product onboarding and deduce lookup for lemonade-cashier" \
  --body "$(cat <<'EOF'
## Summary

- Adds **VisionApiClient** — Dart client for lemonade-vision-server on port 8787, URL derived automatically from the active Lemonade server's host
- Adds **DeduceScreen** — text + voice product lookup with confidence-scored candidate list
- Adds **CaptureScreen** — 5-step product onboarding wizard (mode select → capture → narration → pipeline polling → review → commit)
- Adds **Vision** drawer entry and **VisionHomeScreen** tab bar (Lookup / Onboard)

Connectivity: Lemonade Nexus mesh replaces ngrok. No setup changes needed when both the Strix Halo and iPhone are enrolled on the same Nexus network.

Companion server-side change: `lemonade-vision-server` accepts M4A/iOS audio via ffmpeg conversion (separate PR in that repo).

## Test plan

- [ ] `flutter test` — all unit tests pass
- [ ] `flutter analyze` — no errors
- [ ] On device: open Vision drawer → Lookup tab → type a product name → see ranked candidates
- [ ] On device: Lookup tab → Voice mode → record query → see candidates
- [ ] On device: Onboard tab → Scan New Product → Video mode → record rotation → skip narration → confirm pipeline processes → review draft → commit
- [ ] On device: Onboard tab → Scan New Product → Stills mode → capture front + UPC → continue → commit

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Verify vision-server PR is also open**

The Task 2 server-side fix was committed and pushed to `lemonade-vision-server` main directly. Confirm the audio tests are passing in CI:

```bash
gh run list --repo bong-water-water-bong/lemonade-vision-server --limit 3
```

---

## Self-review checklist

| Spec requirement | Task |
|-----------------|------|
| Fork + clone lemonade-mobile | Task 1 |
| ffmpeg M4A→WAV fix in deduce.py | Task 2 |
| ffmpeg M4A→WAV fix in capture.py | Task 2 |
| Dart types (DeduceCandidate, DraftProduct, CommitRequest, etc.) | Task 3 |
| VisionApiClient.startSession / uploadVideo / uploadStill / uploadNarration / finalize | Task 4 |
| VisionApiClient.pollJob / commitProduct / deduceText / deduceAudio | Task 4 |
| visionClientProvider (port 8787 derived from selected server) | Task 5 |
| DeduceResultTile (confidence color, brand/flavor/matchReason) | Task 6 |
| DraftReviewCard (all editable fields, signal score badges) | Task 6 |
| DeduceScreen (text + voice mode, result list, empty/error states) | Task 7 |
| CaptureScreen (all 5 steps, video + stills modes, narration, processing, review) | Task 8 |
| VisionHomeScreen (Lookup/Onboard tab bar) | Task 9 |
| ChatDrawer Vision entry | Task 9 |
| PR + CI | Task 10 |
