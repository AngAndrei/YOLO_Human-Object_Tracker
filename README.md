# YOLO Human & Object Tracker

A computer vision project that streams 320 × 240 JPEG frames from a Seeed Studio XIAO ESP32-S3 Sense over USB. A Windows laptop decodes the frames with OpenCV and runs YOLO11n on its CPU. The live preview shows bounding boxes and confidence scores, and the same detector can replay saved images or videos.

| Mode | Detected classes |
| --- | --- |
| Default | Person |
| `--room-objects` | Bottle, book, cup |

Room mode detects its three classes in one inference pass per processed frame. The live pipeline uses the newest available frame to keep the preview responsive and resumes after the USB connection is restored.

## Quick start

Open PowerShell in the repository root. Install Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[detection]'
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -OutFile models/yolo11n.pt
```

To run inference without the board, substitute your own image or video path.

```powershell
.\.venv\Scripts\python.exe -m human_tracker replay --source sample.jpg --room-objects --headless --output annotated.png
```

Drop `--room-objects` to detect people. The CLI defaults to a 320-pixel inference size and a confidence cutoff of 0.35. Use `--imgsz`, `--confidence`, and `--model` to override those settings.

## Connect the XIAO camera

1. Open [the camera sketch](firmware/xiao_camera/xiao_camera.ino) in Arduino IDE with the ESP32 Arduino board package installed.
2. Select **XIAO_ESP32S3**, **USB Mode → Hardware CDC and JTAG**, **USB CDC On Boot → Enabled**, and **PSRAM → OPI PSRAM**. Upload the sketch. If upload stalls while connecting, hold **BOOT**, tap **RESET**, release **BOOT**, and select the bootloader port before uploading again.
3. Connect the board with a USB data cable. Find its runtime COM port in Windows Device Manager and close the Arduino Serial Monitor. Replace `COM7` in the following commands with that port.

```powershell
# Verify the camera feed
.\.venv\Scripts\python.exe -m human_tracker --port COM7

# Run one detector at a time after closing the camera preview
.\.venv\Scripts\python.exe -m human_tracker live --source serial:COM7
.\.venv\Scripts\python.exe -m human_tracker live --source serial:COM7 --room-objects

# Measure processed FPS and laptop receipt-to-preview latency
.\.venv\Scripts\python.exe -m human_tracker live --source serial:COM7 --duration 300
```

Focus the preview and press **Q** or **Esc** to close it. The camera sends frames as the four ASCII bytes `XIAO`, a four-byte little-endian JPEG length, and the JPEG payload. The receiver limits each payload to 128 KiB.

## Dataset and evaluation tools

### Person detection and fine-tuning

```powershell
.\.venv\Scripts\python.exe -m human_tracker capture --port COM7 --output datasets/my-study --session room-01 --count 30
.\.venv\Scripts\python.exe -m human_tracker dataset label datasets/my-study
.\.venv\Scripts\python.exe -m human_tracker dataset validate datasets/my-study
```

During capture, press **Space** to save a raw frame or pass `--timed` for automatic capture every two seconds. In the labeler, drag a box around each person and press **Enter** to save or **E** to confirm an empty frame. Assign a distinct session ID to each independent recording.

After collecting and labeling more sessions, export a YOLO dataset. Reserve at least two independent sessions for the test split and replace the example session IDs with your own. Export keeps each recording session in one split and checks that all splits contain positive and negative frames.

```powershell
.\.venv\Scripts\python.exe -m human_tracker dataset export datasets/my-study --output datasets/my-study-v1.zip --test-session test-daylight --test-session test-backlit
```

The [Colab notebook](notebooks/xiao_phase4_colab.ipynb) records the original 360-frame person study and two fine-tuning recipes. Recipe A disables stochastic augmentation. Recipe B uses horizontal flips, HSV changes, translation, and scaling. The notebook verifies the original study archive and starting weights by SHA-256. To rerun that experiment, supply the original `xiao-v1.zip` in the notebook's input folder and use a Colab GPU runtime. The local comparison commands `python -m human_tracker evaluate validate` and `python -m human_tracker evaluate test` also need the executed notebook and its exported A and B checkpoints.

### Room-object evaluation

Capture separate validation and test folds under `datasets/xiao-room/val` and `datasets/xiao-room/test`. Use independent session IDs, including at least two sessions per fold. Tag frames at the intended operating distance with `--tag demo` during capture and include target-free clutter scenes. Aim for at least 20 labeled instances per class at the demo distance and 20 target-free frames per fold.

```powershell
.\.venv\Scripts\python.exe -m human_tracker dataset label datasets/xiao-room/val --room-objects
.\.venv\Scripts\python.exe -m human_tracker room-evaluate inspect
.\.venv\Scripts\python.exe -m human_tracker room-evaluate validate
.\.venv\Scripts\python.exe -m human_tracker dataset label datasets/xiao-room/test --room-objects
.\.venv\Scripts\python.exe -m human_tracker room-evaluate test
```

In the room labeler, press **1** for bottle, **2** for book, or **3** for cup before drawing a box. Validation selects a shared confidence cutoff. The test run uses that cutoff and reports per-class precision, recall, AP50, and the rate of target-free frames with a false positive. Reports go to `build/phase8/` by default.

## Results

In a five-minute live **person detection** run, the laptop processed **16.56 FPS** with **111.4 ms p95 latency** from laptop receipt to preview. On 60 held-out person-study frames, pretrained YOLO11n had the highest mAP50–95 and was retained for deployment. The augmented recipe detected more people at its validation-selected threshold.

| Person model | Test AP50 | Test mAP50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Pretrained YOLO11n | 80.3% | 76.3% | 91.2% | 58.5% |
| Fine-tuned A | 71.4% | 45.6% | 65.9% | 50.9% |
| Fine-tuned B | 79.0% | 39.7% | 75.0% | 73.6% |

See the [validation report](reports/phase5/validation.json) and [held-out test report](reports/phase5/test.json) for the exact settings and thresholds.

In the **room-object** live demo, all three object boxes appeared together and each cleared when its object was removed. A separate 52-frame room validation run at 320 pixels measured the following results at a shared cutoff of 0.537 and IoU 0.50.

| Object | AP50 | Demo-distance recall |
| --- | ---: | ---: |
| Bottle | 92.2% | 21/24 (87.5%) |
| Book | 67.5% | 5/24 (20.8%) |
| Cup | 95.9% | 19/24 (79.2%) |

One of 26 target-free frames produced a false positive. Book recall fell below the 80% demo target, as did cup recall. Held-out room accuracy and timed live room performance are still pending.

## Repository layout

| Path | Purpose |
| --- | --- |
| [firmware/](firmware/) | ESP32-S3 camera firmware |
| [src/human_tracker/](src/human_tracker/) | USB capture, live and replay inference, dataset tools, and evaluation |
| [notebooks/](notebooks/) | Colab fine-tuning notebook and its generator |
| [reports/phase5/](reports/phase5/) | Aggregate person model comparison results |
| [tests/](tests/) | Automated tests |

Run the tests from the repository root after installing the package.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
