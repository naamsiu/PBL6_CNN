# Far-face detection and chin-box fix

- `NANODET_CENTER_ZOOM=1.25`: NanoDet processes the center 80% of the camera frame at its same 512x512 input. Face coordinates are mapped back to the ORIGINAL image. This does not improve optical detail and may exclude faces near the frame edge; use `1.0` to disable it. The full frame is still used for Landmark, pose, API evidence, and displayed footage.
- `LANDMARK_CROP_SCALE=1.33`: modestly expand 256x256 landmark crop (training baseline is 1.25); `LANDMARK_CROP_Y_SHIFT=0.035` shifts it toward the chin. If landmark accuracy gets worse, revert to 1.25 and 0.0.
- `FACE_BOX_PAD_X/TOP/BOTTOM`: only enlarge the visible box (6%, 5%, 16%); *no effect* on logic or detection boxes.
- `NANODET_SCORE_THRESHOLD=0.30`: lower than 0.45 to reduce missed small faces; may cause more false face proposals.
- `EAR_MIN_WIDTH_PX=8`: tries to measure smaller eyes than the previous 10px gate. Eyes that are too small or inconsistent are still UNKNOWN, never counted as closed. Validate against actual video before trusting this safety function.
- The camera prints actual negotiated resolution and warns if below requested. `CAMERA_FOURCC=MJPG` is an OPTIONAL UVC webcam compatibility setting, left blank by default.

## Try on Raspberry Pi

```bash
python cli.py monitor --skip-auth --backend opencv --camera 0 --width 1280 --height 720
```

For cameras offering genuine 1920x1080 mode, compare:

```bash
python cli.py monitor --skip-auth --backend opencv --camera 0 --width 1920 --height 1080
```

1080p on Pi 5 may reduce FPS. Capturing 640x480 and digitally resizing to 1080p adds no real eye detail. If the camera lens is out of focus at the driver's distance, adjust physical focus or use a lens/camera with the appropriate optics. Zoomed detection is most useful when the face is roughly centered. Keep the 1.25 zoom only if it reduces missed faces in a recorded far-distance test; otherwise set 1.0.
