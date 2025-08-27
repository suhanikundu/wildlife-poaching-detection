import io
import os
import time
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import streamlit as st
import requests


# -----------------------------
# UI Styling
# -----------------------------
st.set_page_config(page_title="Wildlife Poacher Detection", page_icon="🦁", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{padding-top:1rem; padding-bottom:2rem; max-width:1200px;}
    .big-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem;}
    .subtle {color:#666;}
    .badge {display:inline-block; padding:4px 10px; border-radius: 999px; background:#eef3ff; color:#223; font-weight:600; font-size:0.8rem;}
    .alert {background: #fff0f0; border: 1px solid #ffc4c4; color: #ff4d4d; padding: 12px 14px; border-radius: 8px;}
    /* High-visibility "no poacher" panel: red text on dark background */
    .ok {background: #1e2533; border: 1px solid #3a4253; color: #ff4d4d; padding: 12px 14px; border-radius: 8px;}
    .pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#f6f6f9; border:1px solid #e5e7f2; margin-right:8px; font-size:0.8rem; color:#000000;}
    .section-title {font-size:1.3rem; font-weight:700; color:#EAEFF8; margin:0.5rem 0 0.25rem}
    .panel {background:#1b2330; border:1px solid #2e3645; color:#eaeff8; padding:12px 14px; border-radius:8px;}
    .topnav {display:flex; gap:12px; margin:6px 0 12px 0;}
    .topnav a {color:#cdd7f1; text-decoration:none; font-weight:600;}
    .topnav a:hover {color:#ffffff;}
    .footer {margin-top:24px; color:#9aa3b2; font-size:0.9rem; border-top:1px solid #273142; padding-top:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Auth & Header
# -----------------------------
if 'authed' not in st.session_state:
    st.session_state.authed = False
    st.session_state.user = None


# Header bar
hdr_l, hdr_r = st.columns([6, 2])
with hdr_l:
    st.markdown("""
    <div class="big-title">🦁 Wildlife Poacher Detection</div>
    <div class="subtle">Detect a person alongside animals in images or videos. Alerts when a potential poacher is present.</div>
    """, unsafe_allow_html=True)
with hdr_r:
    if st.session_state.authed:
        st.write("")
        st.success(f"Signed in as {st.session_state.user}")
        if st.button("Sign out"):
            st.session_state.authed = False
            st.session_state.user = None
            st.rerun()
    else:
        st.write("")
        if st.button("Sign in", type="primary", key="btn_sign_in_header"):
            # Show the login section (do not auto-auth)
            st.session_state.show_login = True
            st.toast("Please sign in below", icon="🔐")


# If not authenticated, show a full login page and exit after render
if not st.session_state.authed:
    st.markdown("<div id='login'></div>", unsafe_allow_html=True)
    col_login, col_visual = st.columns([6,6])
    with col_login:
        st.markdown("<div class='section-title'>Sign in</div>", unsafe_allow_html=True)
        st.write("Enter your name and password to continue.")
        name = st.text_input("Name", key="login_name")
        password = st.text_input("Password", type="password", key="login_password")
        rem_col, btn_col = st.columns([1,1])
        with rem_col:
            remember = st.checkbox("Remember me", value=True, key="chk_remember_me")
        with btn_col:
            submit_login = st.button("Sign in", type="primary", key="btn_login_submit")
        if submit_login:
            if name.strip() and password.strip():
                st.session_state.authed = True
                st.session_state.user = name.strip()
                if remember:
                    st.session_state.remember = True
                st.toast("Signed in successfully", icon="✅")
                st.rerun()
            else:
                st.warning("Please enter both name and password.")
        st.caption("By continuing you agree to the Terms and Privacy Policy.")

    with col_visual:
        # Show requested image on the login page
        try:
            base_dir = os.path.dirname(__file__)
            img_path = os.path.join(base_dir, 'pexels-jakkel-418831.jpg')
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True, caption="Welcome")
            else:
                st.markdown("<div class='panel'>Place 'pexels-jakkel-418831.jpg' in the app folder to show it here.</div>", unsafe_allow_html=True)
        except Exception:
            st.markdown("<div class='panel'>Login image could not be loaded.</div>", unsafe_allow_html=True)

    st.stop()


# -----------------------------
# Utility: Generate a short beep and embed as autoplaying audio (client-side)
# -----------------------------
@st.cache_data
def make_beep_wav_bytes(frequency: int = 950, duration_ms: int = 250, sample_rate: int = 22050) -> bytes:
    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Hanning window to avoid clicks
    tone *= np.hanning(len(tone))
    # Convert to 16-bit PCM WAV in-memory
    audio = np.int16(tone * 32767)
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def play_beep():
    wav_bytes = make_beep_wav_bytes()
    b64 = base64.b64encode(wav_bytes).decode('ascii')
    # Use raw HTML audio with autoplay to ensure it plays without a click
    html = f"""
    <audio autoplay="true">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.components.v1.html(html, height=0)


# -----------------------------
# Detection backends
# -----------------------------
@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x, y, w, h
    conf: float
    class_id: int
    label: str


def _probe_cuda_nms() -> bool:
    try:
        import torch
        import torchvision
        if not torch.cuda.is_available():
            return False
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device='cuda')
        scores = torch.tensor([0.5], device='cuda')
        _ = torchvision.ops.nms(boxes, scores, 0.5)
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_ultralytics_model(model_name: str, prefer_cuda: bool) -> Tuple[object, List[str], str]:
    from ultralytics import YOLO
    import torch
    device = 'cuda' if (prefer_cuda and _probe_cuda_nms()) else 'cpu'
    model = YOLO(model_name)
    try:
        model.to(device)
    except Exception:
        device = 'cpu'
        try:
            model.to('cpu')
        except Exception:
            pass
    # names from model metadata
    try:
        names = [model.model.names[i] for i in range(len(model.model.names))]
    except Exception:
        names = []
    return model, names, device


def detect_ultralytics(model, image_bgr: np.ndarray, names: List[str], device: str, img_size: int, conf_thres: float) -> List[Detection]:
    # Ultralytics auto-handles preprocessing
    res = model(image_bgr, imgsz=img_size, device=device, conf=conf_thres, verbose=False)[0]
    dets: List[Detection] = []
    if getattr(res, 'boxes', None) is not None and len(res.boxes) > 0:
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = int(x2 - x1), int(y2 - y1)
            x, y = int(x1), int(y1)
            conf = float(b.conf[0].item()) if hasattr(b, 'conf') else 0.0
            cid = int(b.cls[0].item()) if hasattr(b, 'cls') else 0
            label = names[cid] if 0 <= cid < len(names) else str(cid)
            dets.append(Detection((x, y, w, h), conf, cid, label))
    return dets


@st.cache_resource(show_spinner=False)
def load_dnn(cfg_path: str, weights_path: str, names_path: str, prefer_cuda: bool):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    if prefer_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    out_names = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(names_path, 'r', encoding='utf-8') as f:
        class_names = [c.strip() for c in f if c.strip()]
    return net, out_names, class_names


def detect_dnn(net, out_names, class_names: List[str], image_bgr: np.ndarray, img_size: int, conf_thres: float, nms_thres: float) -> List[Detection]:
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 1/255.0, (img_size, img_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_names)
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= conf_thres:
                cx, cy, width, height = det[0:4] * np.array([w, h, w, h])
                x = int(cx - width / 2)
                y = int(cy - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    dets: List[Detection] = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            cid = class_ids[i]
            label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            dets.append(Detection((x, y, bw, bh), float(confidences[i]), cid, label))
    return dets


# -----------------------------
# Drawing
# -----------------------------
@st.cache_data
def _colors(n: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    return (rng.random((n, 3)) * 255).astype(np.uint8)


def draw_detections(img_bgr: np.ndarray, detections: List[Detection], names: List[str], thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
    colors = _colors(max(len(names), 1))
    out = img_bgr.copy()
    for d in detections:
        x, y, w, h = d.box
        color = colors[d.class_id % len(colors)].tolist()
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = f"{d.label} {d.conf*100:.0f}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(out, (x, y - th - base), (x + max(tw, 1), y), color, -1)
        cv2.putText(out, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, thickness-1), cv2.LINE_AA)
    return out


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Settings")
# Defaults
backend = "Ultralytics YOLOv8"
prefer_cuda = True
img_size = 608
ultra_model_name = "yolov8n.pt"
# Fixed confidence (hidden from UI)
conf_thres = 0.5

with st.sidebar.expander("Advanced settings", expanded=False):
    backend = st.selectbox("Detection backend", ["Ultralytics YOLOv8", "OpenCV DNN (Darknet)"], index=0)
    prefer_cuda = st.checkbox("Try GPU (auto-fallback)", value=True)
    img_size = st.select_slider("Model input size", options=[320, 416, 512, 608], value=608)
    if backend == "Ultralytics YOLOv8":
        ultra_model_name = st.text_input("Ultralytics model", value="yolov8n.pt")
    else:
        st.caption("Provide Darknet cfg/weights/names")
        dnn_cfg = st.text_input(".cfg path", value=str("darknet/cfg/yolov3.cfg"))
        dnn_weights = st.text_input(".weights path", value=str("data/weights/yolov3.weights"))
        dnn_names = st.text_input("coco.names path", value=str("darknet/cfg/coco.names"))
        nms_thres = st.slider("NMS threshold", 0.1, 0.8, 0.45, 0.05)

# Display controls for interactivity
with st.sidebar.expander("Display", expanded=False):
    play_beep_on_alert = st.checkbox("Play beep on alert", value=True)
    show_table = st.checkbox("Show detections table", value=True)
    box_thickness = st.slider("Box thickness", 1, 5, 2)
    font_scale = st.slider("Label font scale", 0.5, 1.2, 0.6, 0.1)


# Sidebar: alert animal classes (collapsed by default for a cleaner look)
DEFAULT_ANIMALS = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'deer', 'fox', 'lion', 'tiger', 'leopard', 'monkey'
]


# Load model
model = None
names: List[str] = []
device = "cpu"
if backend == "Ultralytics YOLOv8":
    with st.spinner("Loading Ultralytics model..."):
        model, names, device = load_ultralytics_model(ultra_model_name, prefer_cuda)
else:
    with st.spinner("Loading OpenCV DNN (Darknet) model..."):
        try:
            model, out_names, names = load_dnn(dnn_cfg, dnn_weights, dnn_names, prefer_cuda)
        except Exception as e:
            st.error(f"Failed to load Darknet model: {e}")
            st.stop()


# Animal classes selection (defaults for COCO)
available_animals = [n for n in names if n.lower() in DEFAULT_ANIMALS]
if not available_animals and names:
    # fallback: everything except 'person'
    available_animals = [n for n in names if n.lower() != 'person']
with st.sidebar.expander("Alerts: animal classes to monitor", expanded=False):
    selected_animals = st.multiselect("Animal classes", options=names or DEFAULT_ANIMALS, default=available_animals or DEFAULT_ANIMALS,
                                      help="Used for alerting and for showing 'animal' label on boxes.")
ANIMAL_NAME_SET = set(a.lower() for a in selected_animals)


# Session info (collapsed to reduce clutter)
with st.expander("Session info", expanded=False):
    col_si1, col_si2, col_si3 = st.columns(3)
    with col_si1:
        st.markdown(f"**Backend**: {backend}")
    with col_si2:
        st.markdown(f"**Device**: {device}")
    with col_si3:
        model_name_disp = ultra_model_name if backend=='Ultralytics YOLOv8' else os.path.basename(dnn_weights)
        st.markdown(f"**Model**: {model_name_disp}")

# -----------------------------
# Input mode: Image or Video (Tabs)
# -----------------------------
tab_img, tab_vid = st.tabs(["📷 Image", "🎞️ Video"])

with tab_img:
    # Anchor for nav
    st.markdown("<div id='image'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown('<div class="section-title">1) Upload an image file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_upl")
        st.write("")
        if st.button("Use demo image", key="btn_demo_img"):
            try:
                url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
                resp = requests.get(url, timeout=10)
                if resp.ok:
                    st.session_state.demo_img_bytes = resp.content
                    st.toast("Loaded demo image", icon="🖼️")
                else:
                    st.warning("Failed to fetch demo image.")
            except Exception:
                st.warning("Failed to fetch demo image.")

        def _load_image() -> Optional[np.ndarray]:
            if uploaded is None:
                # fallback to demo if available
                if 'demo_img_bytes' in st.session_state:
                    file_bytes = np.asarray(bytearray(st.session_state.demo_img_bytes), dtype=np.uint8)
                    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                return None
            data = uploaded.getvalue()
            file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img

        image_bgr = _load_image()

    # Run detection
    with col_right:
        st.markdown('<div class="section-title">2) Results</div>', unsafe_allow_html=True)
        if image_bgr is None:
            st.markdown('<div class="panel">Upload an image to start.</div>', unsafe_allow_html=True)
        else:
            if backend == "Ultralytics YOLOv8":
                dets = detect_ultralytics(model, image_bgr, names, device, img_size, conf_thres)
            else:
                dets = detect_dnn(model, out_names, names, image_bgr, img_size, conf_thres, nms_thres)

            # Map animal class names to a generic 'animal' label (keep 'person' and others unchanged)
            dets_disp = [
                Detection(d.box, d.conf, d.class_id, ('animal' if (d.label.lower() in ANIMAL_NAME_SET and d.label.lower() != 'person') else d.label))
                for d in dets
            ]

            vis = draw_detections(image_bgr, dets_disp, names, thickness=box_thickness, font_scale=font_scale)

            # Determine alert condition: person + at least one selected animal
            person_detected = any(d.label.lower() == 'person' or d.class_id == 0 for d in dets)
            animal_detected = any(d.label in selected_animals for d in dets)

            # Detection summary
            person_count = sum(1 for d in dets if (d.label.lower()=="person" or d.class_id==0))
            animal_count = sum(1 for d in dets if (d.label.lower() in ANIMAL_NAME_SET and d.label.lower()!="person"))
            st.markdown(f"<div class='pill'>Persons: <b>{person_count}</b></div> <div class='pill'>Animals: <b>{animal_count}</b></div>", unsafe_allow_html=True)

            if person_detected and animal_detected:
                st.markdown('<div class="alert"><b>🚨 Poacher Detected</b> — person and animal present in the same image.</div>', unsafe_allow_html=True)
                st.toast("Poacher Detected", icon="🚨")
                if play_beep_on_alert:
                    play_beep()
                # Overlay big alert text on the image for clarity
                cv2.putText(vis, 'Poacher Detected', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                st.markdown('<div class="ok"><b>✅ No poacher detected</b> — criteria not met.</div>', unsafe_allow_html=True)

            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"Backend: {backend} | Device: {device}")
            # Download annotated image
            ok, buf = cv2.imencode('.png', vis)
            if ok:
                st.download_button("Download annotated image", data=buf.tobytes(), file_name="annotated.png", mime="image/png", key="dl_img_annotated")

            if show_table:
                with st.expander("Detections (table)"):
                    if dets:
                        import pandas as pd
                        df = pd.DataFrame([
                            {
                                'label': dd.label,
                                'conf': round(dd.conf, 4),
                                'x': dd.box[0], 'y': dd.box[1], 'w': dd.box[2], 'h': dd.box[3]
                            }
                            for dd in dets_disp
                        ])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No detections above threshold.")

with tab_vid:
    # Anchor for nav
    st.markdown("<div id='video'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload a video file</div>', unsafe_allow_html=True)
    vfile = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="vid_upl")
    max_frames = st.slider("Max frames to process (0 = all)", 0, 2000, 300, 50)
    process_btn = st.button("Process Video", key="btn_process_video")

    if process_btn:
        if vfile is None:
            st.warning("Please upload a video file.")
            st.stop()

        # Write uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(vfile.getvalue())
            in_path = tmp_in.name

        # Prepare output temp file (MP4 with mp4v codec)
        out_path = tempfile.mktemp(suffix=".mp4")
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            os.unlink(in_path)
            st.stop()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

        progress = st.progress(0)
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        alert_any = False

        count = 0
        limit = max_frames if max_frames > 0 else total
        limit = limit if limit > 0 else 10**9
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection per frame
            if backend == "Ultralytics YOLOv8":
                dets = detect_ultralytics(model, frame, names, device, img_size, conf_thres)
            else:
                dets = detect_dnn(model, out_names, names, frame, img_size, conf_thres, nms_thres)

            # Check alert condition
            person_detected = any(d.label.lower() == 'person' or d.class_id == 0 for d in dets)
            animal_detected = any(d.label in selected_animals for d in dets)
            if person_detected and animal_detected:
                alert_any = True
                cv2.putText(frame, 'Poacher Detected', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            # Map animal class names to a generic 'animal' label for display
            dets_disp = [
                Detection(d.box, d.conf, d.class_id, ('animal' if (d.label.lower() in ANIMAL_NAME_SET and d.label.lower() != 'person') else d.label))
                for d in dets
            ]
            vis = draw_detections(frame, dets_disp, names, thickness=box_thickness, font_scale=font_scale)
            writer.write(vis)

            # Show current frame preview
            frame_placeholder.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame {count+1}")
            if total > 0:
                progress.progress(min((count + 1) / total, 1.0))

            count += 1
            if count >= limit:
                break

        cap.release()
        writer.release()
        info_placeholder.empty()
        progress.empty()

        if alert_any:
            st.markdown('<div class="alert"><b>🚨 Poacher Detected</b> — found person + animal in at least one frame.</div>', unsafe_allow_html=True)
            st.toast("Poacher Detected in video", icon="🚨")
            if play_beep_on_alert:
                play_beep()
        else:
            st.markdown('<div class="ok"><b>✅ No poacher detected</b> across processed frames.</div>', unsafe_allow_html=True)

    # Show processed video
        with open(out_path, 'rb') as f:
            video_bytes = f.read()
            st.video(video_bytes)
            st.download_button("Download processed video", data=video_bytes, file_name="annotated.mp4", mime="video/mp4", key="dl_vid_annotated")

        # Cleanup temp input (keep output while app session runs)
        try:
            os.unlink(in_path)
        except Exception:
            pass


st.caption("Tip: For best accuracy, use a model trained on your specific species. This app defaults to COCO class names when using general models.")

# Simple top navigation (anchors)
st.markdown("""
<div class="topnav">
    <a href="#image">Image</a>
    <a href="#video">Video</a>
    <a href="#about">About</a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div id="about" class="footer">
    <b>Wildlife Poacher Detection</b> • Built with Streamlit & YOLO. Improve safety by detecting people near animals in uploaded media.
</div>
""", unsafe_allow_html=True)
