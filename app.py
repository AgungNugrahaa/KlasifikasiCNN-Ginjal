
import os, io
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# ===== KONFIG =====
MODEL_DIR = "/Users/agungnugraha/Desktop/untitled folder 2/content/save_model"  # folder berisi saved_model.pb
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
IMG_SIZE = 244
THUMB_WIDTH = 300  # ukuran preview di kiri 

st.set_page_config(page_title="Klasifikasi CT Ginjal — CNN", page_icon=None, layout="centered")
st.title("Klasifikasi CT Ginjal — CNN")

st.markdown("""
<style>
/* Sembunyikan box preview file yang muncul setelah upload */
.stFileUploader div[data-testid="stFileUploaderFile"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# CSS: label hijau tebal, confidence putih (badge gelap)
st.markdown("""
<style>
.pred-label { font-size: 2.2rem; font-weight: 800; color: #16a34a; margin: 0.2rem 0; }
.pred-conf  { font-size: 1.1rem; font-weight: 700; color: #ffffff;
              background:#1f2937; padding: 6px 12px; border-radius: 9999px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ===== State untuk reset uploader =====
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ===== Util SavedModel =====
def _ensure_model_dir(path: str) -> str:
    if os.path.isfile(path) and os.path.basename(path) == "saved_model.pb":
        return os.path.dirname(path)
    return path

def _is_saved_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb"))

@st.cache_resource
def load_saved_model(model_dir: str):
    model_dir = _ensure_model_dir(model_dir)
    if not _is_saved_model_dir(model_dir):
        raise FileNotFoundError(f"Folder SavedModel tidak valid: '{model_dir}'")
    loaded = tf.saved_model.load(model_dir)
    sigs = getattr(loaded, "signatures", {})
    if "serving_default" in sigs:
        infer = sigs["serving_default"]
    elif len(sigs) > 0:
        infer = next(iter(sigs.values()))
    else:
        raise RuntimeError("Signature inferensi tidak ditemukan.")
  
    _, kw = infer.structured_input_signature
    input_key = list(kw.keys())[0] if isinstance(kw, dict) and len(kw) >= 1 else None
    return infer, input_key

def preprocess_image(pil_img, img_size):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.resize((img_size, img_size))
    arr = np.asarray(pil_img).astype("float32") / 255.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)  
    return arr

def run_inference(infer_fn, input_key, batch_np):
    x = tf.convert_to_tensor(batch_np)
    try:
        out = infer_fn(**({input_key: x} if input_key is not None else {})) if input_key is not None else infer_fn(x)
    except TypeError:
        out = infer_fn(x) if input_key is not None else infer_fn(inputs=x)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return out.numpy()  # (N, num_classes), sudah softmax

# ===== Load model sekali =====
try:
    infer_fn, input_key = load_saved_model(MODEL_DIR)
except Exception as e:
    st.error(f"Gagal memuat SavedModel: {e}")
    st.stop()

# ===== Uploader single file
upload = st.file_uploader(
    label="", type=["jpg","jpeg","png"],
    accept_multiple_files=False, label_visibility="collapsed",
    key=f"uploader_{st.session_state.uploader_key}"
)

if not upload:
    st.info("Unggah satu gambar untuk memulai.")
else:
    # Baca PIL & prediksi one-shot
    upload.seek(0)
    pil_img = Image.open(io.BytesIO(upload.read()))
    x = preprocess_image(pil_img, IMG_SIZE)
    probs = run_inference(infer_fn, input_key, x)[0].astype(float)
    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])

    # Layout: kiri gambar kecil, kanan hasil besar
    col_img, col_pred = st.columns(2, gap="large")
    with col_img:
        st.image(pil_img, caption=upload.name, width=THUMB_WIDTH)
        # Tombol reset tepat di bawah gambar: kosongkan uploader & ulang dari awal
        if st.button("Reset", use_container_width=True):
            st.session_state.uploader_key += 1  
            try:
                st.rerun()  
            except Exception:
                st.experimental_rerun()  
    with col_pred:
        st.markdown(f"<div class='pred-label'>{CLASS_NAMES[top_idx]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pred-conf'>Confidence: {conf*100:.1f}%</div>", unsafe_allow_html=True)
