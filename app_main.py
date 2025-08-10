
import io
import sys
import traceback
from typing import Optional

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Namo R1 — CPU Visual Chat", page_icon="🧠", layout="wide")
st.title("🧠 Namo R1 — CPU Visual Chat")
st.caption("Runs on CPU · Upload one or more images, add a prompt, and generate a response.")

# --- Environment diagnostics (so we can see what's missing) ------------------
import importlib

def has(pkg: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        ver = md.version(pkg)
        return ver
    except Exception:
        return None

with st.expander("Environment diagnostics", expanded=True):
    st.write({
        "python": sys.version.split()[0],
        "transformers": has("transformers"),
        "accelerate": has("accelerate"),
        "safetensors": has("safetensors"),
        "tokenizers": has("tokenizers"),
        "sentencepiece": has("sentencepiece"),
        "torch": has("torch"),
        "pillow": has("pillow"),
        "namo (PyPI)": has("namo"),
    })
    st.caption("Note: You do NOT need the PyPI package 'namo' if the repo includes a local 'namo/' folder.")

# --- Try importing local Namo first ------------------------------------------
VLInfer = None
_import_err = None
try:
    # Prefer local package (repo code). If the repo is mounted as the working dir,
    # Python should find `namo/` automatically. We also explicitly add cwd to path.
    if "" not in sys.path:
        sys.path.insert(0, "")
    if "." not in sys.path:
        sys.path.insert(0, ".")
    from namo.api.vl import VLInfer  # type: ignore
except Exception as e:
    _import_err = e

if VLInfer is None:
    st.error("❌ Could not import `from namo.api.vl import VLInfer` from the local repo.")
    with st.expander("Show full import error"):
        st.code("".join(traceback.format_exception(type(_import_err), _import_err, _import_err.__traceback__)))
    st.stop()

# --- If we got here, import torch and continue -------------------------------
import torch

st.sidebar.header("Settings")
cuda_available = torch.cuda.is_available()
device = st.sidebar.selectbox("Device", options=["cpu"] + (["cuda:0"] if cuda_available else []), index=0)
st.sidebar.write(f"CUDA available: {'✅' if cuda_available else '❌'}")

model_choice = st.sidebar.selectbox(
    "Model",
    options=["namo", "custom path/checkpoint"],
    help="Use the default small Namo model (downloads automatically) or load from a local checkpoint path."
)
custom_model_path = ""
if model_choice == "custom path/checkpoint":
    custom_model_path = st.sidebar.text_input(
        "Checkpoint path or model name",
        placeholder="e.g., checkpoints/Namo-500M-V1",
        help="Enter a local path or model id if supported."
    )

@st.cache_resource(show_spinner=True)
def load_model(_model_type: str, _device: str):
    return VLInfer(model_type=_model_type, device=_device)

model_type = custom_model_path.strip() if custom_model_path.strip() else "namo"
try:
    model = load_model(model_type, device)
    st.success(f"Model loaded: {model_type} on {device}")
except Exception as e:
    st.error("Failed while constructing VLInfer (likely missing transformers/torch weights).")
    with st.expander("Show full model load error"):
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

st.subheader("1) Upload image(s)")
files = st.file_uploader(
    "Drag & drop images or click to browse",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True
)

images = []
if files:
    cols = st.columns(min(3, len(files)))
    for i, f in enumerate(files):
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images.append(img)
            with cols[i % len(cols)]:
                st.image(img, caption=f.name, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not open {f.name}: {e}")

st.subheader("2) Write your prompt")
prompt = st.text_area("Prompt", value="What is in this image?", height=100)

go = st.button("🚀 Generate", type="primary", disabled=(not images or not prompt.strip()))

st.divider()
st.subheader("Output")

if go:
    try:
        with st.spinner("Thinking on CPU..."):
            out = model.generate(
                images=images if len(images) > 1 else images[0],
                prompt=prompt.strip(),
            )
        st.write(out if isinstance(out, str) else str(out))
    except Exception as e:
        st.error("Generation error.")
        st.exception(e)

st.divider()
with st.expander("Setup notes"):
    st.markdown(
        """
### Requirements
Add this to your **requirements.txt** (do **not** include `namo` here):
```
streamlit>=1.36
transformers>=4.48.0
accelerate>=0.34.0
safetensors>=0.4.5
pillow>=10.0.0
tokenizers>=0.20.0
sentencepiece>=0.2.0
```
If you see wheel issues on Python 3.13 in hosted environments, try Python **3.11**.
        """
    )
