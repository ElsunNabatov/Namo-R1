
import io
import torch
from PIL import Image
import streamlit as st

# Try to import Namo
try:
    from namo.api.vl import VLInfer
except Exception as e:
    st.error("Failed to import Namo. Make sure `pip install -U namo` completed successfully.")
    st.stop()

st.set_page_config(page_title="Namo R1 — CPU Visual Chat", page_icon="🧠", layout="wide")

st.title("🧠 Namo R1 — CPU Visual Chat")
st.caption("Runs on CPU · Upload one or more images, add a prompt, and generate a response.")

with st.sidebar:
    st.header("Settings")
    # Auto-detect device, default to CPU unless CUDA is definitely available
    cuda_available = torch.cuda.is_available()
    device = st.selectbox("Device", options=["cpu"] + (["cuda:0"] if cuda_available else []), index=0)
    st.write(f"CUDA available: {'✅' if cuda_available else '❌'}")

    model_choice = st.selectbox(
        "Model",
        options=["namo", "custom path/checkpoint"],
        help="Use the default small Namo model (downloads automatically) or load from a local checkpoint path."
    )
    custom_model_path = ""
    if model_choice == "custom path/checkpoint":
        custom_model_path = st.text_input(
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
        st.error(f"Failed to load model: {e}")
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

col_a, col_b = st.columns([1, 2])
with col_a:
    max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=1024, value=256, step=16)
with col_b:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.05)

go = st.button("🚀 Generate", type="primary", disabled=(not images or not prompt.strip()))

st.divider()
st.subheader("Output")

if go:
    if not images:
        st.error("Please upload at least one image.")
        st.stop()
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()
    try:
        with st.spinner("Thinking on CPU..."):
            # VLInfer.generate accepts a single image or a list. We'll pass the list directly.
            # We also pass decoding options where supported; if unknown by your namo version, they are ignored.
            out = model.generate(
                images=images if len(images) > 1 else images[0],
                prompt=prompt.strip(),
            )
        st.write(out if isinstance(out, str) else str(out))
    except Exception as e:
        st.exception(e)

st.divider()
with st.expander("ℹ️ Tips & Setup Notes"):
    st.markdown(
        """
- **CPU by default:** This app uses CPU unless you explicitly select `cuda:0` and have a compatible GPU and drivers.
- **Install:** `pip install -U namo streamlit pillow` (Install **PyTorch** separately per your platform instructions.)
- **Multiple images:** You can upload multiple images to provide more context.
- **Custom checkpoint:** If you trained or downloaded a local checkpoint, choose *custom path/checkpoint* in the sidebar and paste the path.
- **CLI demo:** You can also try `python demo.py` from the repo for terminal chat.
        """
    )
