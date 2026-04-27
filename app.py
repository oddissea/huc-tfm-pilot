import time

import streamlit as st
import torch

st.set_page_config(page_title="HUC TFM Pilot — smoke test", page_icon="🩺")

st.title("🩺 HUC TFM Pilot — smoke test")
st.caption("Verificación de que el container Docker tiene PyTorch + CUDA + GPU passthrough.")

st.subheader("PyTorch")
st.write(f"Versión: `{torch.__version__}`")

st.subheader("CUDA")
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    st.success(
        f"CUDA disponible — runtime `{torch.version.cuda}`, "
        f"cuDNN `{torch.backends.cudnn.version()}`"
    )
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        st.write(f"**GPU {i}: {props.name}**")
        st.write(f"- VRAM total: `{props.total_memory / 1024 ** 3:.1f} GB`")
        st.write(f"- Compute capability: `{props.major}.{props.minor}`")
else:
    st.error("CUDA no disponible. Hay un problema de configuración del runtime o passthrough.")

st.subheader("Sanity check: multiplicación 4096×4096 en GPU")
if cuda_ok and st.button("Ejecutar"):
    x = torch.randn(4096, 4096, device="cuda")
    y = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    z = x @ y
    torch.cuda.synchronize()
    dt = time.time() - t0
    st.success(f"OK — {dt * 1000:.1f} ms. Norma del resultado: `{z.norm().item():.3e}`")
