import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain MRI FLAIR Segmentation",
    page_icon="🧠",
    layout="wide"
)

IMAGE_SIZE = 256

def norm_groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)

class ResidualUNetLite(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ResidualBlock(in_channels, base)
        self.enc2 = ResidualBlock(base, base*2)
        self.enc3 = ResidualBlock(base*2, base*4)
        self.enc4 = ResidualBlock(base*4, base*8, dropout=0.05)
        self.bottleneck = ResidualBlock(base*8, base*8, dropout=0.10)
        self.up4, self.dec4 = nn.Conv2d(base*8, base*4, 1), ResidualBlock(base*12, base*4)
        self.up3, self.dec3 = nn.Conv2d(base*4, base*2, 1), ResidualBlock(base*6, base*2)
        self.up2, self.dec2 = nn.Conv2d(base*2, base, 1), ResidualBlock(base*3, base)
        self.final = ResidualBlock(base*2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([self.up4(d4), e4], dim=1))
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(d3), e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.final(torch.cat([d1, e1], dim=1))
        return self.out(d1)

def preprocess_image(image_pil, size=IMAGE_SIZE):
    image_pil = image_pil.convert("L").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(image_pil, dtype=np.float32)
    valid = arr > 0
    if valid.any():
        lo, hi = np.percentile(arr[valid], [1, 99])
        if hi > lo:
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo)
            arr[~valid] = 0.0
        else:
            arr = arr / 255.0
    else:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "brain_tumor_resunet_v2.pth",
        map_location="cpu",
        weights_only=False
    )
    model = ResidualUNetLite(base=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    size = int(checkpoint.get("image_size", IMAGE_SIZE))
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, size, threshold

st.title("🧠 Brain MRI FLAIR Abnormality Segmentation")
st.caption("Lightweight Residual U-Net • Research/education demonstration")
st.warning(
    "This model was trained on 2D FLAIR images from the LGG dataset. "
    "Predictions on T1, T1-contrast, T2, other modalities, scanners or populations "
    "may be unreliable. This is not a diagnostic medical device."
)

uploaded = st.file_uploader(
    "Upload one FLAIR MRI slice",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded:
    try:
        model, size, default_threshold = load_model()
        threshold = st.slider(
            "Segmentation threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(default_threshold),
            step=0.05
        )

        original = Image.open(uploaded).convert("L")
        processed = preprocess_image(original, size)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            probability = torch.sigmoid(model(tensor)).squeeze().numpy()

        small_mask = (probability >= threshold).astype(np.uint8)
        mask_image = Image.fromarray(small_mask * 255).resize(
            original.size, Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image) > 0

        original_rgb = np.asarray(original.convert("RGB")).copy()
        overlay = original_rgb.copy()
        if mask.any():
            overlay[mask] = (
                0.55 * overlay[mask] + 0.45 * np.array([255, 30, 30])
            ).astype(np.uint8)

        c1, c2, c3 = st.columns(3)
        c1.image(original, caption="Uploaded MRI", use_container_width=True)
        c2.image(mask_image, caption="Predicted mask", use_container_width=True)
        c3.image(overlay, caption="Segmentation overlay", use_container_width=True)

        st.image(
            (probability * 255).astype(np.uint8),
            caption="Model probability map (256 × 256)",
            clamp=True,
            width=350
        )
        st.metric("Predicted mask pixel proportion", f"{100 * mask.mean():.2f}%")
        st.caption(
            "This is a pixel proportion, not a physical tumor size or clinical measurement. "
            "A high predicted area can indicate model failure."
        )
    except FileNotFoundError:
        st.error(
            "Model file not found. Upload brain_tumor_resunet_v2.pth "
            "beside app.py in your GitHub repository."
        )
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Upload a single 2D FLAIR MRI slice to generate a segmentation overlay.")
