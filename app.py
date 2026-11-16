
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Brain MRI FLAIR Segmentation",
    page_icon="🧠",
    layout="wide"
)

IMAGE_SIZE = 256
MODEL_PATH = "brain_tumor_resunet_v2.pth"

DEVICE = torch.device("cpu")


# =====================================================
# 2. MODEL ARCHITECTURE
# Must match the Kaggle training notebook exactly
# =====================================================

def norm_groups(channels):
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1, bias=False
        )

        self.norm1 = nn.GroupNorm(
            norm_groups(out_ch), out_ch
        )

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, bias=False
        )

        self.norm2 = nn.GroupNorm(
            norm_groups(out_ch), out_ch
        )

        self.act = nn.SiLU(inplace=True)

        self.drop = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):

        identity = self.skip(x)

        out = self.act(
            self.norm1(self.conv1(x))
        )

        out = self.drop(out)

        out = self.norm2(self.conv2(out))

        return self.act(out + identity)


class ResidualUNetLite(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base=16
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ResidualBlock(
            in_channels, base
        )

        self.enc2 = ResidualBlock(
            base, base * 2
        )

        self.enc3 = ResidualBlock(
            base * 2, base * 4
        )

        self.enc4 = ResidualBlock(
            base * 4, base * 8,
            dropout=0.05
        )

        # Bottleneck
        self.bottleneck = ResidualBlock(
            base * 8, base * 8,
            dropout=0.10
        )

        # Decoder 4
        self.up4 = nn.Conv2d(
            base * 8, base * 4, 1,
            bias=False
        )

        self.dec4 = ResidualBlock(
            base * 12, base * 4
        )

        # Decoder 3
        self.up3 = nn.Conv2d(
            base * 4, base * 2, 1,
            bias=False
        )

        self.dec3 = ResidualBlock(
            base * 6, base * 2
        )

        # Decoder 2
        self.up2 = nn.Conv2d(
            base * 2, base, 1,
            bias=False
        )

        self.dec2 = ResidualBlock(
            base * 3, base
        )

        # Final decoder
        self.final = ResidualBlock(
            base * 2, base
        )

        self.out = nn.Conv2d(
            base, out_channels, 1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder 4
        d4 = F.interpolate(
            b,
            size=e4.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d4 = self.dec4(
            torch.cat(
                [self.up4(d4), e4],
                dim=1
            )
        )

        # Decoder 3
        d3 = F.interpolate(
            d4,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = self.dec3(
            torch.cat(
                [self.up3(d3), e3],
                dim=1
            )
        )

        # Decoder 2
        d2 = F.interpolate(
            d3,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = self.dec2(
            torch.cat(
                [self.up2(d2), e2],
                dim=1
            )
        )

        # Final decoder
        d1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = self.final(
            torch.cat([d1, e1], dim=1)
        )

        return self.out(d1)


# =====================================================
# 3. MRI PREPROCESSING
# Must match Kaggle preprocessing
# =====================================================

def preprocess_image(image_pil, size=IMAGE_SIZE):

    image_pil = image_pil.convert("L").resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(
        image_pil,
        dtype=np.float32
    )

    valid = arr > 0

    if valid.any():

        lo, hi = np.percentile(
            arr[valid],
            [1, 99]
        )

        if hi > lo:

            arr = np.clip(arr, lo, hi)

            arr = (arr - lo) / (hi - lo)

            arr[~valid] = 0.0

        else:
            arr = arr / 255.0

    else:
        arr = arr / 255.0

    return np.clip(
        arr, 0.0, 1.0
    ).astype(np.float32)


# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ResidualUNetLite(
        in_channels=1,
        out_channels=1,
        base=16
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    size = int(
        checkpoint.get("image_size", IMAGE_SIZE)
    )

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    return model, size, threshold


# =====================================================
# 5. APPLICATION HEADER
# =====================================================

st.title(
    "🧠 Brain MRI FLAIR Segmentation"
)

st.caption(
    "Lightweight Residual U-Net | "
    "Medical Imaging Research Project"
)

st.markdown("---")

st.warning(
    "This model was trained on 2D FLAIR MRI images "
    "from the LGG dataset. Predictions on T1, "
    "T1-contrast, T2, other MRI modalities, scanners "
    "or populations may be unreliable. "
    "This is an educational research prototype, "
    "not a clinical diagnostic tool."
)


# =====================================================
# 6. MRI IMAGE UPLOAD
# =====================================================

st.subheader("Upload an MRI Image")

uploaded = st.file_uploader(
    "Upload one FLAIR MRI slice",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)


# =====================================================
# 7. SEGMENTATION
# =====================================================

if uploaded:

    try:

        # Load model
        with st.spinner("Loading trained model..."):

            model, size, default_threshold = load_model()

        # Threshold control
        threshold = st.slider(
            "Segmentation threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(default_threshold),
            step=0.05
        )

        # Load original MRI
        original = Image.open(
            uploaded
        ).convert("L")

        # Preprocess
        processed = preprocess_image(
            original,
            size
        )

        tensor = torch.from_numpy(
            processed
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(DEVICE)

        # Model inference
        with torch.inference_mode():

            logits = model(tensor)

            probability = torch.sigmoid(
                logits
            ).squeeze().cpu().numpy()

        # Generate binary mask
        small_mask = (
            probability >= threshold
        ).astype(np.uint8)

        # Resize mask to original image dimensions
        mask_image = Image.fromarray(
            small_mask * 255
        ).resize(
            original.size,
            Image.Resampling.NEAREST
        )

        mask = np.asarray(
            mask_image
        ) > 0

        # =================================================
        # 8. CREATE SEGMENTATION OVERLAY
        # =================================================

        original_rgb = np.asarray(
            original.convert("RGB")
        ).copy()

        overlay = original_rgb.copy()

        if mask.any():

            overlay[mask] = (
                0.55 * overlay[mask]
                +
                0.45 * np.array(
                    [255, 30, 30]
                )
            ).astype(np.uint8)

        # =================================================
        # 9. DISPLAY RESULTS
        # =================================================

        st.markdown("---")

        st.subheader(
            "Segmentation Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                original,
                caption="Uploaded MRI",
                use_container_width=True
            )

        with col2:

            st.image(
                mask_image,
                caption="Predicted Mask",
                use_container_width=True
            )

        with col3:

            st.image(
                overlay,
                caption="Segmentation Overlay",
                use_container_width=True
            )

        # =================================================
        # 10. PROBABILITY MAP
        # =================================================

        st.markdown("---")

        st.subheader(
            "Tumor Probability Map"
        )

        st.image(
            (probability * 255).astype(np.uint8),
            caption="Model probability map (256 × 256)",
            clamp=True,
            width=350
        )

        # =================================================
        # 11. SEGMENTATION STATISTICS
        # =================================================

        st.markdown("---")

        st.subheader(
            "Segmentation Statistics"
        )

        predicted_area = (
            100 * mask.mean()
        )

        st.metric(
            "Predicted Mask Pixel Proportion",
            f"{predicted_area:.2f}%"
        )

        st.caption(
            "This represents the proportion of image pixels "
            "classified as foreground. It is not a physical "
            "tumor size or clinical measurement."
        )

        if predicted_area > 50:

            st.warning(
                "The predicted foreground covers more than "
                "50% of the image. This may indicate "
                "over-segmentation or an unsuitable input image."
            )

        elif predicted_area == 0:

            st.info(
                "No pixels exceeded the selected threshold. "
                "This does not establish that the MRI is tumor-free."
            )

        else:

            st.success(
                "Segmentation mask generated successfully."
            )

    except FileNotFoundError:

        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Upload the trained checkpoint beside app.py "
            "in your GitHub repository."
        )

    except RuntimeError as exc:

        st.error(
            "Model loading or inference failed. "
            "Check that your Streamlit architecture matches "
            "the Kaggle training notebook."
        )

        st.code(str(exc))

    except Exception as exc:

        st.error(
            "An unexpected error occurred."
        )

        st.exception(exc)


else:

    st.info(
        "Upload a single 2D FLAIR MRI slice "
        "to generate a segmentation result."
    )

    st.markdown(
        """
        **Application workflow**

        1. Upload an MRI image.
        2. Adjust the segmentation threshold.
        3. View the predicted mask.
        4. Inspect the probability map.
        5. Review segmentation statistics.
        """
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Brain MRI Segmentation | Lightweight Residual U-Net | "
    "Educational Research Prototype"
)

st.caption(
    "Not intended for clinical diagnosis, treatment "
    "decisions or replacement of professional medical assessment."
)
