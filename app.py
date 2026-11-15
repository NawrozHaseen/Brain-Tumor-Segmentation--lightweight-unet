import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain MRI Tumor Segmentation", page_icon="🧠", layout="wide")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool=nn.MaxPool2d(2)
        self.e1,self.e2,self.e3=ConvBlock(1,8),ConvBlock(8,16),ConvBlock(16,32)
        self.b=ConvBlock(32,64)
        self.u3,self.d3=nn.ConvTranspose2d(64,32,2,2),ConvBlock(64,32)
        self.u2,self.d2=nn.ConvTranspose2d(32,16,2,2),ConvBlock(32,16)
        self.u1,self.d1=nn.ConvTranspose2d(16,8,2,2),ConvBlock(16,8)
        self.out=nn.Conv2d(8,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2))
        b=self.b(self.pool(e3))
        d3=self.d3(torch.cat([self.u3(b),e3],1))
        d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)

@st.cache_resource
def load_model():
    model=TinyUNet()
    checkpoint=torch.load("brain_tumor_unet.pth",map_location="cpu",weights_only=True)
    if isinstance(checkpoint,dict) and "model_state_dict" in checkpoint:
        state=checkpoint["model_state_dict"]
        image_size=int(checkpoint.get("image_size",256))
        default_threshold=float(checkpoint.get("threshold",0.5))
    else:
        state=checkpoint
        image_size,default_threshold=256,0.5
    model.load_state_dict(state)
    model.eval()
    return model,image_size,default_threshold

st.title("🧠 Brain MRI Tumor Segmentation")
st.caption("Tiny U-Net • LGG FLAIR MRI • Educational research demonstration")
st.warning(
    "Not a medical device. This model is not clinically validated and must not be used "
    "to diagnose, rule out, or guide treatment of any condition."
)

uploaded=st.file_uploader("Upload an MRI slice",type=["png","jpg","jpeg","tif","tiff"])
if uploaded:
    try:
        model,size,default_threshold=load_model()
        threshold=st.slider("Segmentation threshold",0.10,0.90,float(default_threshold),0.05)
        original=Image.open(uploaded).convert("L")
        resized=original.resize((size,size),Image.Resampling.BILINEAR)
        arr=np.asarray(resized,dtype=np.float32)/255.0
        tensor=torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            prob=torch.sigmoid(model(tensor)).squeeze().numpy()

        mask_small=(prob>=threshold).astype(np.uint8)
        mask_img=Image.fromarray(mask_small*255).resize(original.size,Image.Resampling.NEAREST)
        mask=np.asarray(mask_img)>0

        rgb=np.asarray(original.convert("RGB")).copy()
        overlay=rgb.copy()
        if mask.any():
            overlay[mask]=(0.55*overlay[mask]+0.45*np.array([255,30,30])).astype(np.uint8)

        c1,c2,c3=st.columns(3)
        c1.image(original,caption="Uploaded MRI",use_container_width=True)
        c2.image(mask_img,caption="Predicted mask",use_container_width=True)
        c3.image(overlay,caption="Segmentation overlay",use_container_width=True)

        st.image((prob*255).astype(np.uint8),caption="Model probability map",clamp=True,width=350)
        st.metric("Predicted mask pixel proportion",f"{100*mask.mean():.2f}%")
        st.caption("Pixel proportion is not a physical tumor size or a clinical measurement.")
    except FileNotFoundError:
        st.error("Upload brain_tumor_unet.pth beside app.py in your GitHub repository.")
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Upload a single MRI slice to generate a segmentation overlay.")
