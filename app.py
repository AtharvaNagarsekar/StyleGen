import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import base64
LATENT_DIM = 100
IMAGE_SIZE = 128
IMG_CHANNELS = 3
CONDITION_CHANNELS = 1
CHECKPOINT_PATH = 'fashion_cgan_checkpoint_hfds.pth'
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_channels, img_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.latent_proj = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.enc1 = nn.Conv2d(condition_channels, 64, 4, 2, 1, bias=False)
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128))
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256))
        self.enc4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512))
        self.enc5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512))

        self.dec1 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(512 + 512, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.Dropout(0.5))
        self.dec2 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(512 + 512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.Dropout(0.5))
        self.dec3 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128))
        self.dec4 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64))
        self.dec5 = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(64 + 64, img_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, condition, noise):
        latent_feat = self.latent_proj(noise)
        e1 = self.enc1(condition);
        e2 = self.enc2(e1);
        e3 = self.enc3(e2);
        e4 = self.enc4(e3);
        e5 = self.enc5(e4)

        combined = torch.cat([e5, latent_feat], 1)

        d1 = self.dec1(combined); d1 = torch.cat([d1, e4], 1)
        d2 = self.dec2(d1); d2 = torch.cat([d2, e3], 1)
        d3 = self.dec3(d2); d3 = torch.cat([d3, e2], 1)
        d4 = self.dec4(d3); d4 = torch.cat([d4, e1], 1)
        output = self.dec5(d4)
        return output
mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=CONDITION_CHANNELS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * CONDITION_CHANNELS, std=[0.5] * CONDITION_CHANNELS),
])

def denormalize_img(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    denormalized_tensor = tensor * std + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    return denormalized_tensor

@st.cache_resource
def load_generator_model(checkpoint_path, latent_dim, condition_channels, img_channels):
    device = torch.device("cpu")
    model = Generator(latent_dim, condition_channels, img_channels).to(device)

    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint not found at {checkpoint_path}. Please ensure the file is in the correct directory.")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except KeyError:
        st.error(f"Could not find 'generator_state_dict' in the checkpoint file. Please check the key used during model saving.")
        return None
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None

def load_mask_image(filepath):
    if not os.path.exists(filepath):
        st.warning(f"Default mask image not found at {filepath}. Please ensure it's in the same directory as app.py.")
        return None
    try:
        mask_img = Image.open(filepath).convert('L').convert('RGB')
        return mask_img
    except Exception as e:
        st.error(f"Error loading default mask image {filepath}: {e}")
        return None

default_masks = {
    "Mask Image 1": load_mask_image("m1.png"),
    "Mask Image 2": load_mask_image("m2.png"),
    "Mask Image 3": load_mask_image("m3.png"),
    "Mask Image 4": load_mask_image("m4.png"),
    "Mask Image 5": load_mask_image("m5.png"),
    "Mask Image 6": load_mask_image("m6.png"),
    "Mask Image 7": load_mask_image("m7.png"),
}
default_masks = {name: img for name, img in default_masks.items() if img is not None}
st.set_page_config(layout="wide", page_title="StyleGen: Fashion Generator", page_icon="👗")

st.sidebar.title("StyleGen Info")
st.sidebar.subheader("About")
st.sidebar.markdown("""
    StyleGen is a demonstration of a Conditional Generative Adversarial Network (CGAN)
    for generating fashion outfits based on simple mask inputs.
""")
st.sidebar.subheader("How to Use")
st.sidebar.markdown("""
1.  **Choose a Mask:** Select a default mask image or upload your own image (white shape on black background) in the main area.
2.  **Generate:** The application will automatically generate a fashion outfit corresponding to the shape.
3.  **Explore:** Try different masks to see the variety of generated styles!
""")

st.title("✨ StyleGen: Fashion Generator ✨")
st.markdown("""
    Upload a mask image (white shape on a black background) or select a default mask,
    and let StyleGen generate a unique fashion outfit for you!
""")

st.subheader("Select or Upload Your Mask")
mask_option_main = st.radio("Choose your mask source:", ("Upload Custom Mask", "Use Default Mask"))

uploaded_file = None
selected_default_mask = None

if mask_option_main == "Upload Custom Mask":
    uploaded_file = st.file_uploader("Upload your mask image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload a mask image.")
else:
    if not default_masks:
        st.warning("No default mask images found or loaded. Please upload a custom mask.")
    else:
        default_mask_name = st.selectbox("Select a default mask image:", list(default_masks.keys()))
        selected_default_mask = default_masks[default_mask_name]

generator_model = load_generator_model(CHECKPOINT_PATH, LATENT_DIM, CONDITION_CHANNELS, IMG_CHANNELS)

if generator_model is not None:
    mask_image = None
    if uploaded_file is not None:
        try:
            mask_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            uploaded_file = None

    elif selected_default_mask is not None:
         mask_image = selected_default_mask.copy() 

    if mask_image is not None:
        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Input Mask")
                st.image(mask_image, caption="Mask", use_column_width=True)
            mask_tensor = mask_transform(mask_image).unsqueeze(0)
            noise = torch.randn(1, LATENT_DIM, 1, 1)

            mask_tensor = mask_tensor.to("cpu")
            noise = noise.to("cpu")

            with torch.no_grad():
                generated_image_tensor = generator_model(mask_tensor, noise)

            generated_image_pil = transforms.ToPILImage()(denormalize_img(generated_image_tensor).squeeze(0))

            with col2:
                st.subheader("Generated Outfit")
                st.image(generated_image_pil, caption="Generated", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during image processing or generation: {e}")
            st.warning("Please try selecting or uploading the mask again.")

elif generator_model is None:
    st.warning("Generator model could not be loaded. Please ensure the checkpoint file exists and is correct.")


st.markdown("---")
with st.expander("Understanding the Technology"):
    st.markdown("""
    This application is powered by a **Conditional Generative Adversarial Network (CGAN)**.

    * **GANs** consist of two neural networks: a Generator and a Discriminator. The Generator tries to create realistic data (fashion images), while the Discriminator tries to distinguish between real and generated data. They are trained in a competitive game.
    * **Conditional GANs (CGANs)** extend this by adding conditional information (in this case, the mask image) to both the Generator and Discriminator. This allows the Generator to produce outputs that are guided by the input condition, enabling it to generate outfits that fit the provided shape.

    The Generator model takes the mask image and a random noise vector as input. The mask provides the structural outline, and the noise vector allows for generating diverse styles for the same mask. The output is a synthesized image of a fashion outfit.
    """)

st.markdown("<p style='text-align: center;'>Designed by Atharva</p>", unsafe_allow_html=True)
