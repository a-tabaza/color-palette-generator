import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import streamlit as st
from random import sample
import requests
from PIL import Image, ImageColor
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import pyplot as plt



st.set_page_config(
    page_title="Color Palette Generator",
    page_icon="🎨",
    layout="wide",
)

st.title('Color Palette Generator')
st.header('Generates a color palette from an image using KMeans Clustering.')
st.subheader('The range of generated colors is not what I want and I would like to try deep GANs next.')


def image_loader(path):
    try:
        image = Image.open(path)
        bands = list(image.getbands())
    except:
        raise Exception('Image not found, try another path')
    return image, bands

def flatten_image(image):
    df = pd.DataFrame(list(image.getdata()),columns=list(image.getbands()))
    return df

uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image, bands = image_loader(uploaded_file)
    h, w = image.size
    image = image.resize((int(h/3),int(w/3)))
    data = flatten_image(image)
    
class Clusterer():
    def __init__(self, df, colors=8,km=True,mbkm=False):
        self.df = df
        self.models = {}
        self.km = km
        self.mbkm = mbkm

        if km:
            self.models['KMeans'] = KMeans(n_clusters=colors,n_init='auto').fit(self.df)
        if mbkm:
            self.models['MiniBatchKMeans'] = MiniBatchKMeans(n_clusters=colors,n_init='auto').fit(self.df)

    def kmeans_model(self):
        if self.km:
            return self.models['KMeans']
    
    def minibatchkmeans_model(self):
        if self.mbkm:
            return self.models['MiniBatchKMeans']
    
    def kmeans_clusters(self):
        if self.km:
            return self.models['KMeans'].predict(self.df)
    
    def minibatchkmeans_clusters(self):
        if self.mbkm:
            return self.models['MiniBatchKMeans'].predict(self.df)

def generate_rgb_palette(model,colors,norm=False):
    centers = model.cluster_centers_.astype(int)
    luminance = [0.2126*r + 0.7152*g + 0.0722*b for r, g, b in centers]
    sorted_indices = np.argsort(luminance)
    centers_sorted = centers[sorted_indices]

    if norm:
        palette = centers_sorted[:colors] / 255.0
    else:
        palette = centers_sorted[:colors].round().astype(int)
    
    palette_tuple = [tuple(p) for p in palette]
    return palette_tuple

def rgb_to_hex(rgb_value):
    r, g, b = rgb_value[0], rgb_value[1], rgb_value[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def generate_hex_palette(model, colors):
    centers = model.cluster_centers_.astype(int)
    luminance = [0.2126*r + 0.7152*g + 0.0722*b for r, g, b in centers]
    sorted_indices = np.argsort(luminance)
    centers_sorted = centers[sorted_indices]
    palette = centers_sorted[:colors].round().astype(int)
    palette = [rgb_to_hex(p) for p in palette]
    return palette

def hex_to_rgb(hex_value):
    h = hex_value.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

def ensemble_palettes(models,colors):
    rgb_palette = np.mean([generate_rgb_palette(model,colors) for model in models], axis=0).astype(int)
    hex_palette = [rgb_to_hex(p) for p in rgb_palette]
    rgb_palette = [tuple(p) for p in rgb_palette]
    return rgb_palette, hex_palette

def visualize_palette_on_image(palette, image, colors, mode='RGB',reversed=False):
    if reversed:
        palette = palette[::-1]

    if mode not in ['RGB', 'HEX']:
        raise ValueError("mode must be either 'RGB' or 'HEX'")
    
    new_width, _ = image.size
    size = (len(palette), 1)
    palette_img = Image.new('RGB', size, color='white')
    
    pixels = palette_img.load()
    for i in range(len(palette)):
        if mode == 'RGB':
            pixels[i, 0] = palette[i]
        elif mode == 'HEX':
            pixels[i, 0] = ImageColor.getrgb(palette[i])
    
    palette_image = np.asarray(palette_img.resize((new_width, int(new_width/colors)), 
                                                resample= Image.Resampling.NEAREST))
    image = np.asarray(image)

    assert palette_image.shape[1] == image.shape[1], "Image dimension mismatch"
    assert palette_image.shape[2] == image.shape[2], "Image dimension mismatch"
    concat_image = np.concatenate((image,palette_image), axis=0)
    new_image = Image.fromarray(concat_image)
    
    return new_image

def visualize_palette(palette, colors, mode='RGB',reversed=False):
    if reversed:
        palette = palette[::-1]

    if mode not in ['RGB', 'HEX']:
        raise ValueError("mode must be either 'RGB' or 'HEX'")
    
    size = (len(palette), 1)
    palette_img = Image.new('RGB', size, color='white')
    
    pixels = palette_img.load()
    
    for i in range(len(palette)):
        if mode == 'RGB':
            pixels[i, 0] = palette[i]
        elif mode == 'HEX':
            pixels[i, 0] = ImageColor.getrgb(palette[i])
    
    palette_image = palette_img.resize((1024, int(1024/N_COLORS)), resample= Image.Resampling.NEAREST)
    #palette_image = palette_image.rotate(90, expand=True)
    
    return palette_image

if uploaded_file is not None:
    N_COLORS = st.slider('Colors', 1, 16, 8)
    model = Clusterer(data,N_COLORS,km=True)
    km = model.kmeans_model()
    #mbkm = model.minibatchkmeans_model()
    #models = [km,mbkm]
    hex_palette = generate_hex_palette(km,N_COLORS)
    # rgb_palette, hex_palette = ensemble_palettes(models,N_COLORS)
    pal_image = visualize_palette(hex_palette, colors=N_COLORS, mode='HEX')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    with col2:
        st.image(pal_image, caption='Palette', use_column_width=True, width=64)
        # st.text(f'RGB Palette:\n{" ".join(rgb_palette)}')
        st.header(f'HEX Codes:')
        st.subheader(f'{"  ".join(hex_palette)}')
