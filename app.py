import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import io
import base64
import pandas as pd
import streamlit as st
from random import sample
import json
import requests
from PIL import Image, ImageColor
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import pyplot as plt



st.set_page_config(
    page_title="Color Palette Generator",
    page_icon="🎨"
)

st.title('Color Palette Generator')
st.header('Generates a color palette from an image using KMeans Clustering.')
st.caption('By: [Abdulrahman Tabaza](https://github.com/a-tabaza)')


caption = f'Afterthought: the range of colors is limited and sorted by dominance as opposed to prominence in image. Next on my list would be to try a generative adverserial network to generate a better palette.'
st.caption(caption)

@st.cache_data
def image_loader(path):
    try:
        image = Image.open(path)
        bands = list(image.getbands())
    except:
        raise Exception('Image not found, try another path')
    return image, bands

@st.cache_data
def flatten_image(image):
    df = pd.DataFrame(list(image.getdata()),columns=list(image.getbands()))
    return df

uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image, bands = image_loader(uploaded_file)
    h, w = image.size
    image = image.resize((int(h/3),int(w/3)))
    data = flatten_image(image)

@st.cache_data
def Clusterer(df, colors=8):
    df = df
    km = KMeans(n_clusters=colors,n_init='auto').fit(df)
    return km


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

@st.cache_data
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

@st.cache_data
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
    with st.spinner('Wait for it...'):
        model = Clusterer(data,N_COLORS)
        hex_palette = generate_hex_palette(model,N_COLORS)
        new_image = visualize_palette_on_image(hex_palette, image, colors=N_COLORS, mode='HEX')
        pal_image = visualize_palette(hex_palette, colors=N_COLORS, mode='HEX')

    col1, col2 = st.columns(2)

    with col1:

        st.image(new_image, caption='Image', use_column_width=True)
        st.image(pal_image, caption='Palette', use_column_width=True)
    @st.cache_data
    def generate_data(hex_palette):
        names = []
        for code in hex_palette:
            response = requests.get(f'https://www.thecolorapi.com/id?hex={code[1:]}')
            todos = json.loads(response.text)
            names.append(todos['name']['value'])    
        comb = zip(hex_palette,names)
        df = pd.DataFrame({'HEX Code':hex_palette,'Color Name':names})
        csv = df.to_csv().encode('utf-8')
        my_json = df.to_json(orient="records")
        df = df.set_index('HEX Code')
        return df, csv, my_json

    with st.spinner('Wait for it...'):
        
        df, csv, my_json = generate_data(hex_palette)
        with col2:
            st.caption('API Used for Color Names: [THECOLORAPI, Josh Beckman](https://www.thecolorapi.com)')
            st.subheader('HEX Codes and Color Names:')
            st.table(df)          

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.download_button(
            label="Download HEX Codes and Color Names as CSV",
            data=csv,
            file_name='palette.csv',
            mime='text/csv'
        )
    with col4:
        st.download_button(
            label="Download HEX Codes and Color Names as JSON",
            data=my_json,
            file_name='palette.json',
        )
    @st.cache_data
    def save_image(image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        byte = buf.getvalue()
        return byte

    new_byte = save_image(new_image)
    pal_byte = save_image(pal_image)

    with col5:
        st.download_button(
            label="Download Image with Palette as JPEG",
            data=new_byte,
            file_name="image_palette.jpeg",
            mime="image/jpeg",
        )
    with col6:
        st.download_button(
            label="Download Palette as JPEG",
            data=pal_byte,
            file_name="palette.jpeg",
            mime="image/jpeg",
        )
    
