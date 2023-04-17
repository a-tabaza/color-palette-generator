import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

def OnlineImageLoader(url):
    response = requests.get(url, stream=True).raw
    image = Image.open(response) #BytesIO(response.content)
    image.show()
    image = np.array(image)
    # df = pd.DataFrame(image)
    return df

def LocalImageLoader(path):
    image = Image.open(path)
    image.show()
    image = np.array(image)
    
    df = pd.DataFrame(image[:,:,0])
    return image

# OnlineImageLoader('https://www.filmlinc.org/wp-content/uploads/2020/12/Chungking-Express-14.jpg')
LocalImageLoader('./Chungking-Express-14.jpg')

