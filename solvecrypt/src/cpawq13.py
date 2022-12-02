# %%
from PIL import Image
import numpy as np 

# %%
filepath = "my own folder"
img = Image.open(filepath)

# %%
print(img.size)
width, height = img.size

# %%
image_array = np.empty((height, width), dtype=float)
print(image_array)