import os
from PIL import Image

input_dir = "data/jpeg_redundant_f160"
output_dir = "data/jpeg_redundant_f160_scaled"

for fn in os.listdir(input_dir):
    img = Image.open("data/jpeg_redundant_f160/"+fn)
    img.thumbnail((45,45))
    img.save("data/jpeg_redundant_f160_scaled/"+fn)