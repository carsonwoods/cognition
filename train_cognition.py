import os, sys
from PIL import Image

size = 64, 64

for filename in os.listdir("training_data"):
    outfile = os.getcwd() + "/training_data/preprocessed/" + filename + ".jpg"
    print(outfile)
    try:
        im = Image.open("./training_data/" + filename)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile, "JPEG")
    except IOError:
        print("Cannot resize '%s'" % filename)
