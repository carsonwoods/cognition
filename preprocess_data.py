"""
Cleans and preprocesses data for fitting to CNN
"""

import os, sys
from PIL import Image

size = 150, 150

for filename in os.listdir("./data/Carson_Woods/"):
    outfile = os.getcwd() + "/data/Carson_Woods_processed/" + filename
    print(outfile)
    try:
        im = Image.open("./data/Carson_Woods/" + filename)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile, "JPEG")
    except IOError:
        print("Cannot resize '%s'" % filename)
