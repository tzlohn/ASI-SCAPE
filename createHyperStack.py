import tifffile as TFF
import numpy as np
import os,glob
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

dir = filedialog.askdirectory()
os.chdir(dir)
files = glob.glob("*.tif*")

img = TFF.imread(files[0])

output = TFF.memmap("D:\\OpenSPIM.tif",shape=(1+len(files)//89,89,img.shape[0],img.shape[1]), imagej = True, dtype=np.uint16, metadata = {"axes":"TZYX"}, bigtiff = True)
for idx,aFile in enumerate(files):
    t = idx//89
    z = idx%89
    print(t,z)
    img = TFF.imread(aFile)
    output[t,z,:,:] = img
    output.flush()

