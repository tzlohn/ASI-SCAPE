import tifffile as TFF
import numpy as np
import tkinter as tk
from tkinter import filedialog


if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename()

    with TFF.TiffFile(path) as tif:
        for idx in range(0,len(tif.pages),2):
            if idx == 0:
                img1 =TFF.memmap("Channel1.tif",shape = (193,1152,850), dtype=np.uint16, bigtiff = True)
            if idx == 2:
                img2 = TFF.memmap("Channel2.tif",shape = (193,1152,850), dtype=np.uint16, bigtiff = True)
            ImgIdx = idx//4
            if idx%4 == 0:
                if ImgIdx == 0:
                    img1[0,:,:] = tif.pages[ImgIdx*4].asarray()
                else:
                    img1[ImgIdx*2-1,:,:] = tif.pages[ImgIdx*4-1].asarray()
                    img1[ImgIdx*2,:,:] = tif.pages[ImgIdx*4].asarray()
                img1.flush()
            elif idx%4 == 2:
                img2[ImgIdx*2,:,:] = tif.pages[ImgIdx*4+1].asarray()
                img2[ImgIdx*2+1,:,:] = tif.pages[ImgIdx*4+2].asarray()          
                img2.flush()  