import tifffile as TFF
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob, os

root = tk.Tk()
root.withdraw()

def assemble(ImgNames,NamePrefix,MaxProj):
    #ImgNames = self.SortWin.NewFileNames
    Order = "XYZCT"
    ChannelNo = 3
    TimePointNo = 490
    Metadict = dict()


    for idx,name in enumerate(ImgNames):
        print(name)
        Channel = int(name[-14])-1 
        Time = int(name[-11:-8])
        RealName = name
        if MaxProj:
            RealName = "MaxProj_"+RealName
        if not os.path.exists(RealName):
            continue
        with TFF.TiffFile(RealName) as tif:
            print(RealName)
            if idx == 0:
                Layers = len(tif.pages)
                DeskewShape = tif.pages[0].asarray().shape
                if MaxProj:
                    ProjOrder = Order.replace("Z","")
                    NewFileName = "MaxProj_Deskew_"+NamePrefix+".ome.tif"
                    Shape = getShapeByOrder(ProjOrder,TimePointNo,ChannelNo,Layers,DeskewShape)
                    Metadict["axes"] = ProjOrder[::-1]                    
                else:
                    NewFileName = "Deskew_"+NamePrefix+".ome.tif"    
                    Shape = getShapeByOrder(Order,TimePointNo,ChannelNo,Layers,DeskewShape)
                    Metadict["axes"] = Order[::-1]
                img = TFF.memmap(NewFileName,shape = Shape, dtype = np.uint16, metadata = Metadict, bigtiff = True)

            if MaxProj:
                img[Time,Channel,:,:] = tif.pages[0].asarray()
            else: 
                for ind,aFrame in enumerate(tif.pages):                       
                    img[Time,Channel,ind,:,:] = aFrame.asarray()

            tif.close()

        img.flush()

def getShapeByOrder(DimOrder,TimePointNo,ChannelNo,Layers,Shape2D):
    match len(DimOrder):
        case 5:
            shape = (TimePointNo,ChannelNo,Layers,Shape2D[0],Shape2D[1])
        case 4:
            if DimOrder[2] == "Z":
                match DimOrder[3]:
                    case "T":
                        shape = (TimePointNo,Layers,Shape2D[0],Shape2D[1])
                    case "C":
                        shape = (ChannelNo,Layers,Shape2D[0],Shape2D[1])
            else:
                shape = (TimePointNo,ChannelNo,Shape2D[0],Shape2D[1])
        case 3:
            match DimOrder[2]:
                case "T":
                    shape = (TimePointNo,Shape2D[0],Shape2D[1])
                case "C":
                    shape = (ChannelNo,Shape2D[0],Shape2D[1])                

    return shape

if __name__ == "__main__":
    path = filedialog.askdirectory()
    os.chdir(path)
    name = glob.glob("Deskew*.tif")
    NamePrefix = "timelapse_2min-interval_c534"
    assemble(name,NamePrefix,False)
    #assemble(name,True)