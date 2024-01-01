import numpy as np
from scipy.fftpack import fft2,fftshift,ifft2
import tifffile as TFF
import tkinter as tk
from tkinter import filedialog,simpledialog
import multiprocessing as mp
from multiprocessing import get_context
import json,os,glob
from ome_types import to_dict, from_tiff,OME


def calcShift(Frame1,Frame2):
    Ff1 = fft2(Frame1)
    Ff2 = fft2(np.flip(np.flip(Frame2,axis=0),axis=1))
    CorrImg = np.abs(fftshift(ifft2((np.multiply(Ff1,Ff2)))))
    MaxPos = np.where(CorrImg == np.max(CorrImg))
  
    return [1+MaxPos[0][0]-Ff1.shape[0]/2,1+MaxPos[1][0]-Ff1.shape[1]/2]

def calcContrast(img):
    return np.round(np.std(img),decimals=2)

def findIdealShift(z,img):
    i = 0
    shift_result = [0,0]
    """
    while shift_result[0] == 0 and shift_result[1] == 0 and z+i < img.shape[0]-1: 
        i = i+1
        shift_result = calcShift(img[z,:,:],img[z+i,:,:])
    """
    result_x = list()
    result_y = list()
    for i in range(5):
        if z+i+1 == img.shape[0]:
            break
        shift_result = calcShift(img[z,:,:],img[z+i+1,:,:])
        result_x.append(shift_result[0]/(i+1))
        result_y.append(shift_result[1]/(i+1))
        print(i+1,[np.round(shift_result[0]/(i+1),decimals = 2),np.round(shift_result[1]/(i+1),decimals = 2)])
    shift_result[0] = np.round(np.mean(result_x),decimals = 1)
    shift_result[1] = np.round(np.mean(result_y),decimals = 1)
    return shift_result

def findFolderName(string):
    n = -1
    while string[n] != "/":
        n=n-1
    pos = len(string)+n
    return [string[0:pos],string[pos+1:]]

def formNewImage(RawImage,shift):
    match len(RawImage.shape):
        case 3:
            idx_offset = 0
            Shape = [RawImage.shape[0],RawImage.shape[1]+RawImage.shape[0]*abs(shift[0]),RawImage.shape[2]+RawImage.shape[0]*abs(shift[1])]
        case 4:
            idx_offset = 1
            Shape = [RawImage.shape[0],RawImage.shape[1],RawImage.shape[2]+RawImage.shape[1]*abs(shift[0]),RawImage.shape[3]+RawImage.shape[1]*abs(shift[1])]
    NewImage = np.zeros(shape = Shape,dtype=RawImage.dtype)
    layer = RawImage.shape[0+idx_offset]
    x_length = RawImage.shape[1+idx_offset]
    y_length = RawImage.shape[2+idx_offset]
    for z in range(layer):
        if shift[0] > 0:
            offset_x = 0
            start_x = offset_x + z*shift[0]
            end_x = abs(offset_x-x_length)+z*shift[0]
        else:
            offset_x = NewImage.shape[1+idx_offset]
            end_x = offset_x + z*shift[0]
            start_x = abs(offset_x-x_length)+z*shift[0]
        
        if shift[0] > 0:
            offset_y = 0
            start_y = offset_y + z*shift[1]
            end_y = abs(offset_y-y_length)+z*shift[1]
        else:
            offset_y = NewImage.shape[2+idx_offset]
            end_y = offset_y + z*shift[1]
            start_y = abs(offset_y-y_length)+z*shift[1]
        try:
            match len(RawImage.shape):
                case 3:
                    NewImage[z,start_x:end_x,start_y:end_y] = RawImage[z,:,:]
                case 4:
                    NewImage[:,z,start_x:end_x,start_y:end_y] = RawImage[:,z,:,:]    
        except:
            #pass
            print(z,start_x,end_x,start_y,end_y)
        
    return NewImage

def getSliceStep(FolderName):
    FileName = FolderName + "/AcqSettings.txt"

    with open(FileName) as file:
        metadata = json.load(file)
        file.close()
    
    return metadata

def combineMetadata(meta1,meta2):
    meta1.update(meta2)
    meta = str(meta1)
    
    return meta.replace(",","\n")

def calibrateShift(img):
    UpperLayer = simpledialog.askinteger("z range","please enter the first layer for deskewing:\nstart from 1,end at %d"%img.shape[0])-1
    LowerLayer = simpledialog.askinteger("z range","please enter the last layer for deskewing: \nend at %d"%img.shape[0])-1
    
    if img.ndim == 4:
        for n in range(img.shape[1]):
            ColorImg.append(img[UpperLayer:LowerLayer+1,n,:,:])
    else:
        ColorImg.append(img[UpperLayer:LowerLayer+1,:,:])
    
    AllShift_x = list()
    AllShift_y = list()

    ShiftByChannel = list()

    for idx,AImage in enumerate(ColorImg):        
        PoolInput = [(AImage[z,:,:]) for z in range(AImage.shape[0])]
        with get_context("spawn").Pool(processes=4) as pool:
            contrast = pool.map(calcContrast,PoolInput)
            pool.close()
            pool.join()
        
        GoodImageIdx = contrast.index(max(contrast))
        print("Image plane %d has the best contrast"%(GoodImageIdx+1))

        Shift = findIdealShift(GoodImageIdx,AImage)
        print(SliceStep," um: ",Shift)
        Shift[0] = int(Shift[0])
        Shift[1] = int(Shift[1])
        """
        PoolInput = [(AImage[z,:,:],AImage[z+shift_offset,:,:]) for z in range(AImage.shape[0]-shift_offset)]
        with get_context("spawn").Pool(processes=2) as pool:
            Shift = pool.starmap(calcShift,PoolInput)
            pool.close()
            pool.join()
        
        shift_x_list = [shift[0] for shift in Shift]
        #shift_x_list = [i for i in shift_x_list if i != 0]
        shift_y_list = [shift[1] for shift in Shift]
        #shift_y_list = [i for i in shift_y_list if i != 0]
        
        shift_x = int(np.round(np.median(shift_x_list)/shift_offset))
        shift_y = int(np.round(np.median(shift_y_list)/shift_offset))
        print(shift_x_list,shift_x)
        print(shift_y_list,shift_y)
        """
        #shift_x = int(np.round(Shift[0]/shift_offset))
        #shift_y = int(np.round(Shift[1]/shift_offset))

        ShiftByChannel.append(Shift)
    return ShiftByChannel

def getShift(StepSize,img):
    dim_x = img.shape[-1]
    dim_y = img.shape[-2]
    scale = 2048/dim_y
    print("scale : ",scale)
    slope_y = -16/scale # pixel/µm
    offset_y = 0.1857/scale
    slope_x = 0
    offset_x = 0
    shift_x = (StepSize*slope_x + offset_x)
    shift_y = (StepSize*slope_y + offset_y)

    return (shift_x,shift_y)

if __name__ == "__main__":

    isCalibrate = False

    root = tk.Tk()
    root.withdraw()
    
    CoreCount = mp.cpu_count()-1

    #ImgName = filedialog.askopenfilename()
    #"""
    print("This app is for de-shifting the image acquired by the EMBL configuration of ASI-SCAPE")
    print("Author: Tzu-Lun Ohn @EMBL-imaging centre. v0.2 06-12-23")
    FolderName = filedialog.askdirectory(title="please select the folder containing the image and metadata files")
    try:
        AllFolder = os.listdir(FolderName)
    except:
        FolderName = findFolderName(FolderName)
        FolderName = FolderName[0]
        AllFolder = os.listdir(FolderName)
    #for aFolder in AllFolder:
    for _ in range(1):
        os.chdir(FolderName)
        #os.chdir(aFolder)
        cwd = os.getcwd()
        AllTif = glob.glob("*.tif")
        if len(AllTif) > 1:
            for atif in AllTif:
                if atif[0:6] == "Deskew":
                    os.remove(atif)

        iName = AllTif[0]
        ImgName = cwd+"/"+iName
        print(ImgName)
    #"""
    #for i in range(1):
        FileName = findFolderName(ImgName)
        try:
            StackMetadata = getSliceStep(FileName[0])
            SliceStep = StackMetadata["stepSizeUm"]
            ChannelNo = StackMetadata["numChannels"]
            SliceNo = StackMetadata["numSlices"]
            #print("Find slice step",SliceStep)
            #print(StackMetadata)
        except:
            SliceStep = 1.1
            StackMetadata = {"info":"No AcqSettings.txt"}
            print("no AcqSettings.txt found")
        #UpperLayer = simpledialog.askinteger("z range","please enter the first layer for deskewing:\nstart from 1")-1
        #LowerLayer = simpledialog.askinteger("z range","please enter the last layer for deskewing:")-1

        ome = OME()
        """
        OmeXml = from_tiff(ImgName)
        pixels = OmeXml.images[0].pixels
        for p in pixels:
            try:
                if len(p[1])>3:
                    for q in p[1]:
                        print(q)
            except:
                continue
        
        DataDict = to_dict(OmeXml)
        
        for p in DataDict:
            d = DataDict[p]
            c = d[0]
            for b in c:
                if b != "pixels":
                    print(p,b,":",c[b])
                else:
                    for q in c[b]:
                        if q == "tiff_data_blocks":
                            for n in c[b][q]:
                                print(n)
        """
        with TFF.TiffFile(ImgName) as tif:
            #img = tif.asarray()
            #img = TFF.memmap("test.tif",shape = tif.pages[0].shape) 
            OMEmeta = to_dict(tif.ome_metadata)
            MetaList = OMEmeta["images"]
            MetaDict = MetaList[-1]
            MetaDict = MetaDict["pixels"]
            DimOrder = MetaDict["dimension_order"].name
            DimOrder = DimOrder.lower()
            ImageShape = (MetaDict["size_"+DimOrder[0]],MetaDict["size_"+DimOrder[1]],MetaDict["size_"+DimOrder[2]],MetaDict["size_"+DimOrder[3]],MetaDict["size_"+DimOrder[4]])
            print(ImageShape)
            """
            for key in metadict["images"]:
                if type(key) is dict:
                    print(type(key["pixels"]))
                    for a in key["pixels"]:
                        print(a)
                        #print(a,":",key[a])
            """
            n = 0
            for aPage in tif.pages:
                n = n+1

            tags = tif.pages[0].tags#imagej_metadata
            metadata = tif.imagej_metadata

            #print(tags)
            #print(metadata)
            #series = tif.series[0]            
            tif.close()
            with open("OME.xml","w") as xml:
                xml.write(tags["ImageDescription"].value)
                xml.close()
            #print(metadata)
        meta = combineMetadata(json.loads(metadata["Info"]),StackMetadata)
        metadata["Info"] = meta
        
        ColorImg = list()
        #print(img.shape)
        match img.ndim:
            case 4:
                metadata["axes"] = "ZYX"
                for n in range(img.shape[1]):
                    ColorImg.append(img[:,n,:,:])
            case 5:
                metadata["axes"] = "TZYX"
                for n in range(img.shape[2]):
                    ColorImg.append(img[0:10,:,n,:,:])                
            case other:
                if ChannelNo == 1:
                    ColorImg.append(img)
                else:
                    for idx in range(ChannelNo):
                        CurrentChannel = list()
                        for ind in range(0,img.shape[0],ChannelNo):
                            CurrentChannel.append(ind+idx)
                        ColorImg.append(np.asarray(CurrentChannel))
        
        if isCalibrate:
            Shift = calibrateShift(img)
        else:
            Shift = getShift(SliceStep,img)

        if os.path.isdir("Deskew"):
            print("there are old deskewed images, please delete them for the new processing")
            input("finished...")
            exit()
        else:
            os.mkdir("Deskew")
            os.chdir("Deskew")
        
        print("pixel size: %.3f µm"%(0.097))
        
        for idx,AImage in enumerate(ColorImg):
            shift_x = int(np.round(Shift[0]))
            shift_y = int(np.round(Shift[1]))
            #NewFileName = FileName[0]+"/"+"Deskew_"+FileName[1][0:-4]+"_Channel"+str(idx+1)+".tif"
            NewFileName = "Deskew_"+FileName[1][0:-4]+"_Channel"+str(idx+1)+".tif"
            NewImage = formNewImage(AImage,[shift_x,shift_y])
            print("Writing channel",idx+1)
            TFF.imwrite(NewFileName,NewImage,dtype = np.uint16,resolution=(1/0.097,1/0.097),metadata = metadata,ome = True, bigtiff=True)
            #TFF.memmap(NewFileName,NewImage,shape=NewImage.shape,dtype = np.uint16,resolution=(1/0.097,1/0.097),metadata = metadata,ome = True, bigtiff=True)
    
    input("finished...")
        #TFF.imwrite(NewFileName,NewImage)
    #"""
