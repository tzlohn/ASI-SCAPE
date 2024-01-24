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

def getNewPageSize(PageSize,Shift):
    Shape = [PageSize[1]+PageSize[0]*abs(Shift[0]),PageSize[2]+PageSize[0]*abs(Shift[1])]    
    return Shape

def getAssignCoordinate(shift,shape,z,nx_length,ny_length):
    x_length = shape[-2]
    y_length = shape[-1]
    if shift[0] > 0:
        offset_x = 0
        start_x = offset_x + z*shift[0]
        end_x = abs(offset_x-x_length)+z*shift[0]
    else:
        offset_x = nx_length
        end_x = offset_x + z*shift[0]
        start_x = abs(offset_x-x_length)+z*shift[0]
    
    if shift[1] > 0:
        offset_y = 0
        start_y = offset_y + z*shift[1]
        end_y = abs(offset_y-y_length)+z*shift[1]
    else:
        offset_y = ny_length
        end_y = offset_y + z*shift[1]
        start_y = abs(offset_y-y_length)+z*shift[1]
    
    return [int(start_x),int(end_x),int(start_y),int(end_y)]

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
    
    ColorImg = list()
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

def getShift(StepSize,shape):
    dim_x = shape[-1]
    dim_y = shape[-2]
    scale = 2048/dim_y
    print("scale : ",scale)
    slope_y = -16/scale # pixel/µm
    offset_y = 0.1857/scale
    slope_x = 0
    offset_x = 0
    shift_x = (StepSize*slope_x + offset_x)
    shift_y = (StepSize*slope_y + offset_y)

    return (shift_x,shift_y)

def getDataSize(shape,dtype):
    pixel = 1
    for v in shape:
        pixel = v*pixel
    match dtype:
        case "UINT16":
            digit = 2
        case "UINT8":
            digit = 1
    #return kbyte
    return pixel*digit/1024

def getSingleFileSize(DimOrder,Shape,DataSize):
    pass

def createImage(FileName,ImageShape,metadata,MetaDict,OriImageShape,NewSize):
    NewFileName = "Deskew_"+FileName+".ome.tif"
    img =TFF.memmap(NewFileName,shape = ImageShape, dtype=np.uint16, metadata = metadata, bigtiff = True)

    FileName = list()
    TiffPars = list()
    OnePar = list()
    isFirst = True   
    for tiff_data in MetaDict["tiff_data_blocks"]: #"planes" as key
        if not tiff_data["uuid"]["file_name"] in FileName:
            if len(FileName) != 0:
                TiffPars.append(OnePar)              
            FileName.append(tiff_data["uuid"]["file_name"])
            OnePar = list()
        OnePar.append(tiff_data)    
    TiffPars.append(OnePar)

    for tifname,tiffpars in zip(FileName,TiffPars):
        print(tifname)
        with TFF.TiffFile(tifname) as tif:
            DataList = list()
            for page,pars in zip(tif.pages,tiffpars):
                data = page.asarray()
                ChannelNo = pars["first_c"]
                Time = pars["first_t"]
                ZPos = pars["first_z"]
                #print(ChannelNo,Time,ZPos)
                [start_x,end_x,start_y,end_y] = getAssignCoordinate(Shift,OriImageShape,ZPos,int(NewSize[-2]),int(NewSize[-1]))
                #print([start_x,end_x,start_y,end_y])
                try:
                    img[Time,ChannelNo,ZPos,start_x:end_x,start_y:end_y] = data
                    DataList.append(data)
                except:
                    print("OME-metadata has a wrong index",ChannelNo,Time,ZPos)   
            img.flush()
            tif.close()

if __name__ == "__main__":

    isCalibrate = False

    root = tk.Tk()
    root.withdraw()
    
    CoreCount = mp.cpu_count()-1

    #ImgName = filedialog.askopenfilename()
    #"""
    print("This app is for de-shifting the image acquired by the EMBL configuration of ASI-SCAPE")
    print("Author: Tzu-Lun Ohn @EMBL-imaging centre. v0.3 02-01-24")
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


        with TFF.TiffFile(ImgName) as tif:            
            #img = TFF.memmap("test.tif",shape = tif.pages[0].shape) 
            OMEmeta = to_dict(tif.ome_metadata)
            MetaList = OMEmeta["images"]
            MetaDict = MetaList[0]
            MetaDict = MetaDict["pixels"]
            NumType = MetaDict["type"].name
            DimOrder = MetaDict["dimension_order"].name
            DimOrder = DimOrder.lower()
            OriImageShape = [MetaDict["size_t"],MetaDict["size_c"],MetaDict["size_z"],MetaDict["size_y"],MetaDict["size_x"]]
            #ImageShape.reverse()
            #ImageShape = tuple(ImageShape)

            if isCalibrate:
                img = tif.asarray()
                Shift = calibrateShift(img)
            else:
                Shift = getShift(SliceStep,OriImageShape[-3:])
                NewSize = getNewPageSize(OriImageShape[-3:],Shift)
            
            ImageShape = OriImageShape.copy()
            ImageShape[-2] = int(NewSize[-2])
            ImageShape[-1] = int(NewSize[-1])
            ImageShape = tuple(ImageShape)
            print(ImageShape)

            tags = tif.pages[0].tags#imagej_metadata
            metadata = tif.imagej_metadata
            """
            with open("OME.xml","w") as xml:
                xml.write(tags["ImageDescription"].value)
                xml.close()
            """
            tif.close()
            #print(metadata)
        ImageSize = getDataSize(ImageShape,NumType)
        try:
            meta = combineMetadata(json.loads(metadata["Info"]),StackMetadata)
            metadata["Info"] = meta
        except:
            metadata = dict()
        metadata["axes"] = "TCZYX"

        for MetaDict in MetaList:
            FileName = MetaDict["name"]
            MetaDict = MetaDict["pixels"]
            createImage(FileName,ImageShape,metadata,MetaDict,OriImageShape,NewSize)
            
            """            
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
        """
    input("finished...")
        #TFF.imwrite(NewFileName,NewImage)
    #"""
