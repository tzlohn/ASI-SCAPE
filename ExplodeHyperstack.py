from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tifffile as TFF
import numpy as np
import sys,os,glob,json

#HyperOrder = ["XYZC","XYZT","XYZCT","XYZTC","XYCZ","XYTZ"]
HyperOrder = ["XYZCT","XYCZT","XYZTC"]

def sortName(Names:list):
    #This function solves the file name sorting result with *_1.tif,*_10.tif,...*_19.tif,*_2.tif,*_20.tif
    NewList = list()
    for idx,aName in enumerate(Names):
        if idx == 0:
            NewList.append(aName)
            continue
        if len(aName) < len(NewList[-1]):
            n = len(NewList)-1
            while len(NewList[n]) > len(aName) and n > 0:
                n = n-1
            NewList.insert(n+1,aName)
        else:
            NewList.append(aName)
    
    return NewList

class SortWorker(QObject):
    def __init__(self,parent):
        super().__init__()
        self.SortWin = parent
    

    def checkDoneTif(self,AllTif):
        NewDoneTif = list()
        for atif in AllTif:
            if "MMStack" in atif:
                continue
            else:
                NewDoneTif.append(atif)
        return NewDoneTif
    
    def breakdown(self):
        NameRE = self.SortWin.NamePrefix+"*_MMStack_" + "*ome.tif"
        AllTif = glob.glob(NameRE)
        AllTif = sortName(AllTif)

        NewFileNames = dict()
        OutputImgs = dict()

        ZLayerNo = self.SortWin.ZLayerNo.value()
        HyperOrder = self.SortWin.HyperStackOrder.currentText()
        ChannelNo = self.SortWin.ColorChannels.value()
        TimePnts = self.SortWin.TimePointNo.value()
        if HyperOrder[3] == "Z":
            StackNo = ChannelNo*ZLayerNo
            MultiplexCount = ChannelNo
        else:
            StackNo = ZLayerNo
            MultiplexCount = 1
        NameTemplate = self.SortWin.NamePrefix
        Remainder = 0
        FileExistingWarning = False
        #if self.checkDeskewExisting(ChannelNo,TimePnts,self.SortWin.NamePrefix):
        self.SortWin.MainWin.sig_progress.emit(1)
        for idx,aTiff in enumerate(AllTif):
            with TFF.TiffFile(aTiff) as tif:
                for ind,page in enumerate(tif.pages):
                    data = page.asarray()
                    if data.shape[0] > 0:
                        RealPageNo = Remainder%StackNo//MultiplexCount
                        [SNstr,SN] = self.SortWin.getNamePost(Remainder)
                        FileName = NameTemplate+"_"+SNstr+".ome.tif"
                        if not FileName in OutputImgs:
                            NewFileNames[FileName] = SN
                            if os.path.exists(FileName):
                                Remainder = Remainder+1
                                if not FileExistingWarning:
                                    FileExistingWarning = True
                                print("%s exists"%FileName)
                                    #self.SortWin.showFileExistingWarningBox()
                                continue
                            OutputImgs[FileName] = TFF.memmap(FileName,shape=(ZLayerNo,data.shape[0],data.shape[1]), dtype=np.uint16, metadata = {"axes":"ZYX"}, bigtiff = True)
                            #OutputImgs[FileName] = TFF.memmap(FileName,shape=(ZLayerNo,200,data.shape[1]), dtype=np.uint16, metadata = {"axes":"ZYX"}, bigtiff = True)
                        
                        OutputImgs[FileName][RealPageNo,:,:] = data
                        #OutputImgs[FileName][RealPageNo,:,:] = data[900:1100,:]
                        if RealPageNo == ZLayerNo-1:
                            OutputImgs[FileName].flush()
                        Remainder = Remainder+1
                    else:
                        print(ind)                
                tif.close()
            self.SortWin.MainWin.sig_progress.emit(int(round(100*(idx+1)/len(AllTif))))
        #print(NewFileNames)
        self.SortWin.MainWin.sig_openDeskew.emit(NewFileNames)

    def checkDeskewExisting(self,ChannelNo,TimePnt,prefix):
        Name = "Deskew_"+prefix+"*.tif"
        DeskewTiffs = glob.glob(Name)
        print (len(DeskewTiffs),ChannelNo*TimePnt)
        if len(DeskewTiffs) == ChannelNo*TimePnt:
            return True
        else:
            return False

class DeskewWorker(QObject):
    def __init__(self,parent):
        super().__init__()
        self.ShiftWin = parent
    
    def createImage(self):
        ImgNames = self.pars["ImgNames"]
        ImageShape = self.pars["ImageShape"]
        metadata = self.pars["metadata"]
        OriImageShape = self.pars["OriImageShape"]
        NewSize = self.pars["NewSize"]
        Shift = self.pars["Shift"]
        for idx,FileName in enumerate(ImgNames):
            print(FileName)
            if "_MMStack" in FileName:
                continue

            NewFileName = "Deskew_"+FileName
            if os.path.exists(NewFileName):
                if self.ShiftWin.MaxProj.isChecked():
                    MaxProjName = "MaxProj_"+NewFileName
                    if os.path.exists(MaxProjName):
                        continue
                    else:
                        img = TFF.imread(NewFileName)
                        try:
                            TFF.imwrite(MaxProjName,np.max(img,axis = 0))
                            print("Max Project %s..."%NewFileName)
                        except:
                            print(img.shape)
                else: 
                    continue
                
            print("Deskewing %s..."%FileName)
            img =TFF.memmap(NewFileName,shape = ImageShape, dtype=np.uint16, metadata = metadata, bigtiff = True)
                
            with TFF.TiffFile(FileName) as tif:
                for ZPos,page in enumerate(tif.pages):
                    data = page.asarray()
                    [start_x,end_x,start_y,end_y] = self.getAssignCoordinate(Shift,OriImageShape,ZPos,int(NewSize[-2]),int(NewSize[-1]))
                    img[ZPos,start_x+1:end_x-1,start_y+1:end_y-1] = data[1:-1,1:-1]
                tif.close()

            if self.ShiftWin.MaxProj.isChecked():
                MaxProjName = "MaxProj_"+NewFileName
                TFF.imwrite(MaxProjName,np.max(img,axis = 0))
            
            self.ShiftWin.UIwin.MainWin.sig_progress.emit(int(round(100*(idx+1)/len(ImgNames))))

        self.ShiftWin.UIwin.MainWin.sig_openHyper.emit()
        
    def setParameters(self,pars):
        self.pars = dict()
        self.pars["ImgNames"] = pars[0]
        self.pars["ImageShape"] = pars[1]
        self.pars["metadata"] = pars[2]
        self.pars["OriImageShape"] = pars[3]
        self.pars["NewSize"] = pars[4]
        self.pars["Shift"] = pars[5]

    def getAssignCoordinate(self,shift,shape,z,nx_length,ny_length):
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

class HyperstackWorker(QObject):
    def __init__(self,parent):
        super().__init__()
        self.HyperWin = parent
        self.SortWin = self.HyperWin.SortWin
    
    def assembleHyperstack(self):
        MaxProj = self.HyperWin.MaxProjCB.isChecked()
        ImgNames = self.SortWin.NewFileNames
        DimDict = self.SortWin.DimOrderDict
        Order = self.HyperWin.HyperstackType.currentText()
        ChannelNo = self.SortWin.ColorChannels.value()
        TimePointNo = self.SortWin.TimePointNo.value()
        Metadict = dict()

        self.SortWin.MainWin.sig_progress.emit(0)
        for idx,name in enumerate(ImgNames):
            Channel = DimDict[name][0] 
            Time = DimDict[name][1]
            RealName = "Deskew_"+name
            if MaxProj:
                RealName = "MaxProj_"+RealName

            with TFF.TiffFile(RealName) as tif:
                if idx == 0:
                    Layers = len(tif.pages)
                    DeskewShape = tif.pages[0].asarray().shape
                    if MaxProj:
                        ProjOrder = Order.replace("Z","")
                        NewFileName = "MaxProj_Deskew_"+self.SortWin.NamePrefix+".ome.tif"
                        Shape = self.getShapeByOrder(ProjOrder,TimePointNo,ChannelNo,Layers,DeskewShape)
                        Metadict["axes"] = ProjOrder[::-1]                    
                    else:
                        NewFileName = "Deskew_"+self.SortWin.NamePrefix+".ome.tif"    
                        Shape = self.getShapeByOrder(Order,TimePointNo,ChannelNo,Layers,DeskewShape)
                        Metadict["axes"] = Order[::-1]
                    img = TFF.memmap(NewFileName,shape = Shape, dtype = np.uint16, metadata = Metadict, bigtiff = True)

                if MaxProj:
                    img[Time,Channel,:,:] = tif.pages[0].asarray()
                else: 
                    for ind,aFrame in enumerate(tif.pages):                       
                        img[Time,Channel,ind,:,:] = aFrame.asarray()

                tif.close()

            if self.HyperWin.DeleteCB.isChecked():
                os.remove(RealName)
            
            self.SortWin.MainWin.sig_progress.emit(int(round(100*(idx+1)/len(ImgNames))))
            img.flush()


    def getShapeByOrder(self,DimOrder,TimePointNo,ChannelNo,Layers,Shape2D):
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

class Hyperstack(QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.DeskewWin = parent
        self.SortWin = parent.UIwin
        self.MainWin = self.SortWin.MainWin

        self.setTitle("Hyperstack")

        self.TypeLabel = QLabel(parent = self, text = "Hyperstack type:")
        self.HyperstackType = QComboBox(self)
        self.HyperstackType.addItems(["XYZCT"])

        self.MaxProjCB = QCheckBox(self)
        self.MaxProjCB.setText("Stacking max projection images")
        if not self.DeskewWin.MaxProj.isChecked():
            self.MaxProjCB.setDisabled(True)
            
        self.DeleteCB = QCheckBox(self)
        self.DeleteCB.setText("Delete source images")

        self.CreateHyperstack = QPushButton(self)
        self.CreateHyperstack.setText("Create hyperstack image")
        self.CreateHyperstack.clicked.connect(self.createHyperstack)

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.TypeLabel,0,0,1,4)
        self.Layout.addWidget(self.HyperstackType,1,0,1,4)
        self.Layout.addWidget(self.MaxProjCB,2,0,1,4)
        self.Layout.addWidget(self.DeleteCB,3,0,1,4)
        self.Layout.addWidget(self.CreateHyperstack,4,0,1,4)

        self.HyperThread = QThread(self)
        self.HyperWorker = HyperstackWorker(self)
        self.HyperWorker.moveToThread(self.HyperThread)
        self.HyperThread.started.connect(self.HyperWorker.assembleHyperstack)
    
    def createHyperstack(self):
        self.DeskewWin.Deskew.setDisabled(True)

        if self.MaxProjCB.isChecked():
            print("Hyperstacking Max Projection images...")
        else:
            print("Hyperstacking 3D images...")

        if self.HyperThread.isRunning():
            return False
        
        self.HyperThread.start()
        self.HyperThread.quit()
      
class BackShift(QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.UIwin = parent
        self.UIwin.MainWin.sig_openHyper.connect(self.openHyperCreator)

        self.setTitle("Deskew")

        self.SliceStepLabel = QLabel(parent = self, text= "Slice Step:")
        self.SliceStep = QDoubleSpinBox(self)
        self.SliceStep.setMinimum(0)
        self.SliceStep.setDecimals(3)
        self.SliceStep.setSingleStep(0.01)
        """
        if self.UIwin.SliceStep is False:
            self.SliceStep.setValue(0)
        else:
            self.SliceStep.setValue(self.UIwin.SliceStep)
        """
        self.SliceStepCB = QCheckBox(self)
        self.SliceStepCB.stateChanged.connect(self.checkState)

        self.BinningLabel = QLabel(parent = self, text= "Binning:")
        self.Binning = QSpinBox(self)
        self.Binning.setMinimum(1)
        self.Binning.setMaximum(2)
        self.Binning.setSingleStep(1)
        self.Binning.setValue(1)

        self.SlopeLabel = QLabel(parent = self, text = "Step size to angle ratio:")
        self.Slope = QComboBox(self)
        self.Slope.addItems(["70µm/°","45µm/°"])

        self.CamRotateBox = QCheckBox(self)
        self.CamRotateBox.setText("Was the camera rotated (2024 configuration)?")

        self.selectImageLabel = QLabel(parent = self, text = "Select a file for deskew:")
        self.selectImage = QComboBox(self)
        #self.selectImage.addItems(self.UIwin.NewFileNames)

        self.SelectAll = QCheckBox(self)
        self.SelectAll.setText("select all images")
        self.SelectAll.stateChanged.connect(self.setSelectState)

        self.MaxProj = QCheckBox(self)
        self.MaxProj.setText("Max projection") 

        self.Deskew = QPushButton(self)
        self.Deskew.setText("Deskew")
        self.Deskew.clicked.connect(self.deskewImages)
        self.Deskew.setDisabled(True)

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.SliceStepLabel,0,0,1,2)
        self.Layout.addWidget(self.SliceStep,0,2,1,2)
        self.Layout.addWidget(self.SliceStepCB,0,4,1,1)
        self.Layout.addWidget(self.BinningLabel,1,0,1,2)
        self.Layout.addWidget(self.Binning,1,2,1,2)
        self.Layout.addWidget(self.SlopeLabel,2,0,1,2)
        self.Layout.addWidget(self.Slope,2,2,1,2)
        self.Layout.addWidget(self.CamRotateBox,3,0,1,4)
        self.Layout.addWidget(self.selectImageLabel,4,0,1,2)
        self.Layout.addWidget(self.selectImage,5,0,1,4)
        self.Layout.addWidget(self.SelectAll,6,0,1,4)
        self.Layout.addWidget(self.MaxProj,7,0,1,4)
        self.Layout.addWidget(self.Deskew,8,0,1,4)

        self.setLayout(self.Layout)

        self.DeskewThread = QThread(self)
        self.DeskewWorker = DeskewWorker(self)
        self.DeskewWorker.moveToThread(self.DeskewThread)
        self.DeskewThread.started.connect(self.DeskewWorker.createImage)

    def setSelectState(self):
        if self.SelectAll.isChecked():
            self.selectImage.setDisabled(True)
        else:
            self.selectImage.setEnabled(True)
    
    def checkState(self):
        if self.HyperGroup.HyperThread.isRunning():
            return False

        if self.SliceStepCB.isChecked():
            self.Deskew.setEnabled(True)
        else:
            self.Deskew.setDisabled(True)
    
    def deskewImages(self):
        if self.DeskewThread.isRunning():
            MsgBox = QMessageBox(self)
            MsgBox.setWindowTitle("Deskewing")
            MsgBox.setText("A deskewing is undergoing")
            MsgBox.setIcon(QMessageBox.Warning)
            MsgBox.show()
            return False
        self.UIwin.SortHyperStack.setDisabled(True)
        SliceStep = self.UIwin.SliceStep

        if not self.SelectAll.isChecked():
            ImgNames = [self.selectImage.currentText()]
        else:
            ImgNames = self.UIwin.NewFileNames
        
        ImgName = ImgNames[0]
        with TFF.TiffFile(ImgName) as tif:                               
            PageShape = tif.pages[0].asarray().shape
            OriImageShape = [len(tif.pages),PageShape[0],PageShape[1]]
            metadata = dict()
            metadata["axes"] = "ZYX"

            Shift = self.getShift(SliceStep,OriImageShape)
            NewSize = self.getNewPageSize(OriImageShape,Shift)
        
            ImageShape = OriImageShape.copy()
            ImageShape[-2] = int(NewSize[-2])
            ImageShape[-1] = int(NewSize[-1])
            ImageShape = tuple(ImageShape)
            print(ImageShape)
            tif.close()

        self.DeskewWorker.setParameters([ImgNames,ImageShape,metadata,OriImageShape,NewSize,Shift])        

        self.DeskewThread.start()
        self.DeskewThread.quit()
    
    def getNewPageSize(self,PageSize,Shift):
        Shape = [PageSize[1]+PageSize[0]*abs(Shift[0]),PageSize[2]+PageSize[0]*abs(Shift[1])]    
        return Shape

    def getShift(self,StepSize,shape, isRescale = True):
        """
        isRotate is True for all images acquried after 2024, before 2024 is false
        """
        match(self.Binning.value()):
            case 1:
                isBinning = False
            case 2:
                isBinning = True

        match(self.Slope.currentText()):
            case "70µm/°":
                isRescale = True
            case "45µm/°":
                isRescale = False

        dim_x = shape[-1]
        dim_y = shape[-2]
        scale = 2048/dim_y

        if isBinning:
            scale = 2
        else:
            scale = 1

        if isRescale:
            rescaleFactor = 0.4
        else:
            rescaleFactor = 1

        #print("scale : ",scale)
        slope_y = -16*rescaleFactor/scale # pixel/µm
        offset_y = 0.1857/scale
        slope_x = 0
        offset_x = 0
        shift_x = (StepSize*slope_x + offset_x)
        shift_y = (StepSize*slope_y + offset_y)

        if not self.CamRotateBox.isChecked():
            return (shift_x,shift_y)
        else:
            return (-shift_y,shift_x)
    
    @pyqtSlot()
    def openHyperCreator(self):
        self.UIwin.SortHyperStack.setEnabled(True)
        self.HyperGroup.setEnabled(True)
        if not self.MaxProj.isChecked():
            self.HyperGroup.MaxProjCB.setDisabled(True)
        else:
            self.HyperGroup.MaxProjCB.setEnabled(True)
        #self.HyperGroup = Hyperstack(self)
        #self.UIwin.MainWin.Layout.addWidget(self.HyperGroup,0,8,3,4)
        
class BreakHyperstack(QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.MainWin = parent
        self.MainWin.sig_openDeskew.connect(self.openDeskew)

        self.setTitle("Explode OME-TIFF")
        self.FilePathLabel = QLabel(parent = self, text = "Select a folder with images:")
        self.FilePath = QLineEdit(self)
        self.FilePath.setText(os.getcwd())
        self.FilePath.textChanged.connect(self.loadASImeta)
        self.BrowseButton = QPushButton(self) 
        self.BrowseButton.setText("Browse...")
        self.BrowseButton.clicked.connect(self.browseDir)

        self.HyperStackOrder = QComboBox(self)
        self.HyperStackOrder.addItems(HyperOrder)
        self.HyperStackLabel = QLabel(parent = self, text = "Hyperstack order:")
        self.HyperStackOrder.currentTextChanged.connect(self.setEnableStates)
        self.HyperStackCB = QCheckBox(self)
        self.HyperStackCB.stateChanged.connect(self.checkStart)  

        self.ZLayerLabel = QLabel(parent = self, text = "No. of Z layers:")
        self.ZLayerNo = QSpinBox(self)
        self.ZLayerNo.setMinimum(1)
        self.ZLayerNo.setMaximum(100000)
        self.ZLayerNo.setSingleStep(1)
        self.ZLayerNo.setValue(1)
        self.ZLayerCB = QCheckBox(self)
        self.ZLayerCB.stateChanged.connect(self.checkStart)

        self.ColorChannelLabel = QLabel(parent = self, text = "No. of channels:")
        self.ColorChannels = QSpinBox(self)
        self.ColorChannels.setMinimum(1)
        self.ColorChannels.setMaximum(5)
        self.ColorChannels.setSingleStep(1)
        self.ColorChannels.setValue(1)
        self.ColorChannelCB = QCheckBox(self)
        self.ColorChannelCB.stateChanged.connect(self.checkStart)         

        self.TimePointLabel = QLabel(parent = self, text = "No. of time points:")
        self.TimePointNo = QSpinBox(self)
        self.TimePointNo.setMinimum(1)
        self.TimePointNo.setMaximum(100000)
        self.TimePointNo.setSingleStep(1)
        self.TimePointNo.setValue(1) 
        self.TimePointCB = QCheckBox(self)
        self.TimePointCB.stateChanged.connect(self.checkStart)      

        self.SortHyperStack = QPushButton("Break down ome-tiff")
        self.SortHyperStack.clicked.connect(self.startSorting)
        self.SortHyperStack.setDisabled(True)
        self.CheckSkip = QCheckBox(self)
        self.CheckSkip.setText("Skip this step")

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.FilePathLabel,0,0,1,2)
        self.Layout.addWidget(self.BrowseButton,0,3,1,2)
        self.Layout.addWidget(self.FilePath,1,0,1,5)
        self.Layout.addWidget(self.HyperStackLabel,2,0,1,2)
        self.Layout.addWidget(self.HyperStackOrder,2,2,1,2)
        self.Layout.addWidget(self.HyperStackCB,2,4,1,1)
        self.Layout.addWidget(self.ZLayerLabel,3,0,1,2)
        self.Layout.addWidget(self.ZLayerNo,3,2,1,2)
        self.Layout.addWidget(self.ZLayerCB,3,4,1,1)
        self.Layout.addWidget(self.ColorChannelLabel,4,0,1,2)
        self.Layout.addWidget(self.ColorChannels,4,2,1,2)
        self.Layout.addWidget(self.ColorChannelCB,4,4,1,1)
        self.Layout.addWidget(self.TimePointLabel,5,0,1,2)
        self.Layout.addWidget(self.TimePointNo,5,2,1,2)
        self.Layout.addWidget(self.TimePointCB,5,4,1,1)
        self.Layout.addWidget(self.CheckSkip,6,0,1,5)
        self.Layout.addWidget(self.SortHyperStack,7,0,1,5)

        self.setLayout(self.Layout)

        self.BreakDownThread = QThread(self)
        self.BreakDownWorker = SortWorker(self)
        self.BreakDownWorker.moveToThread(self.BreakDownThread)
        self.BreakDownThread.started.connect(self.BreakDownWorker.breakdown)        

    def browseDir(self):
        ImageDir = QFileDialog.getExistingDirectory(self,"select the dir for saving images")
        self.FilePath.setText(ImageDir)
        os.chdir(self.FilePath.text())

    def setEnableStates(self):
        pass

    def checkStart(self):
        if self.DeskewGroup.DeskewThread.isRunning():
            return False

        if not self.HyperStackCB.isChecked():
            self.SortHyperStack.setDisabled(True)
            return False

        if not self.TimePointCB.isChecked():
            self.SortHyperStack.setDisabled(True)
            return False
    
        if not self.ZLayerCB.isChecked():
            self.SortHyperStack.setDisabled(True)
            return False

        if not self.ColorChannelCB.isChecked():
            self.SortHyperStack.setDisabled(True)
            return False

        self.SortHyperStack.setEnabled(True)
        return True

    def getNamePost(self,idx):            
        Order = self.HyperStackOrder.currentText()
        ChannelNo = self.ColorChannels.value()
        TimePointNo = self.TimePointNo.value()
        ZLayer = self.ZLayerNo.value()
        match Order[3]:
            case "C":
                idx = idx//ZLayer
                keyNo = ChannelNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx//keyNo)))+str(idx//keyNo)
                return ["C"+str(1+idx%keyNo)+"_T"+TSN,(idx%keyNo,idx//keyNo)]
            case "T":
                idx = idx//ZLayer
                keyNo = TimePointNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx%keyNo)))+str(idx%keyNo)
                return ["T"+TSN+"_C"+str(1+idx//keyNo),(idx//keyNo,idx%keyNo)]
            case "Z":
                keyNo = ChannelNo*ZLayer
                TSN = "0"*(len(str(TimePointNo))-len(str(idx//keyNo)))+str(idx//keyNo)
                return ["C"+str(1+idx%ChannelNo)+"_T"+TSN,(idx%ChannelNo,idx//keyNo)]

    def startSorting(self):
        CheckState = self.checkStart()
        if not CheckState:
            MsgBox = QMessageBox(self)
            MsgBox.setWindowTitle("Checking start error:")
            MsgBox.setText("Please check every input parameters")
            MsgBox.setIcon(QMessageBox.Critical)
            MsgBox.show()
            return False
        
        self.startBreakdown()

    def startBreakdown(self):
        if self.BreakDownThread.isRunning():
            MsgBox = QMessageBox(self)
            MsgBox.setWindowTitle("Fire Warning")
            MsgBox.setText("The ome-tiff is under explosion!! Be patient!!")
            MsgBox.setIcon(QMessageBox.Warning)
            MsgBox.show()
            return False
        
        self.BreakDownThread.start()
        self.BreakDownThread.quit()

    def loadASImeta(self):
        try:
            PathName = self.FilePath.text()
            StackMetadata = self.getSliceStep(PathName)
            self.SliceStep = StackMetadata["stepSizeUm"]
            ChannelNo = StackMetadata["numChannels"]
            SliceNo = StackMetadata["numSlices"]
            TimePoint = StackMetadata["numTimepoints"]
            self.NamePrefix = StackMetadata["saveNamePrefix"]
            #print("Find slice step",SliceStep)
            #print(StackMetadata)
            self.ColorChannels.setValue(ChannelNo)
            self.ZLayerNo.setValue(SliceNo)
            self.TimePointNo.setValue(TimePoint)
        except:
            self.SliceStep = False
            StackMetadata = {"info":"No AcqSettings.txt"}
            print("no AcqSettings.txt found")

    def getSliceStep(self,FolderName):
        FileName = FolderName + "/AcqSettings.txt"

        with open(FileName) as file:
            metadata = json.load(file)
            file.close()
    
        return metadata
    
    @pyqtSlot(dict)
    def openDeskew(self,NewFileNames):
        self.DimOrderDict = NewFileNames
        self.NewFileNames = list(NewFileNames.keys())
        self.DeskewGroup.setEnabled(True)
        if self.SliceStep is False:
            self.DeskewGroup.SliceStep.setValue(0)
        else:
            self.DeskewGroup.SliceStep.setValue(self.SliceStep)
        self.DeskewGroup.selectImage.addItems(self.NewFileNames)
        #self.DeskewGroup = BackShift(self)
        #self.MainWin.Layout.addWidget(self.DeskewGroup,0,4,4,4)
    
    def showFileExistingWarningBox(self):
        MsgBox = QMessageBox()
        MsgBox.setWindowTitle("Files existing")
        MsgBox.setText("Exploded ome-tiff is existing. \nThe process will still need to run for getting info for next steps, please wait!")
        MsgBox.setIcon(QMessageBox.Warning)
        MsgBox.show()
        
class MainWin(QWidget):
    sig_openDeskew = pyqtSignal(dict)
    sig_openHyper = pyqtSignal()
    sig_progress = pyqtSignal(int)
    def __init__(self):
        super().__init__()

        self.sig_progress.connect(self.updateProgressbar)

        self.setWindowTitle("ASI SCAPE deskewing tool")
        BreakdownGroup = BreakHyperstack(self)
        BreakdownGroup.DeskewGroup = BackShift(BreakdownGroup)
        BreakdownGroup.DeskewGroup.setDisabled(True)
        BreakdownGroup.DeskewGroup.HyperGroup = Hyperstack(BreakdownGroup.DeskewGroup)
        BreakdownGroup.DeskewGroup.HyperGroup.setDisabled(True)
        self.ProgressBar = QProgressBar(self)

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(BreakdownGroup,0,0,4,4)
        self.Layout.addWidget(BreakdownGroup.DeskewGroup,0,4,4,4)
        self.Layout.addWidget(BreakdownGroup.DeskewGroup.HyperGroup,0,8,3,4)
        self.Layout.addWidget(self.ProgressBar,3,8,1,5)
        self.setLayout(self.Layout)
    
    @pyqtSlot(int)
    def updateProgressbar(self,percent):
        self.ProgressBar.setValue(percent)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())