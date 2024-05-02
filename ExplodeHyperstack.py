from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tifffile as TFF
import numpy as np
import sys,os,glob,json

#HyperOrder = ["XYZC","XYZT","XYZCT","XYZTC","XYCZ","XYTZ"]
HyperOrder = ["XYZCT","XYZTC"]

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
    
    def breakdown(self):
        AllTif = glob.glob("*.tif")
        if len(AllTif) > 1:
            for atif in AllTif:
                if atif[0:6] == "Deskew":
                    os.remove(atif)
        AllTif = sortName(AllTif)

        NewFileNames = dict()

        ZLayerNo = self.SortWin.ZLayerNo.value()
        NameTemplate = AllTif[0][0:-8]
        Remainder = 0
        count = 0
        
        for aTiff in AllTif:
            print(aTiff,Remainder)
            with TFF.TiffFile(aTiff) as tif:
                for idx in range(len(tif.pages)):
                    data = tif.pages[idx].asarray()
                    RealPageNo = Remainder%ZLayerNo
                    if RealPageNo == 0:
                        [SNstr,SN] = self.SortWin.getNamePost(count)
                        NewFileName = NameTemplate+"_"+SNstr+".ome.tif"
                        NewFileNames[NewFileName] = SN
                        count = count+1
                        img =TFF.memmap(NewFileName,shape=(ZLayerNo,data.shape[0],data.shape[1]), dtype=np.uint16, metadata = {"axes":"ZYX"}, bigtiff = True)                        
                    img[RealPageNo,:,:] = data
                    Remainder = Remainder+1                
                tif.close()
            img.flush()
        
        self.SortWin.MainWin.sig_openDeskew.emit(NewFileNames)

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
        
        for FileName in ImgNames:
            print("Deskewing %s..."%FileName)
            NewFileName = "Deskew_"+FileName
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
        if self.UIwin.SliceStep is False:
            self.SliceStep.setValue(0)
        else:
            self.SliceStep.setValue(self.UIwin.SliceStep)
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
        self.Slope.addItems(["70µm/°","35µm/°"])

        self.CamRotateBox = QCheckBox(self)
        self.CamRotateBox.setText("Was the camera rotated (2024 configuration)?")

        self.selectImageLabel = QLabel(parent = self, text = "Select a file for deskew:")
        self.selectImage = QComboBox(self)
        self.selectImage.addItems(self.UIwin.NewFileNames)

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
        
        SliceStep = self.UIwin.SliceStep

        if not self.SelectAll.isChecked:
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
            case "35µm/°":
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
        self.HyperGroup = Hyperstack(self)
        self.UIwin.MainWin.Layout.addWidget(self.HyperGroup,0,8,3,4)
        
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

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.FilePathLabel,0,0,1,2)
        self.layout.addWidget(self.BrowseButton,0,3,1,2)
        self.layout.addWidget(self.FilePath,1,0,1,5)
        self.layout.addWidget(self.HyperStackLabel,2,0,1,2)
        self.layout.addWidget(self.HyperStackOrder,2,2,1,2)
        self.layout.addWidget(self.HyperStackCB,2,4,1,1)
        self.layout.addWidget(self.ZLayerLabel,3,0,1,2)
        self.layout.addWidget(self.ZLayerNo,3,2,1,2)
        self.layout.addWidget(self.ZLayerCB,3,4,1,1)
        self.layout.addWidget(self.ColorChannelLabel,4,0,1,2)
        self.layout.addWidget(self.ColorChannels,4,2,1,2)
        self.layout.addWidget(self.ColorChannelCB,4,4,1,1)
        self.layout.addWidget(self.TimePointLabel,5,0,1,2)
        self.layout.addWidget(self.TimePointNo,5,2,1,2)
        self.layout.addWidget(self.TimePointCB,5,4,1,1)
        self.layout.addWidget(self.SortHyperStack,6,0,1,5)

        self.setLayout(self.layout)

        self.BreakDownThread = QThread(self)
        self.BreakDownWorker = SortWorker(self)
        self.BreakDownWorker.moveToThread(self.BreakDownThread)
        self.BreakDownThread.started.connect(self.BreakDownWorker.breakdown)        

    def browseDir(self):
        ImageDir = QFileDialog.getExistingDirectory(self,"select the dir for saving images")
        self.FilePath.setText(ImageDir)
        os.chdir(self.FilePath.text())
        NamePrefix = glob.glob("*pos0.ome*")
        self.NamePrefix = NamePrefix[0][0:-8] 

    def setEnableStates(self):
        pass

    def checkStart(self):
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
        match Order[3]:
            case "C":
                keyNo = ChannelNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx//keyNo)))+str(idx//keyNo)
                return ["C"+str(1+idx%keyNo)+"_T"+TSN,(idx%keyNo,idx//keyNo)]
            case "T":
                keyNo = TimePointNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx%keyNo)))+str(idx%keyNo)
                return ["T"+TSN+"_C"+str(1+idx//keyNo),(idx//keyNo,idx%keyNo)]

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
        self.DeskewGroup = BackShift(self)
        self.MainWin.Layout.addWidget(self.DeskewGroup,0,4,4,4)
        
class MainWin(QWidget):
    sig_openDeskew = pyqtSignal(dict)
    sig_openHyper = pyqtSignal()
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASI SCAPE deskewing tool")
        BreakdownGroup = BreakHyperstack(self)

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(BreakdownGroup,0,0,4,4)
        self.setLayout(self.Layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())