from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tifffile as TFF
import numpy as np
import sys,os,glob,json

#HyperOrder = ["XYZC","XYZT","XYZCT","XYZTC","XYCZ","XYTZ"]
HyperOrder = ["XYZCT","XYZTC"]

def sortName(Names:list):
    NewList = list()
    for idx,aName in enumerate(Names):
        if idx == 0:
            NewList.append(aName)
            continue
        if len(aName) < len(NewList[-1]):
            n = len(NewList)-1
            while len(NewList[n]) > len(aName):
                n = n-1
            NewList.insert(n+1,aName)
        else:
            NewList.append(aName)
    
    return NewList

class BackShift(QGroupBox):
    def __init__(self,parent):
        super().__init__()

        self.UIwin = parent

        self.setTitle("Deskew")
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
        self.CamRotateBox.setText("Was camera rotated (2024 configuration)?")

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

        self.Layout = QGridLayout(self)
        self.Layout.addWidget(self.BinningLabel,0,0,1,2)
        self.Layout.addWidget(self.Binning,0,2,1,2)
        self.Layout.addWidget(self.SlopeLabel,1,0,1,2)
        self.Layout.addWidget(self.Slope,1,2,1,2)
        self.Layout.addWidget(self.CamRotateBox,2,0,1,4)
        self.Layout.addWidget(self.selectImageLabel,3,0,1,2)
        self.Layout.addWidget(self.selectImage,4,0,1,4)
        self.Layout.addWidget(self.SelectAll,5,0,1,4)
        self.Layout.addWidget(self.MaxProj,6,0,1,4)
        self.Layout.addWidget(self.Deskew,7,0,1,4)

        self.setLayout(self.Layout)
    
    def setSelectState(self):
        if self.SelectAll.isChecked():
            self.selectImage.setDisabled(True)
        else:
            self.selectImage.setEnabled(True)
    
    def deskewImages(self):
        SliceStep = self.UIwin.SliceStep

        if not self.SelectAll.isChecked:
            ImgNames = [self.selectImage.currentText()]
        else:
            ImgNames = self.UIwin.NewFileNames

        for ImgName in ImgNames:
            with TFF.TiffFile(ImgName) as tif:            
                #img = TFF.memmap("test.tif",shape = tif.pages[0].shape)                      
                OriImageShape = tif.pages[0].asarray().shape
                metadata = tif.imagej_metadata

                Shift = self.getShift(SliceStep,OriImageShape)
                NewSize = self.getNewPageSize(OriImageShape,Shift)
            
                ImageShape = OriImageShape.copy()
                ImageShape[-2] = int(NewSize[-2])
                ImageShape[-1] = int(NewSize[-1])
                ImageShape = tuple(ImageShape)
                print(ImageShape)
                tif.close()

            self.createImage(ImgName,ImageShape,metadata,OriImageShape,NewSize)
    
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
        print(dim_x,dim_y)
        scale = 2048/dim_y

        if isBinning:
            scale = 2
        else:
            scale = 1

        if isRescale:
            rescaleFactor = 0.4
        else:
            rescaleFactor = 1

        print("scale : ",scale)
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

    def createImage(self,FileName,ImageShape,metadata,OriImageShape,NewSize):
        NewFileName = "Deskew_"+FileName
        img =TFF.memmap(NewFileName,shape = ImageShape, dtype=np.uint16, metadata = metadata, bigtiff = True)
           
        with TFF.TiffFile(FileName) as tif:
            for ZPos,page in enumerate(tif.pages):
                data = page.asarray()
                [start_x,end_x,start_y,end_y] = self.getAssignCoordinate(Shift,OriImageShape,ZPos,int(NewSize[-2]),int(NewSize[-1]))
                img[ZPos,start_x+1:end_x-1,start_y+1:end_y-1] = data[1:-1,1:-1]
            tif.close()
        img.flush()

class UserInput(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Explode OME-TIFF")
        self.FilePathLabel = QLabel(parent = self, text = "select the file folder:")
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

        self.ZLayerLabel = QLabel(parent = self, text = "No. of Z layers:")
        self.ZLayerNo = QSpinBox(self)
        self.ZLayerNo.setMinimum(1)
        self.ZLayerNo.setMaximum(100000)
        self.ZLayerNo.setSingleStep(1)
        self.ZLayerNo.setValue(1)
        self.ZLayerCB = QCheckBox(self)

        self.ColorChannelLabel = QLabel(parent = self, text = "No. of channels:")
        self.ColorChannels = QSpinBox(self)
        self.ColorChannels.setMinimum(1)
        self.ColorChannels.setMaximum(5)
        self.ColorChannels.setSingleStep(1)
        self.ColorChannels.setValue(1)
        self.ColorChannelCB = QCheckBox(self)         

        self.TimePointLabel = QLabel(parent = self, text = "No. of time points:")
        self.TimePointNo = QSpinBox(self)
        self.TimePointNo.setMinimum(1)
        self.TimePointNo.setMaximum(100000)
        self.TimePointNo.setSingleStep(1)
        self.TimePointNo.setValue(1) 
        self.TimePoinitCB = QCheckBox(self)

        self.SortHyperStack = QPushButton("Start sorting")
        self.SortHyperStack.clicked.connect(self.startSorting)

        self.layout = QGridLayout(self)
        self.layout.addWidget(self.FilePathLabel,0,0,1,2)
        self.layout.addWidget(self.BrowseButton,0,2,1,2)
        self.layout.addWidget(self.FilePath,1,0,1,4)
        self.layout.addWidget(self.HyperStackLabel,2,0,1,2)
        self.layout.addWidget(self.HyperStackOrder,2,2,1,2)
        self.layout.addWidget(self.ZLayerLabel,3,0,1,2)
        self.layout.addWidget(self.ZLayerNo,3,2,1,2)
        self.layout.addWidget(self.ZLayerCB,3,4,1,1)
        self.layout.addWidget(self.ColorChannelLabel,4,0,1,2)
        self.layout.addWidget(self.ColorChannels,4,2,1,2)
        self.layout.addWidget(self.ColorChannelCB,4,4,1,1)
        self.layout.addWidget(self.TimePointLabel,5,0,1,2)
        self.layout.addWidget(self.TimePointNo,5,2,1,2)
        self.layout.addWidget(self.TimePoinitCB,5,4,1,1)
        self.layout.addWidget(self.SortHyperStack,6,2,1,2)

        self.setLayout(self.layout)

        self.NewFileNames = ["test1.tif","test2.tif"]
        self.openDeskew()        

    def browseDir(self):
        ImageDir = QFileDialog.getExistingDirectory(self,"select the dir for saving images")
        self.FilePath.setText(ImageDir)

    def setEnableStates(self):
        pass

    def checkStart(self):
        if self.TimePointNo.isEnabled():
            if not self.TimePoinitCB.isChecked():
                return False
        
        if self.ZLayerNo.isEnabled():
            if not self.ZLayerCB.isChecked():
                return False

        if self.ColorChannels.isEnabled():
            if not self.ColorChannelCB.isChecked():
                return False

        return True

    def getNamePost(self,idx):            
        Order = self.HyperStackOrder.currentText()
        ChannelNo = self.ColorChannels.value()
        TimePointNo = self.TimePointNo.value()
        match Order[3]:
            case "C":
                keyNo = ChannelNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx//keyNo)))+str(idx//keyNo)
                return "C"+str(1+idx%keyNo)+"_T"+TSN
            case "T":
                keyNo = TimePointNo
                TSN = "0"*(len(str(TimePointNo))-len(str(idx%keyNo)))+str(idx%keyNo)
                return "T"+TSN+"_C"+str(1+idx//keyNo)
        

    def startSorting(self):
        CheckState = self.checkStart()
        if not CheckState:
            MsgBox = QMessageBox(self)
            MsgBox.setWindowTitle("Checking start error:")
            MsgBox.setText("Please check every input parameters")
            MsgBox.setIcon(QMessageBox.Critical)
            MsgBox.show()
            return False

        os.chdir(self.FilePath.text())
        AllTif = glob.glob("*.tif")
        if len(AllTif) > 1:
            for atif in AllTif:
                if atif[0:6] == "Deskew":
                    os.remove(atif)
        AllTif = sortName(AllTif)

        ZLayerNo = self.ZLayerNo.value()
        NameTemplate = AllTif[0][0:-8]
        Remainder = 0
        count = 0
        self.NewFileNames = list()

        for aTiff in AllTif:
            print(aTiff,Remainder)
            with TFF.TiffFile(aTiff) as tif:
                for idx in range(len(tif.pages)):
                    data = tif.pages[idx].asarray()
                    RealPageNo = Remainder%ZLayerNo
                    if RealPageNo == 0:
                        SN = self.getNamePost(count)
                        #SN = "0"*(len(str(len(AllTif)-1))-len(str(count)))+str(count)
                        NewFileName = NameTemplate+"_"+SN+".tiff"
                        self.NewFileNames.append(NewFileName)
                        count = count+1
                        img =TFF.memmap(NewFileName,shape=(ZLayerNo,data.shape[0],data.shape[1]), dtype=np.uint16, metadata = {"axes":"ZYX"}, bigtiff = True)                        
                    img[RealPageNo,:,:] = data
                    Remainder = Remainder+1                
                tif.close()
            img.flush()

        self.openDeskew()
    
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
            SliceStep = 1.1
            StackMetadata = {"info":"No AcqSettings.txt"}
            print("no AcqSettings.txt found")

    def getSliceStep(self,FolderName):
        FileName = FolderName + "/AcqSettings.txt"

        with open(FileName) as file:
            metadata = json.load(file)
            file.close()
    
        return metadata
    
    def openDeskew(self):
        self.DeskewGroup = BackShift(self)
        self.layout.addWidget(self.DeskewGroup,0,5,7,4)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UserInput()
    win.show()
    sys.exit(app.exec_())