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
        self.layout.addWidget(self.FilePath,1,0,1,5)
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
        self.layout.addWidget(self.SortHyperStack,6,3,1,2)

        self.setLayout(self.layout)        

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
        NewFileNames = list()

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
                        NewFileNames.append(NewFileName)
                        count = count+1
                        img =TFF.memmap(NewFileName,shape=(ZLayerNo,data.shape[0],data.shape[1]), dtype=np.uint16, metadata = {"axes":"ZYX"}, bigtiff = True)                        
                    img[RealPageNo,:,:] = data
                    Remainder = Remainder+1                
                tif.close()
            img.flush()
    
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

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UserInput()
    win.show()
    sys.exit(app.exec_())