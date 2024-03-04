from PIL import Image
#FileName = r"D:\Data_SCAPE\DictyV2TLb\DictyV2TLb_MMStack_Pos0.ome.tif"
FileName = r'F:\Deskew-Test\No1-Pos3-TP_MMStack_Pos0.ome.tif'
FileName = r"F:\Deskew-Test\No1-Pos3-TP_MMStack_Pos0_1.ome.tif"
im = Image.open(FileName)
imageDesc = im.tag[270]
print(imageDesc)