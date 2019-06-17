from PIL import ImageGrab, ImageOps
from numpy import *
import pyautogui, time
#global env
replayBtn=(555,399)
dino=(326,406)
lookBox1=(dino[0]+60,dino[1], dino[0]+110, dino[1]+30)
#reset
pyautogui.click(replayBtn)
def detectObs():
    image = ImageGrab.grab(lookBox1)
    grayImage = ImageOps.grayscale(image)
    print(array(grayImage.getcolors()).sum())
    return array(grayImage.getcolors()).sum()

while True:
    #when detecting the obstacle
    if 1747 != detectObs():
        pyautogui.press("space")
