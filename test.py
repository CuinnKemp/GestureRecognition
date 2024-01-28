# import the opencv library
import cv2
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
from  PIL  import Image
  
  
# define a video capture object
vid = cv2.VideoCapture(0)

data_dir = 'data/train/YES'
directory = os.path.join(data_dir)
os.chdir(directory)
  
framecount = 1
# while(True):
while True:
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    frame = cv2.resize(frame,(50,50))
  
    # Display the resulting frame
    # frame = cv2.resize(frame,(50,50))

    img = frame
    original = img.copy()

    l = int(max(40, 44))
    u = int(min(44, 44))

    ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(img, (21, 51), 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(edges, l, u)

    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    data = mask.tolist()
    sys.setrecursionlimit(10**8)
    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255

    img = Image.fromarray(result)
    img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)

    frame = np.array(img) 
    # Convert RGB to BGR 
    frame = frame[:, :, ::-1].copy() 




    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    cv2.imwrite((str(framecount) + '.png'), frame)
    print(framecount)
    framecount += 1

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



