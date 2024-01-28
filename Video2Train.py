# import the opencv library
import cv2
import time
import os 
  
  
# define a video capture object
vid = cv2.VideoCapture("3.mp4")

fps = int(vid.get(5))
print("fps:", fps)


data_dir = './data/train/NO'
directory = os.path.join(data_dir)
os.chdir(directory)
  
framecount = 101
# while(True):
while framecount <= 10000:
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    frame = cv2.resize(frame,(50,50))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame', frame)

    cv2.imwrite((str(framecount) + '.png'), frame)
    print(framecount)
    framecount += 1

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

