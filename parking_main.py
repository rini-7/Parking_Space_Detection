import cv2 as cv
from util import get_parking_spots_bboxes,empty_or_not
mask_path='/Users/rinivaish/Desktop/utilities/programming/python programming/computer_vision/mask_crop.png'

video_path = '/Users/rinivaish/Desktop/utilities/programming/python programming/computer_vision/ParkingLotDetectorAndCounter/data/parking_crop_loop.mp4'

mask = cv.imread(mask_path,0) #open in grayscale

#BOUNDING BOX
connected_components = cv.connectedComponentsWithStats(mask,4,cv.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

cap = cv.VideoCapture(video_path)
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        print("End of the video")
        break
    #blue bounding boxes
    for spot in spots:
        x1,y1,w,h=spot
        spot_crop = frame[y1:y1+h,x1:x1+w,:] #img having only our parking spot
        spot_status = empty_or_not(spot_crop)
        if spot_status:

            frame = cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2) #width = 2
        else:
            frame = cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2) #width = 2

    cv.imshow('frame',frame)
    if cv.waitKey(25) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()