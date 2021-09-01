# %%
import cv2

def extract(frame_skip):
    name = 397
    for i in range(13,13):
        
        video = cv2.VideoCapture("./bg_videos/{name}.mp4".format(name=i))   
        count = 0 
    
        while(video.isOpened()):
            count += frame_skip

            ret, frame = video.read()
            if ret==True:
                video.set(1, count)
                h, w, _ = frame.shape
                if h > w:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite("./backgrounds/{name}.jpg".format(name=name), frame)
 
                frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
                cv2.imshow('bg', frame)
            else:
                video.release()
                break

            key = cv2.waitKey(1)
            if key != -1 and key != 255:
                video.release()
                cv2.destroyAllWindows()
                return

            name += 1

    video.release()
    cv2.destroyAllWindows()

extract(frame_skip = 20)