# %%
import cv2

def extract(frame_skip):
    for i in range(0,8):
        print(i)
        video = cv2.VideoCapture("./raw_dataset/bg_videos/{name}.mp4".format(name=i))   
        count = 0 
        while(video.isOpened()):
            count += frame_skip
    
            ret, frame = video.read()
            if ret==True:
                video.set(1, count)
                h, w, _ = frame.shape
                if h > w:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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

extract(frame_skip = 30)