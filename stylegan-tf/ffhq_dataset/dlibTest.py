import sys  
  
import dlib  
from skimage import io  
  
  
detector = dlib.get_frontal_face_detector()  
win = dlib.image_window()  
  
for f in sys.argv[1:]:  
    print("Processing file: {}".format(f))  
    img = io.imread(f)  
    # The 1 in the second argument indicates that we should upsample the image  
    # 1 time.  This will make everything bigger and allow us to detect more  
    # faces.  
    dets = detector(img, 1)  
    print("Number of faces detected: {}".format(len(dets)))  
    for i, d in enumerate(dets):  
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(  
            i, d.left(), d.top(), d.right(), d.bottom()))  
  
    win.clear_overlay()  
    win.set_image(img)  
    win.add_overlay(dets)  
    dlib.hit_enter_to_continue()  

if (len(sys.argv[1:]) > 0):  
    img = io.imread(sys.argv[1])  
    dets, scores, idx = detector.run(img, 1, -1)  
    for i, d in enumerate(dets):  
        print("Detection {}, score: {}, face_type:{}".format(  
            d, scores[i], idx[i]))  