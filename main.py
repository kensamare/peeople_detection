# example.py
from reego.bg import remove
import numpy as np
import io
from PIL import Image
import time

# Uncomment the following lines if working with trucated image formats (ex. JPEG / JPG)
# In my case I do give JPEG images as input, so i'll leave it uncommented
# from PIL import ImageFile
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# input_image = 'input.png'
# output_image = 'output-image.png'
#
# f = np.fromfile(input_image)
# start_time = time.time()
# result = remove(f, "u2net_human_seg")
# print("--- %s seconds ---" % (time.time() - start_time))
# img = Image.open(io.BytesIO(result)).convert("RGBA")
# img.save(output_image)

import cv2


def show_webcam(mirror=False):
    avr_time = []
    cam = cv2.VideoCapture('example1.mp4')
    length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        i += 1
        print(f"{i}/{length}")
        ret_val, img = cam.read()
        if not ret_val:
            break
        img = cv2.resize(img, (480, 360))
        # cv2.imwrite(f"orig/{i}.png", img)
        start_time = time.time()
        result = remove(img, "u2net_human_seg", i == 0)
        avr_time.append(time.time()-start_time)
        cv2.imshow('my webcam', img)
        im_bw = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)[1]
        # cv2.imwrite(f"fig/{i}.png", im_bw)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    print(np.average(avr_time))
    print(np.var(avr_time))
    print(np.std(avr_time))


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
