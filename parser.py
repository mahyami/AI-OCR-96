from __future__ import print_function
import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.nan)


def to_int(bytes): 
  return int.from_bytes(bytes, byteorder="little", signed=False)

file_name = 'test_dump'
img_arr = []
label_arr = []
MAX_W = 40
MAX_H = 40
i = 0 
with open('test.cdb', 'rb') as f: 
  yy = f.read(2)
  m = f.read(1)
  d = f.read(1)
  w = f.read(1)
  h = f.read(1)
  tr = int.from_bytes(f.read(4), byteorder="little", signed=False) # Total Rec
  lc = int.from_bytes(f.read(4 * 128), byteorder="little", signed=False) # Letter count 
  it = f.read(1) # image type 
  comment = f.read(256)
  f.read(245)

  print("image type", it)
  # print("comment", comment)
  print("total recs", tr)
  print()

  for idx in range(lc): 
    flag = to_int(f.read(1)) # must be ff

    if not flag == 255: 
      print("STH is wrong", flag )
      break
    label = to_int(f.read(1))
    ww = to_int(f.read(1))
    hh = to_int(f.read(1))

    # if ww > MAX_W: MAX_W = ww
    # if hh > MAX_H: MAX_H = hh 

    f.read(2)
    
    img = [None] * (ww*hh)

    for y in range(hh):
      bwite = True
      counter = 0
      while counter < ww: 
        bw_value = to_int(f.read(1))
        x = 0
        while x < bw_value:
          if bwite: 
            img[y * ww+x+counter] = 0 
          else: 
            img[y * ww+x+counter] = 1
          x += 1
        bwite = not bwite
        counter += bw_value

    img = np.reshape(img, (hh, ww))

    if ww > MAX_W or hh > MAX_H: 
      continue

    # Pad the image 
    img = np.pad(img, [(0, MAX_H-hh), (0, MAX_W-ww)], mode="constant")

    # img_file = Image.new('1', (MAX_H, MAX_H))
    # pixels = img_file.load()
    # for i in range(MAX_H):
    #   for j in range(MAX_W):
    #     pixels[j, i] = img[i][j],

    # filename = "./data/img{}_{}.png".format(idx, label)
    # img_file.save(filename)
    # print("saving " + filename)

    img = np.reshape(img, (MAX_H * MAX_H))
    img_arr.append(img)
    label_arr.append(label)
    i += 1 
    print(i)
    
    # if idx == 10: break

print("Saving numpy dumps...")
np.save(file_name + "_img.bin", img_arr)
np.save(file_name + "_lbl.bin", label_arr)


