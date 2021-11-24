import os
import cv2
import matplotlib.pyplot as plt

data_dir = 'train_label4_awing_final'

filenames = os.listdir(data_dir)

data_paths_dict = {}
for ff in filenames:
    if not 'txt' in ff and os.path.exists(os.path.join(data_dir, ff.split('.')[0]+".txt")):
        if 'FaceDetect' in ff:
            data_paths_dict[ff] = ff.split('.')[0]+".txt"

keys = list(data_paths_dict.keys())

print("Total Indices :", len(keys))
idx = int(input('Choose Index: '))
img = cv2.imread(os.path.join(data_dir, keys[idx]))
height, width, _ = img.shape
with open(os.path.join(data_dir, data_paths_dict[keys[idx]]), 'r') as file:
    label = file.readlines()
    label = [x[3:-1] for x in label]
    label = [x.split(' ') for x in label]
    for idx, ll in enumerate(label):
        ll = [float(x) for x in ll]
        # bbox
        ll[0] *= width          # cx
        ll[1] *= height         # cy
        ll[2] *= width          # bbox width
        ll[3] *= height         # bbox height

        for i in range(4, 14):
            if i%2 == 0: ll[i] *= width
            else: ll[i] *= height
        
        ll[0] -= ll[2]/2
        ll[1] -= ll[3]/2
        ll[2] += ll[0]
        ll[3] += ll[1]
        label[idx] = [int(x) for x in ll]

for ll in label:
    img = cv2.rectangle(img, (ll[0], ll[1]), (ll[2], ll[3]), (0, 255, 0), 2)
    for i in range(5):
        if ll[4] > 0:
            img = cv2.circle(img, (ll[4+2*i], ll[4+2*i+1]), 2, (0, 0, 255), -1)
plt.imshow(img[:,:,::-1])
plt.show()
# import pdb; pdb.set_trace()