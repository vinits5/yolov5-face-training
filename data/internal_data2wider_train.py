import os
from tqdm import tqdm

root_dir = '/home/oem/vinit/FaceDetection/yolov5/face-detection-internal-dataset/version2/train'

img_names_dir = os.path.join(root_dir, 'imgs.txt')
labels_dir = os.path.join(root_dir, 'new_labels2.txt')

with open(img_names_dir, 'r') as file:
    data = file.readlines()
    img_names = [x[6:-1] for x in data]  # eliminate "\n" and "train/"

with open(labels_dir, 'r') as file:
    data = file.readlines()
    data = [x[:-1] for x in data]

data_dict = {}
first_case = True

for dd in data:
    check = len(dd.split(' '))
    update_dict = False
    if check==1:
        try:
            numbers = int(dd)
            if numbers>100:
                update_dict = True
        except:
            update_dict = True
        
        if update_dict:
            if first_case: 
                first_case = False
                pass
            else:
                data_dict.update({img_name: labels})
            img_name = dd
            labels = []
    else:
        dd = [int(x) for x in dd.split(" ")]
        annotation = ""
        annotation += str(dd[0]) + " "    # x
        annotation += str(dd[1]) + " "   # y
        annotation += str(dd[2]) + " "    # w
        annotation += str(dd[3]) + " "    # h

        # Add lx and ly and separate them with 0.0
        for i in range(5):
            annotation += str(float(dd[4+2*i])) + " "
            annotation += str(float(dd[4+2*i+1])) + " "
            if float(dd[4+i]) == -1.0:
                annotation += str(-1.0) + " "
            else:
                annotation += str(0.0) + " "
        annotation = annotation[:-1]
        labels.append(annotation)

count = 0
with open(os.path.join(root_dir, 'wider_label.txt'), 'w') as file:
    for img_name in tqdm(img_names):
        key = img_name.split('/')[-1].split('.')[0]
        try:
            label = data_dict[key]
            file.write("# "+img_name+"\n")
            for ll in label:
                file.write(ll+"\n")
        except:
            count += 1
            pass
# import pdb; pdb.set_trace()

