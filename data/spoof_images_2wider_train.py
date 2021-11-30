import os
from tqdm import tqdm

root_dir = '/home/oem/vinit/FaceDetection/karzads-facedetection-c175c1eb868c'

img_names_dir = os.path.join(root_dir, 'spoof_images/file_names.txt')
failed_imgs = os.path.join(root_dir, 'spoof_images/failure_in_result_with_path.txt')

with open(img_names_dir, 'r') as file:
    data = file.readlines()
    img_names = [x[:-1] for x in data]  # eliminate "\n" and "train/"

with open(failed_imgs, 'r') as file:
    data = file.readlines()
    failed_imgs = [x[:-1] for x in data]
    for i in range(len(failed_imgs)):
        temp = failed_imgs[i].split('/')[6:]
        temp.pop(1)
        failed_imgs[i] = "/".join(temp)

def read_label(label_path):
    with open(label_path, 'r') as file:
        data = file.readlines()
        data = [x[:-1] for x in data]
        num_lines = int(data[1])
        labels = []
        for ii in range(num_lines):
            data_ii = data[ii+2]
            data_ii = data_ii.split(' ')
            data_ii = [float(x) if idx == 4 else int(x) for idx, x in enumerate(data_ii)]
            labels.append(data_ii)
        labels = [x for x in labels if x[4]>0.5]

    final_labels = []
    for dd in labels:
        annotation = ""
        annotation += str(dd[0]) + " "    # x
        annotation += str(dd[1]) + " "   # y
        annotation += str(dd[2]-dd[0]) + " "    # w
        annotation += str(dd[3]-dd[1]) + " "    # h

        # Add lx and ly and separate them with 0.0
        for i in range(5):
            annotation += str(float(dd[5+2*i])) + " "
            annotation += str(float(dd[5+2*i+1])) + " "
            if float(dd[4+i]) == -1.0:
                annotation += str(-1.0) + " "
            else:
                annotation += str(0.0) + " "
        annotation = annotation[:-1]
        final_labels.append(annotation)
    return final_labels

count = 0
with open(os.path.join(root_dir, 'wider_label_spoof_images.txt'), 'w') as file:
    for img_name in tqdm(img_names):
        if not img_name in failed_imgs:
            label_path = img_name.split('/')
            label_path.insert(1, 'result')
            label_path = os.path.join(*label_path)
            label_path = os.path.join(root_dir, label_path)
            label_path = label_path.split('.')[0]+'.txt'

            label = read_label(label_path)
            file.write("# "+img_name+"\n")
            for ll in label:
                file.write(ll+"\n")
        else:
            count += 1

print("Failed Count: ", count)