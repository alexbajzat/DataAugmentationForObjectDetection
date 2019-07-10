import cv2
import xml.etree.ElementTree as ET
from os import listdir, mkdir
from os.path import isfile, isdir
import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import uuid

from data_aug.data_aug import RandomHSV, RandomHorizontalFlip, RandomScale, RandomTranslate, RandomShear, Sequence, \
    draw_rect

classes = []
AUGMENTATION_FOLDER = "AUG_GENERATED"


def run():
    if len(sys.argv) <= 2:
        print("Missing arguments")
        exit(1)

    folder = sys.argv[1]
    replication_factor = int(sys.argv[2])
    print("working directory: ", folder)
    print("number of replications: ", replication_factor)

    target_folder = folder + "/" + AUGMENTATION_FOLDER
    if not isdir(target_folder):
        mkdir(target_folder)

    print("Target folder: ", target_folder)

    for voc_file in listdir(folder):
        path = folder + "/" + voc_file

        if isfile(path):
            if ".xml" in path:
                print("voc file: ", path)

                voc_tree = ET.parse(folder + '/' + voc_file)
                voc_obj = voc_tree.getroot()
                boxes = parse_boxes(voc_obj)
                image = load_image(voc_obj.find('path').text)
                augmented = do_augmentation(image, boxes, replication_factor)
                persist_augmented(augmented, target_folder, voc_tree)


def do_augmentation(image, boxes, replication_factor=1):
    augmented = []
    for i in range(replication_factor):
        seq = Sequence([
            RandomHSV(rd.randint(0, 70), rd.randint(0, 90), rd.randint(0, 90)),
            RandomHorizontalFlip(),
            RandomScale(),
            RandomTranslate(),
            RandomShear()
        ])

        img_, boxes_ = seq(image.copy(), boxes.copy())
        # plotted = draw_rect(img_, boxes_)
        # plt.imshow(plotted)
        # plt.show()
        augmented.append((img_, boxes_))
    return augmented


def persist_augmented(augmented, target_folder, voc_tree):
    voc_root = voc_tree.getroot()
    for img, boxes in augmented:
        target_name = str(uuid.uuid4())
        image_name = target_name + ".jpg"
        voc_root.find('path').text = target_folder + "/" + image_name
        persist_image(img, image_name, target_folder)
        update_boxes(voc_root, boxes)
        persist_augmented_voc(voc_tree, target_folder, target_name)


def persist_augmented_voc(voc_obj, target_folder, target_name):
    voc_obj.write(target_folder + "/" + target_name + ".xml")


def persist_image(image, name, target_folder):
    cv2.imwrite(target_folder + "/" + name, image)


def update_boxes(voc_obj, augmented_boxes):
    idx = 0
    for object in voc_obj.findall("object"):
        if idx >= len(augmented_boxes):
            voc_obj.remove(object)
        else:
            object.find('name').text = classes[int(augmented_boxes[idx][4])]
            object.find('bndbox').find('xmin').text = str(augmented_boxes[idx][0])
            object.find('bndbox').find('ymin').text = str(augmented_boxes[idx][1])
            object.find('bndbox').find('xmax').text = str(augmented_boxes[idx][2])
            object.find('bndbox').find('ymax').text = str(augmented_boxes[idx][3])
        idx += 1


def load_image(path):
    return cv2.imread(path)[:, :, ::-1]


def parse_boxes(root):
    # nothing at the moment
    formatted_boxes = []
    for box in root.findall('object'):
        bnd_box = box.find('bndbox')
        name = box.find('name').text
        if name not in classes:
            classes.append(name)

        formatted_boxes.append(
            [float(bnd_box.find('xmin').text),
             float(bnd_box.find('ymin').text),
             float(bnd_box.find('xmax').text),
             float(bnd_box.find('ymax').text),
             float(classes.index(name))])
    return np.array(formatted_boxes)


run()
