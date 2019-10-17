import os, glob
import random

import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify


def vbb_anno2dict(vbb_file, cam_id):
    # 通过os.path.basename获得路径的最后部分“文件名.扩展名”
    # 通过os.path.splitext获得文件名
    filename = os.path.splitext(os.path.basename(vbb_file))[0]

    # 定义字典对象annos
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]  # 可查看所有类别
    # person index
    person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为‘person’的xml
    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id + 1) + ".jpg"
            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            annos[frame_name]["label"] = "person"
            for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                if not id in person_index_list:  # only use bbox whose label is person
                    continue
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)
            if not annos[frame_name]["bbox"]:
                del annos[frame_name]
    return annos


def seq2img(annos, seq_file, outdir, cam_id):
    cap = cv2.VideoCapture(seq_file)
    index = 1
    # captured frame list
    v_id = os.path.splitext(os.path.basename(seq_file))[0]
    cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in annos.keys()])
    while True:
        ret, frame = cap.read()
        if ret:
            if not index in cap_frames_index:
                index += 1
                continue
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outname = os.path.join(outdir, str(cam_id) + "_" + v_id + "_" + str(index) + ".jpg")
            print("Current frame: ", v_id, str(index))
            cv2.imwrite(outname, frame)
            height, width, _ = frame.shape
        else:
            break
        index += 1
    img_size = (width, height)
    return img_size


def anno2txt(vbb_outdir, filename, anno):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""

    assert anno["label"] is "person"

    video = filename.split("_")[1]
    path_name = os.path.join("data","custom","images",filename)+"\n"
    with open(os.path.join("data","custom","train.txt"), 'a+') as train:
        with open(os.path.join("data","custom","valid.txt"), 'a+') as valid:
            with open(os.path.join("data","custom","total.txt"), 'a+') as f:
                f.write(path_name)
            if random.randint(0, 9) is 0:
                valid.write(path_name)
            else:
                train.write(path_name)


def parse_anno_file(vbb_inputdir, vbb_outputdir):
    # annotation sub-directories in hda annotation input directory
    sub_dirs = os.listdir(vbb_inputdir)  # 对应set00,set01...

    for sub_dir in sub_dirs:
        print("Parsing annotations of camera: ", sub_dir)
        # 获取某一个子set下面的所有vbb文件
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            # 返回一个vbb文件中所有的帧的标注结果
            annos = vbb_anno2dict(vbb_file, sub_dir)
            if annos:
                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    print(filename)
                    if "bbox" in anno:
                        anno2txt(vbb_outputdir, filename, anno)
        #             break
        #     break
        # break


def visualize_bbox(xml_file, img_file):
    import cv2
    tree = etree.parse(xml_file)
    image = cv2.imread(img_file)
    origin = cv2.imread(img_file)
    # 获取一张图片的所有bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow("test", image)
    cv2.imshow('origin', origin)
    cv2.waitKey(0)


def main():
    vbb_inputdir = "/mnt/space-2/DataSet/PedestrianDetection/Caltech/annotations"
    vbb_outputdir = "/home/hzg/code/PyTorch-YOLOv3/data/custom/"
    parse_anno_file(vbb_inputdir, vbb_outputdir)


if __name__ == "__main__":
    main()
