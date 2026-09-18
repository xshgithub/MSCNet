import os
import json
import xml.etree.ElementTree as ET

xml_dir = "/data1/huanglab/xsh/dqdetr/dataset/LLVIP/Annotations"
image_dir = "/data1/huanglab/xsh/dqdetr/dataset/LLVIP/visible"

train_txt = "/data1/huanglab/xsh/dqdetr/dataset/LLVIP/train.txt"
test_txt = "/data1/huanglab/xsh/dqdetr/dataset/LLVIP/test.txt"

def load_split(txt_path):
    with open(txt_path) as f:
        return set(line.strip() for line in f.readlines())

train_set = load_split(train_txt)
test_set = load_split(test_txt)


def convert(split_set, output_json):
    images = []
    annotations = []
    categories = []
    category_dict = {}

    ann_id = 1
    img_id = 1
    cat_id = 1

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        file_id = xml_file.replace(".xml", "")

        if file_id not in split_set:
            continue

        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # images
        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall("object"):
            name = obj.find("name").text

            # 类别映射
            if name not in category_dict:
                category_dict[name] = cat_id
                categories.append({
                    "id": cat_id,
                    "name": name
                })
                cat_id += 1

            category_id = category_dict[name]

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })

            ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"{output_json} 转换完成！")


# 生成
convert(train_set, "train.json")
convert(test_set, "test.json")