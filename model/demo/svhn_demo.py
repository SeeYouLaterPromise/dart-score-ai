import h5py
# import numpy as np
import pandas as pd


def get_image_name(ds, idx):
    """读取第 idx 个样本的图片名称"""
    name_ref = ds['digitStruct']['name'][idx][0]
    return ''.join([chr(c[0]) for c in ds[name_ref]])

def get_bbox(ds, idx):
    """读取第 idx 个样本的所有 bounding box 信息"""
    bbox_ref = ds['digitStruct']['bbox'][idx].item()
    bbox = {}

    def _get_attr(name):
        attr = ds[bbox_ref][name]
        if attr.shape[0] == 1:
            return [attr[0][0]]
        else:
            return [ds[attr[i][0]][()][0][0] for i in range(attr.shape[0])]

    bbox['label'] = _get_attr('label')
    bbox['left'] = _get_attr('left')
    bbox['top'] = _get_attr('top')
    bbox['width'] = _get_attr('width')
    bbox['height'] = _get_attr('height')

    return bbox

def parse_digitStruct(mat_file):
    with h5py.File(mat_file, 'r') as ds:
        num_images = ds['digitStruct']['name'].shape[0]
        data = []
        print(f"Number of images: {num_images}")
        for i in range(num_images):
            name = get_image_name(ds, i)
            bbox = get_bbox(ds, i)
            # 构建每个图像的标注
            for j in range(len(bbox['label'])):
                entry = {
                    'filename': name,
                    'label': int(bbox['label'][j]) % 10,  # SVHN中10代表0
                    'left': float(bbox['left'][j]),
                    'top': float(bbox['top'][j]),
                    'width': float(bbox['width'][j]),
                    'height': float(bbox['height'][j])
                }
                data.append(entry)
            
            if i == 5:
                break  # 仅处理前5个样本以测试
    return data


if __name__ == '__main__':
    
    data = parse_digitStruct('data/SVHN/digitStruct.mat')
    df = pd.DataFrame(data)
    df.to_csv('svhn_labels.csv', index=False)
