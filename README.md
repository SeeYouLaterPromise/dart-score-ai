# Dart score system
Objective: (video -> images -> coordinates_points -> score) 分析飞镖扎在飞盘上的得分

# Task Allocation
- 模型工程师 (Yexin Liu Lu)（处理视觉模型，输入：静态图片；输出：飞镖坐标点列表）
- 几何计算员 (Zhengkai Yan)（处理得分逻辑，输入：飞镖坐标点列表；输出：得分列表）
- 系统集成员 (Zhijun Wang)（建立实时计分系统：做一个程序，调用设备摄像头，输出静态帧列表；测试模型）

# Directory Structure
## work zone
协作过程中不要去修改其他人负责的工作区！
- `model`: model engineering
- `geometry`: calculate the score
- `system`: test and construct front-end presentation

## config
`.yaml` for parameter configuration.

## data
`dataset_address.txt` or `chaoxing` to download the dataset
You should arrange your `data` directory like this:
- `darts_dataset`
- `yolo_dataset` (future, you convert `darts_dataset`and preserve here)

## report
write your work records or thoughts here. 

## model
### Collaborators should read:
* 首先，你需要运行 `model/conver_label.py`：
    - 将数据集(`data/darts_dataset`)标签形式转换成YOLO标准，并放在`data/yolo_dataset`
    - 划分训练集和验证集
    - 文件存储：`data/yolo_dataset`
* `model/predict_darts.py`中的`predict_image`
  - input: 图片
  - output: 
    1. xy: 形如[[], [], [], [], ...]
    前四个是参考点的xy坐标；后面剩下的是飞镖点的xy坐标。
    2. img0: original image

### Yolov5
运行`yolov5`训练:
```
cd model/yolov5
pip install -r requirements.txt
cd ..
cd ..
python model/yolov5/train.py --img 800 --batch 16 --epochs 100 --data config/yolo_data.yaml --weights "" --project runs_dart --name yolo-first-try
```

模型训练后的实验结果会保存在`runs_dart`文件夹下（由`--project`指定）。




## geometry
...

## system
...