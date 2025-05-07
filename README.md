# Dart score system
Objective: (video -> images -> coordinates_points -> score) 分析飞镖扎在飞盘上的得分

# Background
![Dart](dart.avif)
国际标准组织（WDF：World Darts Federation）规定：

`“标准飞镖靶的 20 分区域必须位于正上方。”`

1896 年，Brian Gamlin（一位木工）设计了现在这个数字排列方式，目的:让高分区被低分区夹住，惩罚失误，避免“刷分”。

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
- `setting.yaml` is used to put `global const variable`.
- `yolo_data.yaml` configures the data format of yolo dataset.
- `dart_hyp.yaml` is the configuration file for `data argument`.

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

---
| 特性           | YOLOv5                  | YOLOv8                  |
|--------------|-------------------------|-------------------------|
| **仓库维护**     | 官方已停止更新                 | 官方持续维护                  |
| **API 风格**   | 依赖命令行或 `train.py`       | 统一的 `YOLO()` 接口         |
| **预训练权重兼容性** | 仅支持 `.pt` 格式的 YOLOv5 权重 | 仅支持 `.pt` 格式的 YOLOv8 权重 |

---
### Yolov5
运行`yolov5`训练:
(You can use `commandline` or `train_yolo.py` at your will.)
```bash
cd model/yolov5
pip install -r requirements.txt
cd ..
cd ..
python model/yolov5/train.py --img 800 --batch 16 --epochs 100 --data config/yolo_data.yaml --weights model/yolov5s.pt --project runs_dart --name yolo-first-try
```

 🔍 YOLOv5 要求输入为固定大小（默认是 640×640），因为：

* 模型的卷积结构需要一致输入尺寸；
* 加快训练速度（小图像 -> 快）；
* 减少显存占用；
* 保持 batch 大小不变。

🔧 如果你设置 `--img 800`，图片会被缩放到 800×800，会显著增加计算量与显存消耗。一般不推荐，除非你目标特别小且贴边。

模型训练后的实验结果会保存在`runs_dart`文件夹下（由`--project`指定）。

### Yolov8
✅ 不需要 `git clone` YOLOv8 仓库
> YOLOv8（由 Ultralytics 提供）设计为一个 PyPI 包，你只需要通过 pip install ultralytics 安装它即可，不需要手动克隆仓库，这一点是它相比 YOLOv5 的一大优势。

---

📦 安装后你能直接使用的内容包括：

* 命令行工具：`yolo`
* 模型权重自动下载（如 yolov8s.pt）
* 所有训练/验证/推理/导出等操作

---

🚀 例如，下面这些命令都可以直接用：

```bash
# 查看可用任务
yolo help

# 训练
yolo detect train model=yolov8s.pt data=config/yolo_data.yaml epochs=100 imgsz=640

# 推理
yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg
```

---

🔍 如果你**确实需要源码查看或修改底层代码**（如改网络结构）：

那你可以选择手动 clone：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .  # 开发模式安装，可调试源码
```

---
本任务选择在 COCO 数据集预训练权重 yolov8s.pt 的基础上进行微调训练：

```bash
yolo detect train \
  model=yolov8s.pt \
  data=config/yolo_data.yaml \
  epochs=10 \
  imgsz=640 \
  batch=16 \
  cfg=config/dart_hyp.yaml \
  name=dart-aug-v1 \
  project=runs_dart
```



## geometry
`geometry/dart_scoring`文件实现了两个函数：
* `calculate_dart_scores`: 计算飞镖得分
  - input
    - **calibration_points**: 4个校准点坐标列表，格式为 `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`
    - **dart_points**: 飞镖坐标列表，格式为 `[[x, y], ...]`
  - output
    - **results**: 每个飞镖的得分列表，格式为 `[{'score': 分数, 'details': 描述}, ...]`
    - **total_score**: 总得分
* `label_dart_scores`: 在飞镖盘图片上标记飞镖得分
  - input
    - **image**: numpy数组，飞镖盘图片 (800x800x3)
    - **dart_points**: 飞镖坐标列表，格式为 `[[x, y], ...]`，归一化坐标
    - **scores**: 得分列表，格式为 `[{'score': 分数, 'details': 描述}, ...]`
  - output
    - **labeled_image**: 标记了得分的numpy图片
## system

`evaluate_dart_image.py`模块提供飞镖评分系统的核心交互功能，支持摄像头实时评估与单图处理两种模式。

### `evaluate_dart_camera()` 
- **实时摄像头模式**  
  通过摄像头实时捕获画面，按需分析飞镖得分
- 快捷键：
  - `n`：捕获当前帧进行评分
  - `q`：退出程序
- 支持自定义模型路径

### `evaluate_dart_image()` 
- **单图处理模式**  
  输入图片路径，返回标记得分后的图像及详细结果
- 参数说明：
  ```python
  img_path: str          # 必选，输入图片路径
  model_path: str = None # 可选，自定义模型路径
  show_steps: bool = False # 是否显示处理中间步骤
