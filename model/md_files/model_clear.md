## 2025.0514

> YOLOv5 中 train.py 中使用的 `ModelDetectionModel`，而 predict_darts.py 中使用的 Model 是 `DetectMultiBackend`，为什么不一致？

这是一个非常重要的设计问题，涉及 **YOLOv5 模型训练 vs 推理的架构差异**。

你问的是：

> ❓**为什么 `train.py` 使用的是 `DetectionModel`，而 `predict_darts.py` 使用的是 `DetectMultiBackend`？这两者为何不一致？**

---

## ✅ 简明回答

| 模块               | 用途     | 类名                 | 说明                                                                  |
| ------------------ | -------- | -------------------- | --------------------------------------------------------------------- |
| `train.py`         | 训练模型 | `DetectionModel`     | YOLOv5 原生模型结构，支持多任务训练、损失回传、forward 正常连接       |
| `predict_darts.py` | 推理部署 | `DetectMultiBackend` | 封装模型加载 + ONNX/engine/PT 自动处理，专为推理设计，不涉及训练/grad |

---

## ✅ 两者的根本区别

### 🔷 `DetectionModel`（来自 `models/yolo.py`）

- 是 YOLOv5 原生定义的训练用模型类（`nn.Module` 结构）
- 用于训练时的 `model.forward()`，内部是：

  - backbone + neck + head
  - 包括 Detect(), loss 输出，支持多任务、结构修改等

- 构造自 `.yaml` 文件 + 权重 `.pt`

**示意代码：**

```python
from models.yolo import DetectionModel
model = DetectionModel(cfg='yolov5s.yaml', ch=3, nc=5)
```

---

### 🔷 `DetectMultiBackend`（来自 `models/common.py`）

- 用于**推理部署阶段**的统一接口
- **支持多种格式的模型：**

  - PyTorch `.pt`
  - TorchScript `.torchscript`
  - ONNX `.onnx`
  - OpenVINO `.xml`
  - TensorRT `.engine`

- 自动选择适配 device（cpu/gpu）、自动 warmup、支持推理

**示意代码：**

```python
from models.common import DetectMultiBackend
model = DetectMultiBackend(weights='best.pt', device='cuda')
```

---

## ✅ 为什么训练时不能用 `DetectMultiBackend`？

- 它只提供 `.forward()` for 推理（且封装在 eval 模式）
- 不包含 `loss()`、梯度、参数优化结构
- 不能在 `.backward()` 中参与梯度传播

---

## ✅ 为什么推理时推荐用 `DetectMultiBackend`？

- 你不需要训练能力，只需要：

  - 加载模型
  - 输入图像
  - 返回预测

- 并且它**自动支持加速引擎格式导出（如 TensorRT）**
- 可在 `val.py`、`detect.py`、自定义脚本中复用

---

## ✅ 总结类比

| 功能             | 用于训练             | 用于推理部署                           |
| ---------------- | -------------------- | -------------------------------------- |
| 模型类           | `DetectionModel`     | `DetectMultiBackend`                   |
| 定义位置         | `models/yolo.py`     | `models/common.py`                     |
| 是否支持梯度反传 | ✅ 支持              | ❌ 不支持                              |
| 是否适配格式     | 只支持 `.pt`         | ✅ 支持 `.pt` / `.onnx` / `.engine` 等 |
| 推荐场景         | 训练、loss、结构设计 | 推理、部署、测试、调用 API             |

---

## Figure out what is the output in my dart prediction

请帮我搞清楚我的 YOLOv5 模型的输出是什么。
我通过打印检查了模型的输出（pred）：

```python
pred = model(img, augment=False, visualize=False)

    for i in range(len(pred)):
        p = pred[i]
        if type(p) is torch.Tensor:
            print(f"{i}-index: Tensor type: {p.shape}")
        elif type(p) is list:
            print(f"{i}-index: List length: {len(p)}")
            for c_p in p:
                print(f"type: {type(c_p)}")

    for i, t in enumerate(pred[1]):
        print(f"pred[1][{i}] shape: {t.shape}")
# 输出
# 0-index: Tensor type: torch.Size([1, 39375, 10])
# 1-index: List length: 3
# type: <class 'torch.Tensor'>
# type: <class 'torch.Tensor'>
# type: <class 'torch.Tensor'>
# pred[1][0] shape: torch.Size([1, 3, 100, 100, 10])
# pred[1][1] shape: torch.Size([1, 3, 50, 50, 10])
# pred[1][2] shape: torch.Size([1, 3, 25, 25, 10])
# 三个尺度之间是二分之一的关系？为什么是100*100; 50*50; 25:25?
```

下面的 yaml 是我在训练模型使用的传入--data 的：

```yaml
train: ../../data/yolo_dataset/images/train
val: ../../data/yolo_dataset/images/val

nc: 5 # number of classes
names: ['Board1', 'Board2', 'Board3', 'Board4', 'Dart']
```

我检查了 DetectMultiBackend 里面的 self.model 在使用 PyTorch 权重是会被赋值成什么：

```python
# code snippet of DetectMultiBackend Class in `common.py`
if pt:  # PyTorch
  model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
  stride = max(int(model.stride.max()), 32)  # model stride
  names = model.module.names if hasattr(model, "module") else model.names  # get class names
  model.half() if fp16 else model.float()
  self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
```

进一步查看`model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)`是什么：

```python
# code snippet of attempt_load function in experimental.py
def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
```

你现在非常细致地调查了 **YOLOv5 模型在推理阶段的输出结构**，这是高质量模型理解的典范。我们来逐步 **拆解你实际观察到的输出结构**，并解答你提出的几个关键问题。

---

## ✅ 你观察到的输出结构：

```plaintext
len(pred): 2
pred[0].shape: torch.Size([1, 39375, 10])
pred[1]: list of 3 tensors with shape:
 - [1, 3, 100, 100, 10]
 - [1, 3, 50, 50, 10]
 - [1, 3, 25, 25, 10]
```

---

## ✅ 一、这是什么？

这是你使用的模型中的 `Detect` 模块在 `forward()` 里返回的两个结果：

```python
return torch.cat(z, 1), z if self.export else (torch.cat(z, 1), z)
```

即：

- `pred[0] = torch.cat(z, 1)`：合并后的检测输出，shape 为 `[B, N, C]`，你看到 `[1, 39375, 10]`
- `pred[1] = z`：是 3 个尺度上的原始检测输出，shape 为：

  - `[1, 3, 100, 100, 10]` ← 对应 P3
  - `[1, 3, 50, 50, 10]` ← 对应 P4
  - `[1, 3, 25, 25, 10]` ← 对应 P5

---

## ✅ 二、这代表了什么结构？

### 🔍 pred\[0] → 合并后的检测预测：

- Shape: `[1, 39375, 10]`
- 含义：

  - `B = 1`：batch size
  - `39375 = 3 × (100×100 + 50×50 + 25×25)` → 所有 anchor 数
  - `10 = 4 + 1 + 5`

    - `4`: box 的 `[tx, ty, tw, th]`
    - `1`: objectness score
    - `5`: 你定义的 5 个类别：\['Board1', ..., 'Dart']

✅ 这是你实际用于 NMS 和推理的结果。

---

### 🔍 pred\[1] → 多尺度输出（P3、P4、P5）

- 3 层，每层：

  - `[1, 3, H, W, 10]`，代表每个尺度有 3 个 anchor
  - `H×W`：为每层特征图的分辨率
  - `10`：同样是 `[x, y, w, h, obj] + nc=5`

这 3 层之所以是：

- `[100, 100]`
- `[50, 50]`
- `[25, 25]`

是因为你的输入图像大小为 800，stride 分别为 8, 16, 32：

```plaintext
P3: 800 / 8 = 100
P4: 800 / 16 = 50
P5: 800 / 32 = 25
```

所以这是 YOLO 的标准多尺度输出。

---

## ✅ 三、这说明你的模型输出来自哪里？

在 `models/yolo.py` 的 `Detect.forward()` 中有如下代码：

```python
def forward(self, x):
    ...
    for i in range(self.nl):  # for each scale level
        ...
        z.append(output)

    return (torch.cat(z, 1), z) if not self.export else z[0]
```

这正是你 `pred = model(img)` 拿到的 `tuple(pred0, pred1)`。

---

## ✅ 四、你用的是哪个模型结构？

你加载的是 PyTorch 格式的 `.pt` 模型：

```python
model = DetectMultiBackend(weights='best.pt', device=device)
```

这个 `best.pt` 是你自己训练时通过 `train.py` 训练出来的，它本质上包含一个 `Model`（`models/yolo.py` 中定义），而这个 `Model` 中的最后一层是 `Detect`。

---

## ✅ 五、所以最终怎么用这个输出？

### ✔️ 用 `pred[0]`（\[1, 39375, 10]）做 NMS：

```python
det = non_max_suppression(pred[0], conf_thres=..., iou_thres=...)
```

你已经这样做了，这是完全正确的。

### ✔️ 如果你需要结构信息（比如结构损失）

- 可通过 `pred[1]` 拿到每个尺度的原始输出 `z`：

  - `[1, 3, H, W, 10]`
  - 可以根据 anchor/grid 反解出特定参考点的预测值（这需要 Anchor 解码）

---

## ✅ 总结：你现在的 YOLO 模型输出如下：

| 输出         | Shape              | 用途                                     |
| ------------ | ------------------ | ---------------------------------------- |
| `pred[0]`    | `[1, 39375, 10]`   | 合并后的所有 anchor 预测，用于 NMS       |
| `pred[1][i]` | `[1, 3, H, W, 10]` | 多尺度检测层的原始输出，用于解析结构特征 |

---

## ✅ 你问的几个核心问题解答：

| 问题                                   | 回答                                                                                |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| 为什么 len(pred) == 2？                | 因为 `Detect.forward()` 返回了 `(cat(z), z)` 两个结果                               |
| 为什么 pred\[1] 有三个尺度？           | 因为 YOLOv5 是三尺度检测（P3、P4、P5），对应 stride=8/16/32，分辨率分别为 100/50/25 |
| `[1, 39375, 10]` 中的 39375 从何而来？ | 3 anchors × (100×100 + 50×50 + 25×25) = 39375                                       |
| `[10]` 是什么？                        | `[x, y, w, h, obj, cls1, cls2, cls3, cls4, cls5]`，你有 5 类 → 4+1+5=10             |
| 哪个输出给 NMS 用？                    | `pred[0]`，你已经正确地用了                                                         |

---

如果你需要帮助进一步从 `pred[1]` 中提取参考点预测结构用于结构损失，或者你想**注入结构回归 head**，我可以继续帮你设计。

是否继续下一步？例如：

- 提取 `pred[1]` 中每层的结构点坐标？
- 或设计一个只输出 4 点坐标的新 head？

> attemp_load 是干什么的？为什么感觉很难清晰地看出推理使用的模型结构？

```python
{'nc': 5,
'depth_multiple': 0.33,
'width_multiple': 0.5,
'anchors': [
  [10, 13, 16, 30, 33, 23],
  [30, 61, 62, 45, 59, 119],
  [116, 90, 156, 198, 373, 326]],

'backbone': [
  [-1, 1, 'Conv', [64, 6, 2, 2]],
  [-1, 1, 'Conv', [128, 3, 2]],
  [-1, 3, 'C3', [128]],
  [-1, 1, 'Conv', [256, 3, 2]],
  [-1, 6, 'C3', [256]],
  [-1, 1, 'Conv', [512, 3, 2]],
  [-1, 9, 'C3', [512]],
  [-1, 1, 'Conv', [1024, 3, 2]],
  [-1, 3, 'C3', [1024]],
  [-1, 1, 'SPPF', [1024, 5]]],

  'head': [
    [-1, 1, 'Conv', [512, 1, 1]],
    [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 6], 1, 'Concat', [1]],
    [-1, 3, 'C3', [512, False]],
    [-1, 1, 'Conv', [256, 1, 1]],
    [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]],
    [-1, 3, 'C3', [256, False]],
    [-1, 1, 'Conv', [256, 3, 2]],
    [[-1, 14], 1, 'Concat', [1]],
    [-1, 3, 'C3', [512, False]],
    [-1, 1, 'Conv', [512, 3, 2]],
     [[-1, 10], 1, 'Concat', [1]],
     [-1, 3, 'C3', [1024, False]],
     [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]],

  'ch': 3
}
{0: 'Board1', 1: 'Board2', 2: 'Board3', 3: 'Board4', 4: 'Dart'}
```

![model](yolov5.jpg)

```python
Fusing layers...
Model summary: 157 layers, 7023610 parameters, 0 gradients, 15.8 GFLOPs
DetectionModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (8): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (9): SPPF(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    (10): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (11): Upsample(scale_factor=2.0, mode='nearest')
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (act): SiLU(inplace=True)
    )
    (15): Upsample(scale_factor=2.0, mode='nearest')
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (act): SiLU(inplace=True)
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 30, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 30, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 30, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
```
