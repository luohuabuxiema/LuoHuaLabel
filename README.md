# LuoHuaLabel - 基于 SAM3 的智能图像标注系统


# 前言
用AI打败AI，AI全自动开发，效率翻10倍，由于自己的项目需要标注数据集，之前用过 labelme、labelimg工具，于是自己想了想能不能结合 SAM3 自己开发一个智能标注工具，研究了几天中间也发现了好多问题，最终通过 AI 辅助编程，完成一个属于自己的标注工具。**吊打 labelme、labelimg 工具**。

 源码地址：[https://github.com/luohuabuxiema/LuoHuaLabel](https://github.com/luohuabuxiema/LuoHuaLabel/)

# 系统简介

系统基于 PySide6 构建，集成了最新一代的 **Segment Anything Model 3 (SAM3)**，基于 SAM3 的**点提示与提示词提示**功能，我于是开发了智能点选和提示词选取功能，当系统开启了 SAM3 智能标注功能时候，可以智能<font color=#FFA500>**点选**</font>进行标注或者<font color=#FFA500>**提示词**</font>自动分割，除了点（Point）标注不可以使用 SAM3 智能标注，剩下的➡️矩形（HBB）、多边形（Polygon）、 OBB 旋转框标注都支持开启智能标注，标注完系统还➡️<font color=#FFA500>支持数据集转换和划分。</font>


| <font color = #FF0000>功能 | 界面演示                                |
| -------------------------- | --------------------------------------- |
| 多边形标注—点选            | ![在这里插入图片描述](assets/img.png)   |
| 多边形标注—点选结果        | ![在这里插入图片描述](assets/img_1.png) |
| obb标注—点选               | ![在这里插入图片描述](assets/img_2.png) |
| obb标注—点选结果           | ![在这里插入图片描述](assets/img_3.png) |
| 目标检测—点选结果          | ![在这里插入图片描述](assets/img_4.png) |
| 目标检测—提示词结果        | ![在这里插入图片描述](assets/img_5.png) |
| 数据集处理系统             | ![在这里插入图片描述](assets/img_6.png) |



>注意：点标注有待优化，这个功能待测试

## ✨ 核心功能特性

- **🚀 AI 智能辅助 (SAM3 驱动)**：支持鼠标悬停预览、单点极速提取轮廓、以及输入文本提示词（Prompt）实现全图目标的批量自动化打框。
- **📐 全能标注模式**：支持 矩形 (Rect)、多边形 (Poly)、点 (Point) 以及 **原生 OBB 旋转框** 标注。
- **🔄 极致 OBB 交互**：独创的旋转框控制手柄，支持 360° 无极顺滑旋转、独立拉伸，并自带“贴墙滑动”的严格图像边界碰撞检测。
- **🕰️ 时光机机制**：自带最高 20 步的撤销/重做（Undo/Redo）状态栈，随时纠正误操作。
- **💾 多格式无缝切换**：支持原生保存与读取 JSON、YOLO (.txt, 自动归一化及反推)、XML (Pascal VOC) 格式。
- **🗄️ 内置数据集处理系统**：独立线程运行的数据集处理子系统，支持一键划分训练/验证/测试集，以及 JSON 到 U-Net Mask、JSON/XML 到 YOLO 的格式清洗与转换。

------

## 🛠️ 部署与运行环境

### 1. 环境依赖

推荐使用 Python 3.10+。首先安装必要的 Python 依赖包：

单独安装 torch>=2.5.0，pytorch 官网地址： [https://pytorch.org/](https://pytorch.org/get-started/previous-versions/?_gl=1*r08hqw*_up*MQ..*_ga*MTg1ODQzMTE5LjE3NzU4ODk5NDI.*_ga_469Y0W5V62*czE3NzU4ODk5NDEkbzEkZzAkdDE3NzU4ODk5NDEkajYwJGwwJGgw/)
![在这里插入图片描述](assets/img_9.png)

**💡 PyTorch 安装注意事项（新手必看）**

在进行安装 PyTorch 之前，请大家务必核对以下几点，避免安装后运行报错：

**1. 确认显卡支持与 CUDA 版本（极其重要）**
* **适用系统**：本教程基于 Windows 环境。
* **如何查看**：按下 `Win + R` 键，输入 `cmd` 打开命令提示符，输入 `nvidia-smi` 并回车。在弹出的表格右上角，找到 **CUDA Version**。
![在这里插入图片描述](assets/img_8.png)

* **版本匹配要求**：你下载的 PyTorch CUDA 版本（例如命令中的 `cu118` 或 `cu116`），**必须小于或等于**你电脑刚刚查到的 CUDA Version。如果你的电脑没有独立 N 卡，或者查不到该信息，请到官网选择 **CPU 版本**的安装命令。


**2. Conda 与 Pip 命令二选一即可**


根据自己电脑安装指定版本，安装命令如下，如果你使用 `conda` 命令卡住，可以尝试先在终端配置好国内的清华/中科大 conda 镜像源，然后再删掉命令后面的 `-c pytorch -c nvidia`（因为带上 `-c` 会强制去国外官方频道下载）：


```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

上面的命令，大多数情况下是安装失败的，所以这里推荐使用阿里云镜像源安装，阿里云上镜像，pytorch gpu版的 whl 包可以在此链接查看：链接: [https://mirrors.aliyun.com/pytorch-wheels](https://mirrors.aliyun.com/pytorch-wheels/)

后面 cu 版本需要对应cuda 的版本号，例如安装 cuda11.8 就写  cu118

```bash
-f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
cuda11.8 安装命令：
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
cuda12.1 安装命令：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu121
```

**3. 验证是否安装成功**
安装进度条跑完后，不要急着关掉窗口！在终端里输入 `python` ，输入以下代码：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(f"CUDA：{torch.version.cuda}")
```
如果输出了 `True`，恭喜你，cuda 可用！如果输出 `False`，说明装成了 CPU 版本或者 CUDA 不匹配，可能需要卸载重装。

---


之后在自己的虚拟环境下，使用下面命令安装所需的库

```
pip install -r requirements.txt
```

>注：如果你需要使用 SAM3 的智能辅助功能，请确保你的环境中已经正确配置了 `sam3` 相关的库及其依赖。

### 2. 模型下载与配置

sam3 官方源码地址: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3/)

sam3.pt 官方权重下载地址: [https://huggingface.co/facebook/sam3/tree/main](https://huggingface.co/facebook/sam3/tree/main/)
![在这里插入图片描述](assets/img_7.png)

为了启用 AI 智能标注，你需要下载 SAM3 模型权重文件（ `sam3.pt`）。

下载后，请打开 `main.py`，在 `MainWindow` 的 `__init__` 方法中，修改模型的绝对路径：

```
self.sam_client.load_model_async(r"你的模型绝对路径/sam3.pt") 
```

### 3. 启动系统

在终端或命令行中运行主程序：

Bash

```
python main.py
```

------

## 📖 用户操作指南

### 基本工作流

1. **打开目录**：点击左侧工具栏的“打开目录”，选择包含图片的文件夹。
2. **选择格式**：在左侧下拉菜单中选择你需要的标注与保存格式（JSON / YOLO / XML）。
3. **开始标注**：
   - **手动标注**：在左侧选择对应工具（快捷键 R/P/T/O），在画板上拖拽或点击即可绘制。
   - **智能标注**：打开左下角的 **SAM 智能辅助** 开关。
     - *点选*：鼠标移入目标，出现绿色预览框/多边形后，左键点击确认。
     - *提示词*：在右下角输入目标英文单词（如 `dog`），按回车即可全图批量打框。
4. **修改属性**：绘制完成后，弹出类别选择框。可通过双击已有框（或快捷键 `E`）重新修改标签。
5. **保存导出**：使用快捷键 `Ctrl + S` 手动保存，或在切换下一张图片时系统会自动保存。

### ⌨️ 快捷键大全

- **A / 左方向键**：上一张图片
- **D / 右方向键**：下一张图片
- **Ctrl + S**：手动保存当前标注
- **Q**：开启/关闭 SAM 智能辅助
- **R / P / T / O**：分别切换至 矩形 / 多边形 / 点 / 旋转框标注
- **Del / Backspace**：删除选中的标注框
- **Ctrl + Z**：撤销上一步操作（支持 20 步）
- **Ctrl + Y (或 Ctrl + Shift + Z)**：重做前进
- **Z / X / C / V**：OBB 旋转框快捷微调角度

------

## ⚠️ 注意事项与边界说明

1. **OBB 旋转框出界限制**：为了符合严格的模型训练标准，系统在 OBB 的拖拽与旋转中加入了强物理碰撞检测。旋转框的四个角点**无法被拖出或旋转出图片的实际边界外**，当触碰边界时，框体会展现出“贴墙滑动”的阻尼感。
2. **SAM 初始化延迟**：软件启动后，SAM3 模型会在后台独立线程（`ModelLoadWorker`）中加载。加载完成前，你可以正常进行手动标注。如果在加载期间开启了 SAM 辅助，系统会在加载完毕后自动为你补做特征提取。
3. **空画布保存逻辑**：如果在某张图片上删除了所有的标注框，再次触发保存时，系统会生成一份空文件覆盖原有文件。这是为了确保“彻底清除该图片标注”的操作能够正确落地到硬盘上。
4. **点标注的局限性**：在“点标注”模式下，SAM 智能辅助与提示词提取会自动被禁用置灰，以防止生成无意义的数据。

------

## 💻 核心开发文档

本系统的架构遵循高内聚、低耦合的设计原则，前端 UI 与底层模型推理完全解耦。

### 1. 模块结构

- `main.py`：主控中心。负责 UI 事件绑定、状态快照引擎（撤销/重做）、快捷键路由、数据读取与反推解析。
- `core/canvas.py` (`Canvas`)：核心画板。继承自 `QGraphicsScene`，负责接管鼠标事件、画板状态拦截、SAM 推理结果渲染与预览图层控制。
- `core/shapes.py`：图形基类与派生类。包含所有可视化组件的重写逻辑， `RotatedRectShape`（旋转框及手柄体系）。
- `core/sam_client.py`：AI 通信总线。包含 `SamInferenceWorker`（推理线程），处理与 PyTorch 及 SAM3 模型的张量计算和特征提取。
- `core/exporter.py`：数据序列化引擎。使用 `mapToScene` 提取全局坐标，完成 JSON/YOLO/XML 的格式化导出。
- `main_dataset_tool.py`：数据集处理子系统，多进程独立运行防阻塞。

### 2. 核心方法

#### 1. OBB 极坐标逆向计算与旋转拉伸 

在 `handle_dragged` 中，系统将鼠标全局坐标映射回框体的局部坐标 (`mapFromScene`)。拉伸上下左右胶囊时，不仅改变 `box_w/box_h`，利用 `mapToScene` 计算出中心点在拉伸后的真实世界偏移量（`scene_offset`），实现了“锚点固定拉伸”。旋转则使用了极坐标系计算：

```
angle_deg = math.degrees(math.atan2(dy, dx))
self.setRotation(angle_deg + 90)
```

#### 2. 时光机状态快照引擎 

为了避免复杂的对象深拷贝，系统采用了**数据快照**模式。

每当画布发生 `state_changed` 信号（画完、删除、松开手柄），系统会直接调用 `Exporter.extract_shapes` 将画布的物理状态榨取成纯粹的 Dict 数据存入栈中。并自带 JSON 字符串去重比对，防止用户无意义的点击消耗堆栈内存。

#### 3. SAM 推理结果提取 OBB 

当 SAM 生成出不规则的多边形 Mask 后，为了支持将 AI 预测转化为 OBB 格式，系统在寻边（`findContours`）后，除了计算标准的 `cv2.boundingRect`，同时调用了 `cv2.minAreaRect(largest_contour)`。可以直接抛出带有倾斜角度的精准外接框，极大地提升了遥感目标标注的效率。

更多功能待更新。。。

------

## 🚧 存在不足

LuoHuaLabel 虽然核心的标注流、数据清洗以及 SAM3 智能引标注已跑通并验证可行，但受限于个人精力与测试环境，系统仍存在一些不完善之处。

1. **潜在的边缘 Bug**：在极高频的快捷键切换、或极端形变的旋转框（OBB）物理碰撞测试中，偶尔可能出现图层刷新延迟或极小概率的坐标精度溢出问题。
2. **大尺度图像性能瓶颈**：目前画板基于 PySide6 的 `QGraphicsScene` 构建。在载入超大分辨率图片（如 4K 乃至 8K 无人机遥感图）并叠加成百上千个多边形掩码时，缩放和拖拽的帧率可能会有所下降。
3. **显存占用**：SAM3 模型作为重量级的视觉大模型，对设备的 CUDA 显存有一定要求。在低显存设备上，"提示词全图提取" 功能可能会因为算力不足而导致界面短暂假死。
4. **格式支持相对局限**：目前的导入导出主要针对常用的 YOLO (HBB/OBB)、LabelMe 风格 JSON 和 Pascal VOC XML 进行了深度适配，暂时缺乏对 COCO 格式原生导出以及部分细分领域格式的直接支持。

------

## 🤝 欢迎二次开发 

非常欢迎且鼓励广大开发者对其进行 Fork 和二次开发！

系统的代码架构已经做好了高度的模块化解耦，你可以非常轻松地在以下方向大展身手：

- **🧩 接入更多大模型**：`core/sam_client.py` 作为一个独立的 AI 通信总线，你可以很方便地将 SAM3 替换为轻量级的 MobileSAM、FastSAM，甚至是接入你自己的垂直领域检测模型来实现预标注。
- **🛠️ 丰富数据处理中心**：你可以在 `main_dataset_tool.py` 中编写更多的数据清洗脚本，例如加入马赛克增强（Mosaic）、图像裁切、数据增强（Data Augmentation）等。
- **🎨 定制化 UI 与交互**：基于 Qt 强大的自绘能力，你可以继续完善 `core/shapes.py`，为旋转框加入更多操作逻辑，或是优化右侧的图层树管理。


---
## 总结

对于有帮助可以一键三连，谢谢各位观众老爷！！！。

## 参考文章


 [sam3本地部署](https://nuyoahinuhz.blog.csdn.net/article/details/159932680?spm=1001.2014.3001.5502/)

### 引用

```bash
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```
