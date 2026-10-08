# 第三方组件说明

本仓库主要发布 LabelSystem 自身源码。依赖通过 Python 包管理器安装，不在仓库内重新打包；各组件的许可证以对应发布版本为准。

| 组件 | 用途 | 项目 / 许可说明 |
| --- | --- | --- |
| PyQt5 | 桌面界面、Qt SQL、图形视图 | [Riverbank](https://www.riverbankcomputing.com/software/pyqt/intro)，GPL v3 / 商业授权 |
| Ultralytics | YOLO训练与推理 | [许可说明](https://www.ultralytics.com/license)，AGPL-3.0 / Enterprise |
| PyTorch | 本地深度学习运行时 | [项目](https://github.com/pytorch/pytorch)，以安装版本的LICENSE为准 |
| OpenCV | 图像处理 | [项目](https://github.com/opencv/opencv)，以安装版本的LICENSE为准 |
| NumPy | 数值计算 | [项目](https://github.com/numpy/numpy) |
| Matplotlib | 训练与统计图 | [项目](https://github.com/matplotlib/matplotlib) |
| Pillow | 图片读写与叠加 | [项目](https://github.com/python-pillow/Pillow) |
| pyqtgraph | 实时训练指标绘图 | [项目](https://github.com/pyqtgraph/pyqtgraph) |
| PyYAML | 数据配置解析 | [项目](https://github.com/yaml/pyyaml) |
| openpyxl | 表格数据处理依赖 | [项目](https://openpyxl.readthedocs.io/) |

`LICENSE` 提供标准 GNU AGPL v3.0 文本，未把第三方组件重新授权为本项目自有代码。`res/` 包含本软件现用的小图标与样式，示例图片由 `examples/create_demo_project.py` 生成。

本仓库不附带业务样本、专用模型、商用软件手册或原工作目录的运行数据。使用者自行获取模型与数据并遵守其授权；调用外部AI服务需遵守服务方条款。
