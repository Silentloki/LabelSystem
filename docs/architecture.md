# 代码结构

```text
main.py                         应用启动、字体、样式和多进程初始化
UI/Project.py                   工程管理入口
UI/WorkWindow.py + .ui          原有三页界面
LabelItem/                     标注画布、矩形、多边形交互
Utils/test.py                  主工作台与页面业务集成（历史文件名）
Utils/AnnotationImporter.py    外部标注 / YOLO 数据导入
Utils/Task.py                  数据集生成与同源划分
Utils/DetectionAugment.py      检测标签旋转 / 平移增强
Utils/yoloTool_fixed.py        本地训练与推理线程
Utils/TrainingRunRecord.py     训练记录
Utils/AITrainingAnalysis*      独立训练分析服务与窗口
Utils/AIChat*                  对话窗口、历史、选择和训练图表
Utils/AIWorkAgent.py           模型请求、工具循环与回复校验
Utils/AIWorkTools.py           领域工具注册与范围路由
Utils/AIWorkspace.py           工程内文件、指纹、事务及恢复
Utils/ProjectContext*          项目资源、关系和人工记录
Utils/AICandidates.py          模型候选生成、对照和采用
Utils/AIPrediction*            历史预测统计、附图和报告校验
Utils/AISelectedAnalysis.py    选中照片分页看图
Utils/AIResult*                结果浏览、筛选、隐藏/删除及问题集
tests/                         合成数据与模拟接口测试
examples/                      合成示例工程生成器
```

数据保存在工程目录中，公共源码不依赖预置数据库。`AI.db` 管理本机工程列表；工程内 `ai_workbench/project_context.sqlite` 保存项目资源和人工记录。模型与API配置由使用者提供。

软件计算数量、坐标和指纹；AI 解释这些证据并选择可用工具。正式修改通过具体预览和事务执行。图像分析必须在后续请求实际附送图片，再校验逐图报告。

当前保留 PyQt5 框架和现有页面，不混入已撤回的整套界面改版，也不包含已删除的旧 AI 标注检查产品。
