import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QSizePolicy, QHBoxLayout, QGroupBox,
                             QComboBox, QPushButton, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


# 设置中文显示支持
def setup_chinese_support():
    # 尝试使用系统中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        # 如果系统字体不可用，使用内置字体（需要安装）
        try:
            import matplotlib.font_manager as fm
            font_path = fm.findfont(fm.FontProperties(family=['sans-serif']))
            plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
        except:
            print("中文支持配置失败，图表可能无法正确显示中文")


class ChartApp(QWidget):
    setup_chinese_support()
    def __init__(self):
        super().__init__()
        # self.data = {
        #     '软边': 68, '脏污': 54, '胶粒': 32, '撞角': 23,
        #     '撞伤': 40, '划痕': 35, '折痕': 63, '破损': 18,
        #     '凹陷': 5, '爆角': 9, '凹坑': 2, '卷角': 2
        # }
        self.data = {}
        self.data_sets = {}
        self.current_dimension = None
        self.title = '产品缺陷分布 - 条形图'
        self.xLabel = '缺陷类型'
        self.yLabel = '缺陷数量'
        self.init_ui()

        self.draw_chart(chart_type='bar')

    def init_ui(self):
        # 设置窗口属性
        self.setWindowTitle("产品缺陷分析系统")
        self.setFixedSize(1100, 760)

        # 主布局
        main_layout = QVBoxLayout(self)

        # 控制面板
        control_group = QGroupBox("图表控制")
        control_layout = QHBoxLayout()

        # 统计维度选择
        self.dimension_combo = QComboBox()
        self.dimension_combo.setMinimumWidth(190)
        self.dimension_combo.currentIndexChanged.connect(self.on_dimension_change)

        # 图表类型选择
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.setMinimumWidth(120)
        self.chart_type_combo.addItems(["条形图", "饼图"])
        self.chart_type_combo.setCurrentText("条形图")
        self.chart_type_combo.currentIndexChanged.connect(self.on_chart_type_change)

        # 排序方式选择
        self.sort_combo = QComboBox()
        self.sort_combo.setMinimumWidth(180)
        self.sort_combo.addItems(["按名称排序", "按数值排序(降序)", "按数值排序(升序)"])
        self.sort_combo.setCurrentText("按名称排序")
        self.sort_combo.currentIndexChanged.connect(self.on_sort_change)

        # 刷新按钮
        refresh_btn = QPushButton("刷新图表")
        refresh_btn.clicked.connect(self.refresh_chart)

        # 添加控件到控制面板
        control_layout.addWidget(QLabel("统计维度:"))
        control_layout.addWidget(self.dimension_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("图表类型:"))
        control_layout.addWidget(self.chart_type_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("排序方式:"))
        control_layout.addWidget(self.sort_combo)
        control_layout.addStretch()
        control_layout.addWidget(refresh_btn)
        control_group.setLayout(control_layout)

        # 创建图表区域
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 添加控件到主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.canvas)

    def prepare_data(self):
        """准备数据，根据选择的排序方式排序"""
        sort_option = self.sort_combo.currentText()

        if sort_option == "按名称排序":
            # 按缺陷名称排序
            labels = sorted(self.data.keys())
            values = [self.data[k] for k in labels]
        elif sort_option == "按数值排序(降序)":
            # 按缺陷数量降序排序
            sorted_items = sorted(self.data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
        else:  # 按数值排序(升序)
            sorted_items = sorted(self.data.items(), key=lambda x: x[1])
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

        return labels, values

    def draw_chart(self, chart_type='bar'):
        """根据选择的类型绘制图表"""
        # 清除之前的图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if len(self.data):
            # 准备数据
            labels, values = self.prepare_data()

            # 根据图表类型绘制
            if chart_type == 'bar':
                # 创建条形图
                colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
                bars = ax.bar(labels, values, color=colors)

                # 添加数据标签
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9)

                # 设置图表标题和标签
                ax.set_title(self.title, fontsize=16, fontweight='bold')
                ax.set_xlabel(self.xLabel, fontsize=12)
                ax.set_ylabel(self.yLabel, fontsize=12)

                # 旋转x轴标签以避免重叠
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)

                # 设置y轴范围
                ax.set_ylim(0, max(values) * 1.15)

                # 添加网格线
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            else:  # 饼图
                # 设置饼图爆炸效果（突出显示）
                explode = [0.05 if v > 30 else 0 for v in values]

                # 创建饼图
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
                    explode=explode,
                    shadow=True,
                    startangle=90,
                    pctdistance=0.8,
                    textprops={'fontsize': 9}
                )

                # 设置饼图属性
                ax.set_title('产品缺陷分布 - 饼图', fontsize=16, fontweight='bold')
                ax.axis('equal')  # 确保饼图是圆形

                # 添加上角图例
                ax.legend(wedges, labels,
                          title="缺陷类型",
                          loc="center left",
                          bbox_to_anchor=(1, 0.5),
                          fontsize=9)

            # 优化布局
            self.figure.tight_layout()

            # 更新画布
            self.canvas.draw()

    def on_chart_type_change(self):
        """图表类型改变时更新图表"""
        chart_type = self.chart_type_combo.currentText()
        chart_type = 'bar' if chart_type == '条形图' else 'pie'
        self.draw_chart(chart_type)

    def on_sort_change(self):
        """排序方式改变时更新图表"""
        chart_type = self.chart_type_combo.currentText()
        chart_type = 'bar' if chart_type == '条形图' else 'pie'
        self.draw_chart(chart_type)

    def refresh_chart(self):
        """刷新图表"""
        chart_type = self.chart_type_combo.currentText()
        chart_type = 'bar' if chart_type == '条形图' else 'pie'
        self.draw_chart(chart_type)

    def setdata(self,data):
        self.data = data

    def set_dimension_data(self, data_sets):
        self.data_sets = data_sets or {}
        self.dimension_combo.blockSignals(True)
        self.dimension_combo.clear()
        self.dimension_combo.addItems(list(self.data_sets.keys()))
        self.dimension_combo.blockSignals(False)
        if self.dimension_combo.count() > 0:
            self.dimension_combo.setCurrentIndex(0)
            self.apply_dimension(self.dimension_combo.currentText())

    def apply_dimension(self, name):
        if name not in self.data_sets:
            return
        config = self.data_sets[name]
        self.current_dimension = name
        self.data = config.get('data', {})
        self.title = config.get('title', name)
        self.xLabel = config.get('xLabel', '名称')
        self.yLabel = config.get('yLabel', '数量')
        self.refresh_chart()

    def on_dimension_change(self):
        self.apply_dimension(self.dimension_combo.currentText())


if __name__ == "__main__":
    # 初始化应用
    app = QApplication(sys.argv)

    # 设置中文支持
    setup_chinese_support()

    # 创建并显示窗口
    window = ChartApp()

    keys = ['长外侧', '短外侧', '长内侧', '短内侧', '内底面', '棱', '角', '外底面']
    values = [3282, 2188, 3282, 2188, 547, 5470, 2188, 547]
    for i, key in enumerate(keys):
        window.data[key] = values[i]
    window.title = '坏品位置统计'
    window.xLabel = '位置名称'
    window.yLabel = '采图数量'
    window.show()
    sys.exit(app.exec_())
