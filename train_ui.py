"""
PyQt5训练监控界面，包含训练参数配置、训练过程监控及结果展示功能
"""

import random
import sys
import traceback

import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QTextEdit, QComboBox,
                             QSpinBox, QFormLayout, QFileDialog, QDoubleSpinBox)
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from main import Main

# 配置matplotlib中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class TrainingThread(QThread):
    """
    训练线程类，继承自QThread，用于在后台执行训练任务

    Signals:
        update_log: 日志更新信号，传递字符串格式的日志信息
        update_metrics: 指标更新信号，传递字典格式的训练指标
        update_score: 分数更新信号，传递字典格式的最终评分
        training_finished: 训练完成信号
    """

    update_log = pyqtSignal(str)
    update_metrics = pyqtSignal(dict)
    update_score = pyqtSignal(dict)
    training_finished = pyqtSignal()

    def __init__(self, train_config, env_config):
        """
        初始化训练线程

        Args:
            train_config: 训练配置字典，包含批大小、训练轮次等参数
            env_config: 环境配置字典，包含数据集、设备类型等配置
        """
        super().__init__()
        self.train_config = train_config
        self.env_config = env_config
        self.should_continue = True  # 训练继续标志位

    def run(self):
        """执行训练主逻辑"""
        try:
            main = Main(self.train_config, self.env_config)

            def should_continue():
                """训练继续条件检查"""
                return self.should_continue

            def progress_callback(metrics):
                """训练进度回调函数"""
                self.update_metrics.emit(metrics)
                log = f"Epoch {metrics['epoch']}/{self.train_config['epoch']} - Loss: {metrics['loss']:.4f}"
                self.update_log.emit(log)

            # 启动主训练流程
            main.run(
                progress_callback=progress_callback,
                should_continue=should_continue
            )

            # 收集最终评分
            score = {
                'f1': main.final_scores['f1'],
                'precision': main.final_scores['precision'],
                'recall': main.final_scores['recall'],
                'alerts': main.final_scores.get('alerts', [])
            }
            self.update_score.emit(score)

        except Exception as e:
            # 异常处理及日志记录
            error_msg = f"训练异常:\n{traceback.format_exc()}"
            self.update_log.emit(error_msg)
        finally:
            self.training_finished.emit()

    def stop(self):
        """停止训练"""
        self.should_continue = False


class TrainingWindow(QMainWindow):
    """主界面窗口类，负责UI布局和交互逻辑"""

    def __init__(self):
        super().__init__()
        self.init_ui()  # 初始化界面
        self.setup_connections()  # 设置信号连接
        self.setWindowTitle("STGCN 训练监控系统")
        self.resize(1200, 800)
        self.alert_counts = {'low': 0, 'medium': 0, 'high': 0}  # 告警计数器

    def init_ui(self):
        """初始化用户界面布局"""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧参数面板布局
        param_group = QGroupBox("训练参数配置")
        form_layout = QFormLayout()

        # 参数控件初始化
        self.init_parameter_widgets(form_layout)

        param_group.setLayout(form_layout)
        main_layout.addWidget(param_group, stretch=1)

        # 右侧主区域布局
        right_layout = QVBoxLayout()
        self.setup_right_panel(right_layout)

        main_layout.addLayout(right_layout, stretch=4)
        self.setCentralWidget(main_widget)

    def init_parameter_widgets(self, form_layout):
        """初始化参数配置控件"""
        # 数据集选择
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['swat', 'wadi'])

        # 数值型参数控件
        self.batch_spin = self.create_spinbox(32, 102400, 5120)
        self.epoch_spin = self.create_spinbox(1, 10240, 128)
        self.slide_win_spin = self.create_spinbox(1, 32, 20)
        self.seed_spin = self.create_spinbox(0, 1000, random.randint(0, 1000))
        self.slide_stride_spin = self.create_spinbox(1, 10, 4)
        self.decay_spin = self.create_double_spinbox(0.0, 1.0, 0.01)
        self.mlp_layer_num_spin = self.create_spinbox(2, 16, 2)
        self.topk_spin = self.create_spinbox(1, 100, 16)

        # 设备选择
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])

        # 添加参数到布局
        form_layout.addRow("数据集:", self.dataset_combo)
        form_layout.addRow("批大小 (batch):", self.batch_spin)
        form_layout.addRow("训练轮次 (epoch):", self.epoch_spin)
        form_layout.addRow("滑动窗口:", self.slide_win_spin)
        form_layout.addRow("计算设备:", self.device_combo)
        form_layout.addRow("种子 (seed):", self.seed_spin)
        form_layout.addRow("滑动步长:", self.slide_stride_spin)
        form_layout.addRow("权重衰减:", self.decay_spin)
        form_layout.addRow("预测任务层数:", self.mlp_layer_num_spin)
        form_layout.addRow("Topk:", self.topk_spin)

    def setup_right_panel(self, layout):
        """设置右侧面板组件"""
        # 评分展示区
        self.score_output = QTextEdit()
        self.score_output.setReadOnly(True)
        layout.addWidget(self.score_output, stretch=1)

        # 训练曲线图
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=3)

        # 日志输出区
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output, stretch=2)

        # 控制按钮布局
        self.setup_control_buttons(layout)

    def setup_control_buttons(self, layout):
        """设置控制按钮"""
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.stop_btn = QPushButton("停止训练")
        self.save_btn = QPushButton("保存结果")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        layout.addLayout(control_layout)

    def create_spinbox(self, min_val, max_val, default):
        """创建整数调节框"""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        return spin

    def create_double_spinbox(self, min_val, max_val, default):
        """创建浮点数调节框"""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        return spin

    def setup_connections(self):
        """设置信号与槽的连接"""
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.save_btn.clicked.connect(self.save_results)

    def start_training(self):
        """启动训练流程"""
        # 清空显示区域
        self.clear_display_areas()

        # 构建配置字典
        train_config = self.build_train_config()
        env_config = self.build_env_config()

        # 创建并启动训练线程
        self.thread = TrainingThread(train_config, env_config)
        self.connect_thread_signals()
        self.start_btn.setEnabled(False)
        self.thread.start()

    def clear_display_areas(self):
        """清空所有显示内容"""
        self.score_output.clear()
        self.log_output.clear()
        ax = self.figure.gca()
        ax.clear()
        self.canvas.draw()

    def build_train_config(self):
        """构建训练配置字典"""
        return {
            'batch': self.batch_spin.value(),
            'epoch': self.epoch_spin.value(),
            'slide_win': self.slide_win_spin.value(),
            'dim': 64,
            'seed': self.seed_spin.value(),
            'slide_stride': self.slide_stride_spin.value(),
            'topk': self.topk_spin.value(),
            'decay': self.decay_spin.value(),
            'mlp_layer_num': self.mlp_layer_num_spin.value(),
            'out_layer_inter_dim': 256,
            'comment': ''
        }

    def build_env_config(self):
        """构建环境配置字典"""
        return {
            'dataset': self.dataset_combo.currentText(),
            'save_path': 'msl',
            'device': self.device_combo.currentText(),
            'report': 'best',
            'load_model_path': ''
        }

    def connect_thread_signals(self):
        """连接线程信号到槽函数"""
        self.thread.update_log.connect(self.update_log)
        self.thread.update_metrics.connect(self.update_metrics)
        self.thread.update_score.connect(self.update_score)
        self.thread.training_finished.connect(self.on_training_finished)

    def stop_training(self):
        """停止训练"""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.start_btn.setEnabled(True)

    def save_results(self):
        """保存训练结果"""
        path = QFileDialog.getSaveFileName(self, '保存结果', './results/msl', 'CSV文件 (*.csv)')[0]
        if path:
            with open(path, 'w') as f:
                f.write(self.score_output.toPlainText())

    def update_score(self, score):
        """
        更新评分显示

        Args:
            score: 包含评分和告警信息的字典
        """
        # 解析告警数据
        alert_levels = score.get('alerts', [])
        alert_text = self.parse_alert_levels(alert_levels)

        # 格式化显示文本
        score_text = f"""Results：
        F1 Score: {score['f1']:.4f}
        Precision: {score['precision']:.4f}
        Recall: {score['recall']:.4f}

        === 告警统计 ===
        {alert_text}"""

        self.score_output.setPlainText(score_text)

    def parse_alert_levels(self, alerts):
        """解析告警等级数据"""
        if not alerts:
            return "无告警数据"

        alert_counts = {
            'CRITICAL (Level 2)': alerts.count(2),
            'WARNING (Level 1)': alerts.count(1),
            'NOTICE (Level 0)': alerts.count(0)
        }
        return "\n        ".join([f"{k}: {v}" for k, v in alert_counts.items()])

    def update_metrics(self, metrics):
        """更新训练指标图表"""
        ax = self.figure.gca()
        ax.clear()
        ax.plot(metrics['loss_history'], label='训练损失', color='#1f77b4')
        ax.set_title("训练损失曲线")
        ax.set_xlabel("训练次数")
        ax.set_ylabel("损失值")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def update_log(self, log):
        """更新日志显示"""
        self.log_output.append(log)

    def on_training_finished(self):
        """训练完成处理"""
        self.start_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())
