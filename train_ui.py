# ui/train_ui.py
import random
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QTextEdit, QComboBox,
                             QSpinBox, QFormLayout, QFileDialog, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from main import Main
from matplotlib import rcParams


# 设置中文字体（必须在导入其他matplotlib模块前配置）
rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



class TrainingThread(QThread):
    update_log = pyqtSignal(str)
    update_metrics = pyqtSignal(dict)
    update_score = pyqtSignal(dict)
    training_finished = pyqtSignal()

    def __init__(self, train_config, env_config):
        super().__init__()
        self.train_config = train_config
        self.env_config = env_config
        self._should_continue = True

    def run(self):
        try:
            main = Main(self.train_config, self.env_config)

            def should_continue():
                return self._should_continue

            def progress_callback(metrics):
                self.update_metrics.emit(metrics)
                log = f"Epoch {metrics['epoch']}/{self.train_config['epoch']} - Loss: {metrics['loss']:.4f}"
                self.update_log.emit(log)

            main.run(
                progress_callback=progress_callback,
                should_continue=should_continue
            )

            score = {
                'f1': main.final_scores['f1'],
                'precision': main.final_scores['precision'],
                'recall': main.final_scores['recall']
            }
            self.update_score.emit(score)


        except Exception as e:
            self.update_log.emit(f"训练异常: {str(e)}")
        finally:
            self.training_finished.emit()

    def stop(self):
        self._should_continue = False


class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_connections()
        self.setWindowTitle("STGCN 训练监控系统")
        self.resize(1200, 800)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧参数面板
        param_group = QGroupBox("训练参数配置")
        form_layout = QFormLayout()

        # 新增参数控件
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['msl'])

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 10240)
        self.batch_spin.setValue(10240)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 10240)
        self.epoch_spin.setValue(512)

        self.slide_win_spin = QSpinBox()
        self.slide_win_spin.setRange(1, 30)
        self.slide_win_spin.setValue(16)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])

        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.00000, 1.00000)
        self.decay_spin.setSingleStep(0.001)
        self.decay_spin.setValue(0.005)

        self.out_layer_num_spin = QSpinBox()
        self.out_layer_num_spin.setRange(1, 5)
        self.out_layer_num_spin.setValue(1)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(16)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1000)
        # 生成随机种子（0-1000之间的整数）并设置初始值
        initial_seed = random.randint(0, 1000)
        self.seed_spin.setValue(initial_seed)

        self.slide_stride_spin = QSpinBox()
        self.slide_stride_spin.setRange(1, 10)
        self.slide_stride_spin.setValue(4)

        # 添加参数到表单
        form_layout.addRow("数据集:", self.dataset_combo)
        form_layout.addRow("批大小 (batch):", self.batch_spin)
        form_layout.addRow("训练轮次 (epoch):", self.epoch_spin)
        form_layout.addRow("滑动窗口:", self.slide_win_spin)
        form_layout.addRow("计算设备:", self.device_combo)
        form_layout.addRow("种子 (seed):", self.seed_spin)
        form_layout.addRow("滑动步长:", self.slide_stride_spin)
        form_layout.addRow("权重衰减:", self.decay_spin)
        form_layout.addRow("输出层数:", self.out_layer_num_spin)
        form_layout.addRow("Topk:", self.topk_spin)

        param_group.setLayout(form_layout)
        main_layout.addWidget(param_group, stretch=1)

        # 右侧主区域
        right_layout = QVBoxLayout()

        # 评分展示区
        self.score_output = QTextEdit()
        self.score_output.setReadOnly(True)
        right_layout.addWidget(self.score_output, stretch=1)

        # 训练曲线
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, stretch=3)

        # 日志面板
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_layout.addWidget(self.log_output, stretch=2)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.stop_btn = QPushButton("停止训练")
        self.save_btn = QPushButton("保存结果")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        right_layout.addLayout(control_layout)

        main_layout.addLayout(right_layout, stretch=4)
        self.setCentralWidget(main_widget)

    def setup_connections(self):
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.save_btn.clicked.connect(self.save_results)

    def start_training(self):
        # 清空所有显示区域
        self.score_output.clear()          # 清空评分区
        self.log_output.clear()            # 清空日志区
        ax = self.figure.gca()             # 清空曲线图
        ax.clear()
        self.canvas.draw()

        train_config = {
            'batch': self.batch_spin.value(),
            'epoch': self.epoch_spin.value(),
            'slide_win': self.slide_win_spin.value(),
            'dim': 64,
            'val_ratio': 0.1,
            'seed': self.seed_spin.value(),
            'slide_stride': self.slide_stride_spin.value(),
            'topk': self.topk_spin.value(),
            'decay': self.decay_spin.value(),
            'out_layer_num': self.out_layer_num_spin.value(),
            'out_layer_inter_dim': 256,
            'comment': ''
        }

        env_config = {
            'dataset': 'msl',
            'save_path': 'msl',
            'device': self.device_combo.currentText(),
            'report': 'best',
            'load_model_path': ''
        }

        self.thread = TrainingThread(train_config, env_config)
        self.thread.update_log.connect(self.update_log)
        self.thread.update_metrics.connect(self.update_metrics)
        self.thread.update_score.connect(self.update_score)
        self.thread.training_finished.connect(self.on_training_finished)

        self.start_btn.setEnabled(False)
        self.thread.start()

    def stop_training(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.start_btn.setEnabled(True)

    def save_results(self):
        path = QFileDialog.getSaveFileName(self, '保存结果', './results/msl', 'CSV文件 (*.csv)')[0]
        if path:
            with open(path, 'w') as f:
                f.write(self.score_output.toPlainText())

    def update_score(self, score):
        score_text = f"""最终评估结果：
        F1 Score: {score['f1']:.4f}
        Precision: {score['precision']:.4f}
        Recall: {score['recall']:.4f}"""
        self.score_output.setPlainText(score_text)

    def update_metrics(self, metrics):
        ax = self.figure.gca()
        ax.clear()
        ax.plot(metrics['loss_history'], label='训练损失', color='#1f77b4')
        ax.set_title("训练损失曲线")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("损失值")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def update_log(self, log):
        self.log_output.append(log)

    def on_training_finished(self):
        self.start_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())