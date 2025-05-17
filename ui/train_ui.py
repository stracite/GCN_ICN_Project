# ui/train_ui.py
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QTextEdit, QComboBox,
                             QSpinBox, QFormLayout, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from main import Main


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
            # 初始化主训练模块
            main = Main(self.train_config, self.env_config)

            # 新增中断检查函数
            def should_continue():
                return self._should_continue

            # 重写回调函数获取训练指标
            def progress_callback(metrics):
                self.update_metrics.emit(metrics)
                log = f"Epoch {metrics['epoch']}/{self.train_config['epoch']} - loss: {metrics['loss']:.4f}"
                self.update_log.emit(log)

            # 运行训练并获取最终评分
            main.run(
                progress_callback=progress_callback,
                should_continue=should_continue  # 传入中断检查
            )

            # 获取最终评分结果
            score = {
                'f1': main.test_result[0],
                'precision': main.test_result[1],
                'recall': main.test_result[2]
            }
            self.update_score.emit(score)

        except Exception as e:
            self.update_log.emit(f"训练异常: {str(e)}")
        finally:
            self.training_finished.emit()



    def stop(self):
        self._should_continue = False  # 安全终止方式


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

        # 核心参数配置
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['msl'])

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(32, 4096)
        self.batch_spin.setValue(1024)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(64)

        self.slide_win_spin = QSpinBox()
        self.slide_win_spin.setRange(1, 30)
        self.slide_win_spin.setValue(15)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])

        # 添加参数到表单
        form_layout.addRow("数据集:", self.dataset_combo)
        form_layout.addRow("批大小 (batch):", self.batch_spin)
        form_layout.addRow("训练轮次 (epoch):", self.epoch_spin)
        form_layout.addRow("滑动窗口:", self.slide_win_spin)
        form_layout.addRow("计算设备:", self.device_combo)

        param_group.setLayout(form_layout)
        main_layout.addWidget(param_group, stretch=1)

        # 右侧主区域
        right_layout = QVBoxLayout()

        # 评分展示区
        self.score_output = QTextEdit()
        self.score_output.setReadOnly(True)
        self.score_output.setStyleSheet("font-size: 14px;")
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
        self.statusBar().showMessage("准备就绪")

    def setup_connections(self):
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)  # 新增绑定
        self.save_btn.clicked.connect(self.save_results)  # 新增绑定

    def start_training(self):
        # 收集训练参数
        train_config = {
            'batch': self.batch_spin.value(),
            'epoch': self.epoch_spin.value(),
            'slide_win': self.slide_win_spin.value(),
            'dim': 64,
            'val_ratio': 0.1,
            'seed': 42,
            'slide_stride': 2,
            'topk': 20
        }

        env_config = {
            'dataset': 'msl',
            'save_path': 'msl',
            'device': self.device_combo.currentText(),
            'slide_stride': 2,
            'report': 'best'
        }

        self.thread.training_finished.connect(self.on_training_finished)

        # 初始化训练线程
        self.thread = TrainingThread(train_config, env_config)
        self.thread.update_log.connect(self.update_log)
        self.thread.update_metrics.connect(self.update_metrics)
        self.thread.update_score.connect(self.update_score)
        self.thread.training_finished.connect(self.on_training_finished)

        self.start_btn.setEnabled(False)
        self.thread.start()

    # 新增停止训练方法
    def stop_training(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.quit()
            self.start_btn.setEnabled(True)
            self.statusBar().showMessage("训练已中止", 3000)

    def save_results(self):
        path = QFileDialog.getSaveFileName(self, '保存结果', './results', 'CSV文件 (*.csv)')[0]
        if path:
            with open(path, 'w') as f:
                f.write(self.score_output.toPlainText())

    def update_score(self, score):
        """更新评分展示"""
        score_text = f"""最终评估结果：
        F1 Score: {score['f1']:.4f}
        Precision: {score['precision']:.4f}
        Recall: {score['recall']:.4f}"""
        self.score_output.setPlainText(score_text)

    def update_metrics(self, metrics):
        """更新训练曲线"""
        ax = self.figure.gca()
        ax.clear()

        ax.plot(metrics['loss_history'], label='训练损失', color='#1f77b4')
        ax.set_title("训练损失曲线", fontsize=12)
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("损失值")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())
