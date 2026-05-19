# the final step in this - ui app call
# pip install PySide6 <------------------------------------------------------------
import os
import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Signal Identification UI")
        self.resize(1200, 700)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # left side: data plot + result
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # plot placeholder
        self.plot_area = QFrame()
        self.plot_area.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #444;
                border-radius: 12px;
            }
        """)
        self.plot_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        plot_layout = QVBoxLayout(self.plot_area)

        plot_label = QLabel("Signal Plot Area")
        plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_label.setStyleSheet("color: #bbbbbb; font-size: 22px;")
        plot_layout.addWidget(plot_label)

        # Result label
        self.result_label = QLabel("Result will appear here")
        self.result_label.setMinimumHeight(80)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border-radius: 10px;
                color: white;
                font-size: 20px;
                padding: 15px;
            }
        """)

        left_panel.addWidget(self.plot_area, 5)
        left_panel.addWidget(self.result_label, 1)

        # right side control panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)

        title = QLabel("Controls")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")

        # Load row
        load_row = QHBoxLayout()
        def get_selected_file():
            text = str(self.file_dropdown.currentText())
            self.result_label.setText(text)
        def get_files():
            self.file_dropdown.blockSignals(True)
            self.file_dropdown.clear()
            with os.scandir("./identification_files") as files:
                for file in files:
                    self.file_dropdown.addItem(file.name)
                    print(file.name)
            self.file_dropdown.blockSignals(False)

        self.load_button = QPushButton("Load files")
        self.load_button.clicked.connect(get_files)
        self.file_dropdown = QComboBox()

        load_row.addWidget(self.load_button)
        load_row.addWidget(self.file_dropdown)

        # Other buttons
        self.load_model_button = QPushButton("Load Model")
        self.download_button = QPushButton("Download Measurements")
        self.identification_button = QPushButton("Identification")
        self.identification_button.clicked.connect(get_selected_file)

        # Styling
        button_style = """
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px;
                font-size: 16px;
            }

            QPushButton:hover {
                background-color: #2563eb;
            }

            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """

        combo_style = """
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
        """

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
            }
        """)

        for btn in [
            self.load_button,
            self.load_model_button,
            self.download_button,
            self.identification_button,
        ]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(50)

        self.file_dropdown.setStyleSheet(combo_style)
        self.file_dropdown.setMinimumHeight(50)

        # Add control layout
        right_panel.addWidget(title)
        right_panel.addLayout(load_row)
        right_panel.addWidget(self.load_model_button)
        right_panel.addWidget(self.download_button)
        right_panel.addWidget(self.identification_button)
        right_panel.addStretch()

        # add layout to main window
        main_layout.addLayout(left_panel, 2)   # 2/3
        main_layout.addLayout(right_panel, 1)  # 1/3


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())