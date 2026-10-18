# the final step in this - ui app call
import json
import os
import sys
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from config import *
from database_download import download_from_db
from pathlib import Path
from brics_toolkit.brics_types import *
from model.data_containers.brics_model_wrapper import BRICSModelWrapper
from model.data_containers.feature_data import FeatureData
from model.data_containers.model_data import ModelData
from model.model.brics_model import BRICSModel
from model.model.train_model import train_model
from model.model.train_model_numeric import train_model as train_model_numeric

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # model_data setup
        self.model_data: Optional[ModelData] = None

        self.setWindowTitle("Signal Identification UI")
        self.resize(1200, 700)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # left side: data plot + result
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # plot placeholder
        self.plot_area = QLabel()
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

        self.plot_label = QLabel("Signal Plot Area")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setStyleSheet("color: #bbbbbb; font-size: 22px;")
        plot_layout.addWidget(self.plot_label)

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

        self.load_button = QPushButton("Load files")
        self.file_dropdown = QComboBox()

        load_row.addWidget(self.load_button)
        load_row.addWidget(self.file_dropdown)

        # Train row
        train_row = QHBoxLayout()
        self.train_model_button = QPushButton("Train Model")

        self.train_dropdown = QComboBox()
        self.train_dropdown.clear()
        self.train_dropdown.addItem("Identification")
        self.train_dropdown.addItem("Gender")
        self.train_dropdown.addItem("Height")
        self.train_dropdown.addItem("Weight")

        train_row.addWidget(self.train_model_button)
        train_row.addWidget(self.train_dropdown)

        # Other buttons
        self.load_model_button = QPushButton("Load Model")
        self.download_button = QPushButton("Download Measurements")
        self.identification_button = QPushButton("Identification")

        # Binding functions <----------------------------------------------------
        def train_and_load_model():
            training_type = str(self.train_dropdown.currentText())
            print(training_type)
            if training_type == "Identification" or training_type == "Gender":
                self.model_data = train_model(training_type=training_type)
            elif training_type == "Height" or training_type == "Weight":
                self.model_data = train_model_numeric(training_type=training_type)

            self.result_label.setText(f"Model trained and loaded")
            pixmap = QPixmap("confusion_matrix.jpg")   # path to your image
            self.plot_label.clear()
            self.plot_label.setPixmap(pixmap)
            self.plot_label.repaint()

            reply = QMessageBox.question(
                self,
                "Confirm Action",
                "Do you wat to save the model?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.model_data:
                    self.model_data.model_wrapper.save_model(filepath=Path("model/model/model"))
        def load_model_from_file():
            result: Optional[str]
            if not self.model_data:
                self.model_data = ModelData(feature_data=FeatureData(), model_wrapper=BRICSModelWrapper())
            self.model_data.model_wrapper.load_model(Path("model/model/model"))
            if self.model_data.model_wrapper.model:
                result = "Model succesfully loaded"
            else:
                result = "Model failed to load"
            self.result_label.setText(result)
        def get_files():
            self.file_dropdown.blockSignals(True)
            self.file_dropdown.clear()
            with os.scandir(UI_DATA_SOURCE_DIR_PATH) as files:
                for file in files:
                    self.file_dropdown.addItem(file.name)
                    print(file.name)
            self.file_dropdown.blockSignals(False)
        def download_measurements():
            download_from_db()
        def plot_identified_signal(file, data):
            to_plot = Path(f"./data/identifier/clean/{file.split(".")[0]}_0.jsonl")
            with open(to_plot, "r") as f:
                record = f.readline()
                while record:
                    feature_vector = parse_record(record)
                    data.append(feature_vector)
                    record = f.readline()

            file_data = np.array(data)
            for i in range(5):
                plt.plot(file_data[:,0], file_data[:,i+1])
            plt.xlim((0,20000))
            plt.savefig("./temp.jpg")
            plt.close()
            # set the jpg into label
            pixmap = QPixmap("temp.jpg")   # path to your image
            self.plot_label.clear()
            self.plot_label.setPixmap(pixmap)
            self.plot_label.repaint()
            for file in IDENTIFIER_FEATURES_PATH.iterdir():
                if file.is_file():
                    file.unlink()
            for file in IDENTIFIER_RESULTS_PATH.iterdir():
                if file.is_file():
                    file.unlink()
        def identify():
            file = str(self.file_dropdown.currentText())
            if not file:
                print("No fileselected")
                return
            print(f"\n{file}")
            person: Optional[str] = ""
            most_common_label = None
            data = []
            result = 0
            training_type = str(self.train_dropdown.currentText())
            if training_type == "Height" or training_type == "Weight":
                if self.model_data:
                    predicted_value = self.model_data.model_wrapper.predict_numeric_value(Path("./raw/"+file))
                    set_result_label(person=person, most_common_label=most_common_label, train_type=training_type, result=predicted_value)
                    return
            if self.model_data:
                person, most_common_label = self.model_data.model_wrapper.identify_person(Path("./raw/"+file))
                plot_identified_signal(file=file, data=data)
            else:
                person = "No model loaded"
                training_type = None
            set_result_label(person=person, most_common_label=most_common_label, train_type=training_type, result=result)

        def set_result_label(person, most_common_label, train_type, result):
            if not train_type:
                self.result_label.setText(f"{person}")
            if train_type == "Identification":
                self.result_label.setText(f"predicted person is: {person}")
            elif train_type == "Gender":
                gender = {0: "male", 1: "female"}
                self.result_label.setText(f"predicted person is: {gender[most_common_label]}") # type: ignore
            elif train_type == "Height":
                self.result_label.setText(f"predicted height is: {result}")
            elif train_type == "Weight":
                self.result_label.setText(f"predicted weight is: {result}")
        
        self.train_model_button.clicked.connect(train_and_load_model)
        self.load_model_button.clicked.connect(load_model_from_file)
        self.load_button.clicked.connect(get_files)
        self.download_button.clicked.connect(download_measurements)
        self.identification_button.clicked.connect(identify)
        #self.identification_button.clicked.connect(identify_gender)
        #-------------------------------------------------------------------------

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
            self.train_model_button,
            self.load_model_button,
            self.download_button,
            self.identification_button,
        ]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(50)

        self.file_dropdown.setStyleSheet(combo_style)
        self.file_dropdown.setMinimumHeight(50)
        self.train_dropdown.setStyleSheet(combo_style)
        self.train_dropdown.setMinimumHeight(50)

        # Add control layout
        right_panel.addWidget(title)
        right_panel.addLayout(load_row)
        right_panel.addLayout(train_row)
        right_panel.addWidget(self.load_model_button)
        right_panel.addWidget(self.download_button)
        right_panel.addWidget(self.identification_button)
        right_panel.addStretch()

        # add layout to main window
        main_layout.addLayout(left_panel, 2)   # 2/3
        main_layout.addLayout(right_panel, 1)  # 1/3

def parse_record(record):
    features = json.loads(record)
    feature_vector = []
    for key, val in features.items():
        if type(val) == list:
            for number in val:
                feature_vector.append(number)
        else:
            feature_vector.append(val)
    return feature_vector

def main():
    app = QApplication(sys.argv)

    # model_data = train_model
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()