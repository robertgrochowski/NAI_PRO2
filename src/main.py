import sys
import threading
import time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import matlib as m

from src.neuralNetwork import NeuralNetwork


class View(QDialog):

    selectedFields = []
    neuralNetwork = None
    inputLayout = None
    modelResult = None
    plotCanvas = None
    threadFinishedSignal = pyqtSignal(tuple)
    errorDialogSignal = pyqtSignal(str)
    teaching = False
    taught = False

    def __init__(self):
        super().__init__()
        self.title = 'NAI PRO2'
        self.width = 720
        self.height = 580
        self.horizontalGroupBox = QGroupBox("Grid")

        # initialize 3 main sections
        self.inputGroupBox = QGroupBox("Teach settings")
        self.inputGroupBox.setFixedWidth(self.width//2)
        self.panel = QGroupBox("Classification")
        self.plot = QGroupBox("Error plot")

        self.initUI()

        self.threadFinishedSignal.connect(self.on_teaching_finish)
        self.errorDialogSignal.connect(self.error_dialog)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(600, 300, self.width, self.height)

        # Create main layout
        windowLayout = QGridLayout()
        windowLayout.addWidget(self.inputGroupBox, 0, 0)
        windowLayout.addWidget(self.panel, 0, 1)
        windowLayout.addWidget(self.plot, 1, 0, 1, 2)

        # Add plot to main layout
        self.plotCanvas = PlotCanvas(self)
        layout = QBoxLayout(QBoxLayout.LeftToRight)
        layout.addWidget(self.plotCanvas)
        self.plot.setLayout(layout)

        # Initialize main layouts
        self.create_settings()
        self.create_panel()

        self.setLayout(windowLayout)
        self.show()

    def create_settings(self):
        # Create settings section and add widgets to layout
        self.inputLayout = QFormLayout()
        self.inputLayout.addRow(QLabel("Epochs amount"), QLineEdit("700"))
        self.inputLayout.addRow(QLabel("Error threshold"), QLineEdit("0.5"))
        self.inputLayout.addRow(QLabel("Learning factor"), QLineEdit("0.5"))
        self.inputLayout.addRow(QLabel("Number of hidden layers"), QLineEdit("17"))
        self.inputLayout.addRow(QLabel("Steepness factor"), QLineEdit("1"))

        statusLabel = QLabel("untaught")
        statusLabel.setStyleSheet("color: red; font-weight: bold;")
        self.inputLayout.addRow(QLabel("Model status"), statusLabel)

        teachButton = QPushButton("Teach")
        teachButton.setFixedHeight(40)
        teachButton.clicked.connect(self.teach_click)

        self.inputLayout.addRow(None, teachButton)
        self.inputGroupBox.setLayout(self.inputLayout)

    def create_panel(self):
        # Create settings section and add widgets to layout

        mainGrid = QGridLayout()

        # Generate clickable grid
        clickableGrid = QGridLayout()
        clickableGrid.setSpacing(0)

        for col in range(4):
            for row in range(6):
                id = row * 4 + col
                btn = MyQPushButton("")
                btn.setFixedWidth(25)
                btn.setFixedHeight(25)
                btn.setAccessibleName(str(id))
                btn.setStyleSheet("background-color: white;")
                clickableGrid.addWidget(btn, row, col)
                btn.clicked.connect(self.grid_button_click)
                self.selectedFields.append(0)

        mainGrid.addLayout(clickableGrid, 0, 0)
        mainGrid.setSpacing(10)

        label = QLabel("Draw a digit")
        label.setAlignment(Qt.AlignCenter)
        mainGrid.addWidget(label, 1, 0)

        label = QLabel("Model result")
        label.setAlignment(Qt.AlignCenter)
        mainGrid.addWidget(label, 1, 1)

        label = QLabel("?")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size:60px")
        mainGrid.addWidget(label, 0, 1)
        self.modelResult = label

        classifyButton = QPushButton("Classify")
        classifyButton.setFixedHeight(40)
        classifyButton.clicked.connect(self.classify_click)
        mainGrid.addWidget(classifyButton, 2, 0, 1, 2)

        self.panel.setLayout(mainGrid)

    def teach_click(self):
        if not self.teaching:
            threading.Thread(target=self.execute_teach).start()
        else:
            self.error_dialog('Teaching is in progress!')

    def on_teaching_finish(self, learn_data):
        self.teaching = False
        self.taught = True
        info = "Elapsed time: {:.2f}s \nAchieved error: {:.4f}\nEpochs: {:d}".format(learn_data[2], learn_data[0][len(learn_data[0]) - 1], learn_data[1])

        msgBox = QMessageBox()
        msgBox.setText('Neural network has been taught and is ready to classify!')
        msgBox.setWindowTitle("Teaching successful")
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setDetailedText(info)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()

    def execute_teach(self):
        # Teaching thread
        try:
            # Fetch data from textboxes
            maxEpoch = int(self.inputLayout.itemAt(0, QFormLayout.FieldRole).widget().text())
            maxError = float(self.inputLayout.itemAt(1, QFormLayout.FieldRole).widget().text())
            alpha = float(self.inputLayout.itemAt(2, QFormLayout.FieldRole).widget().text())
            lambd = float(self.inputLayout.itemAt(4, QFormLayout.FieldRole).widget().text())
            hiddenLayers = int(self.inputLayout.itemAt(3, QFormLayout.FieldRole).widget().text())

            # Set model status
            modelStatusText = self.inputLayout.itemAt(5, QFormLayout.FieldRole).widget()
            modelStatusText.setText("Teaching in progress...")
            modelStatusText.setStyleSheet("color: red; font-weight: bold;")
            self.teaching = True

            # Start teaching
            startTime = time.time()
            self.neuralNetwork = NeuralNetwork(hiddenLayers, alpha, maxError, maxEpoch, lambd)
            errors, epochs = self.neuralNetwork.teach()
            elapsedTime = time.time() - startTime

            # Draw plot
            self.plotCanvas.plot(errors)

            # Change model status
            modelStatusText.setText("Taught")
            modelStatusText.setStyleSheet("color: green; font-weight: bold;")

            self.threadFinishedSignal.emit((errors, epochs, elapsedTime))

        except ValueError:
            self.errorDialogSignal.emit('Please enter valid values')
        except:
            self.errorDialogSignal.emit('An error has occurred while teaching')

    def error_dialog(self, msg):
        msgBox = QMessageBox()
        msgBox.setText(msg)
        msgBox.setWindowTitle("Error")
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()

    def classify_click(self):
        if not self.taught:
            if self.teaching:
                self.error_dialog('Please wait until teaching process finishes')
            else:
                self.error_dialog('Please teach neural network before starting classification')
            return

        result = self.neuralNetwork.classify_input(m.mat(self.selectedFields))
        if result == -1:
            self.modelResult.setText("?")
        else:
            self.modelResult.setText(str(result))

    def grid_button_click(self):
        sending_button = self.sender()
        sending_button.setInserted(not sending_button.isInserted())
        self.selectedFields[int(sending_button.accessibleName())] = 1 if sending_button.isInserted() else 0

        sending_button.setStyleSheet("background-color: red;") \
            if sending_button.isInserted() else sending_button.setStyleSheet("background-color: white;")


class MyQPushButton(QPushButton):

    __inserted = False

    def isInserted(self):
        return self.__inserted

    def setInserted(self, inserted):
        self.__inserted = inserted


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None):
        FigureCanvas.__init__(self, Figure())
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)
        self.plot([1])
        self.draw()

    def plot(self, data):
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = View()
    sys.exit(app.exec_())