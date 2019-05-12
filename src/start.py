import sys
import threading

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from numpy import matlib as m

from src.neuralNetwork import NeuralNetwork


class MyQPushButton(QPushButton):

    __inserted = False

    def isInserted(self):
        return self.__inserted

    def setInserted(self, inserted):
        self.__inserted = inserted



class View(QDialog):

    selectedFields = []
    neuralNetwork = None
    inputLayout = None
    modelResult = None

    def __init__(self):
        super().__init__()
        self.title = 'NAI PRO2'
        self.width = 620
        self.height = 480
        self.horizontalGroupBox = QGroupBox("Grid")

        self.inputGroupBox = QGroupBox("Ustawienia")
        self.inputGroupBox.setFixedWidth(self.width//2)
        self.panel = QGroupBox("Klasyfikacja")
        self.plot = QGroupBox("wykres")

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(600, 300, self.width, self.height)

        windowLayout = QGridLayout()
        windowLayout.addWidget(self.inputGroupBox, 0, 0)
        windowLayout.addWidget(self.panel, 0, 1)
        windowLayout.addWidget(self.plot, 1, 0, 1, 2)

        self.createSettings()
        self.createPanel()

        self.setLayout(windowLayout)
        self.show()

    def createSettings(self):
        teachButton = QPushButton("Naucz")
        teachButton.setFixedHeight(40)

        classifyButton = QPushButton("Klasyfikuj")
        classifyButton.setFixedHeight(40)
        classifyButton.clicked.connect(self.classifyClick)
        teachButton.clicked.connect(self.teachClick)

        self.inputLayout = QFormLayout()
        self.inputLayout.addRow(QLabel("Liczba epok"), QLineEdit("700"))
        self.inputLayout.addRow(QLabel("Próg błędu"), QLineEdit("0.5"))
        self.inputLayout.addRow(QLabel("Współczynnik uczenia"), QLineEdit("0.5"))
        self.inputLayout.addRow(QLabel("Ilość warstw ukrytych"), QLineEdit("17"))
        self.inputLayout.addRow(QLabel("Wspolczynnik stromosci"), QLineEdit("1"))
        statusLabel = QLabel("Nienauczony")
        statusLabel.setStyleSheet("color: red; font-weight: bold;")
        self.inputLayout.addRow(QLabel("Status Modelu"), statusLabel)
        self.inputLayout.addRow(teachButton)
        self.inputLayout.addRow(classifyButton)
        self.inputGroupBox.setLayout(self.inputLayout)

    def teachClick(self):
        x = threading.Thread(target=self.execute_teach)
        x.start()


    def execute_teach(self):
        modelStatusText = self.inputLayout.itemAt(5, QFormLayout.FieldRole).widget()
        modelStatusText.setText("trwa uczenie...")
        maxEpoch = int(self.inputLayout.itemAt(0, QFormLayout.FieldRole).widget().text())
        maxError = float(self.inputLayout.itemAt(1, QFormLayout.FieldRole).widget().text())
        alpha = float(self.inputLayout.itemAt(2, QFormLayout.FieldRole).widget().text())
        lambd = float(self.inputLayout.itemAt(4, QFormLayout.FieldRole).widget().text())
        hiddenLayers = int(self.inputLayout.itemAt(3, QFormLayout.FieldRole).widget().text())
        self.neuralNetwork = NeuralNetwork(hiddenLayers, alpha, maxError, maxEpoch, lambd)
        self.neuralNetwork.teach()
        modelStatusText.setText("nauczony")
        modelStatusText.setStyleSheet("color: green; font-weight: bold;")

    def classifyClick(self):
        result = self.neuralNetwork.classify_input(m.mat(self.selectedFields))
        if result == -1:
            self.modelResult.setText("?")
        else:
            self.modelResult.setText(str(result))


    def createPanel(self):
        mainGrid = QGridLayout()

        clickableGrid = QGridLayout()
        clickableGrid.setSpacing(0)

        for col in range(4):
            for row in range(6):
                id = row*4 + col
                btn = MyQPushButton("")
                btn.setFixedWidth(25)
                btn.setFixedHeight(25)
                btn.setAccessibleName(str(id))
                btn.setStyleSheet("background-color: white;")
                clickableGrid.addWidget(btn, row, col)
                btn.clicked.connect(self.buttonClicked)
                self.selectedFields.append(0)

        mainGrid.addLayout(clickableGrid, 0, 0)
        mainGrid.setSpacing(10)

        label = QLabel("Narysuj cyfrę")
        label.setAlignment(Qt.AlignCenter)
        mainGrid.addWidget(label, 1, 0)

        label = QLabel("Wynik modelu")
        label.setAlignment(Qt.AlignCenter)
        mainGrid.addWidget(label, 1, 1)

        label = QLabel("?")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size:60px")
        self.modelResult = label

        mainGrid.addWidget(label, 0, 1)

        self.panel.setLayout(mainGrid)

    def buttonClicked(self):
        sending_button = self.sender()
        sending_button.setInserted(not sending_button.isInserted())
        self.selectedFields[int(sending_button.accessibleName())] = 1 if sending_button.isInserted() else 0

        sending_button.setStyleSheet("background-color: red;") \
            if sending_button.isInserted() else sending_button.setStyleSheet("background-color: white;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = View()
    sys.exit(app.exec_())