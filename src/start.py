from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys


class MyQPushButton(QPushButton):

    __inserted = False

    def isInserted(self):
        return self.__inserted

    def setInserted(self, inserted):
        self.__inserted = inserted


class View(QDialog):

    selectedFields = []

    def __init__(self):
        super().__init__()
        self.title = 'NAI PRO2'
        self.width = 620
        self.height = 480
        self.horizontalGroupBox = QGroupBox("Grid")

        self.inputGroupBox = QGroupBox("Ustawienia")
        self.inputGroupBox.setFixedWidth(self.width//2)
        self.panel = QGroupBox("Panel")
        self.wykres = QGroupBox("Wykres")

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(600, 300, self.width, self.height)

        windowLayout = QGridLayout()
        windowLayout.addWidget(self.inputGroupBox, 0, 0)
        windowLayout.addWidget(self.panel, 0, 1)
        windowLayout.addWidget(self.wykres, 1, 0, 1, 2)

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
        # teachButton.accepted.connect(self.accept)

        layout = QFormLayout()
        layout.addRow(QLabel("Liczba epok"), QLineEdit())
        layout.addRow(QLabel("Prog bledu"), QLineEdit())
        layout.addRow(QLabel("Wspolczynnik uczenia"), QLineEdit())
        layout.addRow(QLabel("Ilosc warstw ukrytych"), QLineEdit())
        layout.addRow(QLabel("Status Modelu"), QLabel("Nienauczony"))
        layout.addRow(teachButton)
        layout.addRow(classifyButton)
        self.inputGroupBox.setLayout(layout)

    def classifyClick(self):
        print(self.selectedFields)

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