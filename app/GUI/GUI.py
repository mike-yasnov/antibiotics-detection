import os
import sys
import numpy as np
import pandas as pd
import requests

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
import pyqtgraph as pg

from app.paths.paths import PATH_TO_UI_FILE
from app.data.data import Ivium
from app.plots.plots import PlotGraph
from app.plots.utils import color_pen


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientaton, role):
        if role == Qt.DisplayRole:
            if orientaton == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientaton == Qt.Vertical:
                return str(self._data.index[section])


class GUI(QMainWindow):
    def __init__(self):
        """
        Load UI from GUI.ui
        """
        super(GUI, self).__init__()
        self.reg = None
        self.clf = None
        self.graph = None
        self.data = None
        self.plots = None
        self.file_name = None
        self.fname = None
        self.model = None
        self.graphWidget = None
        self.current = None
        self.table_data_columns = None
        self.table_data = None
        loadUi(PATH_TO_UI_FILE, self)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet('''
        QTabWidget::tab-bar {
            alignment: center;
        }''')
        self.table_data = pd.DataFrame([
        ], columns=['File name', 'Substance', 'Antibiotic concentration'])
        self.table_data_columns = self.table_data.columns.tolist()
        self.widget.showGrid(x=True, y=True)
        self.current = None
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        self.model = TableModel(self.table_data)
        self.tableView.setModel(self.model)
        self.predict_antibiotic.setEnabled(False)
        self.actionOpen.triggered.connect(self.open_file)
        self.actionSave_as.triggered.connect(self.save_file)
        self.actionExit.triggered.connect(self.quit_app)
        self.predict_antibiotic.clicked.connect(self.predict_antibiotics)

    def quit_app(self):
        reply = QMessageBox.question(
            self, 'Message',
            'Are you sure you want to quit?',
            QMessageBox.Save |
            QMessageBox.Close |
            QMessageBox.Cancel,
            QMessageBox.Save
        )
        if reply == QMessageBox.Close:
            self.close()
        elif reply == QMessageBox.Save:
            self.save_file()
            self.close()
        else:
            pass

    def save_file(self):
        name = QFileDialog.getSaveFileName(self, 'Save File')
        self.table_data.to_csv(name[0] + '.csv')
        QMessageBox.about(self, 'Information', 'File successfully saved')

    def open_file(self):
        fname = QFileDialog.getOpenFileName(None,
                                            'Select file',
                                            '',
                                            'All Files (*);; \
                                            Python Files(*.py);;\
                                            Text files (*.txt)')
        self.fname = os.path.normpath(str(fname[0]))
        self.file_name = fname[0].split(os.sep)[-1]
        try:
            self.plots = Ivium(fname[0])
            self.data = self.plots.data
            self.graph = PlotGraph(self.data)
            self.display_graph()
            self.predict_antibiotic.setEnabled(True)
        except Exception:
            QMessageBox.warning(self,
                                'Error',
                                'Please, select file from ivium')

    def display_graph(self):
        self.widget.clear()
        if self.graph.type_plotting:
            a = 1
            self.widget.addLegend()
            for cycle in self.data:
                self.widget.plot(
                    np.array(self.data[0][self.data.columns[0]]),
                    np.array(cycle[self.data.columns[1]]),
                    name=('cycle ' + str(a)),
                    pen=color_pen[int(a)])
                a = str(int(a) + 1)

            self.widget.setLabel('left', 'Current', units='A')
            self.widget.setLabel('right', 'Current', units='A')
            self.widget.setLabel('bottom', 'Voltage', units='V')
            self.widget.setTitle('Cyclic Voltammetry')
            self.current = np.array(self.data[-1][self.data.columns[1]])
        else:
            self.widget.plot(
                np.array(self.data[self.data.columns[0]]),
                np.array(self.data[self.data.columns[1]]),
                pen='b')
            self.widget.setLabel('left', 'Current', units='A')
            self.widget.setLabel('right', 'Current', units='A')
            self.widget.setLabel('bottom', 'Voltage', units='V')
            self.widget.setTitle('Cyclic Voltammetry')
            self.current = np.array(self.data[self.data.columns[1]])

    def predict_antibiotics(self):
        # Read data
        df = np.array(pd.read_csv(self.fname, index_col=0).T.iloc[1])
        if df.shape == (1040,):
            file = {'file': open(self.fname, 'rb')}
            prediction = requests.post(
                url='http://localhost:5000/antibiotics/predict',
                files=file
                ).json()
            antibiotic = prediction['antibiotic']
            conc = prediction['concentration']
            self.table_data.loc[len(self.table_data.index)] = [
                self.fname,
                antibiotic,
                conc
                ]
            self.tableView.model().layoutChanged.emit()
            self.tableView.resizeColumnsToContents()
            if antibiotic == 'milk':
                QMessageBox.about(self,
                                  'Prediction',
                                  'Predicted pure milk'
                                  )
            else:
                prediction = f'Predicted: {antibiotic}\n' + \
                            f'Concentration: {conc} mg/l'
                QMessageBox.about(self,
                                  'Prediction',
                                  prediction
                                  )
        else:
            QMessageBox.warning(self, 'Error', 'Data has wrong length')


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = GUI()

    application.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
