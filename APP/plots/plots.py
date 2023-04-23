import matplotlib.pyplot as plt


class PlotGraph:
    def __init__(self, data):
        self.fig = None
        self.name = 'cycle '
        self.format = 'svg'
        self.data = data
        self.type_plotting = self.check_type()
        self.several_cycles = self.check_type()
        self.fig = self.plot_data()

    def plot_data(self):
        if self.type_plotting:
            self.fig = self.plot_more_cycles()
        else:
            self.fig = self.plot_one_cycle()
        return self.fig

    def plot_one_cycle(self):
        fig = plt.figure()
        plt.plot(self.data[self.data.columns[0]],
                 self.data[self.data.columns[1]])
        plt.xlabel('Voltage, V')
        plt.ylabel('Current, uA')
        plt.title('Cyclic voltamperometry')
        return fig

    def plot_more_cycles(self):
        fig = plt.figure()
        number = '1'
        for cycle in self.data:
            plt.plot(self.data[0][self.data.columns[0]],
                     cycle[self.data.columns[1]],
                     label=(self.name + number))
            number = str(int(number) + 1)
        plt.xlabel('Voltage, V')
        plt.ylabel('Current, uA')
        plt.title('Cyclic voltamperometry')
        plt.legend()
        return fig

    def check_type(self):
        if type(self.data) is list:
            return True
        else:
            return False
