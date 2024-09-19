import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.metrics import accuracy_score, confusion_matrix


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


class CategoricalMetrics():
    __folds = pd.DataFrame()
    __classes = []
    __label = ''
    __categoricalColection = {}
    __columns = ['fold', 'acc', 'mc_acc', 'm_tpr', 'm_tnr', 'm_ppv', 'm_npv', 'm_fpr', 'm_fnr', 'm_fdr']
    __categoricalLines = ['c_acc', 'tpr', 'tnr', 'ppv', 'npv', 'fpr', 'fnr', 'fdr']

    def __init__(self, classes, label=''):
        self.__classes = classes
        self.__label = label
        self.__folds = pd.DataFrame(columns=self.__columns)

    def __calcule(self):
        std = {'fold': 'std'}
        mean = {'fold': 'mean'}

        for col in self.__columns:
            if col != 'fold':
                std[col] = self.__folds[col].std()
                mean[col] = self.__folds[col].mean()

        self.__folds = self.__folds.append(std, ignore_index=True)
        self.__folds = self.__folds.append(mean, ignore_index=True)

        # to categorical metrics
        self.__categoricalMetrics = pd.DataFrame(columns=self.__classes)

    def __dropCalculate(self):
        for text in self.__folds['fold']:
            if text == 'std':
                self.__folds.drop(self.__folds.tail(2).index, inplace=True)

    def add(self, y_true, y_pred, verbose=False):
        self.__dropCalculate()

        cnf_matrix = confusion_matrix(y_true, y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        metrics = {}
        c_metrics = {}
        numFold = len(self.__folds) + 1
        metrics['fold'] = "k" + str(numFold)

        # Accuracy
        metrics['acc'] = accuracy_score(y_true, y_pred)
        # Overall accuracy for each class
        c_metrics['c_acc'] = (TP + TN) / (TP + FP + FN + TN)
        metrics['mc_acc'] = np.mean(c_metrics['c_acc'])
        # Sensitivity, hit rate, recall, or true positive rate
        c_metrics['tpr'] = TP / (TP + FN)
        metrics['m_tpr'] = np.mean(c_metrics['tpr'])
        # Specificity or true negative rate
        c_metrics['tnr'] = TN / (TN + FP)
        metrics['m_tnr'] = np.mean(c_metrics['tnr'])
        # Precision or positive predictive value
        c_metrics['ppv'] = TP / (TP + FP)
        metrics['m_ppv'] = np.mean(c_metrics['ppv'])
        # Negative predictive value
        c_metrics['npv'] = TN / (TN + FN)
        metrics['m_npv'] = np.mean(c_metrics['npv'])
        # Fall out or false positive rate
        c_metrics['fpr'] = FP / (FP + TN)
        metrics['m_fpr'] = np.mean(c_metrics['fpr'])
        # False negative rate
        c_metrics['fnr'] = FN / (TP + FN)
        metrics['m_fnr'] = np.mean(c_metrics['fnr'])
        # False discovery rate
        c_metrics['fdr'] = FP / (TP + FP)
        metrics['m_fdr'] = np.mean(c_metrics['fdr'])

        self.__folds = self.__folds.append(metrics, ignore_index=True)

        # Gerando metrivas categóricas do fold
        foldDf = pd.DataFrame.from_dict(c_metrics).transpose()
        foldDf.columns = self.__classes
        foldDf.fillna(0)
        self.__categoricalColection['k' + str(numFold)] = foldDf

        self.__folds.fillna(0)
        self.__calcule()

        if verbose:
            print('------------------------------------------------------------------------')
            print('Computing fold ' + str(numFold) + '...')
            print('Accuracy rate of ' + str(metrics['acc'] * 100) + ' %')
            print('Mean accuracy rate for each class ' + str(metrics['mac'] * 100) + ' %')
            print('True positive rate: ' + str(metrics['tpr'] * 100) + ' %')
            print('True negative rate: ' + str(metrics['tnr'] * 100) + ' %')
            print('Positive predictive value: ' + str(metrics['ppv'] * 100) + ' %')
            print('Negative predictive value: ' + str(metrics['npv'] * 100) + ' %')
            print('False positive rate: ' + str(metrics['fpr'] * 100) + ' %')
            print('False negative rate: ' + str(metrics['fnr'] * 100) + ' %')
            print('False discovery rate: ' + str(metrics['fdr'] * 100) + ' %')

    def plot(self, path='', show=True):
        titles = self.__columns.copy()
        titles.remove('fold')

        values = self.__folds.iloc[-1].values
        values = np.delete(values, 0)

        theta = radar_factory(len(titles), frame='polygon')

        fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        ax.set_ylim(bottom=0, top=1)
        ax.plot(theta, values, color='r')
        ax.fill(theta, values, facecolor='r', alpha=0.25)
        ax.set_varlabels(titles)

        fig.text(0.5, 0.965, self.__label, horizontalalignment='center', color='black', weight='bold', size='large')

        if path != '':
            plt.savefig(path + self.__label + "_" + str(self.getMean('acc')) + ".png")

        if show:
            plt.show()

    def getMean(self, feature):
        return self.__folds.iloc[-1][feature]

    def getStd(self, feature):
        return self.__folds.iloc[-2][feature]

    def save(self, path, verbose=False):
        if (verbose):
            print(self.__folds)

        # Salva o arquivo de métricas
        self.__folds.to_csv(path + self.__label + "_" + str(self.getMean('acc')) + ".csv", index=False, header=True,
                            sep=';')

        categoricalMetrics = pd.DataFrame()

        # Salva arquivos por folds
        for key in self.__categoricalColection.keys():
            self.__categoricalColection[key].to_csv(path + '/fold-' + str(key) + ".csv", index=True, header=True,
                                                    sep=';')
            categoricalMetrics = categoricalMetrics.append(self.__categoricalColection[key])

        categoricalMetrics.reset_index(inplace=True)
        categoricalMetrics = categoricalMetrics.groupby('index', axis=0).mean()
        categoricalMetrics.to_csv(path + '/categoricalMetrics.csv', index=True, header=True, sep=';')