import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from scripts.arch_data import ArchDataCollection, CLossV

FIGSIZE = (12, 5)
FIGSIZE_SQUARE = (12, 12)


def is_in_pareto_f(a1, plot_data):
    for a2 in plot_data:
        if a1 is a2:
            continue
        if a2.model_acc > a1.model_acc and a2.macs_count < a1.macs_count:
            return False
    return True


def add_extremities(frontier_elements):
    class DummyArch:
        def __init__(self, model_acc, macs_count):
            self.model_acc = model_acc
            self.macs_count = macs_count

    first = DummyArch(89.8, frontier_elements[0].macs_count)
    last = DummyArch(frontier_elements[-1].model_acc, 8e8)
    frontier_elements.insert(0, first)
    frontier_elements.append(last)


def draw_pareto_frontier(plot_data, ax, color):
    frontier_elements = []
    for a in plot_data:
        if is_in_pareto_f(a, plot_data):
            frontier_elements.append(a)
    frontier_elements = sorted(frontier_elements, key=lambda arch: arch.macs_count)
    add_extremities(frontier_elements)
    ax.plot(
        [a.model_acc for a in frontier_elements], [a.macs_count for a in frontier_elements],
        linestyle='dashed', color=color)


def plot_acc_vs_macs(collection):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plt.xlabel('Model Accuracy (from train.py)')
    plt.ylabel('# MACS')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V4,),
                              'Original': (CLossV.ORIGINAL,)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.macs_count is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.model_acc for a in plot_data], [a.macs_count for a in plot_data],
                   c=[a.closs_w for a in plot_data] if g_name == 'Diff. Loss' else None, cmap='Reds', label=g_name)
        draw_pareto_frontier(plot_data, ax, 'blue' if g_name == 'Original' else 'red')
        for a in plot_data:
            x = a.model_acc
            y = a.macs_count
            s = a.arch_id
            # s += ', w={}'.format(a.closs_w) if hasattr(a, "closs_w") and a.closs_w else ''
            plt.text(x=x, y=y, s=s, fontsize=3)
    ax.legend()
    plt.savefig('pareto.pdf', bbox_inches='tight')


def plot_acc_vs_macs_wo_pareto(collection):
    def _normalize(data):
        min_d = min(data)
        max_d = max(data)
        return [(i - min_d) / (max_d - min_d) for i in data]

    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = fig.add_subplot(111)
    plt.xlabel('Normalized Model Accuracy (from train.py)')
    plt.ylabel('Normalized # MACS')
    for g_name, clv_group in {'Loss-v4': (CLossV.D_LOSS_V4,)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.macs_count is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        x = _normalize([a.model_acc for a in plot_data])
        y = _normalize([a.macs_count for a in plot_data])
        names = [a.arch_id for a in plot_data]
        normalized_collection = zip(x, y, names)
        ax.scatter(x, y,
                   c=[a.closs_w for a in plot_data] if g_name != 'Original' else None,
                   cmap='Reds' if g_name == 'Loss-v3' else 'Greens', label=g_name)
        plot_exp_regression(x, y, ax)
        acc_range = (max([a.model_acc for a in plot_data]), min([a.model_acc for a in plot_data]))
        print('ACC Range:', acc_range)
        print('ACC Delta:', acc_range[0] - acc_range[1])
        for arch_data in normalized_collection:
            x, y, arch_id = arch_data
            plt.text(x=x, y=y, s=arch_id, fontsize=3)
    # ax.legend()
    plt.savefig('acc_vs_macs_wo_pareto.pdf', bbox_inches='tight')


def plot_acc_vs_w(collection, ax):
    ax.set_xlabel('Custom Loss Weight \"w\"')
    ax.set_ylabel('Model Accuracy (from train.py)')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V4,)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.closs_w for a in plot_data], [a.model_acc for a in plot_data], label=g_name)
        for a in plot_data:
            x = a.closs_w
            y = a.model_acc
            s = a.arch_id
            ax.text(x=x, y=y, s=s, fontsize=10)


def plot_macs_vs_w(collection, ax):
    ax.set_xlabel('Custom Loss Weight \"w\"')
    ax.set_ylabel('# MACS')
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V4,)}.items():
        plot_data = list(
            filter(lambda a: a.macs_count is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        ax.scatter([a.closs_w for a in plot_data], [a.macs_count for a in plot_data], label=g_name)
        for a in plot_data:
            x = a.closs_w
            y = a.macs_count
            s = a.arch_id
            ax.text(x=x, y=y, s=s, fontsize=10)


def configure_multiplot():
    fig = plt.figure(figsize=FIGSIZE)
    acc_w_ax = fig.add_subplot(211)
    macs_w_ax = fig.add_subplot(212)
    return fig, acc_w_ax, macs_w_ax


def plot_curve(fit, fit_type, ax, x_range, color='red', y_range=None):
    x_v = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    if fit_type == 'log':
        y = [fit[1] + fit[0] * np.log(x) for x in x_v]
    elif fit_type == 'linear':
        y = [fit[1] + fit[0] * x for x in x_v]
    elif fit_type == 'exp':
        y = [fit[1] + fit[0] * np.exp(x) for x in x_v]
    elif fit_type == 'square':
        y = [(fit[0] * x ** 2) + (fit[1] * x) + fit[2] for x in x_v]
    else:
        raise RuntimeError('Invalid fit type')
    if y_range:
        y = [i for i in y if i >= y_range[0]]
        x_v = x_v[len(x_v) - len(y):]
        y = [i for i in y if i <= y_range[1]]
        x_v = x_v[:len(y)]
    ax.plot(x_v, y, color=color)


def plot_lin_regression(collection, ax):
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V4,)}.items():
        plot_data = list(
            filter(lambda a: a.model_acc is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        x = [a.closs_w for a in plot_data]
        y = [a.model_acc for a in plot_data]
        fit = np.polyfit(x, y, 1)
        print('Lin fit: {} + {} * x'.format(fit[1], fit[0]))
        plot_curve(fit, 'linear', ax, (min(x), max(x)))


def plot_log_regression(collection, ax):
    for g_name, clv_group in {'Diff. Loss': (CLossV.D_LOSS_V4,)}.items():
        plot_data = list(
            filter(lambda a: a.macs_count is not None and a.closs_w is not None and a.closs_v in clv_group,
                   collection.archs.values()))
        x = [a.closs_w for a in plot_data]
        y = [a.macs_count for a in plot_data]
        fit = np.polyfit(np.log(x), y, 1)
        print('Log fit: {} + {} * log(x)'.format(fit[1], fit[0]))
        plot_curve(fit, 'log', ax, (min(x), max(x)))


def plot_exp_regression(x, y, ax):
    def _exp_func(_x, _a, _b, _c):
        return _a * np.exp(-_b * _x) + _c

    def _get_tan_fit(fit):
        _a, _b, _c = fit
        m = 1  # tan(45 degrees)
        numerator = np.log(-m / (_a * _b))
        x_tan = numerator / (- _b)
        y_tan = _exp_func(x_tan, _a, _b, _c)
        print('Tan Coordinates:', x_tan, y_tan)
        y_o = y_tan - m * x_tan
        return m, y_o

    popt, pcov = curve_fit(_exp_func, x, y, p0=(1, 0, 1))
    print('Exp. fit y = ae^(-b*x) + c: ', popt)
    print(pcov)
    curve_x = np.linspace(0.0, 1.0, 100)
    ax.plot(curve_x, [_exp_func(i, *popt) for i in curve_x], color='red')
    plot_curve(_get_tan_fit(popt), 'linear', ax, (0.0, 0.84), color='blue', y_range=(0.0, 1.0))


def plot_data_vs_w():
    fig, acc_w_ax, macs_w_ax = configure_multiplot()
    plot_acc_vs_w(arch_collection, acc_w_ax)
    plot_lin_regression(arch_collection, acc_w_ax)
    plot_macs_vs_w(arch_collection, macs_w_ax)
    plot_log_regression(arch_collection, macs_w_ax)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('data_vs_w.pdf', bbox_inches='tight')


if __name__ == '__main__':
    arch_collection = ArchDataCollection()
    arch_collection.load()
    plot_acc_vs_macs(arch_collection)
    plot_acc_vs_macs_wo_pareto(arch_collection)
    plot_data_vs_w()
