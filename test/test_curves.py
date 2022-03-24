import matplotlib.pyplot as plt
import numpy as np

FIGSIZE = (12, 5)
RANGE = (0, 1)


def plot_curve(ax, fit, fit_type, x_range, color='red'):
    x_v = np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
    if fit_type == 'log':
        y = [fit[1] + fit[0] * np.log(x) for x in x_v]
    elif fit_type == 'linear':
        y = [fit[1] + fit[0] * x for x in x_v]
    elif fit_type == 'exp':
        y = [fit[1] + fit[0] * np.exp(x) for x in x_v]
    else:
        raise RuntimeError('Invalid fit type')
    ax.plot(x_v, y, color=color)
    plt.savefig('test.pdf', bbox_inches='tight')


if __name__ == '__main__':
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    plot_curve(ax, (1, 0), 'exp', RANGE, 'red')
    plot_curve(ax, (1, np.exp(0.5) - 0.5), 'linear', RANGE, 'blue')
