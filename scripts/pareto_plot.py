import matplotlib.pyplot as plt

ARCH_DATA = {'M45': (83.852, 234863408.0),
             'M46': (85.256, 241886000.0),
             'M47': (85.38, 418307904.0),
             'M48': (84.92, 515988288.0),
             'M49': (84.42, 613585728.0),
             'M50': (84.96, 617594688.0),
             'M51': (84.768, 673194816.0),
             'M52': (85.152, 739632960.0),
             'M53': (85.744, 710906688.0),
             'M54': (84.536, 302628672.0),
             'M55': (84.348, 247360304.0)}


def plot_acc_vs_macs(plot_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Accuracy')
    plt.ylabel('# MACS')
    ax.scatter([plot_data[a][0] for a in plot_data], [plot_data[a][1] for a in plot_data], color='darkgreen',
               marker='^')
    for a in plot_data:
        x = plot_data[a][0]
        y = plot_data[a][1]
        plt.text(x=x, y=y, s=a)
    plt.savefig('pareto.svg')


if __name__ == '__main__':
    plot_acc_vs_macs(ARCH_DATA)
