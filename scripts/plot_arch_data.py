import csv
import sys

import matplotlib.pyplot as plt


def plot_fp_vs_perf(plot_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('# floating point ops')
    # ax.scatter([float(d[1])/1000000.0 for d in plot_data], [d[3] for d in plot_data], color='darkgreen', marker='^')
    ax.scatter([float(d[2]) / 1000000.0 for d in plot_data], [d[3] for d in plot_data], color='red', marker='o')
    for i in range(len(plot_data)):
        x = float(plot_data[i][2]) / 1000000.0
        y = plot_data[i][3]
        plt.text(x=x, y=y, s=plot_data[i][0])
    plt.savefig('arch_fp_op_vs_perf.png')


def plot_mac_vs_perf(plot_data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('time (s)')
    plt.ylabel('# MACs using thop')
    ax.scatter([float(d[2]) / 1000000.0 for d in plot_data], [d[4] for d in plot_data], color='blue', marker='x')
    for i in range(len(plot_data)):
        x = float(plot_data[i][2]) / 1000000.0
        y = plot_data[i][4]
        plt.text(x=x, y=y, s=plot_data[i][0])
    plt.savefig('arch_mac_vs_perf.png')


def main():
    plot_data = []
    with open(sys.argv[1], 'r') as fp:
        for arch_info in csv.reader(fp):
            arch_name, cpu_time, cuda_time, fp_op, mac = arch_info
            plot_data.append((arch_name, float(cpu_time), float(cuda_time), int(fp_op), float(mac)))
    plot_fp_vs_perf(plot_data)
    plot_mac_vs_perf(plot_data)


if __name__ == '__main__':
    main()
