import matplotlib.pyplot as plt


def weights_over_steps_plot(counts: [[]], labels: () = ()):
    x = range(len(counts[0]))
    plt.stackplot(x, counts, labels=labels)
    if len(labels) > 0:
        plt.legend(loc='upper left')


def counts_over_steps_plot_to_file(counts: [[]], file: str, labels: () = ()):
    """ Counts are a series of series. The external is for each quantity counted. The internal is for the
    readings of that quantity at each step."""
    weights_over_steps_plot(counts=counts, labels=labels)
    plt.savefig(file, bbox_inches='tight', dpi=600)
    plt.close()
