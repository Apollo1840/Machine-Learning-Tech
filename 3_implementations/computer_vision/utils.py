import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def t_sne_visualize(x, y, n_sample=None, palette=True):
    """
    x for scatter, y for color

    :param x:
    :param y:
    :param n_sample: int, restrict the number to visualize
    :param palette: bool, use husl as palette or not.
    :return:
    """

    x = np.array(x)
    y = np.array(y)

    if n_sample is not None:
        indixes = np.random.choice(range(len(x)), n_sample, replace=False)
        x = x[indixes]
        y = y[indixes]

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj = tsne.fit_transform(x)

    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'classes': y})

    if palette:
        sns.scatterplot(x="X", y="Y",
                        hue="classes",
                        legend='full',
                        size=0.5,
                        alpha=0.5,
                        palette=sns.color_palette("husl", len(np.unique(y))),
                        data=tsne_df)
    else:
        sns.scatterplot(x="X", y="Y",
                        hue="classes",
                        legend='full',
                        size=0.5,
                        alpha=0.5,
                        data=tsne_df)

    plt.show()


# create a plot of generated images (reversed grayscale)
def show_plot(examples, n, with_channel=True):
    """

    :param: number of rows and columns
    """
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        if with_channel:
            # shape = (n_sample, x_axis, y_axis, channel)
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        else:
            # shape = (n_sample, x_axis, y_axis)
            plt.imshow(examples[i], cmap='gray_r')

    plt.show()
