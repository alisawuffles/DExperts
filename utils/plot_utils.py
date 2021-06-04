def label_bars(ax, rects):
    """
    Attach a text label over each bar displaying its height
    """
    for rect in rects:
        width = rect.get_width()
        ax.text(
            rect.get_x() + width / 2.0, rect.get_y() + 0.01,
            f'{width:.2f}',
            ha='center', va='bottom'
        )