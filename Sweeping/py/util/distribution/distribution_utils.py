from collections.abc import Sequence


def ascii_probability_plot(probs: Sequence[float], height: int = 10, width: int = 80) -> str:
    """
    Create an ASCII density plot of a probability distribution.

    The highest-probability events appear on the *left*.

    Parameters:
        probs (sequence of float): probabilities summing to 1.
        height (int): number of text rows.
        width (int): width of the plot (number of bars).
                     If None, uses len(probs).

    Returns:
        str: ASCII plot.
    """

    if len(probs) == 0:
        return "Distribution of probabilities with an empty set of elements."
    max_prob = max(probs)
    if max_prob == 0.0:
        return "Malformed distribution of probability where all elements have 0 probability."
    min_prob = min(probs)
    # Sort probabilities descending (so left = highest bar)
    sorted_probs = sorted(probs, reverse=True)

    # Determine final width
    width = min(len(sorted_probs), width)

    # If too many probabilities, average them into bins
    if len(sorted_probs) > width:
        group_size = len(sorted_probs) / width
        grouped = []
        for i in range(width):
            start = int(i * group_size)
            end = int((i + 1) * group_size)
            end = max(end, start + 1)
            grouped.append(sum(sorted_probs[start:end]) / (end - start))
        sorted_probs = grouped
    else:
        # pad with zeros if fewer than width
        sorted_probs = sorted_probs + [0.0] * (width - len(sorted_probs))

    # Convert probabilities into integer bar heights (rows)
    bar_heights = [int(p / max_prob * height) for p in sorted_probs]

    # Build the plot top → bottom
    bar = "#"
    rows = []
    for y in range(height, 0, -1):
        row = "".join(bar if bh >= y else " " for bh in bar_heights)
        rows.append(row)

    # Add horizontal axis
    rows.append("-" * width)
    rows.append(f"max={max_prob}, min={min_prob}")

    return "\n".join(rows)
