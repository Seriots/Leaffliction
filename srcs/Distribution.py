#!/bin/env python3

import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper


def get_file_count(path: str) -> int:
    """Get a folder and recursively count all files in it"""
    count = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                count += get_file_count(entry.path)
            else:
                count += 1
    return count


def main():
    """Main"""
    args_handler = ArgsHandler(
        'This program take a path as arguments en display the \
distribution amongst all folders in it',
        [
            ArgsObject('folder_path', 'The path of the targeted folder')
        ],
        [
            OptionObject('help', 'Show this help message',
                         name='h',
                         expected_type=bool,
                         default=False,
                         check_function=display_helper
                         )
        ],
        """If the targeted folder have more than 1 depth,
all files are going to count for first folder category
"""
    )

    try:
        user_input = args_handler.parse_args()
        args_handler.check_args(user_input)
    except SystemExit:
        return
    except Exception as e:
        print(e)
        return

    path = user_input['args'][0]

    if not os.path.isdir(path):
        print(f"{path} is not a valid directory")
        return

    # Read directory and count file in it
    categories = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                categories[entry.name] = get_file_count(entry.path)

    data = categories.values()
    labels = categories.keys()
    print(categories)

    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(8, 4))
    fig.canvas.manager.set_window_title(f"File distribution in {path}")

    plt.subplots_adjust(wspace=0.5)

    fig.add_subplot(1, 2, 1)
    plt.pie(data, labels=labels, autopct='%.1f%%')
    plt.title("Pie chart")

    sns.set_theme(font_scale=0.5, palette="tab10")
    fig.add_subplot(1, 2, 2)
    sns.barplot(x=labels, y=data, hue=labels, dodge=False)
    plt.title("Bar chart")
    try:
        plt.show()
    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()
