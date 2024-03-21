import sys

from utils.ArgsHandler import ArgsHandler, ArgsObject, OptionObject
from utils.ArgsHandler import display_helper


def main():
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


if __name__ == "__main__":
    main()
