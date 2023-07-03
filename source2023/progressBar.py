import math
import colorama


def progressBar(progress, total, message='Progress:', endMessage='Completed', color=colorama.Fore.LIGHTYELLOW_EX):
    """
    Print a progress bar
    """
    percent = 100 * (progress / total)
    bar_length = 50
    filled_length = int(bar_length * percent / 100)
    bar = '*' * filled_length + '-' * (bar_length - filled_length)
    message_text = f'{message} ' if message else ''
    print(color + f'\t {message_text}|{bar}| {percent:.2f}%', end='\r')
    if progress == total:
        message_text_2 = f'{endMessage} ' if endMessage else ''
        print(colorama.Fore.GREEN + f'\t {message_text_2}', end='\r')
        print(colorama.Fore.RESET)