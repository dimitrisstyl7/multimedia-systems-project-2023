import colorama
colorama.init()

def progressBar(progress, total, message='Progress:', endMessage='Completed', color=colorama.Fore.LIGHTYELLOW_EX):
    """
        Print a progress bar
    """
    percent = 100 * (progress / total)
    barLength = 50
    filledLength = int(barLength * percent / 100)
    bar = '*' * filledLength + '-' * (barLength - filledLength)
    messageText = f'{message} ' if message else ''
    print(color + f'\t {messageText}|{bar}| {percent:.2f}%', end='\r')
    if progress == total:
        messageText2 = f'{endMessage} ' if endMessage else ''
        print(colorama.Fore.GREEN + f'\t {messageText2}', end='\r')
        print(colorama.Fore.RESET)
