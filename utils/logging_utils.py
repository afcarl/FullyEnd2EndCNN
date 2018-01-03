from colorama import init
from colorama import Fore, Back, Style
from terminaltables import SingleTable
from natsort import natsorted


def print_table(TABLE_DATA):

    table_instance = SingleTable(TABLE_DATA, "")
    table_instance.justify_columns[2] = 'right'
    print(table_instance.table)


def print_bright(s):

    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value=""):

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def print_params(d):

    print_bright("\nRun parameters session:")
    for key in natsorted(d.keys()):
        if "dir" not in key:
            print_green(key, d[key])


def print_optimizer(name, learning_rate):

    print_bright("\nOptimizer:")
    print_green("Type", name)
    print_green("Learning rate", learning_rate)


def print_queues():

    print_bright("\nQueues:")
    print_green("Initiated queue for async CPU/GPU")


def print_shape(X_shape, y_shape):

    print_bright("\nShapes")
    print_green("X", str(X_shape))
    print_green("y", str(y_shape))


def print_start_training():

    print_bright("\nTraining...")


