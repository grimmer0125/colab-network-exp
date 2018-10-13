import matplotlib
import matplotlib.pyplot as plt

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__ 
        if shell == 'ZMQInteractiveShell': 
            print("in jupyter notebook")
            return True   # common Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        elif shell == 'Shell' # Google's colab. # class: google.colab._shell
            return False  # Other type (?)
    except NameError:
        return False

if is_notebook() == False:
    matplotlib.use('TkAgg')

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    try:
        plt.show(block=True)
    except:
        print("plot exception")    