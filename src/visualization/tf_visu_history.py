import os
import matplotlib.pyplot as plt

from src.utils import save_name


def draw_learning_curves(history, showfig=False, savefig=True, savefile=''):
    fig, ax1 = plt.subplots(1, 1, figsize=(12,4))

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.legend(['train','val'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

#     ax2.plot(history.history['r_square'])
#     ax2.plot(history.history['val_r_square'])
#     ax2.legend(['train','val'])
# #     ax2.set_xlabel('Epochs')
#     ax2.set_ylabel('r2')

#     ax3.plot(history.history['root_mean_squared_error'])
#     ax3.plot(history.history['val_root_mean_squared_error'])
#     ax3.legend(['train','val'])
#     ax3.set_xlabel('Epochs')
#     ax3.set_ylabel('RMSE')

    plt.tight_layout()

    if savefig:
        odir = os.path.dirname(savefile)
        ofile = save_name.check(f"{odir}", os.path.basename(savefile))
        savefile = f'{odir}/{ofile}'
        plt.savefig(f"{savefile}")  # , facecolor='white')
        print(f'Saved as : {savefile}')

    if showfig:
        plt.show()

    plt.close()  
