import pandas
import matplotlib.pyplot as plt
import os

files = sorted(os.listdir('files'))


for file in files:
    df = pandas.read_csv(f'files/{file}', index_col=0)
    df = df.sort_values(by=['Input'])
    df.plot()
    plt.savefig(f'plots/{file}.png')
    plt.close()
