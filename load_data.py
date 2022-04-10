from audioop import add
from re import M
import numpy as np

import pandas as pd



def convert_to_df(np_array):
    m,n,r = np_array.shape
    #Taken from stackoverflow and modified to fit shape https://stackoverflow.com/questions/36235180/efficiently-creating-a-pandas-dataframe-from-a-numpy-3d-array
    out_arr = np.column_stack((np.repeat(np.arange(n),r),np_array.reshape(n*r,-1)))

    out_arr = np.delete(out_arr.T,0,0)
    return pd.DataFrame(out_arr)

def add_class_label(df, label, value):
    df[label] = value
    return df

def main():
    music_data = np.load('./Data/music_data.npy')
    other_data = np.load('./Data/other_data.npy')
    music_data_df = convert_to_df(music_data)
    other_data_df = convert_to_df(other_data)
    music_data_df = add_class_label(music_data_df, "is_music", 1)
    other_data_df = add_class_label(other_data_df, "is_music", 0)
    all_data_df = pd.concat([music_data_df, other_data_df], ignore_index=True)


if __name__ == '__main__':
    main()