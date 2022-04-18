from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import load_data as ld



# def flatten_to_potens_2_frequency_band():
#     rows_to_get = [2,4,8,16]
#     music_data = ld.load_npy_file('./Data/music_data.npy')
#     music_data_flattened = np.array(music_data[:,1,:])
#     for i in rows_to_get:
#         np.append(music_data_flattened, music_data[:,i,:])
#     print(music_data_flattened.shape)

def split_dataframe(df, rows_to_get):
    df_ls = []
    # Append first frequencyband to df
    df_ls.append(df.iloc[:,0:79])
    for i in rows_to_get:
        start = (79*(2**i))
        end = start+79
        df_ls.append(df.iloc[:, start : start+end:79])
    return df_ls

        
def flatten_dataframe_to_n2_frequency_band():
    df = ld.get_data_with_labels()
    rows_to_get = [0,2,3,4]
    df_ls = split_dataframe(df, rows_to_get)
    # Append classification to dataframe
    df_ls.append(df.iloc[:,-1])

    return pd.concat(df_ls, axis=1, ignore_index=True)

def flatten_dataframe_to_n2_frequency_band_unlabeled():
    df = ld.get_test_data()
    rows_to_get = [0,2,3,4]
    df_ls = split_dataframe(df, rows_to_get)

    return pd.concat(df_ls, axis=1, ignore_index=True)

def main():
    print(flatten_dataframe_to_n2_frequency_band())

if __name__ == '__main__':
    main()