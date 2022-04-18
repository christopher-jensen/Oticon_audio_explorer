from audioop import add
from re import M
import numpy as np
import matplotlib.pyplot as plt
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

def load_npy_file(file):
    return np.load(file)

def get_data_with_labels():
    music_data = load_npy_file('./Data/music_data.npy')
    other_data = load_npy_file('./Data/other_data.npy')
    music_data_df = convert_to_df(music_data)
    other_data_df = convert_to_df(other_data)
    # music_data_df = add_class_label(music_data_df, "average", music_data_df.mean(axis=1))
    # other_data_df = add_class_label(other_data_df, "average", other_data_df.mean(axis=1))
    music_data_df = add_class_label(music_data_df, "is_music", 1)
    other_data_df = add_class_label(other_data_df, "is_music", 0)
    all_data_df = pd.concat([music_data_df, other_data_df], ignore_index=True)
    return all_data_df

def get_only_other_data():
    other_data = np.load('./Data/other_data.npy')
    other_data_df = convert_to_df(other_data)
    # other_data_df = add_class_label(other_data_df, "average", other_data_df.mean(axis=1))
    other_data_df = add_class_label(other_data_df, "is_music", 0)
    return other_data_df

def get_only_music_data():
    music_data = np.load('./Data/music_data.npy')
    music_data_df = convert_to_df(music_data)
    # music_data_df = add_class_label(music_data_df, "average", music_data_df.mean(axis=1))
    music_data_df = add_class_label(music_data_df, "is_music", 1)
    return music_data_df

def get_test_data():
    test_data = np.load("./Data/test_data.npy")
    test_data_df = convert_to_df(test_data)
    # test_data_df = add_class_label(test_data_df, "average", test_data_df.mean(axis=1))
    # print(test_data_df)
    return test_data_df

def main():
    all_data = get_data_with_labels()
    # print(all_data.iloc[::1000,-2:])
    # plot1 = get_only_music_data().iloc[::2000,:-1:79].transpose().plot()
    plot2 =get_only_music_data().iloc[::10000,0:10].transpose().plot()
    plt.title("music data")
    # plot3 = get_only_other_data().iloc[::2000,:-1:79].transpose().plot()
    plot4 =get_only_other_data().iloc[::10000,0:10].transpose().plot()
    plt.title("other data")

    plot5 = get_test_data().iloc[::2000,0:10].transpose().plot()
    plt.title("unlabled data")
    # plot1.title("frequencybands")
    # plot2.title("frequencytime")

    plt.show()
    

if __name__ == '__main__':
    main()