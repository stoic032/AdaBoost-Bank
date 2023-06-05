import pandas as pd


def table_split(fn):
    # 将表格文件读入到 Pandas DataFrame中

    df = pd.read_csv(rf'..\{fn}.csv')

    # 获取表格的列名
    column_name = df.columns.values

    # 定义每个小表格的行数
    chunk_size = 10000

    # 将大的 DataFrame 拆分为多个小 DataFrame
    df_list = []
    for i in range(1, df.shape[0], chunk_size):
        df_list.append(df.iloc[i: i + chunk_size])

    # 保存每个小 DataFrame 为单独的 CSV 文件
    for i, df_small in enumerate(df_list):
        filename = rf'..\data\{fn}\{fn}_' + str(i + 1) + '.csv'
        df_small.to_csv(filename, index=False, header=column_name)


if __name__ == '__main__':
    for fn in ['train', 'test']:
        table_split(fn)
