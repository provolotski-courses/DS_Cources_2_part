import numpy as np
import pandas as pd


def eda_report(analyzed_dataset):
    """Проведение EDA-анализа над переданным датасетом"""
    """Вывод статистики по дата фрейму"""
    print('***---- head  ------***')
    print(analyzed_dataset.head())
    print('***---- shape  ------***')
    print(analyzed_dataset.shape)
    print('***---- describe  ------***')
    print(analyzed_dataset.describe())
    print('***---- нулевые значения  ------***')
    print(analyzed_dataset.isnull().sum())
    print('***---- Анализ атрибутов  ------***')
    for col in analyzed_dataset.columns:
        print(f'проводим разбор по атрибуту {col}')
        print(f'тип данных: {analyzed_dataset[col].dtypes}')
        print(f'гранулярность данных {analyzed_dataset[col].nunique()}')
        print('Частота появления каждого значения:')
        print(analyzed_dataset[col].value_counts())
    # выводим значения подмен




def preprocessing_input_data (pd):
    eda_report(pd)
    pd = pd.drop('Country or Area Code', axis=1)
    pd = pd.drop('Reference Date', axis=1)
    pd = pd.drop('Area Code', axis=1)
    pd = pd.drop('Area', axis=1)
    pd = pd[pd['Sex Code'] ==0]
    pd = pd.drop('Sex Code', axis=1)
    pd = pd.drop('Sex', axis=1)
    pd = pd.drop('Code of City', axis=1)
    pd = pd.drop('Code of City type', axis=1)
    pd = pd.drop('Record Type Code', axis=1)
    pd = pd.drop('Record Type', axis=1)
    pd = pd.drop('Reliability Code', axis=1)
    pd = pd.drop('Reliability', axis=1)
    pd = pd.drop('Source Year', axis=1)
    pd = pd.drop('Value Footnotes', axis=1)
    pd.to_csv('data/newdataset.csv')


    return pd

def linear_regression_model(pd):
    print('================================new one=========================================')
    eda_report(pd)

def main():
    file_name = 'data/UNdata_Export_20241025_142506414.txt'
    dataset = pd.read_csv(file_name, sep=';')
    dataset = preprocessing_input_data(dataset)
    city = 'MINSK'
    linear_regression_model(dataset[dataset['City'] == city].drop('City', axis=1).drop('Country or Area', axis=1).drop('City type', axis=1))



if __name__ == '__main__':
    main()
