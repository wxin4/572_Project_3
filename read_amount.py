import pandas as pd


def read_amount_data():
    li = []
    for i in range(5):
        with open(r'./MealAmountData/mealAmountData' + str(i + 1) + '.csv', 'rt')as ff:
            df = pd.read_csv(ff, index_col=None, header=None)[:50]
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)

    return frame
