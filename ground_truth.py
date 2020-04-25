import read_amount


def extract_ground_truth():
    df = read_amount.read_amount_data()
    rows = df[0].tolist()
    bins = []

    # categorize into bins
    for i in rows:
        if i == 0:
            bins.append(1)
        elif 0 < i <= 20:
            bins.append(2)
        elif 20 < i <= 40:
            bins.append(3)
        elif 40 < i <= 60:
            bins.append(4)
        elif 60 < i <= 80:
            bins.append(5)
        elif 80 < i <= 100:
            bins.append(6)
        elif 100 < i <= 120:
            bins.append(7)
        elif 120 < i <= 140:
            bins.append(8)
    return bins
