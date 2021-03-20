import pandas as pd
def marge():
    old = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_garmin_labeled_ready.csv'
    new = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_labeled_new.csv'
    save_to = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_ready_v2.csv'

    df1 = pd.read_csv(old)
    df2 = pd.read_csv(new)
    df = pd.concat([df1, df2]).drop_duplicates('File id')

    df.to_csv(save_to)

def save_labels():
    summary_file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_ready_v2.csv'
    df = pd.read_csv(summary_file)
    df = df[['File id','Type']]
    df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/activity_labels.csv')

if __name__ == '__main__':
    save_labels()