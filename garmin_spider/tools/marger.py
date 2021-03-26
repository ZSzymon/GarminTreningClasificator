import pandas as pd
def marge():
    old = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_editet_const.csv'
    new = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_prelabeled_jakub.csv'
    save_to = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_jakob_szymon.csv'

    df1 = pd.read_csv(old)
    df2 = pd.read_csv(new)
    df = pd.DataFrame()
    for row in df1.iterrows():
        df.append(row)
    df.to_csv(save_to)

def getN():
    file = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_prelabeled_jakub.csv'
    save_to = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_garmin_prelabeled_jakob_half.csv'
    save_to2 = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_garmin_prelabeled_jakob_half2.csv'
    df_old = pd.read_csv(file)
    df_old = df_old.sort_values('File id',ascending=False)
    to_save = df_old.iloc[:100,:]
    to_save.to_csv(save_to)
    to_save = df_old.iloc[100:200, :]
    to_save.to_csv(save_to2)

    pass

def save_labels():
    summary_file = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_prelabeled_jakub.csv'
    df = pd.read_csv(summary_file)
    df2 = df.loc[0:179, ['File id','Type']]
    #df = df[['File id','Type']]
    df2.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/activity_labels_jakob.csv')

if __name__ == '__main__':
    #save_labels()
    #marge()
    getN()