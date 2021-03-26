def jakub(clf):
    file = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_no_label_jakub.csv'

    X, y = read_data(file, is_jakub_files=True)

    predictions = clf.predict(X)

    df = pd.read_csv(file)
    df['Type'] = predictions
    i=0
    for row in df.iloc[:,-1]:
        distance = X[i,0]
        if row == 'BC 1' and X[i,0] < 5:
            df.at[i,'Type'] = 'Rozgrzewka'
        i += 1

    df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_prelabeled_jakub.csv')
    stop = 1
    pass