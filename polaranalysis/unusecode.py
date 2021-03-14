import csv
import pandas as pd

def delete_row_if_any_cell_not_exist(self, to_delete_cells: list):
    csv_file = pd.read_csv(self.source_file)
    column_names = list(csv_file.columns)
    is_allowed = True

    with open(self.destination_file, 'w+') as fp:
        new_csv = csv.DictWriter(fp, fieldnames=column_names)
        new_csv.writeheader()
        for row_count, row in csv_file.iterrows():
            for to_delete_cell in to_delete_cells:
                val = row[to_delete_cell]
                if math.isnan(val):
                    is_allowed = False
            if is_allowed:
                dictrow = dict(row)
                new_csv.writerow(dictrow)
            is_allowed = True


new_file = "/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summary_new.csv"

df = pd.read_csv(new_file)

columns_to_drop = ['Fat percentage of calories(%)', 'Average stride length (cm)',
                   'Running index', 'Training load', 'Average power (W)',
                   'Max power (W)', 'Notes']

try:
    df.drop(columns=columns_to_drop, inplace=True)
except KeyError as error:
    print(error)

df.drop(df[df['Total distance (km)'] < 1].index, inplace=True)

df['Average pace (min/km)'] = toSeconds(df['Average pace (min/km)'])
df['Max pace (min/km)'] = toSeconds(df['Max pace (min/km)'])

df.rename(columns={'Average pace (min/km)': 'Average pace (s/km)',
                   'Max pace (s/km)': 'Max pace (s/km)'}, inplace=True)

columns = list(df.columns)

df['Average heart rate (bpm)'].replace('', np.nan, inplace=True)
df.dropna(subset=['Average heart rate (bpm)', 'Total distance (km)'], inplace=True)
df.dropna(subset=['Average heart rate (bpm)', 'Total distance (km)'], inplace=True)

df = df.drop(df[df['Total distance (km)'] < 2].index)
df = df.drop(df[df['Average pace (s/km)'] > 4.2*60].index)
df = df.drop(df[df['Average pace (s/km)'] < 3.5*60].index)
df.plot.scatter(x='Date', y='Average pace (s/km)',linewidth=2)
axes = plt.gca()
axes.set_ylim([3*60,6*60])
plt.show()

debug = True
