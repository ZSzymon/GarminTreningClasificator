
from summarymaker import SummaryMaker

from summarymaker import CSVEditor
if __name__ == '__main__':

    s = SummaryMaker("/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summary_new.csv",
                     "/home/zywko/PycharmProjects/BA_Code/resources/polar_data/csvs")
    s.create()
    editor = CSVEditor('/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summary_v2.csv',
                       '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summary_v3.csv')
    #editor.delete_row_if_any_cell_not_exist(['Average heart rate (bpm)','Max speed (km/h)', 'Average speed (km/h)'])
    editor.run()