import TCXParser2 as tcxparser
import csv
import itertools
if __name__ == '__main__':
    tcx = tcxparser.TCXParser('/home/zywko/PycharmProjects/BA_Code/'
                              'garmin_spider/tmp/tcxs/activity.tcx')
    times = tcx.time_values()
    hrs = tcx.hr_values()

    positions = tcx.position_values()

    els = tcx.time_values()
    distances = tcx.distance_values()
    altitudes = tcx.altitude_points()

    data = itertools.zip_longest(times,positions, hrs, fillvalue='')
    with open('/home/zywko/PycharmProjects/BA_Code/'
              'garmin_spider/tmp/output.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            csv_writer.writerow(line)
        pass
