import os
import shutil
import sys
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def moveFiles():
    """
    Here is a need to chenge os.walk to walklevel(path, level=1)
    still trying to go deeper :/
    :return:
    """
    path = "/home/zywko/PycharmProjects/BA_Code/resources/polar_data"
    for dirname, dirs, files in walklevel(path, level=1):
        for file in files:
            oldFileName = file

            if str(file.upper()).endswith('.ZIP'):
                new_path = os.path.join(dirname, 'zips', file)
                file = os.path.join(dirname, file)
                shutil.move(file, new_path)
                print("Moving: {}  to zips".format(oldFileName))
                continue
            if str(file.upper()).endswith('.TCX'):
                new_path = os.path.join(dirname, 'tcxs', file)
                file = os.path.join(dirname, file)
                shutil.move(file, new_path)
                print("Moving: {}  to tcx".format(oldFileName))
                continue
            if str(file.upper()).endswith('.GPX'):
                new_path = os.path.join(dirname, 'gpxs', file)
                file = os.path.join(dirname, file)
                shutil.move(file, new_path)
                print("Moving: {}  to gpx".format(oldFileName))
                continue
            if str(file.upper()).endswith('.CSV'):
                new_path = os.path.join(dirname, 'csvs', file)
                file = os.path.join(dirname, file)
                shutil.move(file, new_path)
                print("Moving: {}  to csv".format(oldFileName))
                continue

if __name__ == '__main__':
    moveFiles()
