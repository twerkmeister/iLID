import os
import re
import shutil
import argparse

def clean(filename):
    return re.sub("[^a-zA-Z0-9\.-_ ]", "", filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', dest='target', help='Directory for the filenames to be cleaned', required=True)
    args = parser.parse_args()
    
    os.chdir(args.target)
    files = os.listdir(".")
    clean_files = [clean(f) for f in files]
    
    for filename, destinated_filename in zip(files, clean_files):
        if filename != destinated_filename:
            shutil.move(filename, destinated_filename)