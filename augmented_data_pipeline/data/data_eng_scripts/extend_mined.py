# Go through data in "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09/" and add files to "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09 copy/" if they are not already in the copy.

import os
import shutil

def extend_mined():
    mined_dir = "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09/"
    mined_copy_dir = "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09_copy/"
    mined_files = os.listdir(mined_dir)
    mined_copy_files = os.listdir(mined_copy_dir)
    for file in mined_files:
        # skip if file ends in .index
        if file.endswith(".index"):
            continue
        if file[:-3] not in mined_copy_files:
            shutil.copyfile(mined_dir + file, mined_copy_dir + file)
            print("Copied " + file)

if __name__ == "__main__":
    extend_mined()