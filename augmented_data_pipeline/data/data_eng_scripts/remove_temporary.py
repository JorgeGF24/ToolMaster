# Delete files in a directory if they end in .tmp.gz

import os
import shutil

def remove_temporary():
    directory = "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09_copy/"
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".tmp.gz"):
            os.remove(directory + file)
            print("Removed " + file)

if __name__ == "__main__":
    remove_temporary()