import os
import time
from pathlib import Path

import pandas as pd


input_folder = "data/openface/"
output_folder = "data/openface-processed/"

file_paths = [str(f) for f in Path(input_folder).glob('**/*.csv')]

my_file = open("data/openface/openface_column.txt", "r")
columns = my_file.read().split(",")
my_file.close()
print(columns)

# for file_path in file_paths:
#     file_name = file_path.split("/")[-1].split(".")[0]
#     print(f'processing {file_name} ...')
#     openface_ori = pd.read_csv(file_path)
#     print(openface_ori)

#     openface_new = openface_ori[columns]
#     openface_new.to_csv(output_folder+file_name+'.csv', index=False)
#     print(openface_new)
