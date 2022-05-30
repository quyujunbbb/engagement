import csv

import numpy as np
import pandas as pd


def main(path, output_path):
    """Generate clip-level labels.

    Args:
        path        : input_path
        output_path : output_path
    """
    print("Generate clip-level labels")

    df = pd.read_csv(path + "20210309.csv")

    df.begin = pd.to_datetime(df.begin, format='%H:%M:%S')
    df.end = pd.to_datetime(df.end, format='%H:%M:%S')
    df.begin = (df.begin.dt.hour+df.begin.dt.minute)*60 + df.begin.dt.second
    df.end = (df.end.dt.hour+df.end.dt.minute)*60 + df.end.dt.second
    df = df.to_numpy()
    # print(df)
    # print(df.shape)

    for i in range(len(df)-1):
        try:
            assert df[i,4] == df[i+1,3]
        except:
            if df[i+1,3] == 0:
                pass
            else:
                print(f'line {i} has problem: {df[i,4]}, {df[i+1,3]}')

    affective = df[df[:,2]==1]
    visual = df[df[:,2]==2]
    behavioral = df[df[:,2]==3]

    # sessions = ["20201222_01", "20201222_02", "20201222_03", "20201222_04", "20201222_05", "20201222_06", "20201222_07", "20201222_08", "20201222_09", "20201222_10", "20201222_11", "20201222_12", "20201222_13", "20201222_14"]
    # clp_num = [34, 118, 39, 107, 41, 29, 79, 71, 80, 60, 56, 42, 46, 55]

    # sessions = ["20201229_01", "20201229_02", "20201229_03", "20201229_04", "20201229_05", "20201229_06"]
    # clp_num = [63, 74, 206, 43, 80, 74]

    # sessions = ["20210105_01", "20210105_02", "20210105_03", "20210105_04", "20210105_05", "20210105_06", "20210105_07", "20210105_08", "20210105_09", "20210105_10", "20210105_11", "20210105_12", "20210105_13", "20210105_14"]
    # clp_num = [60, 81, 62, 92, 93, 52, 47, 65, 34, 123, 53, 32, 21, 24]

    sessions = ["20210309_01", "20210309_02", "20210309_03", "20210309_04", "20210309_05", "20210309_06", "20210309_07", "20210309_08", "20210309_09"]
    clp_num = [117, 66, 69, 216, 41, 110, 125, 39, 121]

    out = np.zeros([np.sum(clp_num), 6])
    print(out.shape)

    # affective
    idx = 0
    for s in range(len(sessions)):
        affective_session = affective[affective[:,0]==sessions[s]]
        for c in range(clp_num[s]):
            # print(f'session {sessions[s]}, clip {c}')
            clp_begin = c * 32 / 3
            clp_end = clp_begin + 32 / 3
            # print(clp_begin, clp_end)

            for row in affective_session:
                if row[3]<=clp_begin<row[4]:
                    clp_begin_range = row[3:]
                if row[3]<=clp_end<row[4]:
                    clp_end_range = row[3:]
            if clp_begin_range[-1] == clp_end_range[-1]:
                out[idx,:3] = [s+1, c+1, (clp_begin_range[-1] - 1) / 4]
                idx += 1
            else:
                l1 = clp_begin_range[1] - clp_begin
                l2 = clp_end - clp_end_range[0]
                score = (l1 * clp_begin_range[-1] + l2 * clp_end_range[-1]) / (l1 + l2)
                out[idx,:3] = [s+1, c+1, (score - 1) / 4]
                idx += 1

    # visual
    idx = 0
    for s in range(len(sessions)):
        visual_session = visual[visual[:,0]==sessions[s]]
        for c in range(clp_num[s]):
            # print(f'session {sessions[s]}, clip {c}')
            clp_begin = c * 32 / 3
            clp_end = clp_begin + 32 / 3
            # print(clp_begin, clp_end)

            for row in visual_session:
                if row[3]<=clp_begin<row[4]:
                    clp_begin_range = row[3:]
                if row[3]<=clp_end<row[4]:
                    clp_end_range = row[3:]
            if clp_begin_range[-1] == clp_end_range[-1]:
                out[idx,3] = (clp_begin_range[-1] - 1) / 4
                idx += 1
            else:
                l1 = clp_begin_range[1] - clp_begin
                l2 = clp_end - clp_end_range[0]
                score = (l1 * clp_begin_range[-1] + l2 * clp_end_range[-1]) / (l1 + l2)
                out[idx,3] = (score - 1) / 4
                idx += 1

    # behavioral
    idx = 0
    for s in range(len(sessions)):
        behavioral_session = behavioral[behavioral[:,0]==sessions[s]]
        for c in range(clp_num[s]):
            # print(f'session {sessions[s]}, clip {c}')
            clp_begin = c * 32 / 3
            clp_end = clp_begin + 32 / 3
            # print(clp_begin, clp_end)

            for row in behavioral_session:
                if row[3]<=clp_begin<row[4]:
                    clp_begin_range = row[3:]
                if row[3]<=clp_end<row[4]:
                    clp_end_range = row[3:]
            if clp_begin_range[-1] == clp_end_range[-1]:
                out[idx,4] = (clp_begin_range[-1] - 1) / 4
                idx += 1
            else:
                l1 = clp_begin_range[1] - clp_begin
                l2 = clp_end - clp_end_range[0]
                score = (l1 * clp_begin_range[-1] + l2 * clp_end_range[-1]) / (l1 + l2)
                out[idx,4] = (score - 1) / 4
                idx += 1

    out[:,-1] = (out[:,2] + out[:,3] + out[:,4]) / 3
    out[:,2:] = np.around(out[:,2:], decimals=4)


    with open(output_path+"out_20210309.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out)


if __name__ == "__main__":
    input_path = 'data/annotations/start_end/'
    output_path = 'data/annotations/out_generate_label/'

    main(input_path, output_path)
