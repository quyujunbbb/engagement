import numpy as np
import pandas as pd


def main():
    labels = pd.read_csv("data/annotations/labels.csv")
    label = np.array(labels["overall"])

    classes = [0   , 0.0833, 0.1667,
               0.25, 0.3333, 0.4167,
               0.5 , 0.5833, 0.6667,
               0.75, 0.8333, 0.9167, 1.0]
    labels_class = np.zeros((len(label), 2))
    distance = np.zeros(len(classes))

    for i, row in enumerate(label):
        for j, cls in enumerate(classes):
            distance[j] = abs(row - cls)
        idx = np.argmin(distance)
        labels_class[i,0] = classes[idx]
        labels_class[i,1] = idx

    labels_class = pd.DataFrame(labels_class, columns=["class_value", "class"])
    labels["class_value"] = labels_class["class_value"]
    labels["class"] = labels_class["class"]

    labels.to_csv("data/annotations/labels_class.csv", index=False)

    print(labels)


if __name__ == "__main__":
    main()
