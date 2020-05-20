import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

euclidean_stats = [(79.41834451901566, 1.639344262295082),
         (79.7752808988764, 1.6129032258064515),
         (79.41834451901566, 1.639344262295082)]

manhattan_stats = [(79.7752808988764, 1.6129032258064515),
                   (79.7752808988764, 1.6129032258064515),
                   (79.7752808988764, 1.6129032258064515)]
def plot_data(data, axes):
    stat = []
    type = []
    iteration_no = []
    for i in range(6):
        if i < 3:
            stat.append(data[i][0])
            type.append("cluster_one")
            iteration_no.append(i)
        else:
            stat.append(data[i-3][1])
            type.append("cluster_two")
            iteration_no.append(i-3)

    stat_dic = {"iteration_no": iteration_no, "positive_diagnosis_percentages": stat, "cluster #": type}

    df = pd.DataFrame(stat_dic)
    print(str(df.head(6)))
    sns.barplot(x="iteration_no", y="positive_diagnosis_percentages", hue="cluster #", data=df, ax=axes)


fig, axes = plt.subplots(1, 2, figsize=(14, 8))

plot_data(euclidean_stats, axes[0])
axes[0].set_title("Euclidean distance")

plot_data(manhattan_stats, axes[1])
axes[1].set_title("Manhattan distance")

plt.show()

print("Done")

