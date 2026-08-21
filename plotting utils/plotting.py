import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

danskin_array = []
madry_array = []
ensemble_array = []
step = []
store = {"danskin": danskin_array, "madry": madry_array, "ensemble": ensemble_array}
for filename in os.listdir("data/"):
    if "csv" not in filename:
        continue
    data = pd.read_csv(os.path.join("data/", filename))
    mode = filename.split("-")[0]
    store[mode].append(data['loss'].to_list())
    step = data['step'].to_numpy()

plt.style.use("bmh")
for mode in store:
    array = np.array(store[mode])
    plt.plot(step, np.mean(array, axis=0), label=mode)
    plt.fill_between(step, np.mean(array, axis=0) - np.std(array, axis=0),  np.mean(array, axis=0) + np.std(array, axis=0), alpha=0.2)
plt.ylabel("loss")
plt.xlabel("normalized gradient")
plt.legend()
plt.savefig("attack=False-init=start-normalized=true.png", dpi=300)
plt.show()
