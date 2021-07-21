import pandas as pd
import matplotlib.pyplot as plt
import torch
from polyreg import Model

FEATURE = "Fancy Words"
TARGET = "Distance"
df = pd.read_csv("data/test.csv")

# invert bc points are downward sloping
features = 1 / torch.from_numpy(df[[FEATURE]].to_numpy())
actual = torch.from_numpy(df[TARGET].to_numpy())

model = Model(features, actual)
model.learn(0.5, 50000)

predictions = model.predict()
print(Model.cost(actual, predictions))

plt.scatter(df[FEATURE], df[TARGET], color="blue")
plt.plot(df[FEATURE], predictions.detach().numpy(), color="red")
plt.show()
