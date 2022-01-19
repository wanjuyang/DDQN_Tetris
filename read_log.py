# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 08:51:34 2022

@author: notfu
"""
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

ea = event_accumulator.EventAccumulator("tensorboardss/events.out.tfevents.1642003754.DESKTOP-RN11K1G")
ea.Reload()
print(ea.scalars.Keys())
Score = ea.scalars.Items("Train/New_Dueling_DQN_Score")
max = 0
scores = []
for i in Score:
    if max < i[2]:
        print(i[2])
        max = i[2]
    scores.append(i[2])
print(np.mean(np.array(scores)),max)
#%%
epochs = range(0,3000)
plt.plot(epochs, np.array(scores), 'g', label='Score')
# plt.plot(epochs, np.array(, 'b', label='Testing acc')
plt.title('Score')
plt.xlabel('Epochs')
plt.ylabel('score')
plt.legend()
plt.show()