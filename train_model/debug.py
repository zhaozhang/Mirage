import pickle
from datetime import datetime
import os

if __name__=='__main__':

    fname='batch.pickle'

    with open(os.path.join('data', fname), 'rb') as data:
        samples = pickle.load(data)

    raw_input=[]
    raw_target=[]


print(len(samples))
print([s[1] for s in samples])
