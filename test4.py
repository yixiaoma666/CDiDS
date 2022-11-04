import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


lt = [np.random.randn(1,1)[0, 0] for _ in range(5000)]
print(np.mean(lt))
print(np.std(lt))