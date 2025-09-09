from collections import defaultdict
import json
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np


def draw(flops, l, color):
  x_axis = np.array([x["parameters"] for x in l])
  y_axis = np.array([x["final_loss"] for x in l])
  print("==============", flops)
  plt.scatter(x_axis, y_axis, color=color, s=10)
  
  def scaling_law(N, A, alpha, B, beta, C):
    D = flops / (6 * N)
    return A * N**-alpha + B * D**-beta + C
  p0 = [1.0, 0.1, 1.0, 0.1, 0.0]
  bounds = ([0, 0, 0, 0, 0], 
            [np.inf, 1, np.inf, 1, np.inf])
  
  popt, _ = scipy.optimize.curve_fit(scaling_law, x_axis, y_axis, p0=p0, bounds=bounds, maxfev=10000)
  print(popt)

  x_fit = np.logspace(np.log10(x_axis.min()), np.log10(x_axis.max()), 300)
  y_fit = scaling_law(x_fit, *popt)
  plt.scatter(x_fit, y_fit, color=color, s=2)


def main():
  with open("data/isoflops_curves.json") as fp:
    data = json.load(fp)
  d = defaultdict(list)
  for item in data:
    d[item["compute_budget"]].append(item)
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']
  for (k, v), c in zip(d.items(), colors):
    draw(k, v, c)
  plt.xscale("log")
  plt.show()

if __name__ == "__main__":
  main()