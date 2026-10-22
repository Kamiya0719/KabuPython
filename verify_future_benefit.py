import importlib.util
import pandas as pd
import numpy as np

p = r"C:\Users\ojiro\Documents\PythonFolder\KabuPython\Technical.py"
spec = importlib.util.spec_from_file_location("technical", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

s = pd.Series([-99, 0, 1, 2, 3, 4, 1, 2, 3, 4, np.nan], dtype=float)
y = mod._prepare_classification_target(s)
print(y.dropna().tolist())
print(sorted(y.dropna().unique().tolist()))
print("has_minus99", bool((y == -99).any()))
