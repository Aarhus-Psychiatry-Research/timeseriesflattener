"""Example of loading medications."""

import loaders.raw.load_medications as m

if __name__ == "__main__":
    df = m.antipsychotics(n_rows=500)
