"""Example for loading diagnoses."""

import loaders.raw.load_diagnoses as d

if __name__ == "__main__":
    df = d.gerd(n_rows=200)
