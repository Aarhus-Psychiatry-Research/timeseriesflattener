"""Example of loading medications."""

import psycopmlutils.loaders.raw.load_medications as m

if __name__ == "__main__":
    df = m.antipsychotics()
