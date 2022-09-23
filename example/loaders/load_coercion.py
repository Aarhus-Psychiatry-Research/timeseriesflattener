"""Testing loading of coercion functions."""

# pylint: disable=non-ascii-name

import psycopmlutils.loaders.raw.load_coercion as c

if __name__ == "__main__":
    df = c.coercion_duration(n_rows=100)
    farlighed = c.farlighed(n_rows=20)
    baelte = c.baelte(n_rows=100)
