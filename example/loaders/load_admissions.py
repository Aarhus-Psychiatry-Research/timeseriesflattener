"""Example loader for admissions."""
import psycop_feature_generation.loaders.raw as r

if __name__ == "__main__":
    df = r.load_admissions.admissions(n_rows=1000)
    psych = r.load_admissions.admissions_to_psychiatry(n_rows=1000)
    somatic = r.load_admissions.admissions_to_somatic(n_rows=1000)
