"""Example loader for physical visits."""
import psycop_feature_generation.loaders.raw as r

if __name__ == "__main__":
    df = r.load_visits.physical_visits(n_rows=1000)
    psych = r.load_visits.physical_visits_to_psychiatry(n_rows=1000)
    somatic = r.load_visits.physical_visits_to_somatic(n_rows=1000)
