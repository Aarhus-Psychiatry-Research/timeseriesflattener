"""Example loader for admissions."""
import psycop_feature_generation.loaders.raw as r

if __name__ == "__main__":
    df = r.load_admissions.admissions()
    print("Done")
