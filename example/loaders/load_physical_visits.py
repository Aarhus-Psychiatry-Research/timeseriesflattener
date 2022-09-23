"""Example loader for physical visits."""
import psycopmlutils.loaders.raw as r

if __name__ == "__main__":
    df = r.load_visits.physical_visits_to_psychiatry()
