from psycopmlutils.loaders import LoadMedications
from wasabi import msg

if __name__ == "__main__":
    # For testing, don't review

    df = LoadMedications.load(atc_code="A10", output_col_name="antidiabetics")

    msg.info(f"Columns: {df.columns}")

    pass
