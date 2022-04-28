from loaders.load_demographics import LoadDemographics
from wasabi import msg

if __name__ == "__main__":
    # For testing, don't review

    df = LoadDemographics.birthdays()

    msg.info(f"Columns: {df.columns}")

    pass
