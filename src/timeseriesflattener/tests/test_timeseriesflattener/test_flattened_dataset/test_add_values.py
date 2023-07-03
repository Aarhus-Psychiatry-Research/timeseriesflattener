"""Tests for adding values to a flattened dataset."""


import datetime
import random
from dataclasses import dataclass
from typing import Union

import polars as pl
from wasabi import Printer

msg = Printer(timestamp=True)


def create_random_timestamps_series(
    n_rows: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    name: str = "timestamp",
) -> pl.Series:
    start_timestamp_int = int(start_datetime.timestamp())
    end_timestamp_int = int(end_datetime.timestamp())

    return pl.Series(
        name=name,
        values=[
            datetime.datetime.fromtimestamp(
                random.uniform(start_timestamp_int, end_timestamp_int),
            )
            for _ in range(n_rows)
        ],
    )


@dataclass()
class Event:
    timestamp: datetime.datetime
    value: Union[float, int, str]
    event_type: str


@dataclass()
class Entity:
    uuid: int
    events: list[Event]


def test_unpacking_speed():
    for n_patients in (5_000, 10_000, 20_000, 120_000):
        n_events_per_patient = 10
        n_prediction_times_per_patient = 10

        msg.info("Creating prediction_times dataframe")

        prediction_times_df = pl.DataFrame(
            {
                "entity_id": list(range(n_patients)) * n_prediction_times_per_patient,
            },
        ).with_columns(
            create_random_timestamps_series(
                n_rows=n_patients * n_prediction_times_per_patient,
                start_datetime=datetime.datetime(2021, 1, 1),
                end_datetime=datetime.datetime(2022, 1, 1),
            ),
        )

        msg.info("Creating predictor dataframes")
        predictor_df = pl.concat(
            [
                prediction_times_df.lazy().with_columns(
                    pl.Series(
                        [random.random() for _ in range(len(prediction_times_df))],
                    ).alias(
                        "value",
                    ),
                ),
            ]
            * n_events_per_patient,
        )

        predictor_dicts = predictor_df.sort("entity_id").collect().to_dicts()

        msg.info("Starting unpacking")
        timestamp_start = datetime.datetime.now()
        cur_entity: Union[Entity, None] = None
        entities: list[Entity] = []

        for d in predictor_dicts:
            row_entity_id = d["entity_id"]

            if cur_entity is None or cur_entity.uuid != row_entity_id:
                if cur_entity is not None:
                    entities.append(cur_entity)

                cur_entity = Entity(
                    uuid=row_entity_id,
                    events=[
                        Event(
                            timestamp=d["timestamp"],
                            value=d["value"],
                            event_type="test",
                        ),
                    ],
                )
            else:
                cur_entity.events.append(
                    Event(
                        timestamp=d["timestamp"], value=d["value"], event_type="test",
                    ),
                )

        timestamp_end = datetime.datetime.now()
        msg.info("Finished unpacking")
        (timestamp_end - timestamp_start).total_seconds()

        pass
