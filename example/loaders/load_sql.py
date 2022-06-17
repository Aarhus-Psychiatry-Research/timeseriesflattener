from psycopmlutils.loaders.sql_load import sql_load

if __name__ == "__main__":
    view = "[psycop_t2d_train]"
    sql = "SELECT * FROM [fct]." + view
    df = sql_load(sql, chunksize=None, format_timestamp_cols_to_datetime=False)
