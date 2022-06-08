from psycopmlutils.loaders.sql_load import sql_load

if __name__ == "__main__":
    view = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret_2011]"
    sql = "SELECT * FROM [fct]." + view
    df = sql_load(sql, chunksize = None)