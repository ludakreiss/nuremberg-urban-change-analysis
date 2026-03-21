def create_change_label(df):
    df["change"] = (df["built_2021"] - df["built_2020"]) > 0.1
    return df
