

    print("train nan fields ....")
    train_df = train_df.replace(-1,np.nan)
    for c in train_df.columns.tolist():
        if c in ["id","target"]:
            continue
        else:

            nan_val = train_df[c].apply(lambda x:1 if pd.isnull(x) else 0)
            if len(  set(nan_val)) > 1:

                train_df[c + "__nan__"] = train_df[c].apply(lambda x:1 if pd.isnull(x) else 0)
                print( "column %s and nan legth %d" %  (c, len(set(nan_val)) )  )

    for c in train_df.columns:
        if "__nan__" in c:
            train_features.append(c)

    print("test nan fields ...")
    test_df = test_df.replace(-1,np.nan)
    for c in test_df.columns.tolist():
        if c in ["id"]:
            continue
        else:
            nan_val = test_df[c].apply(lambda x:1 if pd.isnull(x) else 0)
            if len(  set(nan_val)) > 1 or c == "ps_car_12":
                test_df[c + "__nan__"] = test_df[c].apply(lambda x:1 if pd.isnull(x) else 0)
                print( "column %s and nan legth %d" %  (c, len(set(nan_val)) )  )
