    # 3. MNIST-784 demo (no preprocessing)
    mn = fetch_openml("mnist_784", version=1)
    mnist_params = heart_params
    run_demo(
        mn.data.values,
        mn.target.astype(int).values,
        "MNIST-784",
        mnist_params
    )

    # 4. UCI Heart Failure demo
    df3 = pd.read_csv("data/heart_failure.csv")
    df3 = clean_df(df3, target_col="DEATH_EVENT")
    run_demo(
        df3.drop(columns=["DEATH_EVENT"]).values,
        df3["DEATH_EVENT"].values,
        "UCI Heart Failure",
        heart_params
    )

