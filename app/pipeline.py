from fuzzywuzzy import fuzz

def detect_exact_duplicates(df):
    return df[df.duplicated(subset=["Employee_ID", "Expense_Amount", "Category", "Date"], keep=False)].to_dict(orient="records")

def detect_near_duplicates(df, threshold=90):
    near_dupes = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if (df.loc[i, "Employee_ID"] == df.loc[j, "Employee_ID"] and
                df.loc[i, "Expense_Amount"] == df.loc[j, "Expense_Amount"] and
                df.loc[i, "Date"] == df.loc[j, "Date"] and
                df.loc[i, "Category"] == df.loc[j, "Category"]):
                score = fuzz.ratio(str(df.loc[i, "Description"]), str(df.loc[j, "Description"]))
                if score >= threshold:
                    near_dupes.append({"pair": (i, j), "score": score})
    return near_dupes

def detect_outliers(df, isolation_model, kmeans_model):
    X = df[["Expense_Amount"]]
    iso_flags = isolation_model.predict(X).tolist()
    kmeans_labels = kmeans_model.predict(X).tolist()
    return {"isolation_forest": iso_flags, "kmeans": kmeans_labels}
