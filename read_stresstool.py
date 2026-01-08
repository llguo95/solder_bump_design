import pandas as pd

# xls = pd.ExcelFile("stresstool_update.xls")

# file = pd.ExcelFile("myfile.xlsx")

with pd.ExcelFile("stresstool_update.xls") as xls:
    df1 = pd.read_excel(xls, "Kriging")
    df2 = pd.read_excel(xls, "Quadratic")
