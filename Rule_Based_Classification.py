# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset():
    #pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    return pd.read_csv("datasets/persona.csv")

df = load_dataset()

# Data Overview
BOLD = '\033[1m'
END = '\033[0m'
print(BOLD + "Head" + END)
df.head()
print(BOLD + "Tail" + END)
df.tail()
print(BOLD + "Info" + END)
df.info()
print(BOLD + "Null Values" + END)
df.isnull().values.any()

# How many unique SOURCEs are there? What are their frequencies?
df["SOURCE"].nunique() # Number of unique SOURCEs
len(set(df["SOURCE"]))
df["SOURCE"].value_counts()

# How many sales from which country?
df["COUNTRY"].value_counts()

# How much was earned in total from sales by country?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# What are the sales numbers according to SOURCE types?
df["SOURCE"].value_counts()
df.groupby("SOURCE")["PRICE"].count()
df.groupby("SOURCE").agg({"PRICE": "count"})

# What are the PRICE averages in the COUNTRY-SOURCE breakdown?
df.groupby(['COUNTRY','SOURCE']).agg({"PRICE":"mean"})
df.groupby(by=["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# Sort the above output according to PRICE.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX","AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)
agg_df.head()

# All variables except price in the output named agg_df are index names.
# We need to convert these names to variable names.
agg_df = agg_df.reset_index()
agg_df.head()

# Convert the numeric variable named age into a categorical variable.
# Construct the intervals as you think will be persuasive.
bins = [0, 19, 24, 31, 41, agg_df["AGE"].max()]

# What the nomenclature means for the dividing points:
labels = ['0_18', '19_24', '24_30', '30_40', '40_' + str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=labels)
agg_df.head()


# Define new level based customers and add them as variables to the dataset.
# A variable named CUSTOMERS_LEVEL_BASED should be defined and added to the dataset.
# After the values are created, these values need to be deduplicated.
# It is possible to see more than one of the following expressions: USA_ANDROID_MALE_0_18

my_values = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].apply(lambda x: '_'.join(x), axis=1).str.upper()
agg_df['CUSTOMERS_LEVEL_BASED'] = my_values

# Remove the unnecessary variables
agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]]
agg_df.head()
agg_df["CUSTOMERS_LEVEL_BASED"].value_counts()

# After groupby according to the segments, we need to get the price averages and deduplicate the segments.
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE": "mean"})
agg_df.head()

# It is in the customers_level_based index.
agg_df = agg_df.reset_index()
agg_df.head()

# Each persona is expected to have one:
agg_df["CUSTOMERS_LEVEL_BASED"].value_counts()
agg_df.head()

# Add segments to agg_df dataframe with SEGMENT feature.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})
agg_df[agg_df["SEGMENT"] == "C"]["PRICE"].describe().T

# What segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to earn on average?
new_user1 = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user1]

# In which segment and how much income on average would a 35-year-old French woman using iOS expect to earn?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user2]