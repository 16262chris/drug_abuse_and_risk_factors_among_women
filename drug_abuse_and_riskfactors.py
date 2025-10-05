"""
Original file is located at
    https://colab.research.google.com/drive/139Qg02k3FwbLrpJJ15a2_mOmQGbf6jCt
"""

#Import data science libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.ensemble import GradientBoostingClassifier

#Load dataset
df = pd.read_excel("/content/drug abuse and risk factors among women.xlsx")

df.head()

df.info()

df.describe()

df.isnull().sum()

duplicates = df.duplicated().sum()
print(duplicates)

# NOTE: Duplicates are not deleted because personal identifying information
# has been removed from the dataset. As a result, multiple participants
# might have identical records.

#Display the unique values for all the columns

for col in df.columns:
    print(f"{col}: {df[col].unique()}\n")

#Remove all trailing whitespaces and convert characters to lower case

df["age"] = df["age"].str.rstrip()
df["education"] = df["education"].str.lower()
df["father"] = df["father"].str.lower()
df["mother"] = df["mother"].str.lower()
df["income"] = df["income"].str.rstrip().str.lower()
df["race"] = df["race"].str.lower()
df["employment"] = df["employment"].str.rstrip().str.lower()

"""Display all unique values for each column to ascertain
that trailing whitespaces have been removed and characters
have been coverted to lower case"""

for col in df.columns:
    print(f"{col}: {df[col].unique()}\n")

"""Apply binary encoding since categorical variables
("alcohol", "cigarettes", "cocaine",
"crack", "heroin", "marijuana", "meth",
"pain_relievers"is binary)
are binary"""

mapping = {"yes":1, "no":0}
df[["alcohol", "cigarettes", "cocaine", "crack", "heroin", "marijuana", "meth", "pain_relievers"]] = df[
    ["alcohol", "cigarettes", "cocaine", "crack", "heroin", "marijuana", "meth", "pain_relievers"]].replace(mapping)

"""Count the number of rows in column 'father' with response:
'does not know whether or not father is present'.
"""

df["father"].value_counts()

"""Drop rows in column 'father' with response:
'doesn’t know whether or not father is present'.
"""

df = df[df["father"] != "doesn’t know whether or not father is present"]

"""Count the number of rows in column 'mother' with response:
'doesn’t know whether or not mother is present'.
"""

df["mother"].value_counts()

"""Drop rows in column 'mother' with response:
'does not know whether or not mother is present'.
"""

df = df[df["mother"] != "does not know whether or not mother is present"]

"""Apply binary encoding since categorical variables
("alcohol", "cigarettes", "cocaine",
"crack", "heroin", "marijuana", "meth",
"pain_relievers"is binary)
are binary"""

mapping = {"yes":1, "no":0}
df[["alcohol", "cigarettes", "cocaine", "crack", "heroin", "marijuana", "meth", "pain_relievers"]] = df[
    ["alcohol", "cigarettes", "cocaine", "crack", "heroin", "marijuana", "meth", "pain_relievers"]].replace(mapping)

#Display a bar plot to show the frequency of drug use

drug_cols = ["alcohol", "cigarettes", "cocaine", "crack", "heroin", "marijuana",
             "meth", "pain_relievers"]

drug_counts = df[drug_cols].sum().sort_values(ascending=False)

total_participants = len(df)

plt.figure(figsize=(10,5))
bars = plt.bar(drug_counts.index, drug_counts.values, color='skyblue')

for bar in bars:
    height = bar.get_height()
    pct = height / total_participants * 100
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{pct:.1f}%',
             ha='center', va='bottom', fontsize=10)

plt.title("Frequency of Drug Use with Percentages")
plt.ylabel("Number of Participants")
plt.xticks(rotation=45)
plt.show()

# Compute correlation between drug usage columns and show using a heatmap

corr_matrix = df[drug_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Drug Usage")
plt.show()

#Display a bar plot to show the distribution of age groups

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "age", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

#Display a heatmap to show the use of drugs by age group

age_drug_usage = df.groupby("age")[drug_cols].mean() * 100

age_order = ['12-17 years old', '18-25 years old', '26-34 years old',
             '35-49 years old', '50-64 years old', '65 years or older']
age_drug_usage = age_drug_usage.reindex(age_order)

plt.figure(figsize = (12, 6))
sns.heatmap(age_drug_usage, annot = True, cmap = "YlOrRd", fmt = ".2f")
plt.title("Drug Use by Age")
plt.ylabel("Age Group")
plt.show()

#Display a bar plot to show the distribution of education among the participants

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "education", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Education")
plt.xlabel("Education")
plt.ylabel("Participants")
plt.xticks(rotation = 90)
plt.show()

""" Display a heatmap to show the distribution of drug use across different
educational level"""

edu_drug_usage = df.groupby("education")[drug_cols].mean() * 100

# Reorder education levels
edu_order = ["12 to 17 year olds", "did not graduate high school",
             "high school graduate", "some college, or associate degree",
             "college graduate"]
edu_drug_usage = edu_drug_usage.reindex(edu_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(edu_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Drug Use by Education Level (%)")
plt.ylabel("Education Level")
plt.xlabel("Drug")
plt.show()

"""Display a bar plot to show the distribution of participants with fathers
in their home"""

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "father", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of the Presence of Fathers in Participants' Home")
plt.xlabel("Presence of Father in the Home")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

"""Display a heatmap to show the distribution of the presence of fathers in
participants home"""

father_presence_drug_usage = df.groupby("father")[drug_cols].mean() * 100

#Reorder mother presence
father_order = ["participant is 18 or older",
                "yes, the father is in the household",
                "no, the father is not in the household"]
father_presence_drug_usage = father_presence_drug_usage.reindex(father_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(father_presence_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("""Drug Use by Participant with Respect to Whether
or Not Father is Present (%)""")
plt.ylabel("Presence of Father")
plt.xlabel("Drug")
plt.show()

"""Display a bar plot to show the distribution of the presence of mothers in
participants' home"""

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "mother", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of the Presence of Mothers in Participants' Home")
plt.xlabel("Presence of Mothers")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

"""Display a heatmap to show the distribution of the presence of mothers in
participants' home and drug usage"""

mother_presence_drug_usage = df.groupby("mother")[drug_cols].mean() * 100

# Reorder mother presence
mother_order = ["participant is 18 or older",
                "yes, the mother is in the household",
                "no, the mother is not in the household"]
mother_presence_drug_usage = mother_presence_drug_usage.reindex(mother_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(mother_presence_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("""Drug Use by Participant with Respect to Whether
or Not Mother is Present (%)""")
plt.ylabel("Presence of Mother")
plt.xlabel("Drug")
plt.show()

"""Display a bar plot to show the distribution of income among the participants"""

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "income", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Income Among Participants")
plt.xlabel("Income")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

"""Display a heatmap to show the distribution of income among participants
 and drug usage"""

income_drug_usage = df.groupby("income")[drug_cols].mean() * 100

# Reorder participants' income level
income_order = ["less than 20000",
                "between 20000 and 49000",
                "between 50000 and 74999",
                "more than 75000"
                ]
income_drug_usage = income_drug_usage.reindex(income_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(income_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Drug Use by Participants' Income Level (%)")
plt.ylabel("Income Level")
plt.xlabel("Drug")
plt.show()

"""Display a bar plot to show the distribution of race among the participants"""

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "race", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Race Among Participants")
plt.xlabel("Race")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

"""Display a heatmap to show the distribution of Race among participants
and drug usage"""

race_drug_usage = df.groupby("race")[drug_cols].mean() * 100

# Randomly arrange participants' race
race_order = ["asian", "white", "black", "hispanic", "multi-racial",
              "native american", "native hawaiian/pacific islander"]
race_drug_usage = race_drug_usage.reindex(race_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(race_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Drug Use by Participants' Race (%)")
plt.ylabel("Race")
plt.xlabel("Drug")
plt.show()

"""Display a bar plot to show the distribution of employment among
 the participants"""

plt.figure(figsize = (8, 5))

ax = sns.countplot(data = df, x = "employment", color = "skyblue")

total = len(df)
for p in ax.patches:
    height = p.get_height()
    pct = height / total * 100
    ax.annotate(f'{pct:.1f}%', (p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Employment Among Participants")
plt.xlabel("Employment")
plt.ylabel("Participants")
plt.xticks(rotation = 45)
plt.show()

"""Display a heatmap to show the distribution of Employment among participants
and drug usage"""

employment_drug_usage = df.groupby("employment")[drug_cols].mean() * 100

# Reorder participants' employment status
employment_order = ["participant is employed full-time",
                    "participant is employed part-time",
                    "other (incl. not in labor force)",
                    "participant is unemployed",
                    "participant is underage; 12-14 years old"]
employment_drug_usage = employment_drug_usage.reindex(employment_order)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(employment_drug_usage, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Drug Use by Participants Based on Employment Status (%)")
plt.ylabel("Employment")
plt.xlabel("Drug")
plt.show()

#Apply label encoding to age since it is ordinal

mapping = {"12-17 years old":1, "18-25 years old":2, "26-34 years old":3,
           "35-49 years old":4, "50-64 years old":5, "65 years or older":6}
df["age"] = df["age"].replace(mapping)

df.head()

#Apply label encoding to education since it is ordinal

mapping = {"did not graduate high school": 1, "high school graduate":2,
           "some college, or associate degree":3,"college graduate":4,
           "12 to 17 year olds":0}
df["education"] = df["education"].replace(mapping)

df.head()

#Apply label encoding to income as it is ordinal

mapping = {"less than 20000":1, "between 20000 and 49000":2,
           "between 50000 and 74999":3, "more than 75000":4}
df["income"] = df["income"].replace(mapping)

df.head()

#Apply one-hot encoding to "father" as it is a nominal categorical variable

mapping = {"participant is 18 or older": "neither_yes/no_participant_>=_18",
           "yes, the father is in the household": "yes",
           "no, the father is not in the household":"no"}
df["father"] = df["father"].replace(mapping)
df = pd.get_dummies(df, columns=["father"], dtype=int)

df.head()

#Apply one-hot encoding to employment as it is a nominal categorical variable

mapping = {"Participant is employed full-time":"employed full-time",
           "Participant is employed part-time":"employed part-time",
           "Other (incl. not in labor force)":"other",
           "Participant is unemployed":"unemployed",
           "Participant is underage; 12-14 years old":"underage"
           }
df["employment"] = df["employment"].replace(mapping)
df = pd.get_dummies(df, columns = ["employment"], dtype = int)

df.head()

# Apply one-hot encoding to race as it a nominal categorical variable

df = pd.get_dummies(df, columns=["race"], dtype=int)

df.head()

#Apply one-hot encoding to "mother" as it is a nominal categorical variable

mapping = {"participant is 18 or older": "neither_yes/no_participant_>=_18",
           "yes, the mother is in the household": "yes",
           "no, the mother is not in the household":"no"}
df["mother"] = df["mother"].replace(mapping)
df = pd.get_dummies(df, columns=["mother"], dtype=int)

df.head()

for target in drug_cols:
    print(f"Predicting {target.upper()} use ")

    x = df.drop(columns = [target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42, stratify = y
    )

    gb = GradientBoostingClassifier(random_state = 42)
    gb.fit(x_train, y_train)

    y_pred = gb.predict(x_test)
    y_proba = gb.predict_proba(x_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
