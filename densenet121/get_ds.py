import pandas as pd

chexpert = pd.read_csv("/media/azka/Seagate Hub/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv")
chexpert.fillna(0, inplace=True) #fill the with zeros
chexpert = chexpert[chexpert['Frontal/Lateral'] == 'Frontal']
chexpert = chexpert.rename(columns={"Path": "img_paths"})
chexpert = chexpert.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
chexpert.head()

from progress_bar import progress_bar

progress = progress_bar(len(chexpert))

    
res = pd.read_csv(f"mimic.csv")
res = res.drop(["Unnamed: 0", "subject_id", "study_id"], axis=1)
for i, (index, row) in enumerate(chexpert.iterrows()):
    if(row["No Finding"] == 1):
        row["labels"] = 0
    else:
        row["labels"] = 1
    row = row.to_frame().T
    print(row.head())

    res = pd.concat([res, row], ignore_index=True)

    progress.update(i, '')
res.to_csv("cxrs.csv")


df = pd.read_csv("cxrs.csv")
zeros = 0
ones = 0
df = df[df["Support Devices"] == 0]
for i, (index, row) in enumerate(df.iterrows()):
    if(row["labels"] == 0):
        zeros += 1
    else:
        ones += 1
print(zeros, ones)

zeros_df = df[df["labels"] == 0]
ones_df = df[df["labels"] == 1]

train_df = pd.concat([zeros_df.iloc[:int(len(zeros_df) * 9/10)], ones_df.iloc[:int(len(zeros_df) * 9/10)]], ignore_index=True)
test_df = pd.concat([zeros_df.iloc[int(len(zeros_df) * 9/10):], ones_df.iloc[int(len(zeros_df) * 9/10):]], ignore_index=True)

from sklearn.utils import shuffle
zeros_df = shuffle(zeros_df)
ones_df = shuffle(ones_df)

zeros = ones = 0
for i, (index, row) in enumerate(train_df.iterrows()):
    if(row["labels"] == 0):
        zeros += 1
    else:
        ones += 1
print(zeros, ones)

zeros = ones = 0
for i, (index, row) in enumerate(test_df.iterrows()):
    if(row["labels"] == 0):
        zeros += 1
    else:
        ones += 1
print(zeros, ones)

train_df = train_df.drop(columns=["Unnamed: 0"])
test_df = test_df.drop(columns=["Unnamed: 0"])

train_df.to_csv("train.csv")
test_df.to_csv("test.csv")