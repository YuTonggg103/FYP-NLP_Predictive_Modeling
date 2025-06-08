import pandas as pd
import os

file_path = r".\Ori_Emotion_Dataset\Emotions.csv"
df = pd.read_csv(file_path)

label_counts = df['label'].value_counts() 

label_data = pd.DataFrame(label_counts).reset_index()
label_data.columns = ['Label', 'Count']
label_data.set_index('Label', inplace=True)

dataset_name = os.path.splitext(os.path.basename(file_path))[0]
print(f"\nDataset Name:\n{dataset_name}")
print("Column Names:        Data Types:")
for column, dtype in df.dtypes.items():
    print(f"{column:<15}  |   {dtype}")

print("\nLabel        Count")
for label, count in label_data['Count'].items():
    print(f"{label:<9} {count}")

print("\nNull Values per Column:")
null_counts = df.isnull().sum()
for column, null_count in null_counts.items():
    print(f"{column:<15}  |   {null_count}")

