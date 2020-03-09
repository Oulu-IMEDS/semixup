from common.oai_most import load_most_dataset, load_oai_most_datasets
import os
import pandas as pd

most_only = False
clean_data = True

# Read meta data
# Pre-process standardized knee images
# Create corresponding dataframe and save it into a csv file
save_meta_dir = "./processed_data/Metadata/"
output_filename = "most_img_patches.csv"
ds = load_most_dataset(root="./data/",
                       img_dir="./data/MOST_OAI_FULL_0_2/",
                       save_meta_dir=save_meta_dir,
                       saved_patch_dir="./processed_data/MOST_00_0_2_cropped",
                       output_filename=output_filename,
                       force_reload=False, force_rewrite=False)

# Filter out unreliable records without OARSI scores
list_rows = []
for i, row in ds.iterrows():
    if row['KL'] >= 0 and row['KL'] <= 4 and row['XRJSL'] >= 0 and row['XRJSL'] <= 3 and row['XRJSM'] >= 0 and \
            row['XRJSM'] <= 3 and row['XROSFL'] >= 0 and row['XROSFL'] <= 3 and row['XROSFM'] >= 0 and \
            row['XROSFM'] <= 3 and row['XROSTL'] >= 0 and row['XROSTL'] <= 3 and \
            row['XROSTM'] >= 0 and row['XROSTM'] <= 3:
        list_rows.append(row)

df_filtered = pd.DataFrame(list_rows)

out_filename = os.path.join(save_meta_dir, output_filename)
print(f'Writing MOST data to {out_filename}')
df_filtered.to_csv(out_filename, index=False, sep='|')
