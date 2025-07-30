import pickle

pkl_path = "client_data/global_val_data_mnli.pkl"
with open(pkl_path, "rb") as f:
    val_data = pickle.load(f)

# Sanity check and report invalid labels
bad_indices = []
for i, example in enumerate(val_data):
    label = example["labels"]

    # Convert label to a plain Python int if possible
    try:
        label = int(label)
    except (ValueError, TypeError):
        print(f" Could not convert label to int: {label} at index {i}")
        bad_indices.append(i)
        continue

    # Check if label is out of expected range
    if label < 0 or label > 2:
        print(f" Invalid label {label} at index {i}")
        bad_indices.append(i)

# Summary
if not bad_indices:
    print(" All labels are valid (0, 1, 2).")
else:
    print(f" Found {len(bad_indices)} invalid labels.")
