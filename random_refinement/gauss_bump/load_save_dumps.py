import pickle
from pathlib import Path
import warnings


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


def load_dump(path_to_dump: Path):
    try:
        with open(path_to_dump, mode='rb') as file:
            return pickle.load(file=file)
    except FileNotFoundError:
        warnings.warn(f"Warning: The file '{path_to_dump}' does not exist.")
        return None  # Return None or any other fallback value
