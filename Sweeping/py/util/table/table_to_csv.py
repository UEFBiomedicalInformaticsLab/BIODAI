import os
from pathlib import Path

from util.table.table import Table


def table_to_csv(table: Table, file_path: str, overwrite: bool = False):
    """Does not create a new file if it exists. Creates also directories if needed.
    Reads the table and writes the file in chunks to save memory."""
    if not overwrite:
        file_exist = os.path.isfile(file_path)
        if file_exist:
            raise ValueError("File exist." + " File path: " + str(file_path))
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    first_chunk = True
    for chunk in table.chunks_df():
        if first_chunk:
            first_chunk = False
            chunk.to_csv(file_path, index=True, mode='w')  # Write the header in the first chunk
        else:
            chunk.to_csv(file_path, index=True, mode='a', header=False)  # Append without writing the header
