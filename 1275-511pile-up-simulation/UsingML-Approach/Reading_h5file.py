import h5py

import numpy as np

def ReadingH5File(filename):
    """
    Liest ein HDF5-File (filename), sucht automatisch das erste Dataset
    (auf Top-Ebene) und gibt dessen Daten und die Spaltennamen zurück.
    """
    with h5py.File(filename, "r") as f:
        # Alle Top-Level-Keys durchgehen:
        dataset_name = None
        
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                # Wir nehmen das erste gefundene Dataset
                dataset_name = key
                break
        
        if dataset_name is None:
            # Keins gefunden -> Abbruch oder Exception
            print("Fehler: Keine Datasets auf Top-Ebene gefunden.")
            return None, None
        
        # Dataset lesen
        dset = f[dataset_name]
        data_read = dset[()]  # oder dset[:] für NumPy-Array
        
        # Falls du ein Attribut "columns" gespeichert hast:
        colnames = dset.attrs.get("columns", None)  # None, falls nicht vorhanden
        
        print(f"Benutze Dataset '{dataset_name}':")
        #print("Eingelesene Daten:\n", data_read)
        if colnames is not None:
            print("Spaltennamen:", colnames)
        else:
            print("Keine Spaltennamen gefunden.")
    
    return data_read, colnames
