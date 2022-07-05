from click import File
import pandas as pd;
import os;

def upload_file(file: File) -> pd.DataFrame:
    complete_name = os.path.splitext(file.name);

    file_name = complete_name[0];
    extension_file = complete_name[1];

    #Verify extension for read file

    if extension_file == '.csv': 
        data = pd.read_csv(file);
    elif extension_file in ('.xls', ".xlsx"):
        data = pd.read_excel(file);
    else: 
        data = pd.read_json(file);

    return data;
