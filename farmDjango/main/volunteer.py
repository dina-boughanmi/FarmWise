def get_matching_column(df,plant_name):
    
    if plant_name in df['ScientificName'].values:
        return 'ScientificName'
    elif plant_name in df['Common name'].values:
        return 'Common name'
    else:
        raise ValueError(f"Plant name '{plant_name}' not found in either 'ScientificName' or 'Common Name'.")

def metadata(df, plant_name, column):
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    row = df[df[column] == plant_name]

    if row.empty:
        raise ValueError(f"Plant name '{plant_name}' not found in column '{column}'.")

    return row.iloc[0].to_dict()
    