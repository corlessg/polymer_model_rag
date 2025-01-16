import pandas as pd
from CifFile import ReadCif
import os

def cif_to_dataframe(file_path):
    # Load the CIF file
    cif = ReadCif(file_path)
    
    # Access the first (and likely only) block
    block_name = list(cif.keys())[0]
    data_block = cif[block_name]
    
    # Extract unit cell parameters
    unit_cell_params = {
        'Cell_Length_A': data_block.get('_cell_length_a', 'N/A'),
        'Cell_Length_B': data_block.get('_cell_length_b', 'N/A'),
        'Cell_Length_C': data_block.get('_cell_length_c', 'N/A'),
        'Cell_Angle_Alpha': data_block.get('_cell_angle_alpha', 'N/A'),
        'Cell_Angle_Beta': data_block.get('_cell_angle_beta', 'N/A'),
        'Cell_Angle_Gamma': data_block.get('_cell_angle_gamma', 'N/A'),
    }
    
    # Extract molecule name (if available)
    molecule_name = data_block.get('_chemical_name_common', 'Unknown Molecule').strip()
    
    # Check for atomic site data
    if all(key in data_block for key in ['_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']):
        # Extract atomic site data
        atom_labels = data_block['_atom_site_label']
        x_coords = data_block['_atom_site_fract_x']
        y_coords = data_block['_atom_site_fract_y']
        z_coords = data_block['_atom_site_fract_z']
        
        # Create a DataFrame
        data = {
            'Molecule_Name': [molecule_name] * len(atom_labels),
            'Cell_Length_A': [unit_cell_params['Cell_Length_A']] * len(atom_labels),
            'Cell_Length_B': [unit_cell_params['Cell_Length_B']] * len(atom_labels),
            'Cell_Length_C': [unit_cell_params['Cell_Length_C']] * len(atom_labels),
            'Cell_Angle_Alpha': [unit_cell_params['Cell_Angle_Alpha']] * len(atom_labels),
            'Cell_Angle_Beta': [unit_cell_params['Cell_Angle_Beta']] * len(atom_labels),
            'Cell_Angle_Gamma': [unit_cell_params['Cell_Angle_Gamma']] * len(atom_labels),
            'Label': atom_labels,
            'Fractional_X': x_coords,
            'Fractional_Y': y_coords,
            'Fractional_Z': z_coords
        }
        df = pd.DataFrame(data)
        return df
    else:
        print("No atomic site data found in the CIF file.")
        return pd.DataFrame()

def process_directory(directory_path, output_csv):
    """Process all CIF files in a directory and stream rows to a CSV file."""
    # Create or overwrite the CSV file with headers
    with open(output_csv, 'w') as f:
        # Define the header structure
        header = [
            'File_Name', 'Molecule_Name',
            'Cell_Length_A', 'Cell_Length_B', 'Cell_Length_C',
            'Cell_Angle_Alpha', 'Cell_Angle_Beta', 'Cell_Angle_Gamma',
            'Label', 'Fractional_X', 'Fractional_Y', 'Fractional_Z'
        ]
        f.write(','.join(header) + '\n')
    
    # Loop through all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.cif'):  # Process only .cif files
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_name}")
            
            try:
                # Convert CIF to DataFrame
                df = cif_to_dataframe(file_path)
                
                if not df.empty:
                    # Append rows to the CSV
                    print(df.head())
                    df.to_csv(output_csv, mode='a', header=False, index=False)
            except Exception as e:
                # Log the error and continue
                print(f"Error processing file {file_name}: {e}")

def main():
    # Path to the directory containing CIF files
    directory_path = "./data"  # Replace with your directory path
    
    # Output CSV file
    output_csv = "./cif_data_streamed.csv"
    
    # Process all CIF files in the directory and save to CSV
    process_directory(directory_path, output_csv)
    print(f"All CIF files processed. Data saved to '{output_csv}'.")


if __name__ == '__main__':
    main()