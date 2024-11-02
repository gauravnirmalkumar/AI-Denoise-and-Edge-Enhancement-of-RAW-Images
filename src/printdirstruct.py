import os

def print_directory_structure(root_dir, output_file):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(root_dir):
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            file.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                file.write(f'{subindent}{f}\n')

if __name__ == "__main__":
    root_directory = "D:\\assignment2"  # Replace with your directory path
    output_file_path = "D:\\assignment2\\directory_structure.txt"
    print_directory_structure(root_directory, output_file_path)
    print(f"Directory structure has been saved to {output_file_path}")
