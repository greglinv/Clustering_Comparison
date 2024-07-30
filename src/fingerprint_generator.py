import os



def load_fsl_fileset(directory):
    """Load files from the FSL fileset directory."""
    fileset = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return fileset

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as file:  # Read as binary
                    content = file.read()
                    fileset.append(content)
                    # Debug: Print the file being loaded
                    print(f"Loaded file: {filename}")
            except UnicodeDecodeError:
                print(f"Failed to load file: {filename}")

    # Debug: Print the total number of files loaded
    print(f"Total files loaded: {len(fileset)}")
    return fileset


def generate_fingerprints(fileset):
    """Generate fingerprints from the FSL fileset."""
    # Implement your fingerprint generation logic here
    fingerprints = [hash(file) for file in fileset]  # Example: hash the content of each file
    return fingerprints
