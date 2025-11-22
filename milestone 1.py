import nbformat

# List of notebooks in the desired order
notebooks = [
    "01_EDA_PartA.ipynb",
    "01_EDA_PartB.ipynb",
    "02_Preprocessing_PartA.ipynb",
    "02_Preprocessing_PartB.ipynb"
]

# Create a new notebook
merged_nb = nbformat.v4.new_notebook()

# Merge cells from all notebooks
for nb_file in notebooks:
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        merged_nb.cells.extend(nb.cells)

# Save the merged notebook
output_file = "Milestone1_Combined.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    nbformat.write(merged_nb, f)

print(f"All notebooks merged successfully into '{output_file}'!")
