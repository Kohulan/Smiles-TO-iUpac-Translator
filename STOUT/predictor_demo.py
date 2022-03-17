import time
from stout import translate_forward, translate_reverse

# STOUT - IUPAC name to SMILES example

file_iupac = open(
    "IUPAC_names_test.txt", "r"
)  # file is available in the Github repository
file_out = open("SMILES_predictions", "w")

start = time.time()
for i, line in enumerate(file_iupac):
    iupac_name = line.strip("\n")
    SMILES = translate_reverse(iupac_name)
    file_out.write(SMILES + "\n")
file_out.flush()
file_out.close()

# STOUT - SMILES to IUPAC names example
file_smiles = open("SMILES_test.txt", "r")  # file is available in the Github repository
file_out = open("IUPAC_predictions", "w")

for i, line in enumerate(file_smiles):
    SMILES = line.strip("\n")
    iupac_name = translate_forward(SMILES)
    file_out.write(iupac_name + "\n")
file_out.flush()
file_out.close()

print("Time taken for per prediction is {} sec\n".format((time.time() - start) / 100))
