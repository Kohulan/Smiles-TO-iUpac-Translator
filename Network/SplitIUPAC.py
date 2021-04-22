import re

checkWords = (
"(", ")", "]", "[", "{", "}", "-", ",", ".", "di", "tri", "tetra", "penta", "mono", "hexa", "hepta", "octo", "nona",
"deca", "oxo", "octa", "methyl", "hydroxy", "benzene", "oxy", "hydr oxy", "chloro", "cyclo", "amino", "bromo",
"methane", "hydro", "fluoro", "methane", "cyano", "amido", "ethene", "phospho", "amide", "butane", "carbono", "hydro",
"sulfane", "iodo", "butane", "sulfino", "iodo", "ethane", "ethyne", "bi", "oxo", "imino", "nitro", "butan", "idene",
"sulfo", "carbon", "propane", "ethen", "acetaldehyde", "benzo", "butan", "oxa", "nitro so", "nitroso", "hydra", "iso",
"butan", "acid", "  ", "  ", "  ", "  ", " \n", "\n ", "yl")
repWords = (
" ( ", " ) ", " ] ", " [ ", " { ", " } ", " - ", " , ", " . ", " di ", " tri ", " tetra ", " penta ", " mono ",
" hexa ", " hepta ", " octo ", " nona ", " deca ", " oxo ", " octa ", " methyl ", " hydroxy ", " benzene ", " oxy ",
" hydroxy ", " chloro ", " cyclo ", " amino ", " bromo ", " methane ", " hydro ", " fluoro ", " methane ", " cyano ",
" amido ", " ethene ", " phospho ", " amide ", " butane ", " carbono ", " hydro ", " sulfane ", " iodo ", " butane ",
" sulfino ", " iodo ", " ethane ", " ethyne ", " bi ", " oxo ", " imino ", " nitro ", " butan ", " idene ", " sulfo ",
" carbon ", " propane ", " ethen ", " acetaldehyde ", " benzo ", " butan ", " oxa ", " nitroso ", " nitroso ",
" hydra ", " iso ", " butan ", " acid ", " ", " ", " ", " ", "\n", "\n", "yl ")

def get_modified_iupac(iupac_string):
	for check, rep in zip(checkWords, repWords):
		iupac_string = iupac_string.replace(check, rep)
	return iupac_string