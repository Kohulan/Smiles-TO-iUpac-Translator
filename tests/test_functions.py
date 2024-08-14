from STOUT import translate_forward, translate_reverse


def test_smilestoiupac():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    expected_result = "1,3,7-trimethylpurine-2,6-dione"
    actual_result = translate_forward(smiles)
    assert expected_result == actual_result


def test_iupactosmiles():
    iupac_name = "1,3,7-trimethylpurine-2,6-dione"
    expected_result = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    actual_result = translate_reverse(iupac_name)
    assert expected_result == actual_result
