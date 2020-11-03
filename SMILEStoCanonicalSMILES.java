/*
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Copyright (c) 2019, Kohulan Rajan
 */

import java.text.DecimalFormat;

import java.io.*;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.io.SDFWriter;
import org.openscience.cdk.layout.StructureDiagramGenerator;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;

public class SMILEStoCanonicalSMILES {
	String temp = "";
	String moleculeTitle = null;
	int moleculeCount = 0;
	boolean verbose = false;

	public static void main(String[] args) throws Exception {
		IAtomContainer molecule = null;
		SmilesGenerator sg = new SmilesGenerator(SmiFlavor.Canonical);
		String line = args[0].toString();
			try {
			SmilesParser smi = new SmilesParser(DefaultChemObjectBuilder.getInstance());
			molecule = smi.parseSmiles(line);

			StructureDiagramGenerator sdg = new StructureDiagramGenerator();
			sdg.setMolecule(molecule);
			sdg.generateCoordinates(molecule);
			molecule = sdg.getMolecule();
			String smi_ori = sg.create(molecule);
			System.out.println(smi_ori);
		}
			 catch(Exception e) { 
			 	System.out.println(e);
			 }
		}
		
	}
