"""
Set genotype as command line script
"""

import os
import sys
sys.path.insert(1, "../")

import argparse
from pathlib import Path

import sqlite3
from lmtanalysis.Util import mute_prints
from lmtanalysis.Animal import AnimalPool
from lmtanalysis.FileUtil import getFilesToProcess



def set_genotype(files):
    if files is None:
        print("No files selected (aborting)")
        return
    for file in files:
        print("\n\nFile: " , file )
        print("-"*80)
        if not os.path.exists(file):
            print ("  ! File does not exist... (skipping)")
            continue
        
        with mute_prints():

            connection = sqlite3.connect( file )

            pool = AnimalPool( )
            pool.loadAnimals( connection )
            
        print(f"Setting genotype of {len(pool.getAnimalList())} mice, press [Enter] to keep existing one):")

        for animal in sorted(pool.getAnimalList(), key=lambda a: a.name):
            
            genotype = input(f"  * {animal.name}_{animal.RFID} [{animal.genotype}]: ")
            genotype = genotype.strip()
            if len(genotype) > 0:
                print(f"     - setting {animal.name}_{animal.RFID} to '{genotype}'")
                animal.setGenotype( genotype )
            else:
                print(f"     - keeping genotype {animal.genotype}")

        print("Genotype saved in database.")
        print("-"*120)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='*', type=Path, help='sqlite file(s)')
    
    p = parser.parse_args()

    if len(p.files) == 0:
        files = getFilesToProcess()
    else:
        files = map(str, p.files)
    set_genotype(files)
        
        
    
        
    
        
        
    
    
