#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#preprocessed
import tensorflow
import argparse as arg
import pandas as pd
import pubchempy as pcp
from pubchempy import *
import numpy as np
import kora.install.rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from pyensembl import EnsemblRelease
from tensorflow.keras.utils import to_categorical
parser = arg.ArgumentParser(description='smiles_n2v')
parser.add_argument('--input', '-i', default='dataset_hard')
parser.add_argument('--data', '-d', default='/train')
args = parser.parse_args(args=[])

from google.colab import drive
#drive.mount('/content/drive')
path_data='/content/drive/My Drive/Khani/multi_DTI/'
for i in range(5):
    #load protein-compound interaction dataset
    data = pd.read_csv(path_data+args.input+'/cv_'+str(i)+args.data+'.csv')

    #save labels
    label = np.array(data['label'], dtype='int32')
    #np.save(path_data+args.input+'/cv_'+str(i)+args.data+'_interaction.npy', label)

    #save Ensembl protein ID (ENSP) for applying to node2vec
    with open(path_data+args.input+'/cv_'+str(i)+args.data+'_proIDs.txt', mode='w') as f:
        f.write('\n'.join(data['protein']))

    #save pubchem ID for applying to node2vec
    cid = np.array(data['chemical'], dtype='int32')
    #np.save(path_data+args.input+'/cv_'+str(i)+args.data+'_chemIDs.npy', cid)

    #convert pubchem ID to CanonicalSMILES
    c_id = data.chemical.tolist()
    pcp.download('CSV',path_data+ args.input+'/cv_'+str(i)+'/ismilesref.csv', c_id, operation='property/IsomericSMILES', overwrite=True)
    smileb =pd.read_csv(path_data+args.input+'/cv_'+str(i)+'/ismilesref.csv')
    smib = []
    for j in smileb['IsomericSMILES']:
        smib.append(Chem.MolToSmiles(Chem.MolFromSmiles(j), kekuleSmiles=False, isomericSmiles=True))
    with open(path_data+args.input+'/cv_'+str(i)+args.data+'.smiles', mode='w') as f:
        f.write('\n'.join(smib))
        
    
    smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, " ":65, ":": 66, ",":67, "p":68, "j": 69}    
     
    def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
        X = np.zeros(MAX_SMI_LEN)
        for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
            if ch=='\n':
               gh=0
            else:
               try:
                   X[i] = smi_ch_ind[ch]
               except:
                   print('jj')

        return X #.tolist()
    #get convert CanonicalSMILES from pubchem chemical ID and convert them to onehot vectors
    file_smiles=pd.read_csv(path_data+'dataset_hard'+'/cv_'+str(i)+'/train.smiles',delimiter='0')
    #cID = np.load(path_data+'dataset_hard/'+'cv_'+str(i)+'/test_chemIDs.npy')
    """tosmiles = []
    for jj in range(0,cID.shape[0]):
        #cs = get_compounds(cID[jj],'cid')
        #c = Compound.from_cid(cID[jj])
        c = Compound.from_cid(int(cID[jj]))
        tosmiles.append(c.canonical_smiles)  """
    #tosmiles=np.load('/content/drive/My Drive/Khani/multi_DTI/smiles_'+str(i)+'.npy')  
    #clen = len(cID)
    #file_smiles=file_smiles.apply(str)
    smiles_mol=[]
    print('injam')
    print(file_smiles)
    for jj in range(0,len(file_smiles)):
        smiles_mol.append(label_smiles(str(file_smiles.iloc[jj]), 100, smiles_dict))
    smiles_mol.append(label_smiles(str(file_smiles.iloc[jj]), 100, smiles_dict))
    #np.save('/content/drive/My Drive/Khani/multi_DTI/smiles_mol'+str(i)+'.npy',smiles_mol) 
    np.save('/content/drive/My Drive/Khani/multi_DTI/smiles_mol'+str(i)+'.npy',smiles_mol)#to_categorical(smiles_mol[0:500],70)) 
    
    #for j in cID:
        #smiles = jsp.chemical_smiles(j) #get smiles from using pubchemy 
        #tosmiles.append(smiles)
        
    """#smiles = ["#", "%" , ")", "(", "+", "-", "/", "." , "1", "0", "3", "2", "5", "4",
    "7", "6", "9", "8", "=", "A", "@", "C", "B", "E" ,
    "D", "G", "F", "I", "H", "K", "M", "L", "O", "N", "P", "S", "R", "U", "T", "W" ,
    "V", "Y", "[", "Z", "]", "\\", "a", "c", "b", "e", "d", "g", "f", "i" , "h" , "m", 
    "l", "o", "n", "s", "r", "u", "t", "y", " ", ":", "," , "p", "j"] #defeine universe of possible input values
    #char_to_int = dict((c, n) for n, c in enumerate(smiles)) #defeine a mapping of chars to integers
    #int_to_char = dict((n, c) for n, c in enumerate(smiles))
    #integer_encoded = []
    #for l in range(clen(tosmiles)):
        #integer_encoded.append([char_to_int[char] for char in tosmiles[l]]) #integer encode input data"""



    """Max_smiles = 200
    onehot_tr = np.empty((clen,Max_smiles,0), dtype='float32')
    for j in range(clen(integer_encoded)):
        b_onehot_com = np.identity(0, dtype='float32')[integer_encoded[j]]
        differ_tr_com = Max_smiles - len(integer_encoded[j])
        b_zeros_com = np.zeros((differ_tr, 0), dtype='float32')
        onehot_tr_com[j] = np.vstack((b_onehot_com, b_zeros_com))
    np.save(args.input+'/cv_'+str(i)+args.data+'_recompound.npy', onehot_tr)"""        
            
    #get amino acid seq from Ensembl protein ID (ENSP) and convert them to onehot vectors 

    with open(path_data+'dataset_hard'+'/cv_'+str(i)+args.data+'_proIDs.txt') as f:
        pID = [s.strip() for s in f.readlines()]
    plen = len(pID)
    ens = EnsemblRelease(93) #release 93 uses human reference genome GRCh38
    toseq = []
    for j in pID:
        seq = ens.protein_sequence(j) #get amino acid seq from ENSP using pyensembl
        toseq.append(seq)

    amino_acid = 'ACDEFGHIKLMNPQRSTVWY' #defeine universe of possible input values
    char_to_int = dict((c, n) for n, c in enumerate(amino_acid)) #defeine a mapping of chars to integers
    int_to_char = dict((n, c) for n, c in enumerate(amino_acid))
    integer_encoded = []
    for l in range(len(toseq)):
        integer_encoded.append([char_to_int[char] for char in toseq[l]]) #integer encode input data
    
    
    
    Max_seq = 2000
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print(len(integer_encoded))
    integer_encoded = pad_sequences(integer_encoded, maxlen=Max_seq)
    #onehot_tr = to_categorical(integer_encoded[0:500],21)
    """onehot_tr = np.empty((plen, Max_seq, 20), dtype='float32')
    for j in range(len(integer_encoded)):
        b_onehot = np.identity(20, dtype='float32')[integer_encoded[j]]
        differ_tr = Max_seq - len(integer_encoded[j])
        b_zeros = np.zeros((differ_tr, 20), dtype='float32')
        onehot_tr[j] = np.vstack((b_onehot, b_zeros))"""
    np.save(path_data+args.input+'/cv_'+str(i)+args.data+'_reprotein.npy', integer_encoded)#integer_encoded)
    print(path_data+args.input+'/cv_'+str(i)+args.data+'_reprotein.npy')
    print('fj')

print(type(cid))
print(cid[0])
c = Compound.from_cid(int(cid[0]))
c.canonical_smiles
print(args.input)
# In[ ]: