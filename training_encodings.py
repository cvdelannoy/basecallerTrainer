import re

valid_encoding_types = ['trimer',
                        'pupy',
                        'hp_5class']

def is_valid_encoding_type(encoding_type):
    if encoding_type in valid_encoding_types:
        return True
    return False


def class_number(kmer, encoding_type):
    if encoding_type == 'trimer':
        return trimer_class_number(kmer)
    if encoding_type == 'pupy':
        return pu_py_class_number(kmer)
    if encoding_type == 'hp_5class':
        return hp_class_number(kmer)
    ValueError('encoding type not recognized')

# Subdivision of all trimers in four ordinal classes
cl1 = ['GGT', 'GGA', 'AGT', 'GGG', 'AGG', 'GAT', 'AGA', 'GAG', 'GAA', 'CGT', 'CGA', 'AAT', 'TGA', 'CGG', 'AAG', 'TGT']
cl2 = ['GGC', 'AAA', 'GAC', 'CAT', 'CAG', 'AGC', 'TGG', 'TAT', 'CAA', 'TAG', 'AAC', 'CGC', 'TAA', 'TGC', 'CAC', 'TAC']
cl3 = ['GCT', 'CCT', 'TCT', 'ACT', 'CCG', 'TTT', 'GTT', 'GCG', 'TCG', 'CTT', 'GCA', 'ACG', 'CCA', 'TCA', 'ATT', 'ACA']
cl4 = ['CCC', 'TTG', 'TCC', 'GTA', 'TTA', 'GTG', 'GCC', 'CTG', 'ACC', 'CTA', 'ATG', 'ATA', 'TTC', 'GTC', 'CTC', 'ATC']


def hp_class_number(kmer):
    """
    Classify homopolymer-containing k-mers using a len(kmer)-class system
    """

    k_length = len(kmer)
    class_list = []
    if 'N' in kmer:  # always classify as 1 if kmer contains unknown base
        return 1
    for base in [kmer[0], kmer[-1]]:
        pat = re.compile(base)
        pat_index = [m.start(0) for m in pat.finditer(kmer)]
        lst = [i in pat_index for i in range(k_length)]
        ccf = 0
        ccr = 0
        boolf = True
        boolr = True
        for i in range(k_length):
            if not lst[i]:
                boolf = False  # If series of trues stops in fwd direction, stop adding
            if not lst[-i-1]:
                boolr = False  # If series of trues stops in bwd direction, stop adding
            if not boolf and not boolr:
                break  # If both series are discontinued, stop iterating
            ccf += boolf
            ccr += boolr
        class_list += [ccf, ccr]
    return max(class_list + [1])  # return Nb in range 1( = no dimer at start) - k( = homopolymer)


def trimer_class_number(kmer):
    mid = len(kmer)//2 + 1
    trimer = kmer[mid-2:mid+1]
    if trimer in cl1:
        return 1
    if trimer in cl2:
        return 2
    if trimer in cl3:
        return 3
    if trimer in cl4:
        return 4
    raise ValueError('trimer not recognized.')


def pu_py_class_number(kmer):
    mid_base = kmer[len(kmer) // 2]
    if mid_base in ['A', 'G']:
        return 1
    if mid_base in ['T', 'C']:
        return 2

def dt_class_number(kmer):
    """
    Classify dinucleotides-containing k-mers using a len(kmer)-class system
    """
    # TODO not finished
    k_length = len(kmer)
    class_list = []
    for base in [kmer[0], kmer[-1]]:
        pat = re.compile(base)
        pat_index = [m.start(0) for m in pat.finditer(kmer)]
        lst = [i in pat_index for i in range(k_length)]
        ccf = 0
        ccr = 0
        boolf = True
        boolr = True
        for i in range(k_length):
            if not lst[i]:
                boolf = False  # If series of trues stops in fwd direction, stop adding
            if not lst[-i-1]:
                boolr = False  # If series of trues stops in bwd direction, stop adding
            if not boolf and not boolr:
                break  # If both series are discontinued, stop iterating
            ccf += boolf
            ccr += boolr
        class_list += [ccf, ccr]
    return max(class_list + [1])  # return Nb in range 1( = no dimer at start) - k( = homopolymer)
