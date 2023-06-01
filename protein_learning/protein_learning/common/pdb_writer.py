#from common.sidechain_constants import ONE_TO_THREE

SC_ATOMS = ['CE3', 'CZ', 'SD', 'CD1', 'NH1', 'OG1', 'CE1', 'OE1', 'CZ2', 'OH', 'CG',
            'CZ3', 'NE', 'CH2', 'OD1', 'NH2', 'ND2', 'OG', 'CG2', 'OE2', 'CD2', 'ND1', 'NE2',
            'NZ', 'CD', 'CE2', 'CE', 'OD2', 'SG', 'NE1', 'CG1']
ATOM_ORDER = ['N', 'CA', 'C', 'O', 'CB'] + list(sorted(SC_ATOMS))


def atom_to_one_letter(atom_ty):
    return atom_ty.upper()[0]


def format_coords(coords):
    coords = list(map(lambda x: format(x, ".3f"), coords))
    space0 = make_space(12 - len(str(coords[0])))
    space1 = make_space(8 - len(str(coords[1])))
    space2 = make_space(8 - len(str(coords[2])))
    return f"{space0}{coords[0]}{space1}{coords[1]}{space2}{coords[2]}"


def make_space(spaces):
    return "".join([" "] * spaces)


def space_for_idx(idx):
    return make_space(7 - len(str(idx)))


def space_for_res_ty(atom_ty):
    return make_space(4 - len(atom_ty))


def space_for_res_idx(res_idx):
    return make_space(4 - len(str(res_idx)))


def format_line(res_ty, atom_ty, coords, res_idx, atom_idx):
    res_ty = ONE_TO_THREE[res_ty]
    line = f"ATOM{space_for_idx(atom_idx)}{atom_idx}  {atom_ty}"
    line += f"{space_for_res_ty(atom_ty)}{res_ty} A{space_for_res_idx(res_idx)}{res_idx}"
    line += format_coords(coords)
    line += f"  1.00  0.00           {atom_to_one_letter(atom_ty)}\n"
    return line


def order_atoms(atom_tys):
    ordered = []
    for a in ATOM_ORDER:
        if a in atom_tys:
            ordered.append(a)
    return ordered


def write_pdb(coord_dict, seq, out_path):
    assert out_path.endswith(".pdb")
    pdb = "REMARK repacked structure by EDGEPack\n"
    atom_idx = 1
    for res_idx, (res, atoms) in enumerate(zip(seq, coord_dict)):
        for atom_ty in order_atoms(atoms):
            coords = atoms[atom_ty]
            pdb += format_line(res_ty=res, atom_ty=atom_ty, coords=coords, res_idx=res_idx + 1, atom_idx=atom_idx)
            atom_idx += 1
    with open(out_path, "w+") as f:
        f.write(pdb + "TER")
