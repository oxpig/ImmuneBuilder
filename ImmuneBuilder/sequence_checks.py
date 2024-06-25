from anarci import validate_sequence, anarci, scheme_short_to_long

def number_single_sequence(sequence, chain, scheme="imgt", allowed_species=['human','mouse']):
    validate_sequence(sequence)

    try:
        if scheme != "raw":
            scheme = scheme_short_to_long[scheme.lower()]
    except KeyError:
        raise NotImplementedError(f"Unimplemented numbering scheme: {scheme}")

    assert len(sequence) > 70, f"Sequence too short to be an Ig domain. Please give whole sequence:\n{sequence}"

    allow = [chain]
    if chain == "L":
        allow.append("K")

    # Use imgt scheme for numbering sanity checks
    numbered, _, _ = anarci([("sequence", sequence)], scheme='imgt', output=False, allow=set(allow), allowed_species=allowed_species)

    assert numbered[0], f"Sequence provided as an {chain} chain is not recognised as an {chain} chain."

    output = [x for x in numbered[0][0][0] if x[1] != "-"]
    numbers = [x[0][0] for x in output]

    # Check for missing residues assuming imgt numbering
    assert (max(numbers) > 120) and (min(numbers) < 10), f"Sequence missing too many residues to model correctly. Please give whole sequence:\n{sequence}"

    # Renumber once sanity checks done
    if scheme == "raw":
        output = [((i+1, " "),x[1]) for i,x in enumerate(numbered[0][0][0]) if x[1] != "-"]
    elif scheme != 'imgt':
        numbered, _, _ = anarci([("sequence", sequence)], scheme=scheme, output=False, allow=set(allow), allowed_species=allowed_species)
        output = [x for x in numbered[0][0][0] if x[1] != "-"]

    return output


def number_sequences(seqs, scheme="imgt", allowed_species=['human','mouse']):
    return {chain: number_single_sequence(seqs[chain], chain, scheme=scheme, allowed_species=allowed_species) for chain in seqs}
