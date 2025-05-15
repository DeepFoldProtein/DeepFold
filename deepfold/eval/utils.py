def read_iter(fasta_string: str):
    header = None
    seq_str_list = []
    for line in fasta_string.splitlines():
        line = line.strip()
        # Ignore empty and comment lines.
        if len(line) == 0 or line[0] == ";":
            continue
        if line[0] == ">":
            # New entry.
            if header is not None:
                yield header, "".join(seq_str_list)
            # Track new header and reset sequence.
            header = line[1:]
            seq_str_list = []
        else:
            seq_str_list.append(line)
    # Yield final entry.
    if header is not None:
        yield header, "".join(seq_str_list)
