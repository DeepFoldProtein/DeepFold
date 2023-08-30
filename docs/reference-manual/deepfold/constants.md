# Constants

---

Some module attributes are frequently used in the package.
`deepfold.common.residue_constants` is the module where they are declared.

There are 20 different amino acids that are used for building proteins.
You can think of each residue as a token, and every type of amino acid as a different kind of token.
However, sometimes we were unable to identify the exact type of some residues.
In this case, we give those residues an `UNK` token.
There are 21 distinct types of `aatype` for this reason.

