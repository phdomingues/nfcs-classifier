"""
This module contains auxiliary methods and classes to manipulate NFCS data.

It contains an Enumerator ease the identification of NFCS regions and a
function to map regions to directory names.
"""


from enum import IntEnum, auto


class NFCS(IntEnum):
    """Enumerator for NFCS face characteristics.

    Args:
        SN: "Sulco Nasolabial" (Nasolabial Furrow)
        FS: "Fronte Saliente" (Orotruding Forehead)
        FP: "Fenda Palpebral aprofundada" (Deepened Palpebral fissure)
        BA: "Boca Aberta" (Open Mouth)
        BE: "Boca Estirada" (Stretched Mouth)
    """

    SN = 0  # noqa: WPS115
    FS = auto()  # noqa: WPS115
    FP = auto()  # noqa: WPS115
    BA = auto()  # noqa: WPS115
    BE = auto()  # noqa: WPS115


def nfcs2dir(region: NFCS) -> str:
    """Map nfcs regions to a directory name.

    Args:
        region (NFCS): The specified region.

    Returns:
        str: Name for the directory
    """
    nfcs_dict = {
        NFCS.SN: 'nasolabial_fold',
        NFCS.FS: 'forehead',
        NFCS.FP: 'eyes',
        NFCS.BA: 'mouth',
        NFCS.BE: 'mouth',
    }
    return nfcs_dict.get(region, None)
