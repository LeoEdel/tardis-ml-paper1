import numpy as np


def loada(afile: str, rec: int, idm: int, jdm: int):
    """
    Read a binary file (.a) a return the numpy array corresponding to the reconrd
    :param afile: path for the a file
    :param rec: index of the record to load in the file (inital rec: 1)
    :param idm: number of raws in the array
    :param jdm: number of columns in the array

    Julien Brajard : 28.04.2022 first version
    julien.brajard@nersc.no
    """

    n2drec = int(np.floor((idm * jdm + 4095) / 4096) * 4096)
    bytes_per_float = 4

    fld_1d = np.fromfile(afile,
                         '>f',  # >f: big-endian single precision
                         count=idm * jdm,
                         sep="",  # empty sep means binary file
                         offset=n2drec * bytes_per_float * (rec - 1),
                         )

    return fld_1d.reshape(jdm, idm)



def loada_seq(afile: str, lrec: list, idm: int, jdm: int, dtype = np.float32):
    """
    Read a binary file (.a) a return the numpy array corresponding to several records
    :param afile: path for the a file
    :param lrec: list  records to load in the file (inital rec: 1)
    :param idm: number of raws in the array
    :param jdm: number of columns in the array

    Julien Brajard : 11.05.2022 first version
    julien.brajard@nersc.no
    """

    n2drec = int(np.floor((idm * jdm + 4095) / 4096) * 4096)
    bytes_per_float = 4
    
    nrec = len(lrec)
    fld_3d = np.empty(shape=(nrec, jdm, idm), dtype=dtype)
    for irec,rec in enumerate(lrec):
        fld_3d[irec] = np.fromfile(afile,
                         '>f',  # >f: big-endian single precision
                         count=idm * jdm,
                         sep="",  # empty sep means binary file
                         offset=n2drec * bytes_per_float * (rec - 1),
                         ).reshape(jdm, idm)
        

    return fld_3d

