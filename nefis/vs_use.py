import os
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union


""" reads file structructure, collect all data groups, and under data groups are variables, 
they are split into categories (e.g. hydronammics, )

returns a dictionary describing the file structure, including metadata and data types
and sizes of the variables in a nefis file.
"""


@dataclass
class GroupData:
    """collections of thing that conceptually belong together, including metadata"""

    Offset: int
    VarDim: Union[bool, int]
    Name: str
    DefName: str
    IANames: List[str] = field(default_factory=list)
    IAValue: List[int] = field(default_factory=list)
    RANames: List[str] = field(default_factory=list)
    RAValue: List[float] = field(default_factory=list)
    SANames: List[str] = field(default_factory=list)
    SAValue: List[str] = field(default_factory=list)
    DataSize: Optional[int] = None
    DefIndex: Optional[int] = None
    SizeDim: Optional[List[int]] = None
    OrderDim: Optional[List[int]] = None


@dataclass
class ElementDefinition:
    """
    description per variable of what it is and it metadata
    """

    Name: str
    Type: str
    SizeVal: int
    SizeElm: int
    Quantity: str
    Units: str
    Description: str
    Size: List[int]


@dataclass
class VSStructure:
    FileType: str = 'NEFIS'
    SubType: str = ''
    FileName: str = ''
    DatExt: str = ''
    DefExt: str = ''
    Format: str = 'b'
    AddressType: str = 'uint64'
    GrpDat: List[GroupData] = field(default_factory=list)
    GrpDef: List = field(default_factory=list)
    CelDef: List = field(default_factory=list)
    ElmDef: List[ElementDefinition] = field(default_factory=list)


class VSUse:
    """main class to read NEFIS files"""

    def __init__(self, *args):
        self.args = list(args)
        self.debug = False
        self.quiet = False
        self.without_def = False
        self.vs_struct = VSStructure()

        self.data_file = ''
        self.def_file = ''
        self.filename = ''

        self._parse_args()
        self._resolve_files()
        self._open_data_file()
        self._read_group_data()
        if not self.without_def:
            self._read_definitions()

    def _parse_args(self):
        args = self.args
        if 'debug' in args:
            self.debug = True
            args.remove('debug')
        if 'quiet' in args:
            self.quiet = True
            args.remove('quiet')
        if 'nodef' in args:
            self.without_def = True
            args.remove('nodef')

        if len(args) == 1:
            self.filename = args[0]
        elif len(args) == 2:
            self.filename = args[0]
            self.data_file = args[0]
            self.def_file = args[1]
        elif len(args) > 2:
            raise ValueError('Too many or invalid input arguments.')

        if not isinstance(self.filename, str):
            raise ValueError('Invalid filename')

    def _resolve_files(self):
        if not self.data_file and not self.def_file:
            base_path = Path(self.filename)
            base_stem = base_path.stem
            base_dir = base_path.parent
            ext = base_path.suffix.lower()
            if ext in ['.dat', '.def']:
                self.data_file = str(base_dir / (base_stem + '.dat'))
                self.def_file = str(base_dir / (base_stem + '.def'))
            elif ext == '':
                self.data_file = str(base_dir / (base_stem + '.dat'))
                self.def_file = str(base_dir / (base_stem + '.def'))
            else:
                raise FileNotFoundError(f'Unknown extension: {ext}. Cannot resolve .dat/.def files.')
        else:
            self.data_file = str(Path(self.data_file).resolve())
            self.def_file = str(Path(self.def_file).resolve())

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f'Data file not found: {self.data_file}')
        if not os.path.exists(self.def_file) and not self.without_def:
            raise FileNotFoundError(f'Definition file not found: {self.def_file}')

    def _open_data_file(self):
        # TODO: this part is not implemented as in matlab. look on how to fix the parsing of the header and the data address.
        # compare with the matlab code

        if self.def_file == '':
            header_lenght = 128
            print('[INFO] No definition file provided. Using default header length of 128 bytes.')
        else:
            header_lenght = 60

        with open(self.data_file, 'rb') as f:
            # print(f.read())
            header = f.read(header_lenght)
            if not header.find(b'NEFIS Data File'):
                raise ValueError('Invalid NEFIS file header.')

            print(header)

            f.seek(header_lenght)  # move pointer to this position, to skip the header
            big_endian8 = struct.unpack('>Q', f.read(8))[0]  #  read moves the pointer

            f.seek(header_lenght)  # to reset the pointer to the header position
            little_endian8 = struct.unpack('<Q', f.read(8))[0]

            f.seek(header_lenght)
            big_endian4 = struct.unpack('>I', f.read(4))[0]

            f.seek(header_lenght)
            little_endian4 = struct.unpack('<I', f.read(4))[0]

            print(f'[DEBUG] big_endian8: {big_endian8}, little_endian8: {little_endian8}')
            print(f'[DEBUG] big_endian4: {big_endian4}, little_endian4: {little_endian4}')

            f.seek(0, 2)  # move pointer to the end of the file
            actual_size = f.tell()  # get current position of the pointer = file size

            if big_endian8 <= actual_size:
                self.vs_struct.AddressType = 'uint64'
                self.vs_struct.Format = 'b'  # Big-endian
            elif big_endian4 <= actual_size:
                self.vs_struct.AddressType = 'uint32'
                self.vs_struct.Format = 'b'  # Big-endian
            elif little_endian8 <= actual_size:
                self.vs_struct.AddressType = 'uint64'
                self.vs_struct.Format = 'l'  # Little-endian
            elif little_endian4 <= actual_size:
                self.vs_struct.AddressType = 'uint32'
                self.vs_struct.Format = 'l'  # Little-endian
            else:
                raise ValueError('Unable to determine address type from file header.')

    def _read_group_data(self):
        fmt = self.vs_struct.Format
        addr_type = self.vs_struct.AddressType
        addr_bytes = 8 if addr_type == 'uint64' else 4
        hash_table_offset = 128 + 3 * addr_bytes * 997
        with open(self.data_file, 'rb') as f:
            f.seek(hash_table_offset)
            bucket_format = ('<' if fmt == 'l' else '>') + ('Q' if addr_type == 'uint64' else 'I')
            hash_table = [struct.unpack(bucket_format, f.read(addr_bytes))[0] for _ in range(997)]
            for offset in filter(lambda x: x != 2 ** (8 * addr_bytes) - 1, hash_table):
                f.seek(offset)
                link = struct.unpack(bucket_format, f.read(addr_bytes))[0]
                size = struct.unpack(bucket_format, f.read(addr_bytes))[0]
                code = f.read(addr_bytes)
                var_dim = code[-1] == ord('5')
                name = f.read(16).decode('ascii').strip('\x00')
                defname = f.read(16).decode('ascii').strip('\x00')
                ianames = [f.read(16).decode('ascii').strip('\x00') for _ in range(5)]
                iavalues = list(struct.unpack(('>' if fmt == 'b' else '<') + '5i', f.read(20)))
                ranames = [f.read(16).decode('ascii').strip('\x00') for _ in range(5)]
                ravalues = list(struct.unpack(('>' if fmt == 'b' else '<') + '5f', f.read(20)))
                sanames = [f.read(16).decode('ascii').strip('\x00') for _ in range(5)]
                savalues = [f.read(16).decode('ascii').strip('\x00') for _ in range(5)]
                data_size = size - (3 * addr_bytes + 392) if not var_dim else None
                self.vs_struct.GrpDat.append(
                    GroupData(
                        Offset=offset,
                        VarDim=var_dim,
                        Name=name,
                        DefName=defname,
                        IANames=ianames,
                        IAValue=iavalues,
                        RANames=ranames,
                        RAValue=ravalues,
                        SANames=sanames,
                        SAValue=savalues,
                        DataSize=data_size,
                    )
                )

    def _read_definitions(self):
        fmt = self.vs_struct.Format
        addr_type = self.vs_struct.AddressType
        addr_bytes = 8 if addr_type == 'uint64' else 4
        endian = '>' if fmt == 'b' else '<'
        with open(self.def_file, 'rb') as f:
            f.seek(128)
            for _ in range(997):
                f.read(addr_bytes)  # Skip element hash table
            for _ in range(997):
                f.read(addr_bytes)  # Skip cell hash table
            for _ in range(997):
                f.read(addr_bytes)  # Read group definition hash table

            # Read element definitions
            while True:
                header = f.read(addr_bytes * 3)
                if not header or len(header) < addr_bytes * 3:
                    break
                name = f.read(16).decode('ascii').strip('\x00')
                el_type = f.read(8).decode('ascii').strip('\x00')
                size_elm = struct.unpack(endian + ('I' if addr_bytes == 4 else 'Q'), f.read(addr_bytes))[0]
                size_val = struct.unpack(endian + 'I', f.read(4))[0]
                desc = f.read(96).decode('ascii')
                quantity = desc[0:16].strip('\x00')
                units = desc[16:32].strip('\x00')
                description = desc[32:].strip('\x00')
                ndim = struct.unpack(endian + 'I', f.read(4))[0]
                size = list(struct.unpack(endian + '5I', f.read(20)))[:ndim]
                self.vs_struct.ElmDef.append(
                    ElementDefinition(
                        Name=name,
                        Type=el_type,
                        SizeVal=size_val,
                        SizeElm=size_elm,
                        Quantity=quantity,
                        Units=units,
                        Description=description,
                        Size=size,
                    )
                )

    def run(self):
        print('[INFO] NEFIS reader initialized with filename:', self.filename)
        print('[INFO] Data file:', self.data_file)
        print('[INFO] Definition file:', self.def_file)
        print('[INFO] Address Type:', self.vs_struct.AddressType)
        print('[INFO] Byte Order Format:', self.vs_struct.Format)
        print('[INFO] Groups Read:', len(self.vs_struct.GrpDat))
        print('[INFO] Group Definitions Read:', len(self.vs_struct.GrpDef))
        print('[INFO] Element Definitions Read:', len(self.vs_struct.ElmDef))
        return self.vs_struct


if __name__ == '__main__':
    # s = VSUse(r'/Users/mgarciaalvarez/devel/sedtrails/sample-data/trim-f34.dat')
    s = VSUse(r'/Users/mgarciaalvarez/devel/sedtrails/sample-data/trim-f34.dat')

    r = s.run()
    print(r)
