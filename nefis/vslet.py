import os
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

"""
return data a numpy array
"""


@dataclass
class GroupData:
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


class VSLet:
    def __init__(self, vs: VSStructure):
        self.vs = vs

    def _find_group_element_indices(self, group_name, element_name):
        group_idx = next((i for i, g in enumerate(self.vs.GrpDat) if g.Name == group_name), None)
        if group_idx is None:
            raise ValueError(f"Group '{group_name}' not found.")
        def_idx = self.vs.GrpDat[group_idx].DefIndex
        cell_idx = self.vs.GrpDef[def_idx].CelIndex if def_idx is not None else None
        elms = self.vs.CelDef[cell_idx].Elm if cell_idx is not None else []
        element_idx = next((e for e in elms if self.vs.ElmDef[e].Name == element_name), None)
        if element_idx is None:
            raise ValueError(f"Element '{element_name}' not found in group '{group_name}'.")
        return group_idx, element_idx

    def _read_element(self, data_file, group_idx, element_idx, g_index=None, e_index=None):
        group = self.vs.GrpDat[group_idx]
        element = self.vs.ElmDef[element_idx]
        offset = group.Offset + 392 + 3 * (8 if self.vs.AddressType == 'uint64' else 4)
        dtype_map = {'CHARACTE': 'c', 'INTEGER': 'i', 'REAL': 'f', 'LOGICAL': '?', 'COMPLEX': 'ff'}
        dtype = dtype_map.get(element.Type.strip(), None)
        if dtype is None:
            raise ValueError(f'Unsupported data type: {element.Type}')

        count = element.SizeVal * max(1, element.SizeElm // element.SizeVal)

        if g_index or e_index:
            # Basic slicing simulation: flatten sizes and slice accordingly (no reordering or stride checks)
            total_size = element.SizeVal * max(1, element.SizeElm // element.SizeVal)
            slice_start = 0
            slice_end = total_size

            if e_index and isinstance(e_index, list) and all(isinstance(e, int) for e in e_index):
                # Only handle simple 1D slicing case for now
                slice_start = min(e_index)
                slice_end = max(e_index) + 1

            count = slice_end - slice_start
            offset += slice_start * element.SizeVal

        with open(data_file, 'rb') as f:
            f.seek(offset)
            raw = f.read(count * element.SizeVal)
        return raw

    def read(
        self,
        group_name: str,
        element_name: str,
        g_index: Optional[List[int]] = None,
        e_index: Optional[List[int]] = None,
    ):
        """to read file in chunks"""

        group_idx, element_idx = self._find_group_element_indices(group_name, element_name)
        data_path = self.vs.FileName + self.vs.DatExt
        return self._read_element(data_path, group_idx, element_idx, g_index, e_index)

    def read_all_elements(self, group_name: str):
        group_idx = next((i for i, g in enumerate(self.vs.GrpDat) if g.Name == group_name), None)
        if group_idx is None:
            raise ValueError(f"Group '{group_name}' not found.")
        def_idx = self.vs.GrpDat[group_idx].DefIndex
        cell_idx = self.vs.GrpDef[def_idx].CelIndex if def_idx is not None else None
        elms = self.vs.CelDef[cell_idx].Elm if cell_idx is not None else []
        data_path = self.vs.FileName + self.vs.DatExt
        out = {}
        for eidx in elms:
            name = self.vs.ElmDef[eidx].Name
            out[name] = self._read_element(data_path, group_idx, eidx)
        return out
