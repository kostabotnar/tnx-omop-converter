"""Read the compression codecs of a Parquet file from its footer.

pyarrow is not a dependency, so this decodes the Thrift compact protocol footer
(FileMetaData.row_groups[].columns[].meta_data.codec) with the standard library.
"""

from pathlib import Path

# parquet.thrift CompressionCodec; Polars writes "lz4" as LZ4_RAW (7)
CODEC_NAMES = {
    0: "uncompressed",
    1: "snappy",
    2: "gzip",
    3: "lzo",
    4: "brotli",
    5: "lz4",
    6: "zstd",
    7: "lz4",
}


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def byte(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return value

    def varint(self) -> int:
        result = shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, kind: int):
        if kind in (1, 2):
            return kind == 1
        if kind == 3:
            return self.byte()
        if kind in (4, 5, 6):
            return self.zigzag()
        if kind == 7:
            self.pos += 8
            return None
        if kind == 8:
            size = self.varint()
            self.pos += size
            return None
        if kind in (9, 10):
            header = self.byte()
            size = header >> 4
            if size == 15:
                size = self.varint()
            elem = header & 0x0F
            return [self.list_element(elem) for _ in range(size)]
        if kind == 12:
            return self.struct()
        raise ValueError(f"unsupported Thrift type {kind}")

    def list_element(self, kind: int):
        if kind in (1, 2):
            return self.byte() == 1
        return self.value(kind)

    def struct(self) -> dict[int, object]:
        fields: dict[int, object] = {}
        field_id = 0
        while True:
            header = self.byte()
            if header == 0:
                return fields
            delta = header >> 4
            field_id = field_id + delta if delta else self.zigzag()
            fields[field_id] = self.value(header & 0x0F)


def column_codecs(path: Path) -> set[str]:
    """Names of the codecs used by all column chunks of the file."""
    data = Path(path).read_bytes()
    if data[-4:] != b"PAR1":
        raise ValueError(f"{path} is not a Parquet file")
    footer_size = int.from_bytes(data[-8:-4], "little")
    footer = _Reader(data[-8 - footer_size : -8]).struct()
    codecs: set[str] = set()
    for row_group in footer[4]:
        for chunk in row_group[1]:
            codecs.add(CODEC_NAMES[chunk[3][4]])
    return codecs
