from boltzgen_flat.data_data import Record
from boltzgen_flat.data_filter_dynamic_filter import DynamicFilter

IGNORE = "6iqh,8vtl,8vtn,8vto,7qfv,7qft,8rw2,8r6g,8y0s,8V6P,8RQP,9B4H,8RQQ,8RH3,8RDP,8RDR,8RDS,8RDT,8RQR,8VSH,8W3A,8Z5L,8XFY,8S5B,8VB1,8IEQ,8V9I,8VR5,8S3X,8VR3,8IEB,8IEI,8IEP,8VEN,8XN7,8VZX,8VZZ,8W00,9IIW,8VHE,8RDX,8VZY,8XFM,9IJA,8RX7,8S1K,8RX3,8RX9,8VHM,8OLA,8OMN,8W02,7H3E,8VL7,7H6R,8VHL,8YHS,8VLQ,9AYH,9AYJ,9AYK,9AYL,8RXR,7H2W,7H3X,7H41,7H4X,8XLQ,8RIV,8RIW,8RIJ,7H2U,7H2V,7H2X,7H31,7H3D,7H3F,7H3Q,7H45,7H4D,7H4I,7H4N,7H4V,7H4W,7H50,6Q4Q,8UIF,7H3T,9IIX,8VSG,8ROX,8ROY,8VLX,8XJK,1FXZ,7GT5,8IED,8VY9,6Q38,8XJN,8RZW,8RZY,8S01,8S04,8S06,8S0H,8S0I,8S0J,8S0K,8S0P,8S0Q,8S0S,8RI2,8YKY,8S07,8S0O,8VKZ,9AX6,7GSA,7GSB,7GSL,7GSM,7GSO,7GSR,7GST,7GSV,7GSW,7GSY,7GT0,7GT1,7GT3,7GT4,7GT6,7GT7,7GT9,7GTA,7GTD,7GTE,7GTG,7GTH,7GTL,7GTM,7GTN,7GTO,7GTT,8HMB,8VET,8VEU,8VEW,8VEX,8VEY,8XLO,7GTK,7GTI,8HMA,9AXH,9AXM,9IJ9,8RV4,8RV6,8RV7,8RV8,8RV9,8RVA,8RVB,8RZC,8RZD,8RZE,8VL9,8VY7,8YHF,8YHL,8SGM,8VQ3,8VQ4,9AXC,9AYA,9AXX,8XZI,9AY7,9AXA,8RRQ,8RRZ,9AVL,9AXY,9ASB,9AVG,8XOF"
IGNORE = IGNORE.strip().split(",")
IGNORE = {i.strip().lower() for i in IGNORE}


class ErrorFilter(DynamicFilter):
    """A filter that filters complexes with error ligands."""

    def filter(self, record: Record) -> bool:
        """Filter complexes with error ligands.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be keps.

        """
        return record.id.lower() not in IGNORE
