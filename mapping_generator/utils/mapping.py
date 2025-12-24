# mapping_generator/utils/mapping.py

class Mapping:
    """A data structure to hold the results of a DFG mapping.
    
    This class stores the placement of logical nodes onto physical resources
    and the routing paths between them.
    """
    def __init__(self, dfg_size: int):
        """Initializes a new Mapping object.

        Args:
            dfg_size (int): The number of nodes in the DFG being mapped.
        """
        self.dfg_size = dfg_size
        self.placement = {} # Dict mapping logical node ID (str) to physical coordinate (tuple)
        self.routing = {}   # Dict mapping (src_id, dst_id) tuple to a list of coordinates (path)
