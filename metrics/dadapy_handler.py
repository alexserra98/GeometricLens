from dadapy.data import Data

class DataAdapter(Data):
    def __init__(self, *args, variation:str = None, maxk: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variation = variation
        self.maxk = maxk
        if variation == "cosine":
            self.additional_behavior()
    
    def additional_behavior(self):
        self.compute_distances(maxk=self.maxk,metric="cosine")
