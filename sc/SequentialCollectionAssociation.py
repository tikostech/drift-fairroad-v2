# (c) Dr Don SMA Liyanage
# DO NOT REMOVE THE COPYRIGHT STATEMENT

import numpy as np
from decimal import Decimal
from pydantic import BaseModel, Field
from typing import List
import uuid
# from kb.Edge import Edge


# Type SequentialCollectionAssociation
class SequentialCollectionAssociation(BaseModel): #Edge):
    i: str = Field(default="-1")
    refE: str = Field(default="-1")
    source: str = Field(default="-1")
    target: str = Field(default="-1")
    associationLabel: List[str] = []
    associationWeight: List[Decimal] = []

    def __init__(self, i="-1", refE="-1", source="-1", target="-1", associationLabel=[], associationWeight=[]):
        super().__init__(i=i, refE=refE, source=source, target=target, associationLabel=associationLabel,
                         associationWeight=associationWeight)
        if i == "-1":
            self.i = str(uuid.uuid4())
        else:
            self.i = i
        self.refE = refE
        self.source = source
        self.target = target
        self.associationLabel = associationLabel
        self.associationWeight = associationWeight


    def updateAssociation(self, refE="-1", source="-1", target="-1", associationLabel=[], associationWeight=[]):
        self.refE = refE
        self.source = source
        self.target = target
        self.associationLabel = associationLabel
        self.associationWeight = associationWeight
        return True

    def likenessWithingAdaptation(self):
        pass

    def getAdaptation(self):
        pass

    def generateReason(self):
        pass
