# (c) Dr Don SMA Liyanage
# DO NOT REMOVE THE COPYRIGHT STATEMENT

import numpy as np
from decimal import Decimal
from pydantic import BaseModel, Field
from typing import List
import uuid
# from kb.Node import Node

class Causality(BaseModel):
    instruction: int = Field(default=-1)
    argument: str = Field(default="")


 # Attribute: NN/synapses logger  Attribute: CBR case/sequential collection
 #  0 – default, will be managed by SC  T - case type
 #  0 – default, will be managed by SC  N - case type sequence
 #  1 – default, will be managed by SC  w - case weight
 #  0 – default, will be managed by SC  Ʒ - case adaptation
 #  Ф - computed  Γ - case causality
 #  S - mapped feature index [for each feature]  S - case item sequence
 #  W - mapped attribute value [for each feature]  W - case item weight
 #  t - mapped feature type index [for each feature]  t - case item weight type
 #  𝑉𝑚 𝑖𝑛 𝐼  - computed [for each feature]  V min  - case item weight variation minimum
 #  𝑉𝑚 𝑎𝑥 𝐼  - computed [for each feature]  V max  - case item weight variation maximum
 #  I - mapped attribute name / index[ for each feature]  I - case item instruction
 #  ϴ - computed[ for each feature]  c - case item condition
 #  Д - computed[ for each feature]  P - case item causation probability


# Type SequentialCollectionCase
class SequentialCollectionCase(BaseModel): #Node):
    i: str = Field(default="-1")
    refV: str = Field(default="-1")
    T: int = Field(default=0) # 1 - from problem space
    N: int = Field(default=0) # 0 - default
    w: Decimal = Field(default=0.0)
    Ʒ: int = Field(default=0)
    Γ: List[str] = []
    Γi: List[int] = []
    Γa: List[str] = []
    Γc: List[str] = []
    S: List[int] = []
    W: List[str] = []
    Wv: List[float] = []
    t: List[int] = []
    Vmin: List[float] = []
    Vmax: List[float] = []
    I: List[str] = []
    c: List[bool] = []
    P: List[float] = []

    ContextualAdaptation: str = Field(default="")

    def __init__(self, i="-1", refV="-1", T=0, N=0, w=1.0, Ʒ=0, Γ=[], Γi=[], Γa=[], Γc=[], S=[], W=[], Wv=[], t=[], Vmin=[], Vmax=[], I=[], c=[], P=[], ContextualAdaptation=""):
        super().__init__(i=i, refV=refV, T=T, N=N, w=w, Ʒ=Ʒ, Γ=Γ, Γi=Γi, Γa=Γa, S=S, W=W, Wv=Wv, t=t, Vmin=Vmin, Vmax=Vmax, I=I, c=c, P=P, contextualReasoning=ContextualAdaptation)
        if i == "-1":
            self.i = str(uuid.uuid4())
        else:
            self.i = i
        self.refV = refV
        self.T = T
        self.N = N
        self.w = w
        self.Ʒ = Ʒ
        self.Γ = Γ
        self.Γi = Γi
        self.Γa = Γa
        self.Γc = Γc
        self.S = S
        self.W = W
        self.Wv = Wv
        self.t = t
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.I = I
        self.c = c
        self.P = P
        self.ContextualAdaptation = ContextualAdaptation

    def updateCase(self, refV="", T=0, N=0, w=1.0, Ʒ=0, Γ=[], Γi=[], Γa=[], Γc=[], S=[], W=[], Wv=[], t=[], Vmin=[], Vmax=[], I=[], c=[], P=[], ContextualAdaptation=""):
        self.refV = refV
        self.T = T
        self.N = N
        self.w = w
        self.Ʒ = Ʒ
        self.Γ = Γ
        self.Γi = Γi
        self.Γa = Γa
        self.Γc = Γc
        self.S = S
        self.W = W
        self.Wv = Wv
        self.t = t
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.I = I
        self.c = c
        self.P = P
        self.ContextualAdaptation = ContextualAdaptation

        return True

    def likenessWithingAdaptation(self):
        pass

    def getAdaptation(self):
        pass

    def setContextualAdaptation(self, ContextualAdaptation):
        self.ContextualAdaptation = ContextualAdaptation

    def getContextualAdaptation(self):
        return self.ContextualAdaptation
