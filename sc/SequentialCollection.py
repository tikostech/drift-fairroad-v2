# (c) Dr Don SMA Liyanage
# DO NOT REMOVE THE COPYRIGHT STATEMENT

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Any
import json
from sc.SequentialCollectionCase import SequentialCollectionCase
from sc.SequentialCollectionAssociation import SequentialCollectionAssociation
from .ModelSpec import ModelSpec
# from llm.llm_loader import invoke_custom_llm_chain, invoke_custom_llm_chain_multi, invoke_backup_llm_chain
# from llm.llm_direct_api import exec_llmserver_array
# from graphviz import Digraph
import uuid
import numpy

# Type SequentialCollection -> Think as Knowledge Base or Super Knowledge Graph
class SequentialCollection(BaseModel):
    def __init__(self):
        super().__init__()
        # Nodes
        self.__caseBase: List[SequentialCollectionCase] = []
        # Edges
        self.__caseAssociation: List[SequentialCollectionAssociation] = []
    def cbLength(self):
        # i = 0
        # for o in self.__caseBase:
        #     if o.i > 0:
        #         i = i + 1
        # return i
        return len(self.__caseBase)


    def assoLength(self):
        # i = 0
        # for o in self.__caseAssociation:
        #     if o.i > 0:
        #         i = i + 1
        # return i
        return len(self.__caseAssociation)

    def createNewCase(self, case: SequentialCollectionCase = None):
        # caseIdex = len(self.__caseBase)
        # objCase = SequentialCollectionCase(caseIdex)
        # self.__caseBase.append(objCase)
        # if case is not None:
        #     self.setCase(caseIdex, case)
        # return caseIdex
        caseIdex = len(self.__caseBase)
        uuitStr = str(uuid.uuid4())
        # objCase = SequentialCollectionCase(caseIdex)
        caseId = str(caseIdex) + "_" + uuitStr
        objCase = SequentialCollectionCase(i=caseId)
        self.__caseBase.append(objCase)
        if case is not None:
            self.setCase(caseIdex, case)
        return caseIdex

    def addCase(self, case: SequentialCollectionCase = None):
        self.__caseBase.append(case)
        return self.__caseBase.index(case)

    def createNewAssociation(self, association: SequentialCollectionAssociation = None):
        assoIdex = len(self.__caseAssociation)
        objAssociation = SequentialCollectionAssociation(assoIdex)
        self.__caseAssociation.append(objAssociation)
        if association is not None:
            self.setAssociation(assoIdex, association)
        return assoIdex

    def updateMerge(self, newSC: "SequentialCollection") -> "SequentialCollection":
        dedupCase = list(set(self.__caseBase + newSC.__caseBase))
        dedupAssociations = list(set(self.__caseAssociation + newSC.__caseAssociation))
        return SequentialCollection(__caseBase=dedupCase, __caseAssociation=dedupAssociations)

    def drawGraph(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")

        for node in self.__caseBase:
            dot.node(str(node.i) + "_" + str(node.refV), node.refV, color='black', i=node.i, idx=node.idx, T=node.T, N=node.N, w=node.w, Ʒ=node.Ʒ, Γ=node.Γ, S=node.S, W=node.W, t=node.t, Vmin=node.Vmin, Vmax=node.Vmax, I=node.I, c=node.c, P=node.P)

        for edge in self.__caseAssociation:
            dot.edge(
                str(edge.source), str(edge.target), i=edge.i, label=edge.label, color=edge.color, associationLabel=edge.associationLabel, associationWeight=edge.associationWeight
            )
        dot.render(prefix, format="png", view=True)

    def createCaseQuery(self) -> List[str] | Any:
        caseScript = []

        for case in self.__caseBase:
            tStrt = f"addV('{case.refV}').property('id', '{case.i}').property('label', 'CASE')"

            pStr = ""
            # pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
            #          f".property('{str('w')}','{case.w}')" + f".property('{str('Ʒ')}','{case.Ʒ}')" +
            #          f".property('{str('Γi')}',{case.Γi})" f".property('{str('Γa')}',{case.Γa})" + f".property('{str('S')}',{case.S})" +
            #          f".property('{str('W')}',{case.W})" + f".property('{str('t')}',{case.t})" +
            #          f".property('{str('Vmin')}',{case.Vmin})" + f".property('{str('Vmax')}',{case.Vmax})" +
            #          f".property('{str('I')}',{case.I})" + f".property('{str('c')}',{case.c})" +
            #          f".property('{str('P')}',{case.P})"
            #          )

            pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
                    f".property('{str('P')}',{case.P})"
                    )

            rtnStr = "g.V().has('id','{idval}').fold().coalesce(unfold(){pStrS},{tStr})".format(
                idval=case.i, pStrS=pStr, tStr=tStrt + pStr)

            # # 3.6+
            # tStr = f"id: '{self.removeCTRLChar(id)}'"
            # pStr = ""
            # for k, v in properties.items():
            #     pStr = pStr + f",{k}:'{self.removeCTRLChar(v)}'"
            #
            # rtnStr = "g.mergeV([{tStr}{pStr}])".format(tStr=tStr, pStr=pStr)

            caseScript.append(rtnStr)

        return caseScript

    def createCaseQuery_tarray(self) -> List[str] | Any:
        caseScript = []
        associationScript = []

        for case in self.__caseBase:
            tStrt = f"addV('{case.refV}').property('id', '{case.refV + '_' + case.i}').property('label', 'CASE').property('refV', '{case.refV}')"

            pStr = ""
            # pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
            #          f".property('{str('w')}','{case.w}')" + f".property('{str('Ʒ')}','{case.Ʒ}')" +
            #          f".property('{str('Γ')}','{str(np.asarray(case.Γ))}')" + f".property('{str('S')}','{str(np.asarray(case.S))}')" +
            #          f".property('{str('W')}','{str(np.asarray(case.W))}')" + f".property('{str('t')}','{str(np.asarray(case.t))}')" +
            #          f".property('{str('Vmin')}','{str(np.asarray(case.Vmin))}')" + f".property('{str('Vmax')}','{str(np.asarray(case.Vmax))}')" +
            #          f".property('{str('I')}','{str(np.asarray(case.I))}')" + f".property('{str('c')}','{str(np.asarray(case.c))}')" +
            #          f".property('{str('P')}','{str(np.asarray(case.P))}')"
            #          )

            # pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
            #         f".property('{str('w')}','{case.w}')" + f".property('{str('Ʒ')}','{case.Ʒ}')" +
            #         f".property('{str('Γ')}','{str(np.asarray(case.Γ)).replace("'", "")}')" + f".property('{str('S')}','{str(np.asarray(case.S)).replace("'","")}')" +
            #         f".property('{str('W')}','{str(np.asarray(case.W)).replace("'","")}')" + f".property('{str('t')}','{str(np.asarray(case.t)).replace("'","")}')" +
            #         f".property('{str('Vmin')}','{str(np.asarray(case.Vmin)).replace("'","")}')" + f".property('{str('Vmax')}','{str(np.asarray(case.Vmax)).replace("'","")}')" +
            #         f".property('{str('I')}','{str(np.asarray(case.I)).replace("'","")}')" + f".property('{str('c')}','{str(np.asarray(case.c)).replace("'","")}')" +
            #         f".property('{str('P')}','{str(np.asarray(case.P)).replace("'","")}')"
            #         )

            pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
                    f".property('{str('w')}','{case.w}')" + f".property('{str('Ʒ')}','{case.Ʒ}')" +
                    f".property('{str('Γ')}','{str(case.Γ).replace("'", '"')}')" + f".property('{str('Γi')}','{str(case.Γi).replace("'", "")}')" +
                    f".property('{str('Γa')}','{str(case.Γa).replace("'", '"')}')" + f".property('{str('Γc')}','{str(case.Γc).replace("'", '"')}')" +
                    f".property('{str('S')}','{str(case.S).replace("'", "")}')" +
                    f".property('{str('W')}','{str(case.W).replace("'", '"')}')" + f".property('{str('Wv')}','{str(case.Wv).replace("'", '"')}')" +
                    f".property('{str('t')}','{str(case.t).replace("'", '"')}')" +
                    f".property('{str('Vmin')}','{str(case.Vmin).replace("'", "")}')" + f".property('{str('Vmax')}','{str(case.Vmax).replace("'", "")}')" +
                    f".property('{str('I')}','{str(case.I).replace("'", '"')}')" + f".property('{str('c')}','{str(case.c).replace("'", "")}')" +
                    f".property('{str('P')}','{str(case.P).replace("'", "")}')"
                    )

            # pStr = (pStr + f".property('{str('T')}','{case.T}')" + f".property('{str('N')}','{case.N}')" +
            #         f".property('{str('w')}','{case.w}')" + f".property('{str('Ʒ')}','{case.Ʒ}')" +
            #         # f".property('{str('I')}','{str(case.I).replace("'","")}')" + f".property('{str('c')}','{str(case.c).replace("'","")}')"
            #         f".property('{str('I')}','{str(np.asarray(case.I)).replace("'","")}')" + f".property('{str('c')}','{str(np.asarray(case.c)).replace("'","")}')"
            #         )

            #TODO
            # Search by ID is not the best option should be by W + refV
            # Applied for both V & E. the W can be unique
            rtnStrCase = "g.V().has('id','{idval}').fold().coalesce(unfold(){pStrS},{tStr}.property('pk','pk'))".format(
                idval=case.refV + '_' + case.i, pStrS=pStr, tStr=tStrt + pStr)

            rtnStrAssociation = "g.V().has('id','{source}').as('v1').V().has('id','{target}').coalesce(__.inE('{relType}').where(outV().as('v1')),addE('{relType}').from('v1'))".format(
                source=case.refV + '_' + case.i, target=case.refV,
                relType="CASE_IMPLEMENTED")

            # # 3.6+
            # tStr = f"id: '{self.removeCTRLChar(id)}'"
            # pStr = ""
            # for k, v in properties.items():
            #     pStr = pStr + f",{k}:'{self.removeCTRLChar(v)}'"
            #
            # rtnStr = "g.mergeV([{tStr}{pStr}])".format(tStr=tStr, pStr=pStr)

            caseScript.append(rtnStrCase)
            associationScript.append(rtnStrAssociation)

        return caseScript, associationScript

    def removeCTRLChar(self, val) -> str:
        return str(val).replace("'", "") #.replace('"', "").replace("“", "").replace("”", "").replace("’", "")

    def setCase(self, index, case):
        self.__caseBase[index] = case
        return True

    def setAssociation(self, index, association):
        self.__caseAssociation[index] = association
        return True

    def getCase(self, index):
        return self.__caseBase[index]

    def getCaseIDC(self, i):
        for x in self.__caseBase:
            if x.i == i:
                return x
                break

        return None

    def getAssociation(self, index):
        return self.__caseAssociation[index]

    def getFullCaseBase(self):
        return self.__caseBase

    def getFullAssociationBase(self):
        return self.__caseAssociation

    def likeness(self, cSS: SequentialCollectionCase, cPS: SequentialCollectionCase, WType="BIN", RType=""):
        objRtnCase = SequentialCollectionCase(Wv=[])

        tWv = 0.0

        if RType == "DEEPCAUSAL_BINNUM":
            for i in range(len(cSS.Wv)):
                if cSS.c[i]:

                    ssW = cSS.W[i]
                    psW = cPS.W[i]

                    if (type(ssW) == int) or (type(ssW) == float) or (
                            type(ssW) == complex) or (type(ssW) == numpy.float64) or (
                            type(ssW) == numpy.float32) or (type(ssW) == numpy.float16) or (
                            type(ssW) == numpy.int32) or (
                            type(ssW) == numpy.int64) or (type(ssW) == numpy.int8) or (
                            type(ssW) == numpy.int16) or (type(ssW) == numpy.uint32) or (
                            type(ssW) == numpy.uint64) or (type(ssW) == numpy.uint8) or (
                            type(ssW) == numpy.uint16):
                        ssW = float(ssW)
                    else:
                        if ssW == "nan" or ssW == "None" or ssW == "" or ssW is None:
                            ssW = 0
                        else:
                            binVAL = 0
                            for i in bytearray(str.upper(ssW), encoding='utf-8'):
                                binVAL = binVAL + int(format(i, '08b'))
                            ssW = binVAL

                    if (type(psW) == int) or (type(psW) == float) or (
                            type(psW) == complex) or (type(psW) == numpy.float64) or (
                            type(psW) == numpy.float32) or (type(psW) == numpy.float16) or (
                            type(psW) == numpy.int32) or (
                            type(psW) == numpy.int64) or (type(psW) == numpy.int8) or (
                            type(psW) == numpy.int16) or (type(psW) == numpy.uint32) or (
                            type(psW) == numpy.uint64) or (type(psW) == numpy.uint8) or (
                            type(psW) == numpy.uint16):
                        psW = float(psW)
                    else:
                        if psW == "nan" or psW == "None" or psW == "" or psW is None:
                            psW = 0
                        else:
                            binVAL = 0
                            for i in bytearray(str.upper(psW), encoding='utf-8'):
                                binVAL = binVAL + int(format(i, '08b'))
                            psW = binVAL

                    tWv = tWv + abs(ssW - psW)
                    objRtnCase.Wv.append(abs(ssW - psW))
                else:
                    objRtnCase.Wv.append(0)
        else:
            for i in range(len(cSS.Wv)):
                if WType == "BIN" or WType == "BINADV":
                    if cSS.c[i]:
                        tWv = tWv + abs(cSS.Wv[i] - cPS.Wv[i])
                        # print(f"ss {cSS.Wv[i]}, ps {cPS.Wv[i]}, diff {(cSS.Wv[i] - cPS.Wv[i])}, diffabs {abs(cSS.Wv[i] - cPS.Wv[i])}, tWv = {tWv}")
                        objRtnCase.Wv.append(abs(cSS.Wv[i] - cPS.Wv[i]))
                    else:
                        objRtnCase.Wv.append(0)
                elif WType == "BINNUM" or WType == "BINNUMADV":
                    if cSS.t[i] in ['int', 'float', 'complex', 'Decimal', 'numpy.float64', 'numpy.float32', 'numpy.float16', 'numpy.float96', 'numpy.float128', 'numpy.int32', 'numpy.int64', 'numpy.int8', 'numpy.int16', 'numpy.uint32', 'numpy.uint64', 'numpy.uint8', 'numpy.uint16']:
                        if cSS.c[i]:
                            objRtnCase.Wv.append(abs(cSS.Wv[i] - cPS.Wv[i]))
                        else:
                            objRtnCase.Wv.append(0)
                    else:
                        objRtnCase.Wv.append(0)
                else:
                    pass

        return objRtnCase

    def likenessWithin(self, cPS, WType="BIN"):
        # A 4.3 118
        cSelected = None
        for cSS in self.__caseBase:
            if WType == "BIN" or WType == "BINADV":
                if cSS.w == cPS.w:
                    currentVal = sum(cPS.Wv)
                    itrMaxSum = (sum([(x) + (y) for x, y in zip(cSS.Wv, cSS.Vmax)]))
                    itrMinSum = (sum([(x) - (y) for x, y in zip(cSS.Wv, cSS.Vmin)]))

                    cSelected = cSS if (itrMinSum <= currentVal <= itrMaxSum) else cSelected
            elif WType == "BINNUM" or WType == "BINNUMADV":
                if cSS.w == cPS.w:
                    currentVal = 0
                    itrMaxSum = 0
                    itrMinSum = 0
                    for i in range(len(cSS.Wv)):
                        if cSS.t[i] in ['int', 'float', 'complex', 'Decimal']:
                            currentVal = currentVal + cPS.Wv[i]
                            itrMaxSum = itrMaxSum + (cSS.Wv[i] + cSS.Vmax[i])
                            itrMinSum = itrMinSum + (cSS.Wv[i] - cSS.Vmin[i])

                    cSelected = cSS if (itrMinSum <= currentVal <= itrMaxSum) else cSelected
            else:
                pass

        return currentVal, cSelected

    def getSimilarityMatchingWithin(self, cPS, WType="BIN", RType=""):
        cSelectedList = []
        currentVal = sum(cPS.Wv)
        print(f"getSimilarityMatchingWithin currentVal {currentVal}")

        for cSS in self.__caseBase:
            # print(f"cSS {cSS}")
            if WType == "BIN" or WType == "BINADV":
                # print(f"cSS {len(cSS.W)} {len(cPS.W)}")
                # if cSS.w == cPS.w:
                if len(cSS.W) > 0:
                    itrMaxSum = (sum([(x) + (y) for x, y in zip(cSS.Wv, cSS.Vmax)]))
                    itrMinSum = (sum([(x) - (y) for x, y in zip(cSS.Wv, cSS.Vmin)]))

                    # print(f"getSimilarityMatchingWithin itrMinSum {itrMinSum}")
                    # print(f"getSimilarityMatchingWithin currentVal {currentVal}")
                    # print(f"getSimilarityMatchingWithin itrMaxSum {itrMaxSum}")

                    if (itrMinSum <= currentVal <= itrMaxSum):
                        cSelectedList.append(cSS)
                        # print(f"getSimilarityMatchingWithin cSS {cSS}")
                        #
                        # print(f"getSimilarityMatchingWithin itrMinSum {itrMinSum}")
                        # print(f"getSimilarityMatchingWithin currentVal {currentVal}")
                        # print(f"getSimilarityMatchingWithin itrMaxSum {itrMaxSum}")

            elif WType == "BINNUM" or WType == "BINNUMADV":
                if len(cSS.W) > 0: #cSS.w == cPS.w:
                    currentVal = 0
                    itrMaxSum = 0
                    itrMinSum = 0
                    for i in range(len(cSS.Wv)):
                        if cSS.t[i] in ['int', 'float', 'complex', 'Decimal', 'bool']:
                            currentVal = currentVal + cPS.Wv[i]
                            itrMaxSum = itrMaxSum + (cSS.Wv[i] + cSS.Vmax[i])
                            itrMinSum = itrMinSum + (cSS.Wv[i] - cSS.Vmin[i])

                    if (itrMinSum <= currentVal <= itrMaxSum):
                        cSelectedList.append(cSS)
            else:
                pass

        if len(cSelectedList) > 0:
            lstSimilarity = []
            for c in cSelectedList:
                objrtn = self.likeness(c, cPS, WType, RType)
                lstSimilarity.append((c.i, sum(objrtn.Wv)))

            return self.getCaseIDC(lstSimilarity.sort(key=lambda tup: tup[1], reverse=False)[0][0])
        else:
            # return None
            return self.getSimilarityMatching(cPS, WType, RType)

    def getSimilarityMatching(self, cPS, WType="BIN", RType=""):
        if cPS is None:
            return SequentialCollectionCase(0)


        lstSimilarity = []
        for c in self.__caseBase:
            objrtn = self.likeness(c, cPS, WType, RType)
            print(f"c.i {c.i}, sum cSS {sum(objrtn.Wv)}, sum cPS {sum(cPS.Wv)}, cSS {objrtn.Wv}, cPS {cPS.Wv}")
            # print(f"Wv {objrtn.Wv}")
            lstSimilarity.append((c.i, sum(objrtn.Wv)))

        lstSimilarity.sort(key=lambda tup: tup[1], reverse=False)
        idx = lstSimilarity[0][0]

        tmpObj = self.getCaseIDC(idx)
        print(f"tmpObj {tmpObj}")

        return self.getCaseIDC(idx)

    def getAggregativeAdaptation(self, cPS, cSS, deltaw):
        # C 7.3.2
        # IMPLY A 7.2/193
        # deltaw C`.LIKENESS-WITHIN-ADAPTATION(C) 117

        ddw = deltaw / len(cPS.Wv)
        print(f"Sc ADD1 0, sW {deltaw}")
        print(f"Sc ADD1 0, sW {len(cPS.Wv)}")

        for i in range(len(cPS.Wv)):

            # print(f"Sc 004.2.1 {i}")
            # print(f"Sc 004.2.2 {i}, {cPS.Wv[i]}, {cSS.Wv[i]}, {deltaw}, {ddw}")
            # print(f"Sc 004.2.1 {i}, {cSS.t[i]}, {cPS.Wv[i]}, {cSS.Wv[i]}")

            # if cPS.t[i] in ['int', 'float', 'complex', 'Decimal', 'bool']:
            #     pW = cPS.Wv[i]

            # if cSS.t[i] in ['int', 'float', 'complex', 'Decimal', 'bool']:
            #     pW = cPS.Wv[i]
            #     tW = cSS.Wv[i]
            if cSS.c[i]:

                print(f"Sc ADD1 {i}, sW {cSS.W[i]}")
                print(f"Sc ADD1 {i}, pW {cPS.W[i]}")

                tW = cSS.Wv[i]
                pW = cPS.Wv[i]

                print(f"Sc ADD1 {i}, tW {tW}")
                print(f"Sc ADD1 {i}, pW {pW}")

                # print(f"Sc 004.2.2.1 {pW}")
                deltaW = (tW + pW) / 2
                sigmaW = ((((tW - deltaW) ** 2) + ((pW - deltaW) ** 2)) / 2) ** 0.5

                print(f"Sc ADD1 {i}, deltaW {deltaW}")
                print(f"Sc ADD1 {i}, sigmaW {sigmaW}")

                if pW != 0:
                    deltaWpB = (sigmaW / pW)
                    deltaWp = deltaWpB * 0.01
                else:
                    deltaWpB = sigmaW
                    deltaWp = deltaWpB * 0.01

                print(f"Sc ADD1 {i}, deltaWp {deltaWp}")

                # print(f"Sc 004.2.3 {deltaWp}")
                # print(f"Sc 004.2.4 {cSS.Γi}")
                # print(f"Sc 004.2.5 {cPS.c}")


                # if cSS.Γi.count(i) > 0:
                #     # ttc = 1
                #     cPS.c[i] = True
                # else:
                #     # ttc = 0
                #     cPS.c[i] = False

                if cSS.Vmin[i] > 0.0:
                    cPS.Vmin[i] = cSS.Vmin[i] * deltaWp
                else:
                    cPS.Vmin[i] = deltaWp

                if cSS.Vmax[i] > 0.0:
                    cPS.Vmax[i] = cSS.Vmax[i] * deltaWp
                else:
                    cPS.Vmax[i] = deltaWp

                print(f"Sc ADD1 {i}, cPS.Vmin[i]  {cPS.Vmin[i]}")
                print(f"Sc ADD1 {i}, cPS.Vmax[i]  {cPS.Vmax[i]}")

                #TODO should be % - re check
                print(f"Sc ADD1 {i}, cSS.P[i] {cSS.P[i]}")
                print(f"Sc ADD1 {i}, cPS.P[i] {cPS.P[i]}")
                if ddw > 0.0:
                    cPS.P[i] = ((float(cSS.P[i] + deltaWpB) * (cSS.Γi.count(i) / ddw)) * 0.01)
                else:
                    cPS.P[i] = ((float(cSS.P[i] + deltaWpB) * (cSS.Γi.count(i) / 1)) * 0.01)


                # print(f"Sc 004.2.6 {i}, {cPS.P[i]}")
                print(f"Sc ADD1 {i}, cPS.P[i] {cPS.P[i]}")
                print(f"Sc ADD1 {i}, ddw {ddw}")
                print(f"Sc ADD1 {i}, cSS.Γi.count(i) {cSS.Γi.count(i)}")
            else:
                print(f"No Adaptation {i}")

        return cPS

    def getComputeCaseCausality(self, cPS, cSS, modelspec: ModelSpec, context, refCaseType):
        # A 6.5, 176
        # A 5.5, 149
        ciCausality = []
        txt_FT = ""
        for i in range(len(cSS.Wv)):
            ciFeature = str(cSS.I[i]).replace(".", " ").replace("_", " ")
            ciFeaturestripped = str(cSS.I[i])
            if "." in str(cSS.I[i]):
                ciFeaturestripped = ciFeaturestripped[ciFeaturestripped.rfind("."):].strip().rstrip().lstrip().rstrip("\n").lstrip("\n")
            elif "_" in str(cSS.I[i]):
                ciFeaturestripped = ciFeaturestripped[ciFeaturestripped.rfind("_"):].strip().rstrip().lstrip().rstrip("\n").lstrip("\n")

            if cSS.c[i]:
                psValue = cPS.Wv[i]
                ssValue = cSS.Wv[i]

                psSValue = cPS.W[i]
                ssSValue = cSS.W[i]

                psAggReasoningValue = cPS.P[i]

                # Feature Reasoning
                spsAggReasoningValue = "0.0"
                if "." in str(psAggReasoningValue * 100):
                    if len(str(psAggReasoningValue * 100)) > 4:
                        spsAggReasoningValue = str(psAggReasoningValue * 100)[:4]
                    else:
                        spsAggReasoningValue = str(psAggReasoningValue * 100)

                if ssValue == psValue:
                    cipsStr = "equal"
                elif ssValue > psValue:
                    # cipsStr = spsAggReasoningValue + " percent " + "higher"
                    cipsStr = "higher"
                else:
                    # cipsStr = spsAggReasoningValue + " percent " + "lower"
                    cipsStr = "lower"

                print(f"Sc 004.3.1.1 {i}, {ciFeature}, {cipsStr}")

                # CONTROL_PROMPT = f'You are an investigative agent. Based on the context provided, what is the relationship of {ciFeature}, and what is the relevence of having {cipsStr} relationship of {ciFeature}, explain in less than 20 words?'
                # CONTROL_PROMPT = f'You are a research agent. Based on the context provided around {refCaseType}, what is the adaptive reason of {ciFeature} of a {refCaseType} having {cipsStr} relevance between vales of {psSValue} and  {ssSValue}, explain in less than 20 words?'
                if cipsStr == "equal":
                    CONTROL_PROMPT = f'Based on the context provided around {refCaseType}, as the values are {cipsStr}, what is the relationship between {ciFeature} and {refCaseType}, explain in less than 50 words'
                    t_PROMPT = f'{ciFeature} value is {ssSValue} and it is {cipsStr} compared to the solution value of {psSValue}, what is the relationship between {ciFeature} and {refCaseType}'
                else:
                    CONTROL_PROMPT = f'Based on the context provided around {refCaseType}, as the values are {cipsStr}, what is the relationship of {ciFeature} and {refCaseType}, provided the predicted as {ssSValue} compared to {psSValue}, explain in less than 50 words'
                    t_PROMPT = f'{ciFeature} value is {ssSValue} and it is {cipsStr} compared to the solution value of {psSValue}, what is the relationship between {ciFeature} and {refCaseType}'


                if txt_FT == "":
                    txt_FT = txt_FT + t_PROMPT
                else:
                    txt_FT = txt_FT + " and, " + t_PROMPT

                print(f"Sc 004.3.1.2 {i}, {CONTROL_PROMPT}")
                # TODO PASS in ModelOption
                # result = exec_llmserver_array(context=context, query=CONTROL_PROMPT, modelOption=1)
                print(f"Sc 004.3.1.2.1 X_TIKOS_MODEL {modelspec.X_TIKOS_MODEL}")
                # result = exec_llmserver_array(context=context, query=CONTROL_PROMPT, modelOption=modelspec.X_TIKOS_MODEL)
                result = ""
                print(f"Sc 004.3.1.3 {i}, {result}")

                ciContext = ""
                inContext = False
                for objResult in result:
                    if objResult != "":
                        inContext = True
                    if ciContext == "":
                        ciContext = objResult
                    else:
                        ciContext = ciContext + ", " + objResult


                print(f"Sc 004.3.1.4 {ciContext}, {inContext}")

                if inContext:
                    ciCausality.append((ciFeature, ciContext))
                else:
                    ciCausality.append((ciFeature, ""))
            else:
                ciCausality.append((ciFeature, "NO ADAPTATION"))

        t_PROMPT = f"You are a specialist reasoning agent, Based on the context provided on {refCaseType}, reason the adaptation of {txt_FT} of a similar {refCaseType} solution"

        return ciCausality, t_PROMPT

    def getAggregativeReasoning(self, cPS, cSS, modelspec: ModelSpec, context, refCaseType):
        # A 7.3, 193
        ciC, t_prompt = self.getComputeCaseCausality(cPS, cSS, modelspec, context, refCaseType)
        print(f"Sc 004.3.1 {ciC}")
        print(f"Sc 004.3.2 {t_prompt}")

        featureContextStr = ""
        for i in range(len(cSS.Wv)):
            if cSS.c[i]:
                ciFeature, ciContext = ciC[i]
                # cPS.Γ[i] = ciContext
                # cPS.Γi[i] = i
                # cPS.Γa[i] = ciContext
                cPS.Γc[i] = ciContext
                featureContextStr = (featureContextStr + ", " + ciFeature) if len(featureContextStr.strip()) == 0 else ciFeature

        # Overall Reasoning
        # CONTROL_PROMPT = f'You are an investigative agent. Based on the context provided, summarise the feature mapping reasonings of: {featureContextStr}; less than 100 words?'
        CONTROL_PROMPT = f'{t_prompt}; in less than 250 words?'
        #TODO pass in ModelOption
        # result = exec_llmserver_array(context=context, query=CONTROL_PROMPT, modelOption=1)
        print(f"Sc 004.3.2.1 X_TIKOS_MODEL {modelspec.X_TIKOS_MODEL}")
        # result = exec_llmserver_array(context=context, query=CONTROL_PROMPT, modelOption=modelspec.X_TIKOS_MODEL)
        result = ""

        reasoningContext = ""

        for objResult in result:
            if reasoningContext == "":
                reasoningContext = objResult
            else:
                reasoningContext = reasoningContext + ", " + objResult

        return (cPS, reasoningContext)

    # ###
    # # Reasoning Base
    # ###
    # # User getSimilarityMatchingWithin as base
    #
    # def superficialFeatureProbabilityAdstraction(self):
    #
    #     sqℚ = [[0] * len(self.inputs[0][0][0]) for _ in range(len(self.inputs[0][0]))]
    #
    #     # input to hidden
    #     slIH = self.generateSynapsesNodeGroupIH()
    #     if slIH is not None:
    #         atIH = np.sum([i for i in slIH], 0)
    #         sqℚ = np.add(sqℚ, atIH)
    #
    #     # hidden to hidden
    #     slHH = self.generateSynapsesNodeGroupHH()
    #     # print("slHH out: \n", slHH)
    #     if slHH is not None:
    #         atHH = np.sum([i for i in slHH], 0)
    #         sqℚ = np.add(sqℚ, atHH)
    #
    #     # hidden to output
    #     slHO = self.generateSynapsesNodeGroupHO()
    #     if slHO is not None:
    #         atHO = np.sum([i for i in slHO], 0)
    #         sqℚ = np.add(sqℚ, atHO)
    #
    #     return sqℚ
    #
    # def caseItemContitionProbability(self):
    #     sqℚ = self.superficialFeatureProbabilityAdstraction()
    #     sqμ = sqℚ / len(self.inputs[0][0])
    #     sqϴ = np.subtract(sqμ, (np.power(np.subtract(sqμ, sqℚ), 2) / len(self.inputs[0][0])))
    #     for i in range(len(self.inputs[0][0])):
    #         # print("range ", i)
    #         for j,v1 in enumerate(sqϴ[i]):
    #             if (sqϴ[i][j] <= sqℚ[i][j]):
    #                 sqϴ[i][j] = 1
    #             else:
    #                 sqϴ[i][j] = 0
    #     return sqϴ
    #
    # def caseCausationAbstration(self):
    #
    #     sqℚ = self.superficialFeatureProbabilityAdstraction()
    #
    #     sqϴ = self.caseItemContitionProbability()
    #
    #     cicФ = [[(0, 0.0, 0.0, 0.0, [])] * len(self.inputs[0][0][0]) for _ in range(len(self.inputs[0][0]))]
    #
    #     for kn in range(len(self.inputs[0][0])):
    #         for n in range(len(self.inputs[0][0][0])):
    #             cicФ[kn][n] = (n, 0.0, 0.0, 0.0, [])
    #
    #     # input to hidden
    #     slIH = self.generateSynapsesNodeGroupIH()
    #
    #     cicФmatrixIH = [[(0, 0.0)] * len(slIH) for _ in range(len(self.inputs[0][0]))]
    #
    #     if slIH is not None:
    #         for epoc in range(len(slIH[0])):
    #             for slng in range(len(slIH)):
    #                 cicФmatrixIH[epoc][slng] = (slIH[slng][epoc].argmax(), np.amax(slIH[slng][epoc]))
    #
    #     for k, v in enumerate(cicФmatrixIH):
    #         for i in v:
    #             tcalcAc = (cicФ[k][i[0]][1] + i[1])
    #             tcalcA = tcalcAc / len(cicФmatrixIH)
    #             tcalcS = ((cicФ[k][i[0]][2] + i[1]) / tcalcAc) / len(cicФmatrixIH)
    #             tcalcF = ((cicФ[k][i[0]][3] + (1.0 - i[1])) / tcalcAc) / len(cicФmatrixIH)
    #             if (cicФ[k][i[0]][4] is None):
    #                 cicФ[k][i[0]][4] = [i[1]]
    #             else:
    #                 cicФ[k][i[0]][4].append(i[1])
    #
    #             cicФ[k][i[0]] = (i[0], tcalcA, tcalcS, tcalcF, cicФ[k][i[0]][4])
    #
    #
    #     # hidden to hidden
    #     slHH = self.generateSynapsesNodeGroupHH()
    #     cicФmatrixHH = [[(0, 0.0)] * len(slHH) for _ in range(len(self.inputs[0][0])) for _ in range(len(self.hiddenN) - 1)]
    #
    #     if slHH is not None:
    #         for epoc in range(len(slHH[0])):
    #             for hl in range(len(self.hiddenN) - 1):
    #                 for slng in range(len(slHH)):
    #                     cicФmatrixHH[epoc][slng][hl] = (slHH[slng][epoc].argmax(), np.amax(slHH[slng][epoc]))
    #
    #     for k, hv in enumerate(cicФmatrixHH):
    #         for j, v in enumerate(hv):
    #             for i in v:
    #                 tcalcAc = (cicФ[k][i[0]][1] + i[1])
    #                 tcalcA = tcalcAc / len(hv)
    #                 tcalcS = ((cicФ[k][i[0]][2] + i[1]) / tcalcAc) / len(hv)
    #                 tcalcF = ((cicФ[k][i[0]][3] + (1.0 - i[1])) / tcalcAc) / len(hv)
    #                 if (cicФ[k][i[0]][4] is None):
    #                     cicФ[k][i[0]][4] = [i[1]]
    #                 else:
    #                     cicФ[k][i[0]][4].append(i[1])
    #                 cicФ[k][i[0]] = (i[0], tcalcA, tcalcS, tcalcF, cicФ[k][i[0]][4])
    #
    #
    #
    #     # hidden to output
    #     slHO = self.generateSynapsesNodeGroupHO()
    #     cicФmatrixHO = [[(0, 0.0)] * len(slHO) for _ in range(len(self.inputs[0][0]))]
    #
    #     if slHO is not None:
    #         for epoc in range(len(slHO[0])):
    #             for slng in range(len(slHO)):
    #                 cicФmatrixHO[epoc][slng] = (slHO[slng][epoc].argmax(), np.amax(slHO[slng][epoc]))
    #
    #     for k, v in enumerate(cicФmatrixHO):
    #         for i in v:
    #             tcalcAc = (cicФ[k][i[0]][1] + i[1])
    #             tcalcA = tcalcAc / len(cicФmatrixHO)
    #             tcalcS = ((cicФ[k][i[0]][2] + i[1]) / tcalcAc) / len(cicФmatrixHO)
    #             tcalcF = ((cicФ[k][i[0]][3] + (1.0 - i[1])) / tcalcAc) / len(cicФmatrixHO)
    #             if (cicФ[k][i[0]][4] is None):
    #                 cicФ[k][i[0]][4] = [i[1]]
    #             else:
    #                 cicФ[k][i[0]][4].append(i[1])
    #             cicФ[k][i[0]] = (i[0], tcalcA, tcalcS, tcalcF, cicФ[k][i[0]][4])
    #
    #     return cicФ
    #
    # def caseItemCausality(self, A = 0.0, S = 0.0, F = 0.0, l = 0):
    #
    #     τ2 = 1 if ((S >= F) and (S != 0.0) and (F != 0.0)) else 0
    #     τ2Str = "required information" if (τ2 == 1)  else "non required information"
    #     iStr = "validated to full network" if (l == 1) else "invalid in the full network"
    #     Ф = (τ2, τ2Str + " " + iStr + " with a relevence of:" + str(A))
    #
    #     return Ф
