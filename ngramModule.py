class NGram:
    def __init__(self, n : int) -> None:
        self.n = n

    def generateGrams(self, text : str) -> list:
        if len(text) == 0:
            return []
        if self.n > len(text):
            text += '#'* (self.n - len(text))
        grams : list = []
        for i in range(0,len(text)-self.n+1):
            grams.append(text[i:i+self.n])
        return grams
    
    def findSimilarity(self, text1: str, text2 : str) -> float:
        text1Grams : set = set(self.generateGrams(text1.lower()))
        if '#'*self.n in text1Grams:
            text1Grams.remove('#'*self.n)
        text2Grams : set = set(self.generateGrams(text2.lower()))

         
        return len(text1Grams.intersection(text2Grams)) / len(text1Grams.union(text2Grams)) 
