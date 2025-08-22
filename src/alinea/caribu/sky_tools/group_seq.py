class group_seq:
    """  Doc... """ 

    def __init__(self):
        pass


    def __call__(self, tab, indice):
        return (list(range(0,len(tab[indice][0])-1)),)
