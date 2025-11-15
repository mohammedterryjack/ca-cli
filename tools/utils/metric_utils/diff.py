from numpy import array, array_equal

def first_differing_row(values1:list[list[int]], values2:list[list[int]]) -> int:
    mat1 = array(values1)
    mat2 = array(values2)
    i = 0 
    for i, (row1, row2) in enumerate(zip(mat1, mat2, strict=True)):
        if not array_equal(row1, row2):
            return i  
    return i 
