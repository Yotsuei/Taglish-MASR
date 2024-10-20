best_value = 92.47 / 100
value = 75.24 / 100
governing_rank = 10

def compute_relative_diff () : 
    # relative_diff = (value - best_value) / value
    relative_diff = (best_value - value) / best_value

    return relative_diff

def compute_sub_rank () : 
    diff = compute_relative_diff ()
    sub_rank = governing_rank - diff

    print (f"\nRelative Difference : {diff}\nSubordiante Rank : {sub_rank}\n")

compute_sub_rank ()