
def distance (A, B):
    dx = A[0] - B[0]
    dy = A[1] - B[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def predict ( target , K):
    data = [( distance ( target , point), label )
            for point, label in zip( points , labels ) ]
    data.sort()
    count = {}
    for dist , label in data [: K]:
        count[ label ] = count.get( label , 0) + 1
    result = max(count, key=lambda label: count[label ])
    return result

def predict_proba( target , K):
    data = [( distance ( target , point), label )
            for point, label in zip( points , labels ) ]
    data. sort()
    count = {}
    for dist , label in data [: K]:
        count[ label ] = count.get( label , 0) + 1
    proba = {label: count[ label ] / K for label in count}
    return proba

if __name__ == "__main__":
    points = [(4, 1), (3, 3), (6, 1), (1, 4),
    (3, 5), (8, 2), (5, 6), (7, 4), (6, 6) ]
    #labels = [2, 0, 1, 0, 0, 2, 2, 1, 1]
    labels = [2, 9, 1, 9, 9, 2, 2, 1, 1]
    target = (5, 3)
    K = 5
    print ( 'Predicted label : ', predict ( target , K))
    print ( 'Predicted probabilities : ', predict_proba( target , K))
