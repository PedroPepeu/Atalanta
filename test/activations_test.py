from nn import activation

def test_relu():
    assert activation.relu(-1) == 0
    assert activation.relu(2) == 2

def test_sigmoid():
    assert activation.sigmoid(0) == 0.5

def test_heavyside():
    assert activation.heavyside(8) == 1
    assert activation.heavyside(-1) == 0
    assert activation.heavyside(0) == 1


