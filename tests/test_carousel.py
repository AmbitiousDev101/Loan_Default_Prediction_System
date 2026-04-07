import pytest
from carousel import Carousel, DLinkedListNode

def test_node_creation():
    """Test standard node initialization."""
    node = DLinkedListNode({"name": "Alice"}, None, None)
    assert node.getData()["name"] == "Alice"
    assert node.getNext() is None
    assert node.getPrevious() is None

def test_carousel_add_and_circularity():
    """Test that the carousel correctly maintains circularity."""
    c = Carousel()
    c.add({"id": 1})
    c.add({"id": 2})
    
    # Check current node
    assert c.getCurrentData()["id"] == 2
    
    # Check circularity (Next)
    c.moveNext()
    assert c.getCurrentData()["id"] == 1
    c.moveNext()
    assert c.getCurrentData()["id"] == 2
    
    # Check circularity (Previous)
    c.movePrevious()
    assert c.getCurrentData()["id"] == 1
    c.movePrevious()
    assert c.getCurrentData()["id"] == 2

def test_carousel_multiple_items():
    """Test carousel with more than 2 items."""
    c = Carousel()
    for i in range(1, 4):
        c.add({"id": i})
    
    # Initial is the last added (3)
    assert c.getCurrentData()["id"] == 3
    
    # Next should be 1 (Wraparound)
    c.moveNext()
    assert c.getCurrentData()["id"] == 1
    
    # Previous from 1 should be 3
    c.movePrevious()
    assert c.getCurrentData()["id"] == 3
    
    # Previous from 3 should be 2
    c.movePrevious()
    assert c.getCurrentData()["id"] == 2

def test_carousel_str_representation():
    """Test the string output of the carousel."""
    c = Carousel()
    c.add(1)
    c.add(2)
    s = str(c)
    assert "[1,2]" in s or "[2,1]" in s # Order depends on implementation of add()
