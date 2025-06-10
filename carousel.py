class DLinkedListNode:
    """
    A node in a doubly linked list.

    Attributes:
        data: The data stored in the node.
        next: A reference to the next node in the list.
        previous: A reference to the previous node in the list.
    """
    def __init__(self, initData, initNext, initPrevious):
        """
        Initializes a new node with data, next, and previous references.

        Args:
            initData: The data to store in the node.
            initNext: The next node in the list.
            initPrevious: The previous node in the list.
        """
        self.data = initData
        self.next = initNext
        self.previous = initPrevious

        if initPrevious is not None:
            initPrevious.next = self

        if initNext is not None:
            initNext.previous = self

    def getData(self):
        """
        Returns the data stored in the node.

        Returns:
            The data stored in the node.
        """
        return self.data

    def getNext(self):
        """
        Returns the next node in the list.

        Returns:
            The next node in the list.
        """
        return self.next

    def getPrevious(self):
        """
        Returns the previous node in the list.

        Returns:
            The previous node in the list.
        """
        return self.previous

    def setData(self, newData):	
        """
        Sets the data stored in the node.

        Args:
            newData: The new data to store in the node.
        """
        self.data = newData

    def setNext(self, newNext):
        """
        Sets the next node in the list.

        Args:
            newNext: The new next node.
        """
        self.next = newNext

    def setPrevious(self, newPrevious):
        """
        Sets the previous node in the list.

        Args:
            newPrevious: The new previous node.
        """
        self.previous = newPrevious

class Carousel:
    """
    A circular doubly linked list implementation representing a carousel.

    Attributes:
        head: The head node of the carousel.
        current: The current node in the carousel.
    """
    def __init__(self):
        """
        Initializes an empty carousel.
        """
        self.head = None
        self.current = None

    def add(self, data):
        """
        Adds a new node with the given data to the carousel.

        Args:
            data: The data to store in the new node.
        """
        temp = DLinkedListNode(data, None, None)
        if self.head is None:
            temp.setNext(temp)
            temp.setPrevious(temp)
            self.head = temp
        else:
            temp.setNext(self.current.getNext())
            temp.setPrevious(self.current)
            self.current.getNext().setPrevious(temp)
            self.current.setNext(temp)
        self.current = temp

    def getCurrentData(self):
        """
        Returns the data of the current node in the carousel.

        Returns:
            The data of the current node.
        """
        return self.current.getData()

    def moveNext(self):
        """
        Moves the current pointer to the next node in the carousel.
        """
        self.current = self.current.getNext()

    def movePrevious(self):
        """
        Moves the current pointer to the previous node in the carousel.
        """
        self.current = self.current.getPrevious()

    def __str__(self):
        """
        Returns a string representation of the carousel.

        Returns:
            A string representing the data in the carousel nodes.
        """
        s = '['
        i = 0
        current = self.head
        end = False
        while current is not None and not end:
            if i > 0:
                s = s + ','
            dataObject = current.getData()
            if dataObject is not None:
                s = s + f"{dataObject}"
                i += 1
            current = current.getNext()
            if current == self.head:
                end = True
        s = s + ']'
        return s